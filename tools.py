from langchain_core.tools import tool
from typing import Literal
from chains import store, docs_to_context, docs_to_sources
from safety import classify_blood_pressure, find_red_flags, CRISIS_SYSTOLIC, CRISIS_DIASTOLIC,is_plausible_reading

SEARCH_NOTE = ("Note: these are the closest passages by meaning. They may not answer the question. "
               "Use only what the passages actually say, and if they do not cover the question, say so.")

BP_BASIS = ("Basis: the category table on the MedlinePlus high blood pressure page; "
            "crisis limits are set in the clinic's safety rules.")

RED_FLAG_BASIS = ("Basis: the clinic's fixed emergency phrase list in the safety rules. "
                  "This is not a knowledge base document.")


def search_kb(query, doc_type = "both"):
    if doc_type == "clinic_policy":
        docs = store.similarity_search(query = query, k = 4, filter= {"doc_type": "clinic_policy"})
    elif doc_type == "education":
        docs = store.similarity_search(query = query, k = 4, filter= {"doc_type": "education"})
    else:
        docs = store.similarity_search(query = query, k = 3, filter = {"doc_type":"education"}) + store.similarity_search(query = query, k=2, filter={"doc_type":"clinic_policy"})

    context = docs_to_context(docs)
    sources = docs_to_sources(docs)
    return context, sources

@tool
def search_knowledge_base(query: str, doc_type: Literal['education',"clinic_policy","both"] = "both") -> str:
    """
    Searches the clinic's knowledge base and returns the nearest passages, each starting with its source and section.
    Use this for any question about heart conditions, symptoms, medicines, or clinic rules. Answer only from what it returns.
    The results are the nearest matches by meaning, so check that a passage actually answers the question before using it.
    Args:
        query: a short search phrase with the key words from the question, for example "refill before travelling".
        doc_type: which part of the knowledge base to search.
            "education" for MedlinePlus pages: conditions, symptoms, causes, treatments, blood thinners, cholesterol, blood pressure.
            "clinic_policy" for Riverbend Heart Clinic's own rules: appointments, prescriptions and refills, the patient portal, when to get help.
            "both" when the question needs medical background and the clinic's guidance together.
    """

    context, sources = search_kb(query, doc_type)
    if not context:
        return "No matching passages were found."
    return context + "\n\n" + SEARCH_NOTE

@tool
def classify_bp(systolic: int, diastolic: int) -> str:
    """
    Classifies one blood pressure reading into its category (Normal, Elevated, High Blood Pressure Stage 1, High Blood Pressure Stage 2, Hypertensive Crisis), using the table from the MedlinePlus high blood pressure page. Also says whether the reading is above the clinic's crisis limits.
    Use this for every question that contains a blood pressure reading. Never classify a reading yourself.
    Args:
        systolic: the first, higher number of the reading, for example 162.
        diastolic: the second, lower number of the reading, for example 98.
    """
    if not is_plausible_reading(systolic, diastolic):
        return f"{systolic}/{diastolic} is not a valid blood pressure reading."
    category = classify_blood_pressure(systolic,diastolic)
    if systolic > CRISIS_SYSTOLIC or diastolic > CRISIS_DIASTOLIC:
        return f"{systolic}/{diastolic}: {category}. ABOVE a crisis limit (systolic above {CRISIS_SYSTOLIC} or diastolic above {CRISIS_DIASTOLIC}): flagged for urgent review.\n{BP_BASIS}"
    else:
        return f"{systolic}/{diastolic}: {category}. Not above the clinic's crisis limits.\n{BP_BASIS}"

@tool
def check_red_flags(text: str) -> str:
    """
    Checks a piece of text against the clinic's fixed list of emergency phrases, such as "chest pain" or "passed out", and returns the phrases it contains.
    Use this when the nurse asks whether a message or a quoted sentence contains emergency wording.
    It checks exact phrases only. Finding none does not mean the text is safe, so never tell the nurse there is no emergency based on this tool alone.
    Args:
        text: the exact text to check, copied as written.
    """
    flags = find_red_flags(text)
    if flags:
        return f"Emergency phrases found: {', '.join(flags)}.\n{RED_FLAG_BASIS}"
    return f"No emergency phrases found. This checks a fixed phrase list only, so it does not rule out an emergency.\n{RED_FLAG_BASIS}"


tools = [search_knowledge_base, classify_bp, check_red_flags]


if __name__ == "__main__":
    for t in tools:
        print(f"name: {t.name}")
        print(f"args: {t.args}")
        print()

    print("--- search_kb (plain function) ---")
    context, sources = search_kb("refill before travelling", "clinic_policy")
    print(context[:300])
    print(sources)

    print("\n--- search_knowledge_base ---")
    print(search_knowledge_base.invoke({"query": "symptoms of heart failure", "doc_type": "education"})[:400])
    print("...")
    print(search_knowledge_base.invoke({"query": "tired with AFib"})[-500:])

    print("\n--- classify_bp ---")
    for systolic, diastolic in [(162, 98), (185, 95), (190, 125), (118, 76), (500, 20)]:
        print(classify_bp.invoke({"systolic": systolic, "diastolic": diastolic}))

    print("\n--- check_red_flags ---")
    print(check_red_flags.invoke({"text": "I nearly passed out yesterday"}))
    print(check_red_flags.invoke({"text": "I feel a bit tired"}))