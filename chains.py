from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from safety import check_safety, EMERGENCY_MESSAGE
from langchain_chroma import Chroma
from schemas import MessageAnalysis
from prompts import analysis_prompt, summary_prompt, medication_prompt, symptom_prompt, info_prompt, followup_prompt
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableBranch

load_dotenv()

store = Chroma(
    collection_name = "cardiodesk",
    embedding_function= OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="kb_db"
)
parser = StrOutputParser()

model = ChatOpenAI(model = "gpt-4o-mini", temperature=0)

analysis_chain = analysis_prompt | model.with_structured_output(MessageAnalysis)

summary_chain = summary_prompt | model | parser

symptom_chain = symptom_prompt | model | parser

medication_chain = medication_prompt | model | parser

info_chain = info_prompt | model | parser

analyze_step = RunnableParallel({
    "message": RunnableLambda(lambda x: x['message']),
    "analysis": analysis_chain,
    "summary": summary_chain
})

def add_safety(x):
    safety = check_safety(x['message'], x['analysis'])
    return {**x, "safety" : safety}

safety_step = RunnableLambda(add_safety)

def is_emergency(x):
    return x['safety']['is_emergency']

def emergency_response(x):
    return {"reply": EMERGENCY_MESSAGE,
            "sources": [],
            "path":"emergency",
            "query": "",
            "filter_used":"none"}

emergency_path = RunnableLambda(emergency_response)

def build_query(x):
    topic = x['analysis'].topic
    summary = x['summary']
    return f"{topic}. {summary}"

def docs_to_context(docs):
    contents = [doc.page_content for doc in docs]
    return "\n\n".join(contents)

def docs_to_sources(docs):
    sources = []
    for doc in docs:
        entry = {"source_title":doc.metadata['source_title'], "section": doc.metadata['section'], "url":doc.metadata['url']}
        if entry not in sources:
            sources.append(entry)
    return sources

def retrieve(x):
    query = build_query(x)
    intent = x['analysis'].intent
    if intent == "admin_request":
        docs = store.similarity_search(query = query , k = 4, filter={"doc_type":"clinic_policy"})
        filter_used = "clinic_policy"
    elif intent == "education_question":
        docs = store.similarity_search(query=query, k = 4, filter= {"doc_type":"education"})
        filter_used = "education"
    else:
        docs = store.similarity_search(query = query, k = 3, filter = {"doc_type":"education"}) + store.similarity_search(query = query, k=2, filter={"doc_type":"clinic_policy"})
        filter_used = "education + clinic_policy"
    context = docs_to_context(docs)
    sources = docs_to_sources(docs)
    return {**x, "query":query, "filter_used":filter_used, "context":context, "sources":sources}

retrieve_step = RunnableLambda(retrieve)

draft_step = RunnableParallel({
    "info": RunnablePassthrough(),
    "reply": RunnableBranch(
        (lambda x: x['analysis'].intent == "symptom_report", symptom_chain),
        (lambda x: x['analysis'].intent == "medication_question", medication_chain),
        info_chain
    )
})

def shape_response(x):
    reply = x['reply']
    sources = x['info']['sources']
    intent = x['info']['analysis'].intent
    if intent == "symptom_report":
        path = "symptom_reply"
    elif intent == "medication_question":
        path = "medication_reply"
    else:
        path = "info_reply"
    query = x['info']['query']
    filter_used = x['info']['filter_used']

    return {"reply" : reply, "sources":sources, "path": path, "query": query, "filter_used":filter_used}

normal_path = retrieve_step | draft_step | RunnableLambda(shape_response)

response_step = RunnableParallel({"data": RunnablePassthrough(),
                                  "response" : RunnableBranch(
    (lambda x: is_emergency(x), emergency_path),
    normal_path
)
})

def build_report(x):
    data = x['data']
    analysis = data['analysis']
    response = x['response']

    message = data['message']
    patient_name = analysis.patient_name
    topic = analysis.topic
    intent = analysis.intent
    urgency = analysis.urgency
    red_flags = analysis.red_flags
    summary = data['summary']
    safety = data['safety']
    path = response['path']
    query = response['query']
    filter_used = response['filter_used']
    sources = response['sources']
    draft_reply = response['reply']
    needs_review = True

    return {"message": message,
            "patient_name": patient_name,
            "topic": topic,
            "intent":intent,
            "urgency": urgency,
            "red_flags": red_flags,
            "summary": summary,
            "is_emergency": data['safety']['is_emergency'],
            "safety": safety,
            "path": path,
            "query": query,
            "filter_used": filter_used,
            "sources": sources,
            "draft_reply": draft_reply,
            "needs_review": needs_review}

report_step = RunnableLambda(build_report)

triage_chain = analyze_step | safety_step | response_step | report_step

def prepare(x):
    chat_history = []
    for entry in x['history']:
        if entry['role'] == "user":
            chat_history.append(HumanMessage(content = entry["content"]))
        elif entry['role'] == "assistant":
            chat_history.append(AIMessage(content = entry['content']))
        else:
            continue
    topic = x['topic']
    question = x['question']
    query = topic + " " + question

    docs = store.similarity_search(query, k = 4)

    context = docs_to_context(docs)
    sources = docs_to_sources(docs)

    return {"question": question, "chat_history":chat_history, "context":context, "sources":sources}

prepare_step = RunnableLambda(prepare)

answer_step = RunnableParallel({
    "answer": followup_prompt | model | parser,
    "sources": (lambda x: x['sources'])
})

followup_chain = prepare_step | answer_step

if __name__ == "__main__":
    import json

    # One triage call, to get a real draft to seed the chat with
    patient_message = "I'm travelling next week. How do I get a refill of my blood pressure tablets?"
    report = triage_chain.invoke({"message": patient_message})

    print("=" * 90)
    print("PATIENT MESSAGE:", patient_message)
    print("-" * 90)
    print("Topic:", report["topic"], "| Path:", report["path"])
    print("Draft reply:")
    print(report["draft_reply"])

    # The history, in the form the web page will send it
    history = [
        {"role": "user", "content": patient_message},
        {"role": "assistant", "content": report["draft_reply"]},
    ]

    questions = [
        "If the patient is travelling in 5 days, is it too late for a refill?",
        "What did the patient originally ask?",
        "Does the clinic offer home delivery of medicines?",
    ]

    print("\n" + "=" * 90)
    print("FOLLOW-UP CHAT")
    print("=" * 90)

    for question in questions:
        result = followup_chain.invoke({
            "question": question,
            "topic": report["topic"],
            "history": history,
        })

        print(f"\nNurse: {question}")
        print(f"Assistant: {result['answer']}")
        print("Sources consulted:")
        for source in result["sources"]:
            print(f"  - {source['source_title']} > {source['section']}")

        try:
            json.dumps(result)
            print("JSON-safe: True")
        except TypeError as error:
            print("JSON-safe: False ->", error)

        # What the page does after every turn
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})

    print("\nMessages in the history at the end:", len(history))