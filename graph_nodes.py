from chains import analysis_chain, summary_chain, store, symptom_chain, info_chain, medication_chain, build_query, docs_to_context, docs_to_sources
from graph_state import CaseState
from safety import check_safety, EMERGENCY_MESSAGE

def analyze(state: CaseState):
    message = state['message']
    analysis = analysis_chain.invoke({"message":message})
    return {"analysis": analysis}

def summarize(state: CaseState):
    message = state['message']
    summary = summary_chain.invoke({"message": message})
    return {"summary": summary}

def check_safety_node(state: CaseState):
    message = state['message']
    analysis = state['analysis']
    safety = check_safety(message, analysis)
    return {"safety": safety}

def emergency_reply(state: CaseState):
    return {"reply": EMERGENCY_MESSAGE, "path": "emergency", "query": "", "filter_used": "none", "sources": []}

def retrieve(state: CaseState):
    query = build_query(state)
    intent = state['analysis'].intent
    if intent == "admin_request":
        docs = store.similarity_search(query, k = 4, filter={"doc_type":"clinic_policy"})
        filter_used = "clinic_policy"
    elif intent == "education_question":
        docs = store.similarity_search(query, k = 4, filter= {"doc_type": "education"})
        filter_used = "education"
    else:
        docs = store.similarity_search(query, k = 3, filter={"doc_type":"education"}) + store.similarity_search(query, k = 2, filter = {"doc_type": "clinic_policy"})
        filter_used = "education + clinic_policy"
    return {"query": query, "filter_used" : filter_used, "context": docs_to_context(docs), "sources": docs_to_sources(docs)}

def symptom_reply(state: CaseState):
    context = state['context']
    message = state['message']
    reply = symptom_chain.invoke({"message": message, "context": context})
    return {"reply": reply, "path": "symptom_reply"}

def medication_reply(state: CaseState):
    context = state['context']
    message = state['message']
    reply = medication_chain.invoke({"context": context, "message": message})
    return {"reply": reply, "path": "medication_reply"}

def info_reply(state: CaseState):
    context = state['context']
    message = state['message']
    reply = info_chain.invoke({"message": message, "context": context})
    return {"reply": reply, "path": "info_reply"}

def build_report(state: CaseState):
    message = state['message']
    analysis= state['analysis']
    summary = state['summary']
    safety = state['safety']
    query = state['query']
    filter_used = state['filter_used']
    sources = state['sources']
    path = state['path']

    patient_name = analysis.patient_name
    topic = analysis.topic
    intent = analysis.intent
    urgency = analysis.urgency
    red_flags = analysis.red_flags
    is_emergency = safety['is_emergency']
    draft_reply = state['reply']
    needs_review = True

    report = {"message": message, "summary": summary, "safety":safety, "path": path, "query": query, "filter_used": filter_used, "sources": sources, "patient_name": patient_name, "topic": topic, "intent": intent, "urgency": urgency, "red_flags": red_flags, "is_emergency": is_emergency, "draft_reply": draft_reply, "needs_review": needs_review}

    return {"report": report}

if __name__ == "__main__":
    import sys
    from schemas import MessageAnalysis

    print("--- free checks ---")
    fake = MessageAnalysis(topic="blood pressure", intent="symptom_report", red_flags=["pounding headache"],
                           urgency="urgent", patient_name="Raj")
    state = {"message": "My BP is 190/125 and I have a pounding headache. - Raj",
             "analysis": fake, "summary": "Patient reports BP 190/125 with a headache."}
    state.update(check_safety_node(state))
    print("ok  " if state["safety"]["is_emergency"] and state["safety"]["triggered_by"] == "rules" else "FAIL",
          "190/125 is an emergency by rules even when the model says urgent:", state["safety"]["bp_readings"])

    state.update(emergency_reply(state))
    print("ok  " if state["reply"] == EMERGENCY_MESSAGE and state["path"] == "emergency" else "FAIL", "fixed emergency reply")

    report = build_report(state)["report"]
    v1_keys = {"message", "patient_name", "topic", "intent", "urgency", "red_flags", "summary", "is_emergency",
               "safety", "path", "query", "filter_used", "sources", "draft_reply", "needs_review"}
    print("ok  " if set(report) == v1_keys else "FAIL", f"report keys match v1 ({len(report)})")

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        print("\n--- two real messages through the nodes (uses the model) ---")
        for message, reply_node in [
            ("My BP is 190/125 and I have a pounding headache. - Raj", None),
            ("Is it safe to take ibuprofen while I am on warfarin? - Meera", medication_reply),
        ]:
            state = {"message": message}
            for node in [analyze, summarize, check_safety_node]:
                state.update(node(state))
            if state["safety"]["is_emergency"]:
                state.update(emergency_reply(state))
            else:
                state.update(retrieve(state))
                state.update(reply_node(state))
            report = build_report(state)["report"]
            print(f"\n{message}")
            for key in ["intent", "urgency", "is_emergency", "path", "filter_used"]:
                print(f"  {key:<13}{report[key]}")
            print(f"  safety       {report['safety']['triggered_by']} {report['safety']['bp_readings']}")
            print(f"  sources      {[s['source_title'] for s in report['sources']]}")