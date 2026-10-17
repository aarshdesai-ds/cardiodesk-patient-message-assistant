from langgraph.graph import StateGraph, START, END
from graph_state import CaseState
from graph_nodes import analyze, summarize, emergency_reply, medication_reply, info_reply, symptom_reply, retrieve, build_report, check_safety_node

def route_by_safety(state: CaseState):
    if state['safety']['is_emergency']:
        return "emergency"
    return "normal" 

def route_by_intent(state: CaseState):
    analysis = state['analysis']
    intent = analysis.intent
    if intent == "medication_question":
        return "medication"
    elif intent == "symptom_report":
        return "symptom"
    else:
        return "info"

def build_graph(checkpointer = None):
    graph = StateGraph(CaseState)

    graph.add_node("analyze", analyze)
    graph.add_node("summarize", summarize)
    graph.add_node("check_safety", check_safety_node)
    graph.add_node("emergency_reply", emergency_reply)
    graph.add_node("retrieve", retrieve)
    graph.add_node("symptom_reply", symptom_reply)
    graph.add_node("medication_reply", medication_reply)
    graph.add_node("info_reply", info_reply)
    graph.add_node("build_report", build_report)

    graph.add_edge(START, "analyze")
    graph.add_edge(START, "summarize")
    graph.add_edge("analyze", "check_safety")
    graph.add_edge("summarize", "check_safety")
    graph.add_edge("emergency_reply", "build_report")
    graph.add_edge("symptom_reply", "build_report")
    graph.add_edge("medication_reply", "build_report")
    graph.add_edge("info_reply", "build_report")

    graph.add_conditional_edges("check_safety", route_by_safety, {
        "emergency": "emergency_reply",
        "normal": "retrieve"
    })

    graph.add_conditional_edges("retrieve", route_by_intent, {
        "medication": "medication_reply",
        "symptom": "symptom_reply",
        "info": "info_reply"
    })

    graph.add_edge("build_report", END)

    return graph.compile(checkpointer=checkpointer)


workflow = build_graph()

if __name__ == "__main__":
    import sys

    print("--- nodes (free) ---")
    print(sorted(workflow.get_graph().nodes))

    print("\n--- routers (free) ---")
    from schemas import MessageAnalysis
    def fake(intent):
        return MessageAnalysis(topic="t", intent=intent, red_flags=[], urgency="routine", patient_name="")
    cases = [
        ("safety: emergency",   route_by_safety, {"safety": {"is_emergency": True}},  "emergency"),
        ("safety: normal",      route_by_safety, {"safety": {"is_emergency": False}}, "normal"),
        ("intent: symptom",     route_by_intent, {"analysis": fake("symptom_report")},      "symptom"),
        ("intent: medication",  route_by_intent, {"analysis": fake("medication_question")}, "medication"),
        ("intent: education",   route_by_intent, {"analysis": fake("education_question")},  "info"),
        ("intent: admin",       route_by_intent, {"analysis": fake("admin_request")},       "info"),
    ]
    for name, router, state, want in cases:
        got = router(state)
        print(f"{'ok  ' if got == want else 'FAIL'} {name:<20} -> {got}")

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        print("\n--- two messages through the graph (uses the model) ---")
        for message in ["My BP is 190/125 and I have a pounding headache. - Raj",
                        "Can I move my echo appointment to next week? - Sunil"]:
            config = {"run_name": "cardiodesk-graph-test", "tags": ["cardiodesk", "v2", "test"]}
            report = workflow.invoke({"message": message}, config=config)["report"]
            print(f"\n{message}")
            print(f"  path {report['path']}  filter {report['filter_used']}  emergency {report['is_emergency']}"
                  f"  ({report['safety']['triggered_by']})")