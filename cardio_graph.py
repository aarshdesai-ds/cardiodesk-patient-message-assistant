from langgraph.graph import StateGraph, START, END
from graph_state import CaseState
from storage import checkpointer
from graph_nodes import analyze, summarize, emergency_reply, medication_reply, info_reply, symptom_reply, retrieve, build_report, check_safety_node, nurse_review

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
    graph.add_node("nurse_review", nurse_review)
    graph.add_node("build_report", build_report)

    graph.add_edge(START, "analyze")
    graph.add_edge(START, "summarize")
    graph.add_edge("analyze", "check_safety")
    graph.add_edge("summarize", "check_safety")
    graph.add_edge("emergency_reply", "build_report")
    graph.add_edge("symptom_reply", "nurse_review")
    graph.add_edge("medication_reply", "nurse_review")
    graph.add_edge("info_reply", "nurse_review")
    graph.add_edge("nurse_review", "build_report")

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


workflow = build_graph(checkpointer=checkpointer)

if __name__ == "__main__":
    import sys
    from schemas import MessageAnalysis

    print("--- nodes (free) ---")
    print(sorted(workflow.get_graph().nodes))

    print("\n--- routers (free) ---")
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

    print("\n--- emergency never waits (free, reads the wiring) ---")
    edges = [(e.source, e.target) for e in workflow.get_graph().edges]
    emergency_exits = [t for s, t in edges if s == "emergency_reply"]
    reply_exits = {s: t for s, t in edges if s in ("symptom_reply", "medication_reply", "info_reply")}
    print("ok  " if emergency_exits == ["build_report"] else "FAIL", "emergency_reply ->", emergency_exits)
    print("ok  " if set(reply_exits.values()) == {"nurse_review"} and len(reply_exits) == 3 else "FAIL",
          "all 3 replies ->", reply_exits)

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        from uuid import uuid4
        from langgraph.types import Command
        from storage import config_for

        print("\n--- Raj (must not pause) and Meera (must pause), uses the model ---")
        for message in ["My BP is 190/125 and I have a pounding headache. - Raj",
                        "Is it safe to take ibuprofen while I am on warfarin? - Meera"]:
            config = {**config_for(f"test-{uuid4().hex[:6]}"),
                      "run_name": "cardiodesk-review-test", "tags": ["cardiodesk", "v2", "review"]}
            result = workflow.invoke({"message": message}, config=config)
            paused = bool(result.get("__interrupt__"))
            if paused:
                print(f"\n{message}\n  paused, draft: {result['__interrupt__'][0].value['draft'][:90]}...")
                result = workflow.invoke(Command(resume="approve"), config=config)
            else:
                print(f"\n{message}\n  did not pause")
            report = result["report"]
            print(f"  path {report['path']}  review_decision {report['review_decision']}")