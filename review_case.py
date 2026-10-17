from cardio_graph import workflow
from storage import config_for
import sys
from uuid import uuid4
from langgraph.types import Command

def make_config():
    config = config_for(f"cli-{uuid4().hex[:6]}")
    config = {**config, "run_name": "cardiodesk-cli", "tags":['cardiodesk','v2','cli']}
    return config

def show_payload(payload):
    sources = ", ".join(payload['sources']) if payload['sources'] else "none"
    print(f"NURSE REVIEW ({payload['intent']}, urgency {payload['urgency']})")
    print(f"Patient: {payload['message']}")
    print(f"Summary: {payload['summary']}")
    print(f"Sources: {sources}")
    print(f"Draft reply:\n{payload['draft']}")

def ask_nurse():
    print("Type 'approve' to send as is, or type a corrected reply:")
    while True:
        answer = input("> ").strip()
        if answer == "":
            print("Please Type 'approve' to send as is, or type a corrected reply:")
            continue
        return answer

def run(message, auto_answer = None):
    config = make_config()
    result = workflow.invoke({"message": message}, config = config)
    while result.get("__interrupt__"):
        payload = result['__interrupt__'][0].value
        if auto_answer is None:
            show_payload(payload)
        answer = auto_answer or ask_nurse()
        result = workflow.invoke(Command(resume=answer), config=config)
    return result['report']

def print_report(report):
    if report['is_emergency']:
        print(f"EMERGENCY: fixed message sent at once, no review needed (triggered by {report['safety']['triggered_by']})")
    for key in ["path", "intent", "urgency", "is_emergency", "review_decision"]:
        print(f"{key:<16}{report[key]}")
    print(f"{'triggered_by':<16}{report['safety']['triggered_by']}") 
    print(f"reply:\n{report['draft_reply']}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
    else:
        message = input("Paste the patient message: ").strip()

    report = run(message)
    print("\n--- report ---")
    print_report(report)