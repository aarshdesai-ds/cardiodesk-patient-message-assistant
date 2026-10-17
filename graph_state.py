from typing import TypedDict, Literal
from schemas import MessageAnalysis

class CaseState(TypedDict):
    message: str
    analysis: MessageAnalysis
    summary: str
    safety: dict
    query: str
    filter_used: str
    context: str
    sources: list[dict]
    reply: str
    path: Literal['emergency', "symptom_reply", "medication_reply", "info_reply"]
    report: dict
    review_decision: Literal["approved", "edited", "not_needed"]

if __name__ == "__main__":
    from typing import get_args

    keys = CaseState.__annotations__
    expected = ["message", "analysis", "summary", "safety", "query", "filter_used",
                "context", "sources", "reply", "path", "report", "review_decision"]
    print("keys:", list(keys))
    print("ok  " if list(keys) == expected else "FAIL", "12 keys in order")
    print("ok  " if keys["analysis"] is MessageAnalysis else "FAIL", "analysis is MessageAnalysis")
    print("ok  " if set(get_args(keys["path"])) == {"emergency", "symptom_reply", "medication_reply", "info_reply"} else "FAIL",
          "path values")