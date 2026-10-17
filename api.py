from fastapi import FastAPI, HTTPException
from chains import triage_chain, store
from cardio_graph import workflow
from langgraph.types import Command
from agent import ask
from uuid import uuid4
from storage import config_for
from graph_nodes import make_review_payload
from pydantic import BaseModel, Field
from typing import Literal

class TriageRequest(BaseModel):

    message: str = Field(..., min_length=1, max_length=2000)

class ChatTurn(BaseModel):

    role: Literal['user', "assistant"]
    content: str

class FollowupRequest(BaseModel):

    question: str = Field(..., min_length=1, max_length=500)
    topic: str
    history: list[ChatTurn]

class Source(BaseModel):
    source_title: str
    section: str
    url: str

class SafetyInfo(BaseModel):

    is_emergency: bool
    bp_readings: list[str]
    matched_phrases: list[str]
    model_urgency: str
    triggered_by: str

class TriageResponse(BaseModel):
    message: str
    patient_name: str
    topic: str
    intent: str
    urgency: str
    red_flags: list[str]
    summary: str
    is_emergency: bool
    safety: SafetyInfo
    path: str
    query: str
    filter_used: str
    sources: list[Source]
    draft_reply: str
    needs_review: bool
    review_decision: str = "not_needed"

class FollowupResponse(BaseModel):
    answer: str
    sources: list[Source]
    tools_used: list[str]

class CaseRequest(BaseModel):
    message: str = Field(... ,min_length=1, max_length=2000)

class ReviewRequest(BaseModel):
    answer: str = Field(..., min_length=1, max_length=2000)

class ReviewPayload(BaseModel):
    message: str
    summary: str
    intent: str
    urgency: str
    draft: str
    sources: list[str]

class CaseResponse(BaseModel):
    case_id: str
    status: Literal['waiting_for_nurse', 'done']
    review: ReviewPayload | None = None
    response: TriageResponse | None = None

def case_response(case_id):
    config = config_for(case_id)
    snapshot = workflow.get_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="case not found")
    if snapshot.next == ("nurse_review",):
        review = make_review_payload(snapshot.values)
        return {"case_id":case_id, "status":"waiting_for_nurse", 'review': review, "response":None}
    else:
        if "report" in snapshot.values:
            return {"case_id": case_id, "status": "done", "review": None, "response":snapshot.values['report']}
        else:
            raise HTTPException(status_code=500, detail="case is in an unexpected state")

app = FastAPI(title = "CardioDesk API", description= 'This is a demo using synthetic data, and all the drafts are for clinician review. This demo is not intended to be a used as a medical device.')

@app.get("/health")
def health():
    return {"status":"ok", "chunks": len(store.get()['ids'])}

@app.post("/triage", response_model=TriageResponse)
def triage(request: TriageRequest):
    try: 
        report = triage_chain.invoke({"message":request.message})
        return report
    except Exception as e:
        raise HTTPException(status_code=502, detail = "The triage service is temporarily unavailable due to some error.")

@app.post("/followup", response_model=FollowupResponse)
def followup(request: FollowupRequest):
    history = []
    for chat in request.history:
        history.append(chat.model_dump())
    try:
        result = ask(request.question, request.topic,history)
        return result
    except Exception as e:
        raise HTTPException(status_code=502, detail = "The triage service is temporarily unavailable due to some error.")

@app.post("/cases", response_model=CaseResponse)
def create_case(request: CaseRequest):
    case_id = uuid4().hex[:8]
    try:
        workflow.invoke({"message": request.message}, config=config_for(case_id))
    except Exception as e:
        raise HTTPException(status_code=502, detail = "The case service is temporarily unavailable due to some error.")
    return case_response(case_id)

@app.get("/cases/{case_id}", response_model=CaseResponse)
def get_case(case_id):
    return case_response(case_id)

@app.post("/cases/{case_id}/review", response_model=CaseResponse)
def review_case(case_id, request: ReviewRequest):
    current = case_response(case_id)
    if current['status'] != "waiting_for_nurse":
        raise HTTPException(status_code=409, detail = "case is not waiting for review")
    try:
        workflow.invoke(Command(resume=request.answer.strip()), config = config_for(case_id))
    except Exception:
            raise HTTPException(status_code = 502, detail = "The review service is temporarily unavailable...")
    return case_response(case_id)


