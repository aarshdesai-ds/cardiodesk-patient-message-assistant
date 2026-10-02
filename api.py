from fastapi import FastAPI, HTTPException
from chains import triage_chain, followup_chain, store
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

class FollowupResponse(BaseModel):
    answer: str
    sources: list[Source]


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
        result = followup_chain.invoke({"question":request.question, "topic":request.topic,"history":history})
        return result
    except Exception as e:
        raise HTTPException(status_code=502, detail = "The triage service is temporarily unavailable due to some error.")

