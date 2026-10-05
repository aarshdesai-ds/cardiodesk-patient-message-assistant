"""CardioDesk - the nurse's screen.

This page talks to the CardioDesk API over HTTP. It does not import chains.py.

Run the API first:   uvicorn api:app --reload
Then this page:      streamlit run app.py
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("CARDIODESK_API_URL", "http://127.0.0.1:8000")
TIMEOUT_SECONDS = 60

SAMPLE_MESSAGES = {
    "Chest pressure": "I've had pressure in my chest for the last 30 minutes and my left arm hurts. Should I book an appointment? - Meera",
    "AFib and tiredness": "I was told I have AFib. I feel very tired all the time. Is that normal? - Arjun",
    "Warfarin and aspirin": "I take warfarin. Can I take aspirin for a headache?",
    "Reading of 135/85": "My blood pressure reading was 135/85. What category is that?",
    "Refill before travel": "I'm travelling next week. How do I get a refill of my blood pressure tablets?",
    "Ashwagandha": "Is it safe to take ashwagandha with my heart failure medicines?",
    "Reading of 190/125": "My blood pressure is 190/125 and I have a bad headache.",
}

PATH_LABELS = {
    "emergency": "Fixed emergency message (not generated)",
    "symptom_reply": "Symptom reply",
    "medication_reply": "Medication reply",
    "info_reply": "Information reply",
}

TOOL_LABELS = {
    "classify_bp": "Blood pressure category check",
    "check_red_flags": "Emergency phrase check",
    "search_knowledge_base[education]": "Knowledge base search (education pages)",
    "search_knowledge_base[clinic_policy]": "Knowledge base search (clinic policies)",
    "search_knowledge_base[both]": "Knowledge base search (education pages and clinic policies)",
}


def call_api(method, path, payload=None):
    """Call the API. Returns (data, error_message); exactly one of them is None."""
    try:
        if method == "GET":
            response = requests.get(API_URL + path, timeout=10)
        else:
            response = requests.post(API_URL + path, json=payload, timeout=TIMEOUT_SECONDS)
    except requests.exceptions.RequestException:
        return None, "Could not reach the CardioDesk API. Check that it is running."

    if response.status_code == 200:
        return response.json(), None
    if response.status_code == 422:
        return None, "The request was not accepted. Check that the text is not empty or too long."
    return None, "The service is temporarily unavailable. Please try again."


def show_sources(sources):
    if not sources:
        st.caption("No sources were consulted.")
        return
    for source in sources:
        label = f"{source['source_title']} › {source['section']}"
        if source["url"]:
            st.markdown(f"- [{label}]({source['url']})")
        else:
            st.markdown(f"- {label}")


def show_text(text):
    """Show text with its line breaks kept."""
    st.markdown(text.replace("\n", "  \n"))


def show_tools(tools_used):
    """Show which checks the assistant ran for an answer, without repeats."""
    if not tools_used:
        st.caption("Checks run: none")
        return
    labels = []
    for tool in tools_used:
        label = TOOL_LABELS.get(tool, tool)
        if label not in labels:
            labels.append(label)
    st.caption("Checks run: " + "; ".join(labels))


def use_sample(text):
    st.session_state["message_box"] = text


# ---------- Page setup ----------

st.set_page_config(page_title="CardioDesk", page_icon="🫀", layout="wide")

for key, default in [("message_box", ""), ("report", None), ("history", []), ("chat", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------- Sidebar ----------

with st.sidebar:
    st.title("CardioDesk")
    st.caption("Patient-message assistant for a heart clinic")

    st.warning(
        "Demo only. Uses synthetic messages - do not enter real patient information. "
        "Replies are drafts for a clinician to review. This is not medical advice."
    )

    health, health_error = call_api("GET", "/health")
    if health_error:
        st.error("API not reachable")
    elif health["chunks"] == 0:
        st.error("API connected, but the knowledge base is empty. Run ingest.py.")
    else:
        st.success(f"API connected - {health['chunks']} knowledge base chunks")

    st.subheader("Sample messages")
    for label, text in SAMPLE_MESSAGES.items():
        st.button(label, on_click=use_sample, args=(text,), use_container_width=True)

# ---------- Message box and triage ----------

st.header("Patient message")
st.text_area(
    "Patient message",
    key="message_box",
    height=120,
    max_chars=2000,
    placeholder="Paste a synthetic patient message, or pick a sample from the sidebar.",
    label_visibility="collapsed",
)

if st.button("Triage message", type="primary"):
    text = st.session_state["message_box"].strip()
    if not text:
        st.warning("Enter a message first.")
    else:
        with st.spinner("Analysing the message..."):
            report, error = call_api("POST", "/triage", {"message": text})
        if error:
            st.error(error)
        else:
            st.session_state["report"] = report
            st.session_state["history"] = [
                {"role": "user", "content": report["message"]},
                {"role": "assistant", "content": report["draft_reply"]},
            ]
            st.session_state["chat"] = []

# ---------- The report ----------

report = st.session_state["report"]

if report:
    st.divider()
    safety = report["safety"]

    if report["is_emergency"]:
        reasons = []
        if safety["matched_phrases"]:
            reasons.append("matched phrases: " + ", ".join(safety["matched_phrases"]))
        if safety.get("bp_readings"):
            reasons.append("blood pressure reading above a crisis limit: " + ", ".join(safety["bp_readings"]))
        if safety["model_urgency"] == "emergency":
            reasons.append("the model rated it an emergency")
        st.error("EMERGENCY - act on this message now. Flagged because " + "; ".join(reasons) + ".")

    col_patient, col_topic, col_intent, col_urgency = st.columns(4)
    col_patient.markdown(f"**Patient**  \n{report['patient_name'] or 'Unknown'}")
    col_topic.markdown(f"**Topic**  \n{report['topic']}")
    col_intent.markdown(f"**Intent**  \n{report['intent'].replace('_', ' ')}")
    col_urgency.markdown(f"**Urgency**  \n{report['urgency']}")

    st.markdown(f"**Summary:** {report['summary']}")
    if report["red_flags"]:
        st.markdown("**Warning signs noted:** " + ", ".join(report["red_flags"]))

    st.subheader("Draft reply")
    if report["is_emergency"]:
        st.caption("Pre-approved emergency message. This text is fixed, not generated.")
    else:
        st.caption("Draft only - a nurse must review this before anything is sent.")
    with st.container(border=True):
        show_text(report["draft_reply"])

    with st.expander(f"Sources consulted ({len(report['sources'])})"):
        show_sources(report["sources"])

    with st.expander("How this was produced"):
        st.markdown(f"**Path:** {PATH_LABELS.get(report['path'], report['path'])}")
        st.markdown(f"**Knowledge base searched:** {report['filter_used']}")
        st.markdown(f"**Search text:** {report['query'] or 'none'}")
        st.markdown(f"**Safety check triggered by:** {safety['triggered_by']}")
        st.markdown(f"**Phrases matched by the rules:** {', '.join(safety['matched_phrases']) or 'none'}")
        st.markdown(f"**Readings above a crisis limit:** {', '.join(safety.get('bp_readings', [])) or 'none'}")
        st.markdown(f"**Model's urgency rating:** {safety['model_urgency']}")

    # ---------- Follow-up chat ----------

    st.divider()
    st.subheader("Ask about this case")

    for turn in st.session_state["chat"]:
        with st.chat_message(turn["role"]):
            show_text(turn["content"])
            if turn["role"] == "assistant":
                show_tools(turn.get("tools_used", []))
            if turn.get("sources"):
                with st.expander("Sources consulted"):
                    show_sources(turn["sources"])

    question = st.chat_input("Ask a follow-up question about this case", max_chars=500)

    if question:
        with st.chat_message("user"):
            show_text(question)

        with st.spinner("Looking that up..."):
            result, error = call_api("POST", "/followup", {
                "question": question,
                "topic": report["topic"],
                "history": st.session_state["history"],
            })

        if error:
            st.error(error)
        else:
            tools_used = result.get("tools_used", [])
            with st.chat_message("assistant"):
                show_text(result["answer"])
                show_tools(tools_used)
                if result["sources"]:
                    with st.expander("Sources consulted"):
                        show_sources(result["sources"])

            st.session_state["history"].append({"role": "user", "content": question})
            st.session_state["history"].append({"role": "assistant", "content": result["answer"]})
            st.session_state["chat"].append({"role": "user", "content": question})
            st.session_state["chat"].append({"role": "assistant", "content": result["answer"],
                                             "sources": result["sources"], "tools_used": tools_used})
