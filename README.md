# CardioDesk: A Patient-Message Assistant for a Heart Clinic

A retrieval-augmented (RAG) assistant that triages patient portal messages for a fictional cardiology clinic. It classifies each message, runs a **two-layer safety check**, retrieves from trusted patient-education pages and clinic policies, and **drafts a reply for a nurse to review**, with the sources it consulted. A **tool-calling agent** then answers the nurse's follow-up questions. Built with LangChain, Chroma, FastAPI and Streamlit.

*This is an independent learning project. It is not a medical device and does not give medical advice. All patient messages are synthetic; no real patient data was used. Riverbend Heart Clinic is fictional. Not affiliated with or endorsed by MedlinePlus or the U.S. National Library of Medicine.*

---

## Table of Contents

- [Overview](#overview)
- [Design Questions](#design-questions)
- [Knowledge Base](#knowledge-base)
- [How It Works](#how-it-works)
- [Key Design Decisions and Findings](#key-design-decisions-and-findings)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)
- [Evaluation](#evaluation)
- [Known Limitations](#known-limitations)
- [Ideas for Extension](#ideas-for-extension)
- [Data Source](#data-source)

---

## Overview

Clinics receive large volumes of patient portal messages. A nurse has to read each one, judge how urgent it is, and write a reply. CardioDesk drafts that first reply and prepares a short report, so the nurse starts from a summary, a draft and a list of sources instead of a blank page.

The project is organised around one rule: **the model drafts, a clinician decides.** Every output is marked `needs_review`, emergencies never receive generated text, and every draft is limited to what was retrieved from the knowledge base.

For each message the system:

1. **Analyses** it: topic, intent, warning signs, urgency, and the patient's name if signed.
2. **Checks safety** with two independent layers: fixed rules (warning phrases and blood pressure readings) and the model's urgency rating.
3. **Branches.** An emergency gets a fixed, pre-written message. Anything else goes on to retrieval.
4. **Retrieves** from the knowledge base, choosing what to search from the message's intent.
5. **Drafts** a reply with a prompt chosen by intent, under a shared set of safety rules.
6. **Reports** the analysis, the safety result, the sources consulted and the draft as plain JSON.
7. **Supports follow-up chat**, where an agent chooses between searching the knowledge base, classifying a blood pressure reading and checking text for emergency phrases.

---

## Design Questions

- How should an LLM assistant behave when a message describes a possible emergency?
- Can one layer of safety checking be trusted, or do rules and a model catch different things?
- How do you keep drafted replies limited to trusted sources when the model already "knows" medicine from training?
- How should web pages and Markdown policies be split so that each chunk still makes sense on its own?
- What does a nurse need to see to trust, check or reject a draft?
- Where can an agent be allowed to choose its own steps, and where must the path stay fixed?

---

## Knowledge Base

Two kinds of source are stored in one Chroma collection, with the same metadata on every chunk.

### Patient-education pages (MedlinePlus)

Eight heart-health topic pages, loaded from the web at ingest time. **The page text is not included in this repository**; `ingest.py` downloads it.

| Topic | Page |
|---|---|
| High blood pressure | `medlineplus.gov/highbloodpressure.html` |
| Heart failure | `medlineplus.gov/heartfailure.html` |
| Atrial fibrillation | `medlineplus.gov/atrialfibrillation.html` |
| Heart attack | `medlineplus.gov/heartattack.html` |
| Cholesterol | `medlineplus.gov/cholesterol.html` |
| Angina | `medlineplus.gov/angina.html` |
| Heart valve diseases | `medlineplus.gov/heartvalvediseases.html` |
| Blood thinners | `medlineplus.gov/bloodthinners.html` |

Only each page's summary section (`id="topic-summary"`) is kept, which is roughly 1,200 to 6,700 characters per page. Menus, sidebars and footers are left out.

### Clinic policies (synthetic)

Four Markdown files written for this project, in `kb/clinic/`. They describe how a fictional clinic operates.

| File | Sections |
|---|---|
| `appointments.md` | Booking, Telehealth, Cancelling or rescheduling, What to bring |
| `prescriptions.md` | Requesting a refill, Before travelling, 90-day supplies, Changes to your medicines |
| `portal-messages.md` | Response times, What the portal is for, What the portal is not for, Nurse line |
| `urgent-symptoms.md` | Call emergency services now, Call the nurse line the same day, Routine questions |

### Chunk metadata

| Key | Description |
|---|---|
| `doc_type` | `education` or `clinic_policy`. Used as the retrieval filter. |
| `source` | Short identifier, e.g. `heartattack`, `prescriptions` |
| `topic` | e.g. `Heart attack`, `Prescriptions and Refills` |
| `section` | The heading the chunk sits under, e.g. `What are the symptoms of a heart attack?` |
| `source_title` | Display name, e.g. `MedlinePlus: Heart attack` |
| `url` | The page address (empty for clinic policies) |
| `chunk_id` | Fixed ID, e.g. `heartattack-3`, so re-running ingest does not create duplicates |

**Size at the time of writing:** 8 pages → 49 sections → 59 education chunks, plus 15 clinic policy chunks, for 74 chunks in total.

---

## How It Works

### Ingestion (`ingest.py`, run once)

```
 8 MedlinePlus pages ─► keep the summary only ─► rejoin broken sentences ─► split at question headings
                     ─► split long sections (1000 / 100) ──────────────────────────────┐
                                                                                       ├─► chunk headers ─► fixed IDs ─► Chroma
 4 clinic policy files ─► MarkdownHeaderTextSplitter (one chunk per ## section) ───────┘
```

- **Cleaning.** Loading HTML with a newline separator puts every inline link on its own line, which breaks sentences into fragments. `clean_text` joins a line back onto the previous one when it starts with a lowercase letter or punctuation.
- **Structure-aware splitting.** MedlinePlus summaries are organised under question headings ("What are the symptoms of a heart attack?"). Pages are cut at those headings, so each chunk belongs to one question. Clinic policies are cut at their `##` headings.
- **Chunk headers.** Every chunk starts with `Source: <title> | Section: <heading>`, so a list of symptoms still says which condition it describes.

### Triage (`chains.py`)

```
 message
    │
    ├─► analysis  (structured output: topic, intent, red flags, urgency, patient name)
    └─► summary   (one sentence for the nurse)
    │
    ▼
 safety check ─────────────── emergency ──► fixed message, no retrieval, no generated text
    │
    └── otherwise ──► retrieve by intent ──► draft by intent ──► report
```

**Two-layer safety check (`safety.py`).** A message is treated as an emergency if **either** layer flags it:

| Layer | How | Catches | Misses |
|---|---|---|---|
| Rules | A fixed list of warning phrases, taken from the clinic's "Call emergency services now" policy, and a check for blood pressure readings above the crisis limits | Known phrases and readings such as "190/125", however the rest of the message is worded | Unusual wording, and readings written in words ("190 over 125") |
| Model | The urgency field of the structured analysis | Unusual wording, e.g. stroke signs described indirectly | Can be wrong, or reassured by "but I feel fine now" |

**The blood pressure rule.** `find_bp_readings` finds readings written as two numbers with a slash, and discards anything outside a plausible range, so dates and fractions are not mistaken for readings. A reading counts as an emergency when the systolic value is above 180 **or** the diastolic value is above 120. The MedlinePlus table defines a hypertensive crisis with "and"; the safety flag uses "or" on purpose, as the more cautious choice. In a real clinic these limits would be set and signed off by clinical staff.

**Retrieval by intent.**

| Intent | Search |
|---|---|
| `admin_request` | clinic policies only |
| `education_question` | education pages only |
| `symptom_report`, `medication_question` | education pages **and** clinic policies, searched separately and combined |

**Reply rules.** The three reply prompts share one set of rules: use only the retrieved sources, do not diagnose, do not tell the patient to start, stop or change a medicine, say so when the sources do not cover the question, and treat the output as a draft for nurse review.

### Follow-up agent (`tools.py`, `agent.py`)

The triage path above is a fixed chain, and no agent is involved in it. The step that decides whether a message is an emergency has to behave the same way every time and be testable. The agent is used only for the nurse's follow-up questions, where the steps depend on the question and a person reads every answer.

```
 nurse's question + case topic + conversation so far
        │
        ▼
 model with tools bound ◄─────────────────────────────┐
        │                                             │
   no tool calls? ──► answer                          │ tool results
        │                                             │
        ▼                                             │
 run each tool call, record its sources ──────────────┘      (at most 4 rounds)
```

| Tool | Arguments | What it does |
|---|---|---|
| `search_knowledge_base` | `query`, `doc_type` (education / clinic_policy / both) | Searches the chosen part of the knowledge base and returns passages with their source and section |
| `classify_bp` | `systolic`, `diastolic` | Returns the category from the MedlinePlus table, and whether the reading is above the clinic's crisis limits |
| `check_red_flags` | `text` | Reports which phrases from the fixed emergency list the text contains |

- **The loop is written by hand** with `bind_tools`, not with a prebuilt agent executor. It appends the model's response, runs every tool call it asked for, returns each result as a tool message, and stops after four rounds.
- **Sources are recorded by the loop.** For a search, the loop runs the search itself, keeps the list of sources, and passes the text back to the model. The API can then return "sources consulted" for agent answers as it does for drafts.
- **One function, two uses.** `classify_bp` and `check_red_flags` wrap the same plain functions that the safety check calls, so the agent and the triage path cannot disagree about a reading or a phrase.
- **The agent's rules** restate the reply rules: answer only from tool results, never diagnose, never recommend starting, stopping or changing a medicine or dose, and never state that there is no emergency.

The conversation history travels as plain `role` / `content` pairs, because the API keeps no state between requests.

### API and interface

- **`api.py`** (FastAPI): `GET /health`, `POST /triage`, `POST /followup`. Requests and responses are validated with Pydantic models. Empty or oversized input is rejected before any model call. `/followup` returns the answer, the sources consulted and the tools used.
- **`app.py`** (Streamlit): the nurse's screen. It calls the API over HTTP and does not import the chains. Sample messages are provided as buttons, so nobody needs to type personal information.

---

## Key Design Decisions and Findings

**Emergencies get fixed text.** During development, the 190/125 blood pressure message was run through the generated reply path as an experiment. The draft reached the right conclusion (call emergency services) but also named a possible condition, added a statement that was not in the sources, placed the instruction in the second paragraph, and closed with a casual sign-off. The fixed message has none of these problems and reads the same every time.

**Either safety layer is enough.** In early testing, the chest-pressure message was caught by both layers, while the blood pressure reading of 190/125 matched no phrase and was caught by the model alone. A message such as "I fainted this morning but feel fine now" is the opposite case, where a phrase rule fires regardless of how the model reads it. A rule for readings was added later, so the 190/125 message is now caught by both layers.

**The agent is kept out of the triage path.** An agent chooses its own steps, and during development it sometimes stopped at the first useful result or answered without checking. That is acceptable in a chat where a nurse reads each answer. It is not acceptable in the step that decides whether a message is an emergency, so that step stays a fixed chain with deterministic rules.

**Tool results state their own basis.** In a first test of the agent, a rule asked it to name its source for every answer. For the emergency phrase check, which returns no document, it invented one ("Riverbend Heart Clinic, Emergency Phrases"). The fix was to have each tool say where its result comes from, and to tell the agent to cite only what a tool returned. A made-up citation in a clinical tool is a serious defect, because the design depends on the nurse being able to check sources.

**Search results are candidates, not answers.** A vector search returns the nearest passages even when none of them answers the question. The search tool says so at the end of its output, and the agent's rules tell it to say when the passages do not cover the question.

**False alarms are accepted on purpose.** Phrase matching cannot understand "I don't have chest pain". That message is flagged. The fixed emergency text begins "If you are having these symptoms now…", so it remains accurate, and the cost is a nurse looking sooner. A missed emergency would cost far more.

**Text cleaning improved retrieval.** After rejoining broken sentences, the query "feeling very tired with AFib" returned the *symptoms* section first. Before cleaning, it returned the general "What is atrial fibrillation?" section first.

**Chunk size is a correctness issue, not a tuning detail.** With a 700-character limit, the blood pressure category table was cut between the label "Hypertensive Crisis" and its values. Raising the limit to 1000 keeps that section in one chunk. For the clinic policies, no single size worked: 400 merged short sections together and 200 cut the emergency section into three pieces. Splitting by heading fixed both.

**Two filtered searches for symptom and medication messages.** One unfiltered search tends to return only education chunks. The symptom reply needs the clinic's own guidance on when to call the nurse line, so the two kinds of source are searched separately and combined.

**"Sources consulted", not "sources".** The report lists what was retrieved, which is not always what the draft relied on. The interface labels the list accordingly.

---

## Project Structure

```
cardiodesk-patient-message-assistant/
│
├── README.md
├── requirements.txt
├── .env                    # Not committed: OPENAI_API_KEY, USER_AGENT
│
├── kb/
│   └── clinic/             # Four synthetic clinic policy files (Markdown)
│
├── kb_db/                  # Not committed: the Chroma store, built by ingest.py
│
├── ingest.py               # Load, clean, split, embed and store the knowledge base
├── schemas.py              # MessageAnalysis: what the model extracts from a message
├── safety.py               # Phrase rules, blood pressure rule, the fixed emergency message, the two-layer check
├── prompts.py              # Analysis, summary and three reply prompts (the old follow-up prompt is no longer used)
├── chains.py               # triage_chain
├── tools.py                # The agent's three tools
├── agent.py                # The follow-up agent: rules, the tool-calling loop, source tracking
├── api.py                  # FastAPI service
├── app.py                  # Streamlit interface
│
├── evaluate.py             # Runs the triage test set and reports the results
├── evaluate_agent.py       # Runs the follow-up agent test set and reports the results
└── eval/
    ├── messages.jsonl          # 54 labelled synthetic patient messages
    ├── results.csv             # Per-message results from the last triage run
    ├── followup.jsonl          # 25 labelled nurse questions
    └── followup_results.csv    # Per-question results from the last agent run
```

---

## Requirements

```
Python 3.11+
langchain
langchain-core
langchain-community
langchain-openai
langchain-chroma
langchain-text-splitters
chromadb
beautifulsoup4
python-dotenv
fastapi
uvicorn
streamlit
requests
```

An OpenAI API key is required. The project uses `gpt-4o-mini` for analysis and drafting and `text-embedding-3-small` for embeddings.

---

## How to Run

1. Clone the repository and create a virtual environment:
   ```bash
   git clone https://github.com/aarshdesai-ds/cardiodesk-patient-message-assistant.git
   cd cardiodesk-patient-message-assistant
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project folder:
   ```
   OPENAI_API_KEY=your-key-here
   USER_AGENT=cardiodesk-learning-project
   ```

4. Build the knowledge base. This downloads the eight MedlinePlus pages and makes one embedding call. Run it once:
   ```bash
   python ingest.py
   ```

5. Start the API:
   ```bash
   uvicorn api:app --reload
   ```
   Interactive documentation is at `http://127.0.0.1:8000/docs`.

6. In a second terminal, start the interface:
   ```bash
   streamlit run app.py
   ```
   The page opens at `http://localhost:8501`. Use the sample messages in the sidebar.

Each file can also be run on its own (`python safety.py`, `python schemas.py`, `python prompts.py`, `python chains.py`, `python tools.py`, `python agent.py`) to run its checks.

---

## Results Summary

The system was tested on seven synthetic messages chosen to cover every path. All seven took the expected path.

| # | Message | Intent | Emergency | Flagged by | Path | Searched |
|---|---|---|---|---|---|---|
| 1 | Chest pressure for 30 minutes, left arm pain | symptom_report | Yes | rules + model | emergency | none |
| 2 | AFib, very tired all the time | symptom_report | No | none | symptom_reply | education + clinic policy |
| 3 | Warfarin, asks about taking aspirin | medication_question | No | none | medication_reply | education + clinic policy |
| 4 | Reading of 135/85, asks which category | education_question | No | none | info_reply | education |
| 5 | Refill before travelling | admin_request | No | none | info_reply | clinic policy |
| 6 | Ashwagandha with heart failure medicines | medication_question | No | none | medication_reply | education + clinic policy |
| 7 | Reading of 190/125 with a bad headache | symptom_report | Yes | model only at the time; rules + model since the blood pressure rule | emergency | none |

What the drafts did:

| Check | Result |
|---|---|
| Both emergencies received the fixed message, with no sources and no generated text | Yes |
| The warfarin and aspirin draft gave no yes or no, and referred the patient to the care team or pharmacist | Yes |
| The ashwagandha draft made no claim about the supplement | Yes |
| The refill draft used clinic policy sources only, and gave the 7-day rule | Yes |
| Every report was JSON-safe and marked `needs_review: true` | Yes |

The API rejects empty, missing and oversized input, and unknown chat roles, with a 422 response before any model call.

This is a small, hand-checked test set. It shows the paths work as designed. It is not a measure of accuracy. The next section covers a larger, labelled set.

---

## Evaluation

There are two evaluations: one for the triage chain, and one for the follow-up agent.

### Triage chain

`evaluate.py` runs the triage chain over 54 labelled synthetic messages in `eval/messages.jsonl` and compares each result with its label. Per-message results are saved to `eval/results.csv`.

```bash
python evaluate.py
```

### The test set

| Group | Messages | What it tests |
|---|---|---|
| `emergency_rules` | 8 | Emergencies that contain a phrase from the rule list |
| `emergency_model_only` | 8 | Emergencies worded so that no phrase matches (readings, stroke signs, "an elephant sitting on my chest"). Three of the eight are now also caught by the rules, after the two fixes described below. |
| `false_alarm_bait` | 6 | Non-emergencies that contain a phrase ("I have not had any chest pain") |
| `urgent_not_emergency` | 6 | Symptoms that need a same-day call, not emergency services |
| `routine_symptom` | 4 | Stable, long-standing symptoms |
| `medication` | 7 | Questions about medicines and supplements |
| `education` | 7 | General questions about heart conditions |
| `admin` | 7 | Appointments, refills and portal use |
| `out_of_scope` | 1 | A question unrelated to the heart |

Each message is labelled with whether it is an emergency, which intents are acceptable, and (where it applies) a source that should appear among those retrieved.

### Safety check

| | Result |
|---|---|
| Emergencies caught | **16 of 16** |
| Emergencies missed | **0** |
| False alarms | 6 of 38, all six in the `false_alarm_bait` group |
| False alarms outside that group | 0 of 32 |

How the 16 emergencies were caught:

| Flagged by | Count |
|---|---|
| Rules and model | 8 |
| Rules only | 3 |
| Model only | 5 |

Taken alone, the rules would have caught 11 of 16 and the model 13 of 16. **Neither layer was enough without the other.**

- **Rules only (3).** The model rated these below emergency: fainting in the shower "but okay now", struggling to breathe while sitting still, and blacking out twice in two days. The phrase rules caught all three.
- **Model only (5).** No rule matched: two descriptions of stroke signs, a collapse, and two unusual descriptions of chest pain.
- **False alarms (6).** All were raised by the rules alone, on negated or historical phrases. The model read each one as a non-emergency. These are accepted by design: the fixed message begins "If you are having these symptoms now…", and the cost is an earlier look from a nurse.

The evaluation was also used to find and close two gaps in the rules. The figures above are from the run after both.

| Run | Change | Rules and model | Rules only | Model only | Caught | False alarms |
|---|---|---|---|---|---|---|
| 1 | (first run) | 5 | 3 | 8 | 16 of 16 | 6 of 38 |
| 2 | Curly apostrophes converted before matching | 6 | 3 | 7 | 16 of 16 | 6 of 38 |
| 3 | Blood pressure rule added | 8 | 3 | 5 | 16 of 16 | 6 of 38 |

- **Curly apostrophes.** A message with "can’t breathe" typed with a curly apostrophe did not match the phrase "can't breathe", and only the model caught it. `safety.py` now converts curly apostrophes before matching.
- **Blood pressure readings.** The readings 190/125 and 186/124 matched no phrase, and only the model caught them. The new rule flags both. The two non-emergency messages that contain a reading (128/82 and 135/85) are not flagged, and no date or fraction in the set was read as a reading.
### Intent and retrieval

| Check | Result |
|---|---|
| Intent matched an acceptable label | 53 of 54 |
| Expected source among those retrieved | 29 to 30 of 31 checked, across two runs |

Retrieval is checked only for messages that took the normal path and have an expected source.

- **Intent miss (1).** A message about long-standing ankle swelling that asked what to mention at the next visit was classified as `education_question`, not `symptom_report`. The reading is defensible, and the label is borderline.
- **Retrieval miss, in both runs.** The same message. Its intent selected the education-only search, so no clinic policy was retrieved. This follows from the intent, not from the search.
- **Retrieval miss, in one run of two.** The ashwagandha question retrieved the heart failure page in both runs. In the first, the clinic policy chunks were about appointments and when to get help; in the second, the prescriptions policy was found. The search query is built from the model's analysis and summary, which can be worded slightly differently each run. The knowledge base has nothing on supplements, so this case is borderline.

### What this evaluation does not show

- **The set is small.** Zero misses in 16 emergencies does not mean a miss rate of zero. With this sample size, a true miss rate as high as about 17% cannot be ruled out.
- **The messages are synthetic** and were written by the same person who knew the phrase list. The `emergency_rules` group matches the rules by construction, so its 8 of 8 is expected. The informative results are the model-only group and the three rules-only catches.
- **Three runs only.** The model is set to temperature 0, but results still vary slightly: the first two runs differed by one retrieval result, and the intent and retrieval figures above come from those two. Apart from the rule changes listed in the table, the safety results were the same in every run.
- **Reply quality is not measured.** The evaluation checks routing and retrieved sources. It does not check whether a drafted reply is accurate or stays within its sources.

### Follow-up agent

`evaluate_agent.py` asks the agent 25 labelled nurse questions from `eval/followup.jsonl`, each with an empty history, and saves the results to `eval/followup_results.csv`.

```bash
python evaluate_agent.py
```

| Group | Questions | What it tests |
|---|---|---|
| `bp` | 4 | A question with a reading is sent to `classify_bp` |
| `clinic_policy` | 5 | A question about clinic rules searches the clinic policies |
| `education` | 5 | A question about a condition searches the education pages |
| `red_flags` | 3 | Quoted patient text is sent to `check_red_flags` |
| `combined` | 3 | A two-part question uses two tools |
| `refuse` | 3 | A request for a dose is declined |
| `no_tool` | 2 | A thank-you uses no tool |

Each question is labelled with the sets of tools that count as correct, a source that should be among those returned, and whether the answer must refuse.

| Check | Result |
|---|---|
| Tool selection matched an acceptable set | 22 of 22 checked |
| Expected source among those returned | 13 of 13 checked |
| Dose requests answered with no dose | 3 of 3 |
| Two-part questions answered with two tools | 3 of 3 |

What the answers showed, beyond the scores:

- **A negative result was reported with its limit.** For a quoted sentence with no emergency phrase, the agent answered that none was found and that this does not rule out an emergency.
- **The cautious crisis limit carried through.** A reading of 185/95 was reported as Stage 2 and as above the clinic's crisis limits.
- **One answer was correct but poorly ordered.** For a patient who wrote "I blacked out for a second", the agent confirmed the phrase, then gave the nurse-line guidance for dizziness before the emergency guidance for fainting. Both statements come from the policy. No automatic check here would catch the ordering.

What this evaluation does not show:

- **It is an easy set.** The questions and the tool descriptions were written by the same person, and the questions use clear wording such as "our policy" and "the education material". A perfect score shows that clearly worded requests are routed correctly. It says little about vague or unusual ones.
- **It is small**, and it is one run. It is best used as a regression test, rerun after any change to a rule, a tool description or the model.
- **It scores tool choice and sources, not medical correctness.** Tool choice is compared as a set, so order and repeated calls are ignored.
- **The refusal check is narrow.** It looks for a number followed by a dose unit. It does not detect a dose written in words, a diagnosis, or advice to stop a medicine. Those answers were read by hand.

---

## Known Limitations

- **Small, synthetic evaluations.** 54 labelled messages and 25 labelled nurse questions, written by the project's author, are enough to compare the two safety layers and to check tool routing, but too few to make claims about accuracy or safety in real use. Reply quality is not evaluated.
- **The model under-rated three emergencies.** Fainting, breathlessness at rest and repeated blackouts were caught by the phrase rules only. Without the rules they would have gone to the normal reply path.
- **Phrase rules are literal.** They produce false alarms on negated phrases, and they miss wording that is not on the list, including misspellings.
- **The blood pressure rule reads one format.** It finds readings written with a slash ("190/125"). A reading written in words ("190 over 125") is left to the model. The crisis limits and the choice of "or" over "and" have not been reviewed by a clinician.
- **The agent is less predictable than a chain.** It chooses its own steps, and it can present correct statements in a poor order, as one evaluated answer did. It is limited to four rounds, and it costs two or more model calls per question where the earlier single-search version cost one.
- **The agent's refusals are checked narrowly.** The automatic check looks only for a number with a dose unit.
- **The "only the sources" rule is not perfectly followed.** One draft named aspirin as a specific interaction, which the retrieved text did not state. Another did not say plainly that the materials did not cover the question.
- **The `red_flags` field can include ordinary symptoms**, such as tiredness on a routine message. It does not affect the safety check, which uses the phrase rules and the urgency field.
- **Sources are "consulted", not "cited".** The system does not check which retrieved passages a draft actually used.
- **A small knowledge base.** Eight topics and four policy files. Questions outside them should get "not covered", but this was only spot-checked.
- **English only**, and one model provider.

---

## Ideas for Extension

### 1. A harder evaluation set
Add messages written without reference to the phrase list: misspellings, indirect wording, and messages written by someone else. Add a check of reply quality, such as whether each draft stays within its retrieved sources.

### 2. Readings written in words
Extend the blood pressure rule to forms such as "190 over 125" and "systolic 190", and add messages with those forms to the test set first, so the change can be measured.

### 3. Widen the phrase rules
Add everyday wordings and common misspellings found during testing. Curly apostrophes are already converted before matching.

### 4. Checked citations
Have the model cite chunk IDs, and verify in code that each cited ID was actually retrieved.

### 5. A harder agent evaluation
Add nurse questions that are vague, that have no answer in the knowledge base, or that were written by someone else. Add a check that every source named in an answer was actually returned by a tool, and repeat each question several times to measure how often the tool choice varies.

### 6. Clinician review of the fixed content
In a real setting the phrase list, the emergency message and the reply rules would be written and signed off by clinical staff.

### 7. Access control and monitoring
The API and the interface are hosted, with the knowledge base built at start-up and the API key held as a server secret. The next steps are an access token on the API, and logging of tool choices and errors without writing any message text to the logs.

---

## Data Source

Patient-education content: [MedlinePlus](https://medlineplus.gov), produced by the U.S. National Library of Medicine. The eight topic pages are downloaded at ingest time and are not redistributed in this repository. See MedlinePlus for its own terms of use. Clinic policy files and all patient messages were written for this project and are fictional.
