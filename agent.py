from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from tools import search_kb, tools, SEARCH_NOTE
from chains import model

model_with_tools = model.bind_tools(tools)

MAX_ROUNDS = 4

tools_by_name = {t.name: t for t in tools}

SYSTEM_RULES = """You assist a nurse at Riverbend Heart Clinic who is reviewing one patient portal message. You are talking to the nurse, not to the patient.

Rules:
1. Answer only from tool results. Do not use outside medical knowledge, even if you are sure of it.
2. Never diagnose. Never recommend starting, stopping or changing a medicine or a dose. Those decisions belong to the clinician. If asked, say so.
3. For any blood pressure reading, use classify_bp. Never classify a reading yourself.
4. For questions about heart conditions, symptoms, medicines or clinic rules, use search_knowledge_base and choose the doc_type that fits the question.
5. If the tools do not cover the question, say plainly that the knowledge base does not cover it. Do not guess.
6. Never state that there is no emergency or that a patient is safe. Report what a tool found and what it does not check.
7. Keep answers short. For search results, name the source and section exactly as they appear in the passage. For other tools, state the basis the tool gives. Never invent a source name or cite a document that a tool did not return.
8. If no tool is needed, for example when the nurse says thank you, reply briefly without calling one.

Case topic:"""

FALLBACK_STRING = "I couldn't finish that within the step limit. Please try rephrasing the question."


def to_messages(history):
    messages = []
    for entry in history:
        if entry['role'] == "user":
            messages.append(HumanMessage(content = entry['content']))
        elif entry['role'] == "assistant":
            messages.append(AIMessage(content = entry['content']))
        else:
            continue
    return messages

def run_agent(messages):
    sources = []
    tools_used = []

    for i in range(MAX_ROUNDS):
        response = model_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return response.content, sources, tools_used

        for tool_call in response.tool_calls:
            label = tool_call['name']
            if tool_call['name'] == "search_knowledge_base":
                label = label + f"[{tool_call['args'].get('doc_type','both')}]"
            tools_used.append(label)
            tool = tools_by_name.get(tool_call['name'])
            if not tool:
                messages.append(ToolMessage(content = "Unknown tool call", tool_call_id = tool_call['id']))
                continue
            if tool_call['name'] == "search_knowledge_base":
                context, found = search_kb(**tool_call['args'])
                for entry in found:
                    if entry not in sources:
                        sources.append(entry)
                content = context + "\n\n" + SEARCH_NOTE if context else "No matching passages were found."

                messages.append(ToolMessage(content = content, tool_call_id = tool_call['id']))

            else:
                messages.append(tool.invoke(tool_call))

    return FALLBACK_STRING, sources, tools_used

def ask(question, topic, history):

    messages = [SystemMessage(content=SYSTEM_RULES + " " + topic)] + to_messages(history) + [HumanMessage(content=question)]
    answer, sources, tools_used = run_agent(messages)
    return {"answer":answer, "sources":sources, "tools_used":tools_used}



if __name__ == "__main__":
    topic = "atrial fibrillation and tiredness"

    questions = [
        "The patient just called with a reading of 162/98. What category is that?",
        "And what does our policy say about when they should call the nurse line?",
        "What are the common symptoms of atrial fibrillation?",
        "They also wrote \"I nearly passed out yesterday\". Does that match our emergency phrases?",
        "What dose of metoprolol should they take?",
        "Thanks.",
    ]

    history = []
    for number, question in enumerate(questions, start=1):
        print(f"\n{'=' * 70}")
        print(f"QUESTION {number}: {question}")
        print("=" * 70)
        result = ask(question, topic, history)
        print(f"TOOLS USED: {result['tools_used']}")
        print(f"SOURCES:    {[s['source_title'] + ' | ' + s['section'] for s in result['sources']]}")
        print(f"\nANSWER: {result['answer']}")
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})
