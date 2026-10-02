RED_FLAG_PHRASES = ["chest pain", "chest pressure", "pressure in my chest", "tightness in my chest", "chest tightness", "pain in my chest",
                    "can't breathe", "cannot breathe", "can not breathe", "struggling to breathe", "trouble breathing", "gasping",
                    "fainted", "passed out", "blacked out", "lost consciousness",
                    "face drooping", "face is drooping", "slurred speech", "can't speak", "arm weakness", "arm is weak", "numb on one side"]

def find_red_flags(message):
    flags = []
    message = message.lower()
    for phrase in RED_FLAG_PHRASES:
        if phrase in message:
            flags.append(phrase)
    return flags

EMERGENCY_MESSAGE = """Your message mentions symptoms that may need emergency care. If you are having these symptoms now,
call your local emergency number immediately. Do not wait for a reply on the portal.
A nurse has been alerted to your message."""


def check_safety(message, analysis):
    matched = find_red_flags(message)
    by_rules = bool(matched)
    by_model = analysis.urgency == "emergency"
    if by_rules and by_model:
        triggered_by = "both"
    elif by_rules:
        triggered_by = "rules"
    elif by_model:
        triggered_by = "model"
    else:
        triggered_by = "none"
    return {"is_emergency": by_rules or by_model,
            "matched_phrases": matched,
            "model_urgency": analysis.urgency,
            "triggered_by": triggered_by}
    