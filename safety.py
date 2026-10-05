import re

RED_FLAG_PHRASES = ["chest pain", "chest pressure", "pressure in my chest", "tightness in my chest", "chest tightness", "pain in my chest",
                    "can't breathe", "cannot breathe", "can not breathe", "struggling to breathe", "trouble breathing", "gasping",
                    "fainted", "passed out", "blacked out", "lost consciousness",
                    "face drooping", "face is drooping", "slurred speech", "can't speak", "arm weakness", "arm is weak", "numb on one side"]

CRISIS_SYSTOLIC  = 180
CRISIS_DIASTOLIC = 120
SYSTOLIC_MIN,  SYSTOLIC_MAX   = 70, 260
DIASTOLIC_MIN, DIASTOLIC_MAX  = 40, 160

def is_plausible_reading(systolic, diastolic):
    if SYSTOLIC_MIN <= systolic <= SYSTOLIC_MAX and DIASTOLIC_MIN <= diastolic <= DIASTOLIC_MAX and systolic > diastolic:
        return True
    return False

def find_bp_readings(message):
    final_matches = []
    pattern = r"\b(\d{2,3})\s*/\s*(\d{2,3})\b"

    matches = re.findall(pattern, message)
    for first, second in matches:
        first, second = int(first), int(second)
        if is_plausible_reading(first, second):
            final_matches.append((first,second))
    return final_matches

def classify_blood_pressure(systolic, diastolic):
    if systolic > 180  and  diastolic > 120:
        return "Hypertensive Crisis"
    elif systolic >= 140  or  diastolic >= 90:
        return "High Blood Pressure Stage 2"
    elif systolic >= 130  or  diastolic >= 80:
        return "High Blood Pressure Stage 1"
    elif systolic >= 120  and diastolic < 80:
        return "Elevated"
    else:
        return "Normal"

def find_red_flags(message):
    flags = []
    message = message.lower().replace("\u2019","'")
    for phrase in RED_FLAG_PHRASES:
        if phrase in message:
            flags.append(phrase)
    return flags

EMERGENCY_MESSAGE = """Your message mentions symptoms that may need emergency care. If you are having these symptoms now,
call your local emergency number immediately. Do not wait for a reply on the portal.
A nurse has been alerted to your message."""


def check_safety(message, analysis):
    matched = find_red_flags(message)
    readings = find_bp_readings(message)
    crisis_readings = [f"{systolic}/{diastolic}" for systolic, diastolic in readings if systolic > CRISIS_SYSTOLIC  or  diastolic > CRISIS_DIASTOLIC]

    by_rules = bool(matched) or bool(crisis_readings)
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
            "triggered_by": triggered_by,
            "bp_readings":crisis_readings}
    