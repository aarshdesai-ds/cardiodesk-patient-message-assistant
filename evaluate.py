import json
import csv
from chains import triage_chain
from collections import Counter

rows = []
with open("eval/messages.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

def evaluate_row(row):
    result = {**row, 
           "got_emergency": None, 
           "triggered_by":  "" ,
           "got_intent":    "" , 
           "got_path":      "" , 
           "source_titles": [],
           "emergency_ok":  False,
           "intent_ok":     False,
           "source_ok":     None,
           "error":         ""}
    try: 
        report = triage_chain.invoke({"message":row['message']})
        got_emergency = report['is_emergency']
        triggered_by = report['safety']['triggered_by']
        got_intent = report['intent']
        got_path = report['path']
        source_titles = [source['source_title'] for source in report['sources']]

        emergency_ok = got_emergency == row['expected_emergency']
        intent_ok = got_intent in row['acceptable_intents']

        expected_source = row['expected_source']

        if not expected_source:
            source_ok = None
        elif got_path == "emergency":
            source_ok = None
        else:
            source_ok = any(expected_source.lower() in title.lower() for title in source_titles)

        result['got_emergency'] = got_emergency
        result['triggered_by'] = triggered_by
        result['got_intent'] = got_intent
        result['got_path'] = got_path
        result['source_titles'] = source_titles
        result['emergency_ok'] = emergency_ok
        result['intent_ok'] = intent_ok
        result['source_ok'] = source_ok
    except Exception as e:
        result['error'] = str(e)

    return result


def summarise_safety(results):
    results = [row for row in results if not row['error']]
    caught = [row for row in results if row['expected_emergency'] and row['got_emergency']]
    missed = [row for row in results if row['expected_emergency'] and not row['got_emergency']]
    false_alarms = [row for row in results if not row['expected_emergency'] and row['got_emergency']]
    correct = [row for row in results if not row['expected_emergency'] and not row['got_emergency']]

    emergencies = len(caught) + len(missed)
    non_emergencies = len(false_alarms) + len(correct) 
    triggered_counts = Counter(row['triggered_by'] for row in caught)

    return {"caught":caught, "missed":missed, "false_alarms": false_alarms, "correct":correct, "triggered_counts":triggered_counts , "emergencies":emergencies, "non_emergencies":non_emergencies}

def summarise_quality(results):
    results = [row for row in results if not row['error']]
    intent_hits = [row for row in results if row['intent_ok']]
    intent_misses = [row for row in results if not row['intent_ok']]
    checked = [row for row in results if row['source_ok'] is not None]
    source_hits = [row for row in checked if row['source_ok']]
    source_misses = [row for row in checked if not row['source_ok']]

    return {"intent_hits": intent_hits, "intent_misses": intent_misses, "checked":checked, "source_hits": source_hits, "source_misses":source_misses}

def summarise_groups(results):
    results = [row for row in results if not row['error']]
    groups = {}
    for row in results:
        groups.setdefault(row['group'],[])
        groups[row['group']].append(row)
    return groups

def save_csv(results,path):
    if not results:
        return 
    fieldnames = list(results[0].keys())
    csv_rows = []
    for row in results:
        csv_rows.append({**row, "acceptable_intents": " | ".join(row['acceptable_intents']) , "source_titles": " | ".join(row['source_titles'])})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f = f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

if __name__ == "__main__":
    print(f"Loaded {len(rows)} messages")

    # While testing: two rows from each of three groups. Six model calls.
    test_groups = ["emergency_model_only", "false_alarm_bait", "admin"]
    test_rows = []
    for group in test_groups:
        test_rows.extend([row for row in rows if row["group"] == group][:2])

    # For the full run, change this line to:  rows_to_run = rows
    rows_to_run = rows

    results = []
    for number, row in enumerate(rows_to_run, start=1):
        result = evaluate_row(row)
        results.append(result)
        if result["error"]:
            status = "ERROR"
        elif result["emergency_ok"]:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"[{number}/{len(rows_to_run)}] {result['id']}  {result['group']:<22} {status}  "
              f"(triggered_by: {result['triggered_by']})")
    save_csv(results, "eval/results.csv")
    print(f"\nSaved {len(results)} rows to eval/results.csv")

    safety = summarise_safety(results)
    quality = summarise_quality(results)
    errors = [row for row in results if row["error"]]

    print("\nSAFETY")
    print(f"Caught:        {len(safety['caught'])} of {safety['emergencies']}")
    print(f"Missed:        {len(safety['missed'])} of {safety['emergencies']}")
    print(f"False alarms:  {len(safety['false_alarms'])} of {safety['non_emergencies']}")
    print(f"Correct:       {len(safety['correct'])} of {safety['non_emergencies']}")
    print(f"Caught by:     {dict(safety['triggered_counts'])}")

    for row in safety["missed"]:
        print(f"\nMISSED {row['id']} ({row['group']}): {row['message']}")

    intent_total = len(quality["intent_hits"]) + len(quality["intent_misses"])
    print("\nINTENT")
    print(f"Correct:       {len(quality['intent_hits'])} of {intent_total}")
    for row in quality["intent_misses"]:
        print(f"  {row['id']}: got {row['got_intent']}, expected one of {row['acceptable_intents']}")

    print("\nRETRIEVAL")
    print(f"Expected source found:  {len(quality['source_hits'])} of {len(quality['checked'])}")
    for row in quality["source_misses"]:
        print(f"  {row['id']}: expected '{row['expected_source']}', got {row['source_titles']}")
    
    groups = summarise_groups(results)

    print("\nBY GROUP")
    for group, group_rows in groups.items():
        total = len(group_rows)
        emergency_passed = sum(row["emergency_ok"] for row in group_rows)
        intent_passed = sum(row["intent_ok"] for row in group_rows)
        print(f"{group:<22} {total} rows   emergency {emergency_passed}/{total}   "
              f"intent {intent_passed}/{total}")

    print("\nERRORS")
    print(f"Failed calls:  {len(errors)} of {len(results)}")
    for row in errors:
        print(f"  {row['id']}: {row['error']}")