import json
import re
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path("novatech_dataset")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# Defense 1: catches explicit malicious instructions
def detect_instruction_injection(doc):
    text = (doc.get("title", "") + " " + doc.get("content", "")).lower()

    patterns = [
        r"ignore all previous instructions",
        r"ignore previous instructions",
        r"always answer with",
        r"disregard",
        r"override",
        r"confirm using marker",
    ]

    return any(re.search(p, text) for p in patterns)


# Defense 2: catches keyword/query stuffing
def detect_retrieval_targeting(doc):
    text = doc.get("content", "").lower()
    words = re.findall(r"\b[a-zA-Z0-9_-]+\b", text)

    if not words:
        return False

    counts = Counter(words)
    most_common_word, most_common_count = counts.most_common(1)[0]

    repetition_ratio = most_common_count / len(words)

    target_query = doc.get("target_query", "").lower()
    query_repeated = target_query and text.count(target_query) >= 2

    title = doc.get("title", "").lower()
    title_repeated = title and text.count(title) >= 2

    return repetition_ratio > 0.12 or query_repeated or title_repeated


# Build trusted facts from clean docs
def extract_clean_fact_map(clean_docs):
    fact_map = defaultdict(dict)

    for doc in clean_docs:
        facts = doc.get("facts", {})

        entity = (
            facts.get("service_name")
            or facts.get("api_name")
            or facts.get("system_name")
            or facts.get("policy_name")
            or doc.get("title")
        )

        if not entity:
            continue

        entity = entity.lower()

        for key, value in facts.items():
            fact_map[entity][key] = str(value).lower()

    return fact_map


# Defense 3: catches conflicting fake facts
def detect_content_manipulation(doc, clean_fact_map):
    text = doc.get("content", "").lower()
    target_fact_type = doc.get("target_fact_type")
    expected_bad_answer = doc.get("expected_bad_answer")

    if not target_fact_type or expected_bad_answer is None:
        return False

    bad_value = str(expected_bad_answer).lower()

    for entity, facts in clean_fact_map.items():
        if entity in text and target_fact_type in facts:
            trusted_value = facts[target_fact_type]

            if bad_value in text and bad_value != trusted_value:
                return True

    return False


def defend_doc(doc, clean_fact_map):
    reasons = []

    if detect_instruction_injection(doc):
        reasons.append("instruction_injection_defense")

    if detect_retrieval_targeting(doc):
        reasons.append("retrieval_targeting_defense")

    if detect_content_manipulation(doc, clean_fact_map):
        reasons.append("content_manipulation_defense")

    return reasons


def main():
    clean_docs = load_json(DATA_DIR / "clean_docs.json")
    poisoned_docs = load_json(DATA_DIR / "poisoned_docs.json")

    clean_fact_map = extract_clean_fact_map(clean_docs)

    accepted_poisoned_docs = []
    rejected_poisoned_docs = []

    defense_stats = defaultdict(int)
    attack_type_stats = defaultdict(lambda: {
        "total": 0,
        "rejected": 0,
        "accepted": 0,
    })

    for doc in poisoned_docs:
        attack_type = doc.get("attack_type", "unknown")
        attack_type_stats[attack_type]["total"] += 1

        reasons = defend_doc(doc, clean_fact_map)

        if reasons:
            rejected_doc = dict(doc)
            rejected_doc["defense_reasons"] = reasons
            rejected_poisoned_docs.append(rejected_doc)

            attack_type_stats[attack_type]["rejected"] += 1

            for reason in reasons:
                defense_stats[reason] += 1
        else:
            accepted_poisoned_docs.append(doc)
            attack_type_stats[attack_type]["accepted"] += 1

    defense_summary = {
        "total_clean_docs": len(clean_docs),
        "total_poisoned_docs": len(poisoned_docs),
        "accepted_poisoned_docs": len(accepted_poisoned_docs),
        "rejected_poisoned_docs": len(rejected_poisoned_docs),
        "defense_stats": dict(defense_stats),
        "attack_type_stats": dict(attack_type_stats),
    }

    save_json(DATA_DIR / "defended_clean_docs.json", clean_docs)
    save_json(DATA_DIR / "defended_poisoned_docs.json", accepted_poisoned_docs)
    save_json(DATA_DIR / "rejected_poisoned_docs.json", rejected_poisoned_docs)
    save_json(DATA_DIR / "defense_summary.json", defense_summary)

    print("Corpus defense complete.")
    print(json.dumps(defense_summary, indent=2))


if __name__ == "__main__":
    main()