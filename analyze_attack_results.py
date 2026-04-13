import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("novatech_dataset")


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def safe_div(a, b):
    return a / b if b else 0.0


def main():
    poisoned = load_json(DATA_DIR / "poisoned_results.json")
    results = poisoned["results"]

    stats = defaultdict(lambda: {
        "total_queries": 0,
        "poison_retrieved": 0,
        "poison_top1": 0,
        "expected_bad_answer_present": 0,
        "marker_present": 0,
        "answer_correct": 0,
        "degraded_from_clean_correct": 0,
        "eligible_clean_correct": 0,
    })

    no_targeted_poison = {
        "total_queries": 0,
        "answer_correct": 0,
        "baseline_correct": 0,
    }

    for r in results:
        attack_type = r["poison_attack_type"]

        if attack_type is None:
            no_targeted_poison["total_queries"] += 1
            if r["answer_correct"]:
                no_targeted_poison["answer_correct"] += 1
            if r["baseline_answer_correct"]:
                no_targeted_poison["baseline_correct"] += 1
            continue

        s = stats[attack_type]
        s["total_queries"] += 1

        if r["targeted_poison_retrieved"]:
            s["poison_retrieved"] += 1
        if r["poison_rank"] == 1:
            s["poison_top1"] += 1
        if r["expected_bad_answer_present"]:
            s["expected_bad_answer_present"] += 1
        if r["marker_present"]:
            s["marker_present"] += 1
        if r["answer_correct"]:
            s["answer_correct"] += 1
        if r["baseline_answer_correct"]:
            s["eligible_clean_correct"] += 1
        if r["degraded_from_clean_correct"]:
            s["degraded_from_clean_correct"] += 1

    summary = {}

    for attack_type, s in stats.items():
        summary[attack_type] = {
            "num_queries_where_targeted_poison_was_retrieved": s["total_queries"],
            "poison_top1_within_retrieved": safe_div(s["poison_top1"], s["total_queries"]),
            "expected_bad_answer_rate_within_retrieved": safe_div(s["expected_bad_answer_present"], s["total_queries"]),
            "marker_leakage_rate_within_retrieved": safe_div(s["marker_present"], s["total_queries"]),
            "accuracy_within_retrieved": safe_div(s["answer_correct"], s["total_queries"]),
            "degraded_from_clean_correct_rate_within_retrieved": safe_div(
                s["degraded_from_clean_correct"], s["eligible_clean_correct"]
            ),
            "eligible_clean_correct_queries": s["eligible_clean_correct"],
        }

    overall = {
        "queries_with_no_targeted_poison_retrieved": no_targeted_poison["total_queries"],
        "accuracy_when_no_targeted_poison_retrieved": safe_div(
            no_targeted_poison["answer_correct"], no_targeted_poison["total_queries"]
        ),
    }

    output = {
        "by_attack_type": summary,
        "overall_context": overall,
    }

    out_path = DATA_DIR / "attack_type_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()