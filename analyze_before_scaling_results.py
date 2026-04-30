import json
from pathlib import Path

DATA_DIR = Path("novatech_dataset")
OUT_PATH = DATA_DIR / "before_scaling_analysis_summary.json"


def load_json(name):
    path = DATA_DIR / name
    if not path.exists():
        print(f"Missing: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def safe_div(a, b):
    return a / b if b else 0.0


def get_results(data):
    if not data:
        return []
    if isinstance(data, list):
        return data
    return data.get("results", [])


def get_summary(data):
    if isinstance(data, dict):
        return data.get("summary", {})
    return {}


def count_true(results, key):
    return sum(1 for r in results if r.get(key, False))


def analyze_result_file(label, filename):
    data = load_json(filename)
    results = get_results(data)
    summary = get_summary(data)

    if not results:
        return {
            "file": filename,
            "note": "No results found",
            "original_summary": summary,
        }

    total = len(results)

    answer_correct = sum(
        1 for r in results
        if r.get("answer_correct", r.get("correct", False))
    )

    poison_retrieved = sum(
        1 for r in results
        if r.get("targeted_poison_retrieved", r.get("poison_retrieved", False))
    )

    poison_top1 = sum(
        1 for r in results
        if r.get("poison_rank") == 1
    )

    expected_bad = count_true(results, "expected_bad_answer_present")
    marker = sum(
        1 for r in results
        if r.get("marker_present", r.get("poison_marker_seen", False))
    )

    degraded = count_true(results, "degraded_from_clean_correct")

    return {
        "file": filename,
        "num_queries": total,
        "answer_accuracy": safe_div(answer_correct, total),
        "poison_retrieval_rate": safe_div(poison_retrieved, total),
        "poison_top1_rate": safe_div(poison_top1, total),
        "attack_success_rate_expected_bad_answer": safe_div(expected_bad, total),
        "marker_leakage_rate": safe_div(marker, total),
        "degraded_from_clean_correct_rate": safe_div(degraded, total),
        "original_summary": summary,
    }


def analyze_attack_type_analysis():
    data = load_json("attack_type_analysis.json")
    if not data:
        return {}

    return data


def analyze_corpus_defense():
    data = load_json("defense_summary.json")
    if not data:
        return {}

    attack_stats = data.get("attack_type_stats", {})
    output = {}

    for attack_type, stats in attack_stats.items():
        total = stats.get("total", 0)
        rejected = stats.get("rejected", 0)
        accepted = stats.get("accepted", 0)

        output[attack_type] = {
            "total_poison_docs": total,
            "rejected": rejected,
            "accepted": accepted,
            "rejection_rate": safe_div(rejected, total),
            "acceptance_rate": safe_div(accepted, total),
        }

    return {
        "raw_defense_summary": data,
        "per_attack_filtering": output,
    }


def analyze_context_sanitization():
    data = load_json("context_sanitized_results.json")
    if not data:
        return {}

    return {
        "summary": data.get("summary", {}),
        "by_attack_type_when_retrieved": data.get("by_attack_type_when_retrieved", {}),
    }


def analyze_post_generation():
    data = load_json("post_generation_defense_results.json")
    if not data:
        return {}

    return {
        "summary": data.get("summary", {}),
        "by_attack_type_when_retrieved": data.get("by_attack_type_when_retrieved", {}),
    }


def main():
    metadata = load_json("metadata.json") or {}

    report = {
        "dataset_metadata": metadata,

        "baseline": analyze_result_file(
            "baseline",
            "baseline_results.json",
        ),

        "poisoned_no_defense": analyze_result_file(
            "poisoned_no_defense",
            "poisoned_results.json",
        ),

        "attack_type_analysis_before_defense": analyze_attack_type_analysis(),

        "corpus_defense": {
            "document_filtering": analyze_corpus_defense(),
            "rag_after_filtering": analyze_result_file(
                "corpus_defense",
                "defended_results.json",
            ),
        },

        "context_sanitization": analyze_context_sanitization(),

        "post_generation_defense": analyze_post_generation(),

        "human_in_loop": analyze_result_file(
            "human_in_loop",
            "human_in_loop_results.json",
        ),

        "manual_human_review": analyze_result_file(
            "manual_human_review",
            "manual_human_review_results.json",
        ),

        "interpretation": {
            "attack_success_rate_expected_bad_answer": "How often the model repeated the attacker's intended wrong answer.",
            "poison_retrieval_rate": "How often poisoned documents appeared in retrieved context.",
            "poison_top1_rate": "How often poisoned document was ranked first.",
            "marker_leakage_rate": "How often FLAG/poison marker leaked into the answer.",
            "answer_accuracy": "Correctness against gold answer.",
            "degraded_from_clean_correct_rate": "How often a query that was correct in clean baseline became wrong after attack or defense.",
            "corpus_defense": "Blocks poisoned documents before indexing.",
            "context_sanitization": "Allows retrieval but removes suspicious text before generation.",
            "post_generation_defense": "Checks generated answer after generation and blocks/corrects suspicious outputs."
        }
    }

    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved before-scaling analysis to: {OUT_PATH}")
    print("\nMain comparison:\n")

    for name in [
        "baseline",
        "poisoned_no_defense",
        "human_in_loop",
        "manual_human_review",
    ]:
        item = report.get(name, {})
        print(name)
        print(json.dumps(item, indent=2))
        print()

    print("Corpus defense RAG after filtering:")
    print(json.dumps(report["corpus_defense"]["rag_after_filtering"], indent=2))

    print("\nContext sanitization summary:")
    print(json.dumps(report["context_sanitization"].get("summary", {}), indent=2))

    print("\nPost-generation defense summary:")
    print(json.dumps(report["post_generation_defense"].get("summary", {}), indent=2))


if __name__ == "__main__":
    main()