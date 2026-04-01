from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "bfcl_eval" / "data"
SINGLE_TURN_CATEGORIES = ("simple_python", "simple_java", "simple_javascript", "live_simple", "multiple", "live_multiple")
RECENT_CATEGORIES = ("simple_java", "simple_javascript", "live_multiple")
TURN_CATEGORIES = ("multi_turn_base", "memory_kv")
SPILLOVER_SETUPS = {"guidance", "transformers"}
DATASET_FILE_ALIASES = {
    "memory_kv": "BFCL_v4_memory.json",
}

SCORE_FILE_RE = re.compile(r"BFCL_v4_(.+)_score\.json$")

INSTRUCTION_LEAK_MARKERS = (
    "Please note",
    "Note:",
    "###",
    "Function Calls",
    "Explanation",
    "If you provide",
    "The function requires",
    "requires both",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze BFCL score artifacts for spillover, AST decoder failures, "
            "multi-turn failure modes, and pairwise solved-set overlap."
        )
    )
    parser.add_argument(
        "score_dirs",
        nargs="*",
        help=(
            "Optional score directories. Defaults to all local score_* directories "
            "(for example: score_transformers score_tool_only score_guidance)."
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help=(
            "Optional exact category filter such as simple_java simple_javascript "
            "live_multiple multi_turn_base memory_kv."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional exact model directory names to include.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5,
        help="Number of sample IDs/questions to print per overlap side. Default: 5.",
    )
    parser.add_argument(
        "--overlap-setups",
        nargs=2,
        default=("transformers", "tool_only"),
        metavar=("BASELINE", "CONSTRAINED"),
        help=(
            "Pair of setup labels to compare in the overlap section. Labels should "
            "match directory suffixes such as transformers, tool_only, guidance."
        ),
    )
    parser.add_argument(
        "--figure-dir",
        default=None,
        help=(
            "Optional directory for overlap figures. Writes stacked overlap bar charts "
            "using matplotlib when set."
        ),
    )
    return parser.parse_args()


def default_dirs(prefix: str) -> list[Path]:
    return sorted(path for path in ROOT.iterdir() if path.is_dir() and path.name.startswith(prefix))


def resolve_dirs(candidates: list[str] | None, prefix: str) -> list[Path]:
    if candidates:
        return [Path(candidate).resolve() for candidate in candidates]
    return default_dirs(prefix)


def extract_category(path: Path, pattern: re.Pattern[str]) -> str | None:
    match = pattern.fullmatch(path.name)
    if not match:
        return None
    return match.group(1)


def include_artifact(path: Path, category: str, models: set[str] | None, categories: set[str] | None) -> bool:
    model = path.relative_to(path.parents[2]).parts[0]
    if models and model not in models:
        return False
    if categories and category not in categories:
        return False
    return True


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_score_runs(
    score_dirs: list[Path],
    models: set[str] | None,
    categories: set[str] | None,
) -> dict[tuple[str, str, str], dict]:
    runs: dict[tuple[str, str, str], dict] = {}
    for score_dir in score_dirs:
        if not score_dir.is_dir():
            raise SystemExit(f"Score directory not found: {score_dir}")
        setup = score_dir.name.removeprefix("score_")
        for path in sorted(score_dir.rglob("*_score.json")):
            category = extract_category(path, SCORE_FILE_RE)
            if not category or not include_artifact(path, category, models, categories):
                continue
            records = load_jsonl(path)
            if not records:
                continue
            metadata = {}
            failures: list[dict] = records
            if "accuracy" in records[0] and "id" not in records[0]:
                metadata = records[0]
                failures = records[1:]
            model = path.relative_to(score_dir).parts[0]
            runs[(setup, model, category)] = {
                "path": path,
                "metadata": metadata,
                "failures": failures,
                "failures_by_id": {record["id"]: record for record in failures if "id" in record},
            }
    return runs


def load_dataset(category: str) -> dict[str, dict]:
    file_name = DATASET_FILE_ALIASES.get(category, f"BFCL_v4_{category}.json")
    path = DATA_DIR / file_name
    if not path.exists():
        return {}
    raw = path.read_text().strip()
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            entries = [parsed]
        else:
            entries = parsed
        return {entry["id"]: entry for entry in entries if isinstance(entry, dict) and "id" in entry}
    except json.JSONDecodeError:
        with path.open() as handle:
            return {
                entry["id"]: entry
                for entry in (json.loads(line) for line in handle if line.strip())
                if isinstance(entry, dict) and "id" in entry
            }


def load_datasets(categories: Iterable[str]) -> dict[str, dict[str, dict]]:
    return {category: load_dataset(category) for category in categories}


def classify_error_type(record: dict) -> str:
    error_type = record.get("error_type")
    if error_type:
        return str(error_type)
    error = record.get("error")
    if isinstance(error, dict):
        nested = error.get("error_type")
        if nested:
            return str(nested)
    return ""


def shorten(text: str, limit: int = 120) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def print_section(title: str) -> None:
    print(f"\n## {title}")


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    if not rows:
        print("(no rows)")
        return
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))
    print(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)))


def get_question(entry: dict) -> str:
    question = entry.get("question", [])
    if not question:
        return ""
    first_turn = question[0]
    if not first_turn:
        return ""
    for message in first_turn:
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return str(first_turn[0].get("content", ""))


def prompt_function_names(entry: dict) -> set[str]:
    return {function["name"] for function in entry.get("function", [])}


def prompt_param_names(entry: dict) -> set[str]:
    names = set()
    for function in entry.get("function", []):
        parameters = function.get("parameters", {}).get("properties", {})
        names.update(parameters.keys())
    return names


def extract_quoted_chunks(raw: str) -> list[str]:
    return [chunk for _quote, chunk in re.findall(r"""(['"])(.*?)\1""", raw, re.DOTALL)]


def spillover_reasons(raw: str, prompt_entry: dict) -> set[str]:
    reasons = set()
    function_names = prompt_function_names(prompt_entry)
    param_names = prompt_param_names(prompt_entry)
    for chunk in extract_quoted_chunks(raw):
        if any(marker in chunk for marker in INSTRUCTION_LEAK_MARKERS):
            reasons.add("instruction_text")
        if any(name + "(" in chunk for name in function_names):
            reasons.add("function_call_text")
        for param_name in param_names:
            if re.search(rf"(^|[\s,\n]){re.escape(param_name)}\s*=", chunk):
                reasons.add("parameter_name_text")
                break
    return reasons


def spillover_summary(
    score_runs: dict[tuple[str, str, str], dict],
    datasets: dict[str, dict[str, dict]],
) -> None:
    print_section("Score-Level Parameter Spillover (Guidance vs Transformers)")
    rows: list[list[object]] = []
    sample_rows: list[list[object]] = []
    matched_setups = set()

    for (setup, model, category), run in sorted(score_runs.items()):
        if category not in SINGLE_TURN_CATEGORIES or setup not in SPILLOVER_SETUPS:
            continue
        matched_setups.add(setup)
        dataset = datasets.get(category, {})
        counts = Counter()
        samples: list[tuple[str, str]] = []
        for record in run["failures"]:
            prompt_entry = dataset.get(record["id"])
            if not prompt_entry:
                continue
            raw = str(record.get("model_result_raw", ""))
            if not raw:
                continue
            reasons = spillover_reasons(raw, prompt_entry)
            if not reasons:
                continue
            counts["spillover_any"] += 1
            for reason in reasons:
                counts[reason] += 1
            if len(samples) < 3:
                samples.append((record["id"], shorten(raw, 140)))

        total = int(run["metadata"].get("total_count", 0))
        rows.append(
            [
                setup,
                model,
                category,
                total,
                counts["spillover_any"],
                f"{counts['spillover_any'] / total:.3f}" if total else "NA",
                counts["parameter_name_text"],
                counts["instruction_text"],
                counts["function_call_text"],
            ]
        )
        for sample_id, result_text in samples:
            sample_rows.append([setup, model, category, sample_id, result_text])

    print_table(
        [
            "setup",
            "model",
            "category",
            "entries",
            "spillover_any",
            "spill_rate",
            "param_name",
            "instruction",
            "function_text",
        ],
        rows,
    )
    print()
    print("Sample spillover outputs:")
    print_table(["setup", "model", "category", "id", "result"], sample_rows)
    if "guidance" not in matched_setups:
        print()
        print("- guidance score artifacts for the selected recent categories were not found, so Guidance rows were skipped.")


def ast_summary(score_runs: dict[tuple[str, str, str], dict]) -> None:
    print_section("AST Decoder Errors")
    rows: list[list[object]] = []
    setup_order = ("transformers", "guidance", "tool_only")
    grouped: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)

    for key in sorted(score_runs):
        setup, model, category = key
        if category not in SINGLE_TURN_CATEGORIES:
            continue
        run = score_runs[key]
        failures = run["failures"]
        exact_failed = sum(classify_error_type(record) == "ast_decoder:decoder_failed" for record in failures)
        exact_total = int(run["metadata"].get("total_count", 0))
        grouped[(model, category)][setup] = (
            f"{exact_failed}/{exact_total} ({exact_failed / exact_total:.3f})" if exact_total else "-"
        )

    for model, category in sorted(grouped):
        row = [model, category]
        for setup in setup_order:
            row.append(grouped[(model, category)].get(setup, "-"))
        rows.append(row)

    print_table(
        [
            "model",
            "category",
            "transformers",
            "guidance",
            "tool_only",
        ],
        rows,
    )


def has_exact_repeat(model_result: list) -> bool:
    for turn in model_result or []:
        seen = set()
        for step in turn:
            if not isinstance(step, str):
                continue
            if step in seen:
                return True
            seen.add(step)
    return False


def collect_step_counts(model_result: list) -> list[int]:
    counts = []
    for turn in model_result or []:
        if isinstance(turn, list):
            counts.append(len(turn))
    return counts


def multi_turn_summary(score_runs: dict[tuple[str, str, str], dict]) -> None:
    print_section("Multi-Turn And Agentic Failure Modes")
    rows: list[list[object]] = []
    missing_notes: list[str] = []

    for category in TURN_CATEGORIES:
        matched = [key for key in sorted(score_runs) if key[2] == category]
        if not matched:
            continue
        for key in matched:
            setup, model, _category = key
            run = score_runs[key]
            metadata = run["metadata"]
            error_counts = Counter(classify_error_type(record) for record in run["failures"])
            repeated_entries = sum(has_exact_repeat(record.get("model_result", [])) for record in run["failures"])
            step_counts = []
            for record in run["failures"]:
                step_counts.extend(collect_step_counts(record.get("model_result", [])))
            median_steps = f"{median(step_counts):.1f}" if step_counts else "-"
            rows.append(
                [
                    setup,
                    model,
                    category,
                    f"{float(metadata.get('accuracy', 0.0)):.3f}",
                    metadata.get("total_count", 0),
                    error_counts.get("multi_turn:force_terminated", 0),
                    error_counts.get("multi_turn:instance_state_mismatch", 0),
                    error_counts.get("multi_turn:execution_response_mismatch", 0),
                    error_counts.get("multi_turn:empty_turn_model_response", 0),
                    error_counts.get("multi_turn:inference_error", 0),
                    error_counts.get("agentic:no_last_message", 0),
                    error_counts.get("agentic:inference_error", 0),
                    repeated_entries,
                    median_steps,
                ]
            )

    score_setups = {setup for setup, _model, _category in score_runs}
    if "tool_only" in score_setups and not any(
        key for key in score_runs if key[0] == "tool_only" and key[2] == "memory_kv"
    ):
        missing_notes.append(
            "tool_only memory_kv: no score artifact found, consistent with the run being intentionally skipped."
        )

    print_table(
        [
            "setup",
            "model",
            "category",
            "accuracy",
            "total",
            "force_term",
            "state_mismatch",
            "exec_mismatch",
            "empty_turn",
            "mt_infer_err",
            "no_last_msg",
            "agentic_infer_err",
            "repeat_entries",
            "median_steps",
        ],
        rows,
    )
    if missing_notes:
        print()
        for note in missing_notes:
            print(f"- {note}")


def solved_ids_for_run(
    run: dict,
    dataset_ids: set[str],
) -> set[str]:
    return dataset_ids - set(run["failures_by_id"])


def overlap_rows(
    score_runs: dict[tuple[str, str, str], dict],
    datasets: dict[str, dict[str, dict]],
    baseline_setup: str,
    constrained_setup: str,
) -> tuple[list[list[object]], dict[tuple[str, str], dict]]:
    rows: list[list[object]] = []
    details: dict[tuple[str, str], dict] = {}
    categories = {category for _setup, _model, category in score_runs}
    for category in sorted(categories):
        dataset = datasets.get(category, {})
        if not dataset:
            continue
        dataset_ids = set(dataset)
        models = {
            model
            for setup, model, item_category in score_runs
            if item_category == category and setup in {baseline_setup, constrained_setup}
        }
        for model in sorted(models):
            baseline = score_runs.get((baseline_setup, model, category))
            constrained = score_runs.get((constrained_setup, model, category))
            if not baseline or not constrained:
                continue
            baseline_solved = solved_ids_for_run(baseline, dataset_ids)
            constrained_solved = solved_ids_for_run(constrained, dataset_ids)
            rows.append(
                [
                    model,
                    category,
                    len(baseline_solved & constrained_solved),
                    len(baseline_solved - constrained_solved),
                    len(constrained_solved - baseline_solved),
                ]
            )
            details[(model, category)] = {
                "baseline_only": sorted(baseline_solved - constrained_solved),
                "constrained_only": sorted(constrained_solved - baseline_solved),
                "baseline_failures": baseline["failures_by_id"],
                "constrained_failures": constrained["failures_by_id"],
                "dataset": dataset,
            }
    return rows, details


def write_overlap_figure(
    figure_dir: Path,
    pair: tuple[str, str],
    category: str,
    category_rows: list[list[object]],
) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", str((ROOT / ".mplconfig").resolve()))
    figure_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    labels = [row[0] for row in category_rows]
    both = [row[2] for row in category_rows]
    baseline_only = [row[3] for row in category_rows]
    constrained_only = [row[4] for row in category_rows]

    fig, ax = plt.subplots(figsize=(10, max(3, 0.7 * len(labels))))
    y = range(len(labels))
    ax.barh(y, both, label="both solved", color="#6c8ebf")
    ax.barh(y, baseline_only, left=both, label=f"{pair[0]} only", color="#d08770")
    ax.barh(
        y,
        constrained_only,
        left=[left + delta for left, delta in zip(both, baseline_only)],
        label=f"{pair[1]} only",
        color="#8fbc8f",
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("count")
    ax.set_title(f"{category}: solved-set overlap")
    ax.legend()
    fig.tight_layout()
    output_path = figure_dir / f"overlap_{pair[0]}_vs_{pair[1]}_{category}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def overlap_summary(
    score_runs: dict[tuple[str, str, str], dict],
    datasets: dict[str, dict[str, dict]],
    pair: tuple[str, str],
    sample_limit: int,
    figure_dir: Path | None,
) -> None:
    baseline_setup, constrained_setup = pair
    print_section(f"Overlap: {baseline_setup} vs {constrained_setup}")
    rows, details = overlap_rows(score_runs, datasets, baseline_setup, constrained_setup)
    print_table(
        ["model", "category", "both_solved", f"{baseline_setup}_only", f"{constrained_setup}_only"],
        rows,
    )

    constrained_only_error_types = Counter()
    baseline_only_error_types = Counter()
    sample_rows: list[list[object]] = []

    for (model, category), item in sorted(details.items()):
        dataset = item["dataset"]
        for sample_id in item["constrained_only"]:
            failure = item["baseline_failures"][sample_id]
            constrained_only_error_types[classify_error_type(failure)] += 1
        for sample_id in item["baseline_only"]:
            failure = item["constrained_failures"][sample_id]
            baseline_only_error_types[classify_error_type(failure)] += 1

        for sample_id in item["constrained_only"][:sample_limit]:
            failure = item["baseline_failures"][sample_id]
            error_type = classify_error_type(failure)
            sample_rows.append(
                [
                    category,
                    model,
                    constrained_setup,
                    sample_id,
                    error_type,
                    shorten(get_question(dataset[sample_id]), 120),
                ]
            )
        for sample_id in item["baseline_only"][:sample_limit]:
            failure = item["constrained_failures"][sample_id]
            error_type = classify_error_type(failure)
            sample_rows.append(
                [
                    category,
                    model,
                    baseline_setup,
                    sample_id,
                    error_type,
                    shorten(get_question(dataset[sample_id]), 120),
                ]
            )

    print()
    print(f"Top losing error types on {constrained_setup}-only wins:")
    for error_type, count in constrained_only_error_types.most_common(10):
        print(f"- {count:4d} {error_type}")

    print()
    print(f"Top losing error types on {baseline_setup}-only wins:")
    for error_type, count in baseline_only_error_types.most_common(10):
        print(f"- {count:4d} {error_type}")

    print()
    print("Representative asymmetric wins:")
    print_table(["category", "model", "winner", "id", "loser_error", "question"], sample_rows)

    if figure_dir:
        paths = []
        by_category = defaultdict(list)
        for row in rows:
            by_category[row[1]].append(row)
        for category, category_rows in sorted(by_category.items()):
            paths.append(write_overlap_figure(figure_dir, pair, category, category_rows))
        print()
        print("Wrote overlap figures:")
        for path in paths:
            print(f"- {path}")


def main() -> None:
    args = parse_args()
    score_dirs = resolve_dirs(args.score_dirs, "score_")
    category_filter = set(args.categories) if args.categories else None
    model_filter = set(args.models) if args.models else None

    score_runs = load_score_runs(score_dirs, model_filter, category_filter)
    categories = {
        category
        for _setup, _model, category in score_runs
    }
    datasets = load_datasets(categories)

    print("Using score dirs:")
    for path in score_dirs:
        print(f"- {path}")

    spillover_summary(score_runs, datasets)
    # ast_summary(score_runs)

    # multi_turn_summary(score_runs)
    # figure_dir = Path(args.figure_dir).resolve() if args.figure_dir else None
    # overlap_summary(
    #     score_runs,
    #     datasets,
    #     tuple(args.overlap_setups),
    #     args.sample_limit,
    #     figure_dir,
    # )


if __name__ == "__main__":
    main()
