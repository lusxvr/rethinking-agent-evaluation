"""Stage 2: turn one Stage 1 dimension's findings into a small general category system
(extract-then-cluster, Chirkova et al., arXiv:2506.09147). Same algorithm for every dimension.

Phases (--phase all runs all four):
- cluster: embed findings and cluster with HDBSCAN via BERTopic, pooling all grids; noise points
  are reassigned to their nearest cluster. No LLM.
- label: one LLM call per cluster names its general behavior; specifics go in the criterion.
- merge: one LLM call partitions all labels into at most MAX_CATEGORIES categories. Deliberately
  coarse: finer designs gave unstable category counts (7 to 30) across variants.
- valence: one LLM call scores each category from -1 (bad) to +1 (good), stored in categories.json.

Usage:
  uv run python -m scripts.taxonomy.stage2_cluster --dimension error_category \
    --stage1-dirs scripts/taxonomy/stage1_output/<grid> [<grid> ...] --phase all
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from scripts.analysis.utils import grid_label_from_dirs
from scripts.taxonomy import judge_server
from scripts.taxonomy.stage1_extract import DIMENSIONS, REPO_ROOT, SAMPLING

CLIENT_TIMEOUT_S = 300.0  # short prompts (a handful of categories/examples), unlike stage1's 1200s
CLUSTERABLE_DIMENSIONS = sorted(DIMENSIONS)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # SBERT default, trained on paraphrase/STS data
MAX_CATEGORIES = 8  # headroom; final runs used 1-4
WHOLE_LIST_MERGE_MAX = 300  # sanity bound on clusters per merge call (largest seen: 201)
NOISE_LABEL = "uncategorized"


def _dimension_label(dimension: str) -> str:
    return DIMENSIONS[dimension]["label"]


# All dimensions share the finding/evidence_quote/why schema. finding_only drops "why"; tested and
# rejected (1.5-2x more categories).
def _description(parsed: dict, finding_only: bool = False) -> str:
    if finding_only:
        return parsed.get("finding") or ""
    return f"{parsed.get('finding')}: {parsed.get('why')}"


def load_descriptions(stage1_dirs: list[Path], dimension: str, finding_only: bool = False) -> list[dict]:
    """One dimension's findings across Stage 1 grid dirs; "run" is "<grid-dir>/<run-name>"."""
    rows = []
    for stage1_dir in stage1_dirs:
        path = stage1_dir / f"{dimension}.jsonl"
        if not path.is_file():
            raise SystemExit(f"{path} not found -- run stage1_extract.py for this dimension/grid first")
        grid_name = stage1_dir.name
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("ok") or r.get("parse_error") or r.get("parsed") is None:
                continue  # Stage 1's own parse failures -- can't cluster a finding that isn't there
            rows.append({"run": f"{grid_name}/{r['run']}", "description": _description(r["parsed"], finding_only)})
    return rows


def _call_and_parse(client: OpenAI, model: str, sampling: dict, messages: list[dict], max_tokens: int) -> dict | None:
    """One API call, returns the parsed JSON or None on any failure (API-level or a malformed
    response) -- never raises, so callers can retry instead of crashing."""
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=sampling["temperature"], top_p=sampling["top_p"],
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}, **sampling["extra_sampling"]},
        )
    except Exception as exc:
        print(f"warning: API call failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    content = (response.choices[0].message.content or "").strip()
    text = content.strip("`")
    if text.startswith("json"):
        text = text[4:]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


# --- cluster: embeddings plus HDBSCAN (BERTopic), no LLM. Grids are pooled by default
# (--cluster-per-grid to split); pooling gave fewer, more general categories once merge was mandatory.

def run_cluster(args, out_dir: Path) -> None:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from hdbscan.prediction import all_points_membership_vectors
    from sentence_transformers import SentenceTransformer
    from umap import UMAP

    embed_model = SentenceTransformer(args.embedding_model)
    all_descriptions: list[dict] = []
    all_topics: list[int] = []
    next_cluster_id = 0  # global id space -- each grid's own HDBSCAN restarts local ids at 0

    stage1_dirs = [Path(d) for d in args.stage1_dirs]
    groups = [stage1_dirs] if args.cluster_combined else [[d] for d in stage1_dirs]

    for group in groups:
        descriptions = load_descriptions(group, args.dimension, finding_only=args.finding_only)
        docs = [r["description"] for r in descriptions]
        group_label = " + ".join(d.name for d in group)
        print(f"[{args.dimension}] {group_label}: {len(docs)} findings loaded")

        # Explicit UMAP with a fixed random_state, for reproducible clusters.
        hdbscan_model = HDBSCAN(min_cluster_size=args.min_cluster_size or 10, metric="euclidean",
                                cluster_selection_method=args.cluster_selection_method, prediction_data=True)
        kwargs = {
            "hdbscan_model": hdbscan_model,
            "umap_model": UMAP(n_neighbors=args.n_neighbors or 15, n_components=5, min_dist=0.0,
                               metric="cosine", random_state=args.seed),
        }
        topic_model = BERTopic(embedding_model=embed_model, calculate_probabilities=False, verbose=True, **kwargs)
        local_topics, _ = topic_model.fit_transform(docs)
        n_noise_before = sum(1 for t in local_topics if t == -1)

        if args.reassign_noise and n_noise_before and hdbscan_model.labels_.max() >= 0:
            # Reassign noise points to their highest soft-membership cluster (no threshold; merge absorbs errors).
            soft_clusters = all_points_membership_vectors(hdbscan_model)
            local_topics = [int(soft_clusters[i].argmax()) if t == -1 else t for i, t in enumerate(local_topics)]
            n_reassigned = n_noise_before - sum(1 for t in local_topics if t == -1)
            print(f"[{args.dimension}] {group_label}: reassigned {n_reassigned}/{n_noise_before} noise point(s)")

        # Noise (-1) stays one shared bucket; only real cluster ids are remapped.
        local_ids = sorted(set(local_topics) - {-1})
        remap = {local_id: next_cluster_id + i for i, local_id in enumerate(local_ids)}
        next_cluster_id += len(local_ids)

        all_descriptions.extend(descriptions)
        all_topics.extend(remap.get(t, -1) for t in local_topics)

    out_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = out_dir / f"{args.dimension}.raw_assignments.jsonl"
    with assignments_path.open("w") as f:
        for r, t in zip(all_descriptions, all_topics):
            f.write(json.dumps({"run": r["run"], "cluster_id": int(t), "description": r["description"]}) + "\n")

    members_by_cluster: dict[int, list[str]] = {}
    grid_counts_by_cluster: dict[int, dict[str, int]] = {}
    for r, t in zip(all_descriptions, all_topics):
        members_by_cluster.setdefault(t, []).append(r["description"])
        grid = r["run"].split("/", 1)[0]
        grid_counts = grid_counts_by_cluster.setdefault(t, {})
        grid_counts[grid] = grid_counts.get(grid, 0) + 1

    # Random, not the first N: members are in stage1-loading order (grid by grid), so a
    # first-N slice of a cross-grid or large cluster under-represents whichever part sorted later.
    example_rng = random.Random(args.seed)
    clusters = [
        {"cluster_id": cid, "count": len(members), "per_grid_counts": grid_counts_by_cluster[cid],
         "examples": example_rng.sample(members, min(args.n_examples, len(members)))}
        for cid, members in sorted(members_by_cluster.items())
    ]
    (out_dir / f"{args.dimension}.raw_clusters.json").write_text(json.dumps(clusters, indent=2))
    n_real = sum(1 for c in clusters if c["cluster_id"] != -1)
    n_noise = next((c["count"] for c in clusters if c["cluster_id"] == -1), 0)
    print(f"[{args.dimension}] {n_real} clusters found (+ {n_noise} noise points) across "
          f"{len(args.stage1_dirs)} grid(s), wrote to {out_dir}")


# --- label: one LLM call per cluster on random examples; general label, specifics in the criterion.

# Per-dimension (too specific, general) label pairs for the label and merge prompts. Both sides name
# the same behavior, since few-shot style outweighs instructions.
DIMENSION_LABEL_EXAMPLES = {
    "result": ("Finished with a real attempt but scored below the trivial baseline", "Genuine Failure"),
    "error_category": ("Used random sampling instead of the documented greedy decoding setting", "Protocol Deviation"),
    "execution_quality": ("Extracted the answer letter from the wrong character position in the output", "Output Parsing Bug"),
    "model_usage": ("Skipped the documented preprocessing step and used custom code instead", "Off-Protocol Model Usage"),
    "planning_exploration": ("Committed early to using the model only as a frozen feature extractor", "Premature Single-Approach Commitment"),
    "verification_behavior": ("Checked that the output file had the correct number of rows and columns", "Format-Only Verification"),
}


def _dimension_examples(dimension: str) -> tuple[str, str]:
    return DIMENSION_LABEL_EXAMPLES[dimension]


LABEL_SYSTEM_PROMPT_TEMPLATE = (
    "You are naming one cluster of findings for the \"{label}\" dimension of an outcome taxonomy "
    "over autonomous research-agent trajectories drawn from MULTIPLE different tasks.\n\n"
    "Label: a short noun phrase, AT MOST 4 words, naming the GENERAL behavior -- the kind of thing "
    "worth its own category in a taxonomy of at most {max_categories} categories for this "
    "dimension, not the specific technical mechanism. Put the specific mechanism in the criterion "
    "instead. For example, a description as specific and wordy as \"{bad_example}\" should become "
    "the label \"{good_example}\" -- same behavior, named at the general level a whole taxonomy "
    "category needs, with the specific detail kept in the criterion.\n\n"
    "Pick one specific phrasing and commit to it -- never join two near-synonymous ways of naming "
    "the same behavior with a slash (e.g. \"No Error / Successful Completion\") or \"and\"; that "
    "hedging belongs in the criterion, not stacked into the label itself.\n\n"
    "Sometimes the examples don't actually share one behavior -- most might report a bug but a few "
    "report none, or most describe a failed run but a few describe a success. When that happens, "
    "write the label and criterion for whichever behavior most of the examples show, and ignore "
    "the rest -- never write one label that tries to cover both (e.g. never something like "
    "\"succeeded or failed\").\n\n"
    "Respond with a single JSON object matching exactly the fields requested, and nothing else. "
    "Make sure the JSON is syntactically valid: escape any quotation marks that appear inside a "
    "string value."
)


def run_label(args, base_url: str, model: str, out_dir: Path) -> None:
    clusters_path = out_dir / f"{args.dimension}.raw_clusters.json"
    if not clusters_path.exists():
        raise SystemExit(f"{clusters_path} not found -- run --phase cluster first")
    clusters = json.loads(clusters_path.read_text())
    dim_label = _dimension_label(args.dimension)
    bad_example, good_example = _dimension_examples(args.dimension)
    system_prompt = LABEL_SYSTEM_PROMPT_TEMPLATE.format(
        label=dim_label, max_categories=MAX_CATEGORIES, bad_example=bad_example, good_example=good_example)
    sampling = SAMPLING[args.tier]
    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=CLIENT_TIMEOUT_S, max_retries=0)

    def label_one(cluster: dict) -> dict:
        example_text = "\n".join(f"  - {e}" for e in cluster["examples"])
        user = (f"Cluster size: {cluster['count']} (spanning grids: {', '.join(cluster['per_grid_counts'])})\n"
                f"Example findings:\n{example_text}\n\n"
                "Respond with a JSON object with exactly these fields: {\"label\": \"...\", "
                "\"criterion\": \"...\"}")
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
        result = _call_and_parse(client, model, sampling, messages, args.max_tokens)
        if result is None:
            result = _call_and_parse(client, model, sampling, messages, args.max_tokens)  # one retry
        if result is None:
            return {**cluster, "label": f"cluster_{cluster['cluster_id']}", "criterion": "", "ok": False}
        return {**cluster, "label": result["label"], "criterion": result["criterion"], "ok": True}

    out_path = out_dir / f"{args.dimension}.labeled_clusters.jsonl"
    done_ids = {json.loads(l)["cluster_id"] for l in out_path.read_text().splitlines() if l.strip()} \
        if out_path.exists() else set()
    todo = [c for c in clusters if c["cluster_id"] != -1 and c["cluster_id"] not in done_ids]
    n_real = sum(1 for c in clusters if c["cluster_id"] != -1)
    print(f"[{args.dimension}] {len(todo)}/{n_real} cluster(s) to label ({len(done_ids)} already done)")

    if todo:
        with out_path.open("a") as f, ThreadPoolExecutor(max_workers=args.max_num_seqs) as pool:
            futures = {pool.submit(label_one, c): c for c in todo}
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                f.write(json.dumps(result) + "\n")
                f.flush()
                os.fsync(f.fileno())
                print(f"[label] {i}/{len(todo)} done (cluster {result['cluster_id']} -> {result['label']!r})")
    print(f"[{args.dimension}] labeling finished, wrote to {out_path}")


# --- merge: one LLM call assigns every label to one of at most MAX_CATEGORIES categories.

MERGE_SYSTEM_PROMPT_TEMPLATE = (
    "You are grouping a list of category labels for the \"{label}\" dimension of an outcome "
    "taxonomy over autonomous research-agent trajectories, into AT MOST {max_categories} general, "
    "high-level behavioral categories. Each label was written independently for one cluster of "
    "similar findings; the criterion is context, not part of the label. Partition every label into "
    "at most {max_categories} groups -- prefer fewer, broader groups over preserving fine "
    "distinctions: when in doubt, merge. A category should be the kind of thing worth reporting on "
    "its own in a paper: a label as specific and wordy as \"{bad_example}\" belongs folded into a "
    "general group like \"{good_example}\", with the specific detail kept in that group's "
    "criterion, not given its own category.\n\n"
    "Label style: a short noun phrase, AT MOST 4 words, naming the general behavior -- not a "
    "sentence, not a specific technical mechanism. Pick one specific phrasing and commit to it -- "
    "never join two near-synonymous ways of naming the same behavior with a slash (e.g. \"No Error "
    "/ Successful Completion\") or \"and\"; that hedging belongs in the group's criterion, not "
    "stacked into its label.\n\n"
    "Respond with a single JSON object with exactly this field: {{\"groups\": [{{\"label\": \"...\", "
    "\"criterion\": \"...\", \"member_labels\": [...]}}, ...]}}. member_labels lists, for each "
    "group, every original label (copied EXACTLY from before its first \":\" below) assigned to "
    "it. Every original label must appear in exactly one group's member_labels -- there is no "
    "\"leave it standing alone\" option; assign every label to its single best-fitting group, even "
    "a loose fit. Make sure the JSON is syntactically valid: escape any quotation marks that appear "
    "inside a string value."
)


def _merge_whole_list(labeled_clusters: list[dict], dim_label: str, args, base_url: str,
                       model: str, out_path: Path) -> list[dict]:
    """One merge call over all labels, resumable from out_path; reports failure reasons, unlike _call_and_parse."""
    if out_path.exists():
        return json.loads(out_path.read_text())["groups"]

    sampling = SAMPLING[args.tier]
    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=CLIENT_TIMEOUT_S, max_retries=0)
    bad_example, good_example = _dimension_examples(args.dimension)
    blocks = [f"- {c['label']}: {c['criterion']}" for c in labeled_clusters]
    system_prompt = MERGE_SYSTEM_PROMPT_TEMPLATE.format(
        label=dim_label, max_categories=MAX_CATEGORIES, bad_example=bad_example, good_example=good_example)
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": "## Cluster labels\n\n" + "\n".join(blocks)}]
    n_attempts = 3  # connection failures were frequent, and this call is expensive to lose
    for attempt in range(n_attempts):
        # Catch connection errors too: a crash here would lose the whole dimension's labels.
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=sampling["temperature"], top_p=sampling["top_p"],
                max_tokens=args.merge_max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}, **sampling["extra_sampling"]})
        except Exception as exc:
            print(f"[merge] attempt {attempt + 1}/{n_attempts} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        choice = response.choices[0]
        text = (choice.message.content or "").strip().strip("`")
        if text.startswith("json"):
            text = text[4:]
        try:
            result = json.loads(text.strip())
            groups = result["groups"]
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"[merge] attempt {attempt + 1}/{n_attempts} failed to parse ({exc}) -- finish_reason="
                  f"{choice.finish_reason}, completion_tokens={response.usage.completion_tokens}, "
                  f"reasoning_tokens={response.usage.completion_tokens_details.reasoning_tokens}",
                  file=sys.stderr)
            continue
        if len(groups) > MAX_CATEGORIES:
            print(f"[merge] warning: model returned {len(groups)} groups, exceeds MAX_CATEGORIES={MAX_CATEGORIES}",
                  file=sys.stderr)
        out_path.write_text(json.dumps(result, indent=2))
        return groups
    raise SystemExit(f"merge whole-list call failed to return valid JSON in {n_attempts} attempts")


def run_merge(args, base_url: str, model: str, out_dir: Path) -> None:
    label_path = out_dir / f"{args.dimension}.labeled_clusters.jsonl"
    raw_assignments_path = out_dir / f"{args.dimension}.raw_assignments.jsonl"
    if not label_path.exists():
        raise SystemExit(f"{label_path} not found -- run --phase label first")
    labeled_clusters = [json.loads(l) for l in label_path.read_text().splitlines() if l.strip()]
    dim_label = _dimension_label(args.dimension)

    final_label_of: dict[int, str] = {c["cluster_id"]: c["label"] for c in labeled_clusters}
    final_criterion_of: dict[str, str] = {c["label"]: c["criterion"] for c in labeled_clusters}
    final_label_of[-1] = NOISE_LABEL
    final_criterion_of.setdefault(NOISE_LABEL, "Did not fit any coherent cluster of similar findings.")

    if len(labeled_clusters) < 2:
        print(f"[{args.dimension}] only {len(labeled_clusters)} labeled cluster(s), nothing to merge")
    elif len(labeled_clusters) > WHOLE_LIST_MERGE_MAX:
        raise SystemExit(f"[{args.dimension}] {len(labeled_clusters)} labeled clusters exceeds "
                          f"WHOLE_LIST_MERGE_MAX ({WHOLE_LIST_MERGE_MAX}) -- raise the limit or "
                          f"batch the list instead")
    else:
        out_path = out_dir / f"{args.dimension}.merge_proposal.json"
        groups = _merge_whole_list(labeled_clusters, dim_label, args, base_url, model, out_path)
        print(f"[{args.dimension}] whole-list merge: {len(labeled_clusters)} labeled clusters -> {len(groups)} groups")

        cluster_ids_by_label: dict[str, list[int]] = {}
        for c in labeled_clusters:
            cluster_ids_by_label.setdefault(c["label"], []).append(c["cluster_id"])

        n_matched, n_unmatched = 0, 0
        for group in groups:
            group_label = group["label"]
            final_criterion_of[group_label] = group.get("criterion", "")
            for member_label in group["member_labels"]:
                cluster_ids = cluster_ids_by_label.get(member_label) or cluster_ids_by_label.get(member_label.split(":", 1)[0])
                if cluster_ids:
                    for cid in cluster_ids:
                        final_label_of[cid] = group_label
                    n_matched += 1
                else:
                    n_unmatched += 1
                    print(f"[merge] warning: could not match label {member_label!r} to any labeled "
                          f"cluster, leaving it unmerged", file=sys.stderr)
        if n_unmatched:
            print(f"[{args.dimension}] {n_matched}/{n_matched + n_unmatched} merge-group labels matched")

    raw_rows = [json.loads(l) for l in raw_assignments_path.read_text().splitlines() if l.strip()]
    assignments_path = out_dir / f"{args.dimension}.assignments.jsonl"
    category_counts: dict[str, int] = {}
    with assignments_path.open("w") as f:
        for row in raw_rows:
            label = final_label_of.get(row["cluster_id"], f"cluster_{row['cluster_id']}")
            category_counts[label] = category_counts.get(label, 0) + 1
            f.write(json.dumps({"run": row["run"], "label": label, "description": row["description"]}) + "\n")

    final_categories = sorted(
        ({"label": l, "criterion": final_criterion_of.get(l, ""), "count": c} for l, c in category_counts.items()),
        key=lambda c: -c["count"])
    (out_dir / f"{args.dimension}.categories.json").write_text(json.dumps(final_categories, indent=2))
    print(f"[{args.dimension}] merge finished: {len(labeled_clusters)} labeled clusters -> "
          f"{len(final_categories)} final categories, wrote to {out_dir}")


# --- valence: one LLM call scores each category from -1 to +1, stored in categories.json.

VALENCE_SYSTEM_PROMPT_TEMPLATE = (
    "You are rating how positive or negative each category is for the \"{label}\" dimension of an "
    "outcome taxonomy over autonomous research-agent trajectories. For each category, give a "
    "valence score from -1.0 to 1.0: +1.0 means the behavior it describes is clearly good (the "
    "agent succeeded, complied with the documented protocol, or verified its work properly); -1.0 "
    "means it's clearly bad (a failure, a bug, or a protocol violation); 0.0 means neutral, mixed, "
    "or ambiguous -- not clearly good or bad on its own. Don't just sort categories into good/bad -- "
    "think about how severe or consequential each one's behavior actually is on its own terms. A "
    "category whose behavior fully invalidates the run (no usable output, ground-truth access) "
    "should score more negatively than one describing a minor, recoverable issue, even if both are "
    "technically errors. Use the full range rather than clustering everything at the extremes.\n\n"
    "Respond with a single JSON object with exactly this field: {{\"valences\": [{{\"label\": "
    "\"...\", \"valence\": 0.0}}, ...]}}, one entry per category listed below, using \"label\" "
    "exactly as given. Make sure the JSON is syntactically valid."
)


def run_valence(args, base_url: str, model: str, out_dir: Path) -> None:
    categories_path = out_dir / f"{args.dimension}.categories.json"
    if not categories_path.exists():
        raise SystemExit(f"{categories_path} not found -- run --phase merge first")
    categories = json.loads(categories_path.read_text())
    if categories and all("valence" in c for c in categories):
        print(f"[{args.dimension}] valence already scored for all {len(categories)} categories")
        return

    dim_label = _dimension_label(args.dimension)
    sampling = SAMPLING[args.tier]
    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=CLIENT_TIMEOUT_S, max_retries=0)
    blocks = [f"- {c['label']}: {c['criterion']}" for c in categories]
    messages = [{"role": "system", "content": VALENCE_SYSTEM_PROMPT_TEMPLATE.format(label=dim_label)},
                {"role": "user", "content": "## Categories\n\n" + "\n".join(blocks)}]
    # 3 attempts with backoff for transient server errors when several dimensions finish together.
    result = None
    for attempt in range(3):
        result = _call_and_parse(client, model, sampling, messages, args.max_tokens)
        if result is not None:
            break
        if attempt < 2:
            print(f"[{args.dimension}] valence attempt {attempt + 1}/3 failed, retrying after a short wait", file=sys.stderr)
            time.sleep(10)
    if result is None or "valences" not in result:
        print(f"[{args.dimension}] warning: valence call failed after 3 attempts, categories.json left unscored", file=sys.stderr)
        return

    valence_of = {v["label"]: v["valence"] for v in result["valences"] if "label" in v and "valence" in v}
    for c in categories:
        if c["label"] in valence_of:
            c["valence"] = valence_of[c["label"]]
        else:
            print(f"[{args.dimension}] warning: no valence returned for {c['label']!r}", file=sys.stderr)
    categories_path.write_text(json.dumps(categories, indent=2))
    n_scored = sum(1 for c in categories if "valence" in c)
    print(f"[{args.dimension}] valence: scored {n_scored}/{len(categories)} categories, wrote to {categories_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dimension", required=True, choices=CLUSTERABLE_DIMENSIONS)
    parser.add_argument("--phase", default="all", choices=["cluster", "label", "merge", "valence", "all"])
    parser.add_argument("--stage1-dirs", nargs="+", default=None,
                         help="Stage 1 grid directories, combined into one category system (required for cluster/all)")
    parser.add_argument("--output-dir", default=None,
                         help="default: stage2_output/<grid_label of --stage1-dirs>; required without --stage1-dirs")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME, help="cluster only")
    parser.add_argument("--min-cluster-size", type=int, default=None,
                         help="cluster only: HDBSCAN min_cluster_size (default: BERTopic's 10)")
    parser.add_argument("--n-examples", type=int, default=10,
                         help="cluster only: random findings stored per cluster; also the label phase's examples")
    parser.add_argument("--n-neighbors", type=int, default=None,
                         help="cluster only: UMAP param -- default None keeps BERTopic's own "
                              "default (15)")
    parser.add_argument("--cluster-selection-method", default="eom", choices=["eom", "leaf"],
                         help="cluster only: HDBSCAN eom (default, fewer larger clusters) or leaf (smaller, purer clusters)")
    parser.add_argument("--seed", type=int, default=0, help="cluster only: seed for UMAP and example sampling")
    parser.add_argument("--cluster-per-grid", action="store_true",
                         help="cluster each grid separately instead of pooling (comparison runs)")
    parser.add_argument("--no-reassign-noise", action="store_true",
                         help="leave HDBSCAN noise points uncategorized (comparison runs)")
    parser.add_argument("--finding-only", action="store_true",
                         help="cluster on \"finding\" alone, without \"why\" (tested and rejected)")
    parser.add_argument("--tier", default="deepseek-v4-flash-0731-fp8")
    parser.add_argument("--max-num-seqs", type=int, default=4, help="client request concurrency; merge is sequential, label is parallel (e.g. 10)")
    parser.add_argument("--server-max-num-seqs", type=int, default=None,
                         help="server batch size when this invocation submits a shared server (see run_stage2.sh)")
    parser.add_argument("--max-model-len", type=int, default=81920)
    parser.add_argument("--gpu-mem-util", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=8192, help="label's per-item completion budget")
    parser.add_argument("--merge-max-tokens", type=int, default=65536, help="merge's whole-list call only")
    parser.add_argument("--ready-timeout-s", type=int, default=86400,
                         help="give up on (and cancel, unless --keep-server) a server not ready after this many seconds; queues can be long")
    parser.add_argument("--label", default="stage2")
    parser.add_argument("--keep-server", action="store_true",
                         help="do not cancel the judge server on exit; needed when sharing one server (see run_stage2.sh)")
    parser.add_argument("--attach-job-id", default=None,
                         help="use this running judge server job instead of submitting one (see run_stage2.sh)")
    args = parser.parse_args()
    if args.tier not in SAMPLING:
        raise SystemExit(f"no SAMPLING entry for tier {args.tier!r}")
    if args.phase in ("cluster", "all") and not args.stage1_dirs:
        raise SystemExit(f"--phase {args.phase} requires --stage1-dirs")
    args.cluster_combined = not args.cluster_per_grid
    args.reassign_noise = not args.no_reassign_noise
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.stage1_dirs:
        out_dir = REPO_ROOT / "scripts/taxonomy/stage2_output" / grid_label_from_dirs(args.stage1_dirs)
    else:
        raise SystemExit("--output-dir required when --stage1-dirs isn't given")

    # cluster runs before (and for --phase cluster, without) the judge server.
    if args.phase in ("cluster", "all"):
        run_cluster(args, out_dir)
    if args.phase == "cluster":
        return 0

    cache_root_result = judge_server._run(["bash", "-c", "set -a; . .env; set +a; echo $CACHE_ROOT"], cwd=REPO_ROOT)
    cache_root = cache_root_result.stdout.strip()
    if not cache_root:
        raise SystemExit("could not resolve CACHE_ROOT from .env")

    job_id = args.attach_job_id or judge_server.submit_server(args)
    try:
        log_path = judge_server.wait_ready(args, job_id, cache_root, args.ready_timeout_s)
        addr = Path(cache_root, f"taxonomy-judge-{job_id}.addr").read_text().strip()
        resources = judge_server.capture_resources(args, job_id, log_path, addr)
        if resources["served_model_id"] is None:
            raise SystemExit(f"[{args.label}] server answered readiness probe but /v1/models had no data")
        model = resources["served_model_id"]
        base_url = f"http://{addr}/v1"

        if args.phase in ("label", "all"):
            run_label(args, base_url, model, out_dir)
        if args.phase in ("merge", "all"):
            run_merge(args, base_url, model, out_dir)
        if args.phase in ("valence", "all"):
            run_valence(args, base_url, model, out_dir)
    finally:
        if not args.keep_server:
            print(f"[{args.label}] tearing down job {job_id}")
            judge_server._run(["scancel", job_id])
    return 0


if __name__ == "__main__":
    sys.exit(main())
