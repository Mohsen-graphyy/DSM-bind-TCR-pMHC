"""Paired affinity ranking evaluation; labels never enter structure scoring."""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from .benchmark_io import read_csv, write_csv, write_json

SCORE_FIELDS = ("score_sum", "peptide_score_sum", "score_mean", "score_per_contact")
PRIMARY_SCORE = "score_sum"  # fixed before labels; never select sign/aggregation on test


def finite_float(value, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {label}")
    return number


def rank_metrics(truth: list[float], scores: list[float]) -> dict:
    n = len(truth)
    eligible = n >= 3 and len(set(truth)) > 1 and len(set(scores)) > 1
    reason = ("fewer_than_3_units" if n < 3 else "constant_truth" if len(set(truth)) < 2
              else "constant_prediction" if len(set(scores)) < 2 else "")
    concordant = 0.0
    comparisons = 0
    for i in range(n):
        for j in range(i):
            if truth[i] == truth[j]:
                continue
            comparisons += 1
            product = (truth[i] - truth[j]) * (scores[i] - scores[j])
            concordant += 1.0 if product > 0 else 0.5 if product == 0 else 0.0
    top = [i for i, value in enumerate(scores) if value == max(scores)] if n else []
    return {"n_units": n, "eligible_rank_correlation": eligible, "ineligible_reason": reason,
            "spearman": float(stats.spearmanr(truth, scores).statistic) if eligible else None,
            "kendall_tau_b": float(stats.kendalltau(truth, scores).statistic) if eligible else None,
            "pairwise_accuracy": concordant / comparisons if comparisons else None,
            "non_tied_truth_pairs": comparisons,
            "top1_credit": (sum(truth[i] == max(truth) for i in top) / len(top)) if n >= 2 else None}


def load_scores(directory: Path, expected_structures: set[str]) -> dict[str, dict]:
    complexes = read_csv(directory / "complex_scores.csv")
    if len(complexes) != len({r["structure_id"] for r in complexes}):
        raise ValueError(f"Duplicate complex score IDs in {directory}")
    if {r["structure_id"] for r in complexes} != expected_structures:
        raise ValueError(f"Missing/extra scored structures in {directory}")
    pairs = read_csv(directory / "pair_scores.csv")
    seen, summed, peptide, counts, contacts = set(), defaultdict(float), defaultdict(float), defaultdict(int), defaultdict(int)
    for r in pairs:
        key = (r["structure_id"], r["binder_chain"], r["target_chain"])
        if key in seen or r["structure_id"] not in expected_structures:
            raise ValueError(f"Duplicate/unknown pair {key}")
        seen.add(key)
        value = finite_float(r["symmetric_score"], "pair score")
        forward = finite_float(r["forward_score"], "forward score")
        reverse = finite_float(r["reverse_score"], "reverse score")
        if not math.isclose(value, forward + reverse, rel_tol=2e-6, abs_tol=2e-3):
            raise ValueError(f"Forward/reverse sum mismatch for {key}")
        sid = r["structure_id"]
        summed[sid] += value
        counts[sid] += 1
        contacts[sid] += int(r["contact_count"])
        if r["target_chain"] == "P":
            peptide[sid] += value
    result = {}
    for r in complexes:
        sid = r["structure_id"]
        total = finite_float(r["score_sum"], "complex score")
        if (counts[sid] != int(r["pair_count"]) or contacts[sid] != int(float(r["total_contacts"]))
                or not math.isclose(total, summed[sid], rel_tol=2e-6, abs_tol=2e-3)):
            raise ValueError(f"Pair/complex aggregation mismatch for {sid}")
        for field, expected_value in (("score_mean", total / counts[sid]),
                                      ("score_per_contact", total / max(contacts[sid], 1))):
            if not math.isclose(finite_float(r[field], field), expected_value, rel_tol=2e-6, abs_tol=2e-3):
                raise ValueError(f"{field} aggregation mismatch for {sid}")
        result[sid] = {field: finite_float(r[field], field) for field in SCORE_FIELDS if field != "peptide_score_sum"}
        result[sid].update(peptide_score_sum=peptide[sid], pair_count=counts[sid],
                           has_peptide_contacts=any(key[0] == sid and key[2] == "P" for key in seen))
    return result


def aggregate_units(observations: list[dict], scores: dict, grouping: str) -> list[dict]:
    grouped = defaultdict(list)
    for r in observations:
        if r.get("delta_g_kcal_mol", "") == "":
            continue
        group = r["peptide"] if grouping == "peptide" else r.get("strict_group_id", "")
        if group:
            grouped[(group, r["unit_id"])].append(r)
    units = []
    for (group, uid), rows in sorted(grouped.items()):
        energies = [finite_float(r["delta_g_kcal_mol"], "delta G") for r in rows]
        structures = sorted({r["structure_id"] for r in rows})
        unit = {"grouping": grouping, "group_id": group, "unit_id": uid,
                "peptide": rows[0]["peptide"], "n_observations": len(rows), "n_structures": len(structures),
                "structure_ids": ";".join(structures), "observation_ids": ";".join(r["observation_id"] for r in rows),
                "delta_g_median": statistics.median(energies), "delta_g_min": min(energies),
                "delta_g_max": max(energies), "affinity_strength": -statistics.median(energies),
                "mixed_assay_context": len({r.get("strict_group_id", "") for r in rows}) > 1}
        for model in ("pretrained", "finetuned"):
            for field in SCORE_FIELDS:
                unit[f"{model}_{field}"] = statistics.mean(scores[model][sid][field] for sid in structures)
        units.append(unit)
    return units


def evaluate(bundle: Path, work: Path, output: Path, bootstrap: int = 2000, seed: int = 7) -> dict:
    from .benchmark_io import verify_bundle
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    structures, observations = verify_bundle(bundle)
    expected = {r["structure_id"] for r in structures}
    scores = {name: load_scores(work / name, expected) for name in ("pretrained", "finetuned")}
    # Same interfaces must have been scored for both checkpoints.
    structural_fields = ("structure_id", "binder_chain", "target_chain", "contact_count", "min_distance")
    pair_lists = [read_csv(work / name / "pair_scores.csv") for name in ("pretrained", "finetuned")]
    if {tuple(r[k] for k in structural_fields) for r in pair_lists[0]} != {
        tuple(r[k] for k in structural_fields) for r in pair_lists[1]
    }:
        raise ValueError("The two checkpoints were scored on different interfaces")
    geometry = defaultdict(int)
    for r in read_csv(work / "geometry_pairs.csv"):
        geometry[r["structure_id"]] += int(r["atom_pairs_14A"])
    if set(geometry) != expected:
        raise ValueError("Geometry baseline does not cover exactly the scored structures")
    units_all, group_metrics, rankings = [], [], []
    for grouping in ("peptide", "assay_context"):
        units = aggregate_units(observations, scores, grouping)
        for u in units:
            sids = u["structure_ids"].split(";")
            u["geometry_n14"] = statistics.mean(geometry[sid] for sid in sids)
        units_all.extend(units)
        groups = defaultdict(list)
        for unit in units:
            groups[unit["group_id"]].append(unit)
        for group, rows in sorted(groups.items()):
            truth = [r["affinity_strength"] for r in rows]
            for model in ("pretrained", "finetuned", "geometry"):
                for field in SCORE_FIELDS if model != "geometry" else ("n14",):
                    predictions = [r[f"{model}_{field}"] for r in rows]
                    metric = rank_metrics(truth, predictions)
                    group_metrics.append({"grouping": grouping, "group_id": group, "model": model,
                                          "score_field": field, "primary": field == PRIMARY_SCORE, **metric})
            truth_ranks = stats.rankdata(-np.asarray(truth), method="average")
            pre_ranks = stats.rankdata(-np.asarray([r[f"pretrained_{PRIMARY_SCORE}"] for r in rows]), method="average")
            fine_ranks = stats.rankdata(-np.asarray([r[f"finetuned_{PRIMARY_SCORE}"] for r in rows]), method="average")
            for i, row in enumerate(rows):
                rankings.append({**row, "experimental_rank": float(truth_ranks[i]),
                                 "pretrained_rank": float(pre_ranks[i]), "finetuned_rank": float(fine_ranks[i]),
                                 "pretrained_abs_rank_error": float(abs(truth_ranks[i] - pre_ranks[i])),
                                 "finetuned_abs_rank_error": float(abs(truth_ranks[i] - fine_ranks[i]))})
    summaries = []
    for grouping in ("peptide", "assay_context"):
        for model in ("pretrained", "finetuned", "geometry"):
            for field in SCORE_FIELDS if model != "geometry" else ("n14",):
                rows = [r for r in group_metrics if (r["grouping"], r["model"], r["score_field"]) == (grouping, model, field)]
                eligible = [r for r in rows if r["eligible_rank_correlation"]]
                pair_rows = [r for r in rows if r["non_tied_truth_pairs"]]
                summaries.append({"grouping": grouping, "model": model, "score_field": field,
                                  "primary": field == PRIMARY_SCORE, "total_groups": len(rows),
                                  "rank_eligible_groups": len(eligible),
                                  "constant_prediction_groups": sum(r["ineligible_reason"] == "constant_prediction" for r in rows),
                                  "macro_spearman": statistics.mean(r["spearman"] for r in eligible) if eligible else None,
                                  "macro_kendall_tau_b": statistics.mean(r["kendall_tau_b"] for r in eligible) if eligible else None,
                                  "macro_pairwise_accuracy": statistics.mean(r["pairwise_accuracy"] for r in pair_rows) if pair_rows else None,
                                  "comparable_unit_pairs": sum(r["non_tied_truth_pairs"] for r in pair_rows)})
    comparisons = []
    rng = np.random.default_rng(seed)
    for grouping in ("peptide", "assay_context"):
        for field in SCORE_FIELDS:
            lookup = {(r["group_id"], r["model"]): r for r in group_metrics
                      if r["grouping"] == grouping and r["score_field"] == field}
            shared = sorted(g for g, m in lookup if m == "pretrained"
                            and lookup[(g, m)]["eligible_rank_correlation"]
                            and lookup[(g, "finetuned")]["eligible_rank_correlation"])
            differences = np.asarray([lookup[(g, "finetuned")]["spearman"] - lookup[(g, "pretrained")]["spearman"] for g in shared])
            ci = None
            if len(shared) >= 2 and bootstrap:
                draws = rng.choice(differences, size=(bootstrap, len(differences)), replace=True).mean(axis=1)
                ci = [float(x) for x in np.quantile(draws, [.025, .975])]
            comparisons.append({"grouping": grouping, "score_field": field, "shared_eligible_groups": len(shared),
                                "mean_delta_spearman_finetuned_minus_pretrained": float(differences.mean()) if len(shared) else None,
                                "paired_group_bootstrap_95ci": ci,
                                "bootstrap_note": "Descriptive group bootstrap; groups may share TCRs/templates. Not an independence proof."})
    expanded = []
    for r in observations:
        item = dict(r)
        for model in ("pretrained", "finetuned"):
            item.update({f"{model}_{k}": v for k, v in scores[model][r["structure_id"]].items()})
        expanded.append(item)
    geometry_diagnostics = []
    for name in ("pretrained", "finetuned"):
        x = np.asarray([geometry[s] for s in sorted(expected)], dtype=float)
        y = np.asarray([scores[name][s][PRIMARY_SCORE] for s in sorted(expected)])
        coefficient = float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) else None
        total_variation = float(((y - y.mean()) ** 2).sum())
        geometry_diagnostics.append({"model": name, "pearson_score_vs_n14": float(stats.pearsonr(x, y).statistic)
                                     if len(x) >= 2 and len(set(x)) > 1 and len(set(y)) > 1 else None,
                                     "coefficient": coefficient,
                                     "r2_counting_approximation": float(1 - ((y - coefficient * x) ** 2).sum() / total_variation)
                                     if coefficient is not None and total_variation > 0 else None})
    result = {"primary_score": PRIMARY_SCORE, "score_direction": "higher", "truth": "-experimental delta G",
              "observations": len(observations), "labeled_observations": sum(r.get("delta_g_kcal_mol", "") != "" for r in observations),
              "unique_structures": len(expected), "biological_units": len({r["unit_id"] for r in observations}),
              "repeat_policy": "median delta G per biological unit within each group; mean score across distinct structures, never across repeated rows",
              "summaries": summaries, "paired_comparisons": comparisons, "geometry_diagnostics": geometry_diagnostics,
              "bootstrap_samples": bootstrap, "seed": seed,
              "limitations": ["Affinity ranking benchmark, not IMMREP functional reactivity or its official metric.",
                              "Peptide-only groups can mix HLA, assay, temperature and publication; assay_context is a separate analysis.",
                              "Pretraining/fine-tuning corpus overlap is unknown; ATLAS is not certified independent.",
                              "Checkpoint audit concerns remain; this benchmark measures them rather than repairs training.",
                              "No absolute predicted delta G, probability, RMSE or unit conversion is claimed.",
                              "No sign or score aggregation is selected to maximize ATLAS performance."]}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "observation_scores.csv", expanded)
    write_csv(output / "unit_scores.csv", units_all)
    write_csv(output / "rankings.csv", rankings)
    write_csv(output / "per_group_metrics.csv", group_metrics)
    write_json(output / "metrics.json", result)
    lines = ["# ATLAS / DSMBind paired ranking benchmark", "", "Primary score: score_sum; higher is stronger. Experimental ranking: -delta G.",
             "", f"Observations: {len(observations)}; unique structures: {len(expected)}; biological units: {result['biological_units']}.",
             "", "Repeated measurements use median experimental delta G; duplicate structures are scored once. Ranks use average ties, rank 1 = strongest.",
             "", "| Grouping | Model | Score | Eligible groups | Macro Spearman | Macro pairwise accuracy |", "|---|---|---|---:|---:|---:|"]
    def display(value):
        return "undefined" if value is None else f"{value:.5f}"
    for r in summaries:
        lines.append(f"| {r['grouping']} | {r['model']} | {r['score_field']} | {r['rank_eligible_groups']}/{r['total_groups']} | {display(r['macro_spearman'])} | {display(r['macro_pairwise_accuracy'])} |")
    lines += ["", "## Paired changes (same eligible groups)", ""]
    for r in comparisons:
        lines.append(f"- {r['grouping']} / {r['score_field']}: delta Spearman={display(r['mean_delta_spearman_finetuned_minus_pretrained'])}; groups={r['shared_eligible_groups']}; descriptive 95% CI={r['paired_group_bootstrap_95ci']}.")
    lines += ["", "## Geometry diagnostics", "", "Do learned scores add information beyond the N14 atom-pair counting baseline?", ""]
    for r in geometry_diagnostics:
        lines.append(f"- {r['model']}: Pearson(score,N14)={display(r['pearson_score_vs_n14'])}; counting-approximation R2={display(r['r2_counting_approximation'])}.")
    lines += ["", "## Interpretation and limits", ""] + [f"- {s}" for s in result["limitations"]]
    lines += ["- Constant predictions have undefined rank correlation and are counted explicitly, not assigned a favorable score.",
              "- Groups with fewer than three units do not contribute to macro rank correlation; pairwise accuracy still includes non-tied pairs and gives tied predictions half credit.",
              "- Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC5860664/", "",
              "## Files", "", "- rankings.csv: unit-level experimental and model ranks by peptide and assay context.",
              "- observation_scores.csv: all original observations preserved, including repeated measurements.",
              "- metrics.json: all metrics, eligibility, paired bootstrap and caveats.",
              "- per_group_metrics.csv: all groups, including ineligible/constant groups."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
