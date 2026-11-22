"""Benchmark: label-masking only vs bidirectional order-swapping.

Real LLM calls (NOT mocked). Three phases, each resumable:

  generate  Make N Pro/Con argument pairs (one per hardcoded topic) and save
            them, so every judging run sees the SAME arguments.
  judge     Judge every pair in BOTH orders (pro_first, con_first), R repeats
            each, and append one JSON line per judge call.
  report    Read the saved judgments and compute the metrics. Needs no API
            keys. Every number printed here is computed from the saved files.

Run from the repo root:
    python tests/evals/benchmark_bidirectional.py all --n 24 --repeats 4
or one phase at a time:
    python tests/evals/benchmark_bidirectional.py generate --n 24
    python tests/evals/benchmark_bidirectional.py judge --n 24 --repeats 4
    python tests/evals/benchmark_bidirectional.py report

Outputs go to eval_runs/ (pairs.json, judgments.jsonl, summary.json,
pair_level.csv, top_discrepancies.csv).

Definitions (m = pro_total - con_total, on a 0-30 scale per side):
  Before  = label masking only, ONE presentation order per judgment.
  After   = bidirectional: average of the pro_first and con_first judgments.
  Second-position advantage = (second speaker's total) - (first speaker's).
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

CRITERIA = (
    "speaker_a_logic", "speaker_a_evidence", "speaker_a_persuasion",
    "speaker_b_logic", "speaker_b_evidence", "speaker_b_persuasion",
)
ORDERS = ("pro_first", "con_first")
THRESHOLD = 1.0  # same pass threshold as the old position_bias eval

TOPICS = [
    "Universal basic income should be adopted worldwide",
    "Social media should be regulated as a public utility",
    "Nuclear power should be expanded to fight climate change",
    "Standardized testing should be abolished in schools",
    "Remote work should be the default for office jobs",
    "Cities should ban private cars from their centers",
    "Genetically modified crops should be widely adopted",
    "University education should be free for everyone",
    "Zoos should be phased out",
    "Artificial intelligence should be regulated by an international body",
    "The four-day work week should become standard",
    "Homework should be eliminated in primary schools",
    "Space exploration deserves more public funding than it gets today",
    "Animal testing for cosmetics should be banned worldwide",
    "Cryptocurrencies should be treated as legal tender",
    "Professional athletes are overpaid",
    "Plastic bags should be banned nationwide",
    "Voting should be compulsory",
    "School uniforms should be mandatory",
    "Self-driving cars should be allowed on all public roads",
    "Fast fashion should be taxed heavily",
    "Online anonymity should be restricted",
    "Tourism caps should be imposed on popular historic cities",
    "Open-source AI model weights should be freely published",
]


# ---------------------------------------------------------------- paths / io

class Paths:
    def __init__(self, out_dir: Path):
        self.out = out_dir
        self.pairs = out_dir / "pairs.json"
        self.judgments = out_dir / "judgments.jsonl"
        self.summary = out_dir / "summary.json"
        self.pair_csv = out_dir / "pair_level.csv"
        self.top_csv = out_dir / "top_discrepancies.csv"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_pairs(paths: Paths) -> dict:
    if paths.pairs.exists():
        return json.loads(paths.pairs.read_text(encoding="utf-8"))
    return {"meta": {}, "pairs": []}


def _save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _load_judgments(paths: Paths) -> list[dict]:
    if not paths.judgments.exists():
        return []
    rows = []
    with open(paths.judgments, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _with_backoff(fn, retries: int = 6, base: float = 10.0):
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < retries - 1:
                delay = base * (2 ** attempt) + random.uniform(0, 1)
                print(f"  retry in {delay:.1f}s ({str(exc)[:120]})")
                time.sleep(delay)
    raise last  # type: ignore[misc]


def _base_state(topic: str, messages: list | None = None) -> dict:
    return {
        "session_id": "benchmark",
        "topic": topic,
        "position_pro": f"For: {topic}",
        "position_con": f"Against: {topic}",
        "round": 1,
        "max_rounds": 3,
        "messages": messages or [],
        "round_scores": [],
        "winner": None,
        "trace_id": None,
    }


# ------------------------------------------------------------------ generate

def cmd_generate(args, paths: Paths) -> None:
    from debate_gpt.agents import build_llms, make_con_node, make_pro_node

    paths.out.mkdir(parents=True, exist_ok=True)
    data = _load_pairs(paths)
    have = {p["pair_id"] for p in data["pairs"]}
    llms = build_llms()
    data["meta"].update(
        pro_model=str(getattr(llms["pro"], "model_name", "?")),
        con_model=str(getattr(llms["con"], "model_name", "?")),
        judge_model=str(getattr(llms["judge"], "model_name", "?")),
    )
    pro_node = make_pro_node(llms["pro"])
    con_node = make_con_node(llms["con"])

    n = min(args.n, len(TOPICS))
    for i, topic in enumerate(TOPICS[:n]):
        if i in have:
            continue
        print(f"[generate {i + 1}/{n}] {topic}")
        pro_msg = _with_backoff(lambda: pro_node(_base_state(topic)))["messages"][0]
        con_msg = _with_backoff(
            lambda: con_node(_base_state(topic, messages=[pro_msg]))
        )["messages"][0]
        data["pairs"].append({
            "pair_id": i,
            "topic": topic,
            "pro_text": pro_msg.content,
            "con_text": con_msg.content,
            "pro_words": len(pro_msg.content.split()),
            "con_words": len(con_msg.content.split()),
            "generated_at": _now(),
        })
        _save_json(paths.pairs, data)  # save after every pair
        time.sleep(args.sleep)
    print(f"pairs saved: {len(data['pairs'])} -> {paths.pairs}")


# --------------------------------------------------------------------- judge

def cmd_judge(args, paths: Paths) -> None:
    from debate_gpt.agents import build_llms, judge_once

    data = _load_pairs(paths)
    pairs = data["pairs"][: args.n]
    if not pairs:
        sys.exit("No pairs found. Run the 'generate' phase first.")
    done = {r["key"] for r in _load_judgments(paths)}
    judge_llm = build_llms()["judge"]
    data["meta"]["judge_model"] = str(getattr(judge_llm, "model_name", "?"))
    data["meta"]["judge_client"] = type(judge_llm).__name__  # e.g. ChatGroq
    _save_json(paths.pairs, data)

    total = len(pairs) * len(ORDERS) * args.repeats
    count = 0
    # repeat is the OUTER loop so an interrupted run still has complete
    # repeat sets for every pair.
    for r in range(args.repeats):
        for p in pairs:
            for order in ORDERS:
                count += 1
                key = f"{p['pair_id']}|{order}|{r}"
                if key in done:
                    continue
                stats: dict = {}
                state = _base_state(p["topic"])
                swap = order == "con_first"
                try:
                    score = _with_backoff(
                        lambda: judge_once(
                            judge_llm, state, p["pro_text"], p["con_text"],
                            swap=swap, stats=stats,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    # API/network failure: NOT written, so a re-run retries it.
                    print(f"  SKIPPED {key}: API error: {str(exc)[:150]}")
                    continue
                rec = {
                    "key": key, "pair_id": p["pair_id"], "order": order,
                    "repeat": r, "ts": _now(),
                    "tier": stats.get("tier"),
                    "llm_calls": stats.get("llm_calls"),
                    "errors": stats.get("errors", []),
                }
                if "parse_error" in score:
                    rec.update(ok=False, parse_error=score["parse_error"],
                               raw=str(score.get("raw", ""))[:500])
                else:
                    rec.update(
                        ok=True,
                        **{c: score[c] for c in CRITERIA},
                        pro_score=score["pro_score"],
                        con_score=score["con_score"],
                        round_winner=score["round_winner"],
                        reasoning=score["reasoning"],
                    )
                with open(paths.judgments, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
                print(f"[judge {count}/{total}] pair={p['pair_id']} {order} "
                      f"r={r} tier={rec['tier']}")
                time.sleep(args.sleep)


# -------------------------------------------------------------------- report

def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


_WIN = {1: "pro", -1: "con", 0: "tie"}


def _sd(vals) -> float | None:
    vals = list(vals)
    return statistics.stdev(vals) if len(vals) > 1 else None


def _mean(vals) -> float | None:
    vals = list(vals)
    return statistics.fmean(vals) if vals else None


def _r(x, nd=3):
    return None if x is None else round(x, nd)


def _delta_summary(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": _r(statistics.fmean(vals)),
        "median": _r(statistics.median(vals)),
        "max": _r(max(vals)),
        "share_ge_threshold": _r(sum(v >= THRESHOLD for v in vals) / len(vals)),
    }


def _flip_summary(flags: list[bool]) -> dict:
    return {"n": len(flags), "flip_rate": _r(sum(flags) / len(flags)) if flags else None}


def _boot_ci(vals: list[float], n_boot: int = 2000, seed: int = 0):
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    k = len(vals)
    means = sorted(statistics.fmean(rng.choices(vals, k=k)) for _ in range(n_boot))
    return [_r(means[int(0.025 * n_boot)]), _r(means[int(0.975 * n_boot) - 1])]


def _corr(xs: list[float], ys: list[float]):
    if len(xs) < 3:
        return None
    try:
        return _r(statistics.correlation(xs, ys))
    except Exception:  # noqa: BLE001  (constant input etc.)
        return None


def cmd_report(args, paths: Paths) -> None:
    data = _load_pairs(paths)
    info = {p["pair_id"]: p for p in data["pairs"][: args.n]}
    excluded: list[dict] = []
    if args.valid_only:
        # Rule fixed in advance from the debater prompt (180-220 words):
        # drop any pair where Pro or Con is outside the allowed range
        # (this includes empty arguments). Not chosen from the results.
        for pid in sorted(info):
            pw, cw = info[pid]["pro_words"], info[pid]["con_words"]
            if not (args.min_words <= pw <= args.max_words
                    and args.min_words <= cw <= args.max_words):
                excluded.append({"pair_id": pid, "pro_words": pw, "con_words": cw})
                del info[pid]
        paths.summary = paths.out / "summary_valid.json"
        paths.pair_csv = paths.out / "pair_level_valid.csv"
        paths.top_csv = paths.out / "top_discrepancies_valid.csv"
    all_recs = [r for r in _load_judgments(paths) if r["pair_id"] in info]
    if not all_recs:
        sys.exit("No judgments found. Run the 'judge' phase first.")

    J = {(r["pair_id"], r["order"], r["repeat"]): r for r in all_recs if r.get("ok")}
    PF, CF = ORDERS

    def reps(pid, order):
        return sorted(rep for (p, o, rep) in J if p == pid and o == order)

    def m(rec):  # margin = pro - con
        return rec["pro_score"] - rec["con_score"]

    pids = sorted(info)
    both = {pid: sorted(set(reps(pid, PF)) & set(reps(pid, CF))) for pid in pids}

    # ---- Metric 1: position bias ------------------------------------------
    before_legacy, before_gap, before_flip = [], [], []   # order flipped
    noise_legacy, noise_flip = [], []                     # same order repeated
    after_legacy, after_flip = [], []                     # bidirectional repeats
    ctrl_legacy, ctrl_flip = [], []                       # same-order, 2-call avg

    for pid in pids:
        for rep in both[pid]:
            a, b = J[(pid, PF, rep)], J[(pid, CF, rep)]
            before_legacy.append(max(abs(a["pro_score"] - b["pro_score"]),
                                     abs(a["con_score"] - b["con_score"])))
            before_gap.append(abs(m(a) - m(b)))
            before_flip.append(_sign(m(a)) != _sign(m(b)))

        for order in ORDERS:
            rs = reps(pid, order)
            for i in range(len(rs)):
                for j in range(i + 1, len(rs)):
                    x, y = J[(pid, order, rs[i])], J[(pid, order, rs[j])]
                    noise_legacy.append(max(abs(x["pro_score"] - y["pro_score"]),
                                            abs(x["con_score"] - y["con_score"])))
                    noise_flip.append(_sign(m(x)) != _sign(m(y)))
            if len(rs) >= 4:  # control: two independent 2-call averages
                h1 = [J[(pid, order, rs[0])], J[(pid, order, rs[1])]]
                h2 = [J[(pid, order, rs[2])], J[(pid, order, rs[3])]]
                p1, c1 = _mean(x["pro_score"] for x in h1), _mean(x["con_score"] for x in h1)
                p2, c2 = _mean(x["pro_score"] for x in h2), _mean(x["con_score"] for x in h2)
                ctrl_legacy.append(max(abs(p1 - p2), abs(c1 - c2)))
                ctrl_flip.append(_sign(p1 - c1) != _sign(p2 - c2))

        bi = {}
        for rep in both[pid]:
            a, b = J[(pid, PF, rep)], J[(pid, CF, rep)]
            bi[rep] = ((a["pro_score"] + b["pro_score"]) / 2,
                       (a["con_score"] + b["con_score"]) / 2)
        keys = sorted(bi)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                (p1, c1), (p2, c2) = bi[keys[i]], bi[keys[j]]
                after_legacy.append(max(abs(p1 - p2), abs(c1 - c2)))
                after_flip.append(_sign(p1 - c1) != _sign(p2 - c2))

    # signed second-position advantage + side advantage, one value per pair
    spa_pair, side_pair, pair_rows = [], [], []
    for pid in pids:
        rp, rc = reps(pid, PF), reps(pid, CF)
        if not rp or not rc:
            continue
        m_pf = statistics.fmean(m(J[(pid, PF, x)]) for x in rp)
        m_cf = statistics.fmean(m(J[(pid, CF, x)]) for x in rc)
        spa = (m_cf - m_pf) / 2      # >0 means the SECOND speaker scores higher
        side = (m_pf + m_cf) / 2     # >0 means Pro scores higher, order-free
        spa_pair.append(spa)
        side_pair.append(side)
        b_by_rep = {x: (m(J[(pid, PF, x)]) + m(J[(pid, CF, x)])) / 2 for x in both[pid]}
        pair_rows.append({
            "pair_id": pid, "topic": info[pid]["topic"],
            "pro_words": info[pid]["pro_words"], "con_words": info[pid]["con_words"],
            "word_diff_pro_minus_con": info[pid]["pro_words"] - info[pid]["con_words"],
            "n_reps_pro_first": len(rp), "n_reps_con_first": len(rc),
            "mean_margin_pro_first": _r(m_pf, 2), "mean_margin_con_first": _r(m_cf, 2),
            "order_gap": _r(abs(m_pf - m_cf), 2),
            "second_speaker_advantage": _r(spa, 2),
            "bidir_margin_mean": _r(side, 2),
            "bidir_margin_range": _r(max(b_by_rep.values()) - min(b_by_rep.values()), 2)
            if len(b_by_rep) > 1 else None,
            "bidir_winner_by_repeat": ",".join(_WIN[_sign(v)] for _, v in sorted(b_by_rep.items())),
            "_pf_reason": str(J[(pid, PF, rp[0])]["reasoning"])[:200],
            "_cf_reason": str(J[(pid, CF, rc[0])]["reasoning"])[:200],
        })

    # ---- Metric 2: consistency sigma ---------------------------------------
    crit_sd = {c: [] for c in CRITERIA}
    worst = (0.0, None)
    tot_sd_single, tot_sd_bidir = [], []
    for pid in pids:
        for order in ORDERS:
            rs = reps(pid, order)
            if len(rs) < 2:
                continue
            for c in CRITERIA:
                s = _sd(J[(pid, order, x)][c] for x in rs)
                crit_sd[c].append(s)
                if s > worst[0]:
                    worst = (s, f"pair {pid} {order} {c}")
            tot_sd_single.append(_sd(J[(pid, order, x)]["pro_score"] for x in rs))
            tot_sd_single.append(_sd(J[(pid, order, x)]["con_score"] for x in rs))
        if len(both[pid]) >= 2:
            tot_sd_bidir.append(_sd((J[(pid, PF, x)]["pro_score"] + J[(pid, CF, x)]["pro_score"]) / 2
                                    for x in both[pid]))
            tot_sd_bidir.append(_sd((J[(pid, PF, x)]["con_score"] + J[(pid, CF, x)]["con_score"]) / 2
                                    for x in both[pid]))
    crit_means = {c: _r(_mean(v)) for c, v in crit_sd.items() if v}
    all_crit = [x for v in crit_sd.values() for x in v]

    # ---- Metric 3: schema validity -----------------------------------------
    tiers = {"structured": 0, "fallback_first": 0, "fallback_retry": 0, "failed": 0}
    for rec in all_recs:
        tiers[rec.get("tier") or "failed"] = tiers.get(rec.get("tier") or "failed", 0) + 1
    n_tot = len(all_recs)

    def pct(x):
        return _r(100 * x / n_tot, 1) if n_tot else None

    schema = {
        "judge_calls_total": n_tot,
        "tier_counts": tiers,
        "first_try_valid_pct": pct(tiers["structured"]),
        "valid_after_fallback_first_pct": pct(tiers["structured"] + tiers["fallback_first"]),
        "valid_after_all_retries_pct": pct(n_tot - tiers["failed"]),
        "failed_pct": pct(tiers["failed"]),
        "mean_model_calls_per_judgment": _r(_mean(r.get("llm_calls") or 0 for r in all_recs)),
    }

    # ---- Metric 4: error analysis -------------------------------------------
    for_rank = [p for p in pair_rows]
    top_gap = sorted(for_rank, key=lambda p: -p["order_gap"])[: args.top]
    top_inst = sorted([p for p in for_rank if p["bidir_margin_range"] is not None],
                      key=lambda p: -p["bidir_margin_range"])[: args.top]
    length_corr = _corr([p["word_diff_pro_minus_con"] for p in pair_rows],
                        [p["bidir_margin_mean"] for p in pair_rows])

    summary = {
        "generated_at": _now(),
        "python": platform.python_version(),
        "models": data.get("meta", {}),
        "valid_only": bool(args.valid_only),
        "word_range": [args.min_words, args.max_words] if args.valid_only else None,
        "excluded_pairs": excluded,
        "pairs_requested": len(info),
        "pairs_with_both_orders": len(pair_rows),
        "judgments_ok": len(J),
        "judgments_total": n_tot,
        "metric1_position_bias": {
            "unit": "points on a 0-30 per-side scale (sum of 3 criteria)",
            "threshold": THRESHOLD,
            "before_order_flipped_single_judge": {
                "legacy_max_side_delta": _delta_summary(before_legacy),
                "margin_gap": _delta_summary(before_gap),
                "winner_flip": _flip_summary(before_flip),
            },
            "noise_floor_same_order_repeated": {
                "legacy_max_side_delta": _delta_summary(noise_legacy),
                "winner_flip": _flip_summary(noise_flip),
            },
            "control_same_order_2call_average": {
                "legacy_max_side_delta": _delta_summary(ctrl_legacy),
                "winner_flip": _flip_summary(ctrl_flip),
            },
            "after_bidirectional_two_independent_runs": {
                "legacy_max_side_delta": _delta_summary(after_legacy),
                "winner_flip": _flip_summary(after_flip),
            },
            "second_position_advantage_single_order": {
                "mean": _r(_mean(spa_pair)), "ci95_bootstrap": _boot_ci(spa_pair),
                "note": "After (bidirectional) is 0 by construction: each side is first once and second once.",
            },
            "pro_side_advantage_order_free": {
                "mean": _r(_mean(side_pair)), "ci95_bootstrap": _boot_ci(side_pair),
            },
        },
        "metric2_consistency_sigma": {
            "per_criterion_mean_sd": crit_means,
            "mean_sd_all_criteria": _r(_mean(all_crit)),
            "max_of_criterion_means": max(crit_means.values()) if crit_means else None,
            "worst_single_sd": {"value": _r(worst[0]), "where": worst[1]},
            "side_total_sd_single_order_mean": _r(_mean(tot_sd_single)),
            "side_total_sd_bidirectional_mean": _r(_mean(tot_sd_bidir)),
        },
        "metric3_schema_validity": schema,
        "metric4_length_check": {
            "pearson_r_word_diff_vs_bidir_margin": length_corr, "n_pairs": len(pair_rows),
        },
    }
    _save_json(paths.summary, summary)

    pub_cols = [k for k in pair_rows[0] if not k.startswith("_")] if pair_rows else []
    with open(paths.pair_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pub_cols)
        w.writeheader()
        for row in pair_rows:
            w.writerow({k: row[k] for k in pub_cols})
    with open(paths.top_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ranking"] + pub_cols + ["pro_first_reasoning", "con_first_reasoning"])
        w.writeheader()
        for label, rows in (("order_gap", top_gap), ("bidir_instability", top_inst)):
            for row in rows:
                out = {"ranking": label, **{k: row[k] for k in pub_cols},
                       "pro_first_reasoning": row["_pf_reason"],
                       "con_first_reasoning": row["_cf_reason"]}
                w.writerow(out)

    _print_report(summary, top_gap, top_inst, paths)


def _print_report(s: dict, top_gap: list, top_inst: list, paths: Paths) -> None:
    m1, m2, m3 = s["metric1_position_bias"], s["metric2_consistency_sigma"], s["metric3_schema_validity"]
    print("\n" + "=" * 72)
    print("BENCHMARK REPORT")
    print("=" * 72)
    if s["valid_only"]:
        print(f"VALID-ONLY mode, word range {s['word_range']}; excluded pairs: "
              f"{[(e['pair_id'], e['pro_words'], e['con_words']) for e in s['excluded_pairs']]}")
    print(f"pairs requested: {s['pairs_requested']} | with both orders: {s['pairs_with_both_orders']}")
    print(f"judgments ok: {s['judgments_ok']} / {s['judgments_total']} | models: {s['models']}")

    print("\n[1] POSITION BIAS (points; threshold < %.1f)" % THRESHOLD)
    rows = [
        ("Before: order flipped (single judge)", m1["before_order_flipped_single_judge"]),
        ("Noise floor: same order repeated", m1["noise_floor_same_order_repeated"]),
        ("Control: same order, 2-call average", m1["control_same_order_2call_average"]),
        ("After: bidirectional, two runs", m1["after_bidirectional_two_independent_runs"]),
    ]
    print(f"  {'condition':<38}{'n':>5}{'mean':>8}{'max':>8}{'>=1.0':>8}{'flip%':>8}")
    for name, d in rows:
        L, F = d["legacy_max_side_delta"], d["winner_flip"]
        if not L.get("n"):
            print(f"  {name:<38}{'-':>5}  (not enough data)")
            continue
        fr = F["flip_rate"]
        print(f"  {name:<38}{L['n']:>5}{L['mean']:>8}{L['max']:>8}"
              f"{L['share_ge_threshold']:>8}{'' if fr is None else round(100 * fr, 1):>8}")
    sp = m1["second_position_advantage_single_order"]
    sd_ = m1["pro_side_advantage_order_free"]
    print(f"  second-position advantage (single order): mean={sp['mean']} CI95={sp['ci95_bootstrap']}")
    print(f"  Pro-side advantage (order-free):          mean={sd_['mean']} CI95={sd_['ci95_bootstrap']}")

    print("\n[2] CONSISTENCY (sample std dev across repeats)")
    print(f"  mean sd over all criteria: {m2['mean_sd_all_criteria']}")
    print(f"  max of per-criterion mean sd: {m2['max_of_criterion_means']}")
    print(f"  worst single sd: {m2['worst_single_sd']}")
    print(f"  side-total sd, single order: {m2['side_total_sd_single_order_mean']}")
    print(f"  side-total sd, bidirectional: {m2['side_total_sd_bidirectional_mean']}")

    print("\n[3] SCHEMA VALIDITY")
    print(f"  judge calls: {m3['judge_calls_total']} | tiers: {m3['tier_counts']}")
    print(f"  valid on first try (structured output): {m3['first_try_valid_pct']}%")
    print(f"  valid after first plain-text parse:     {m3['valid_after_fallback_first_pct']}%")
    print(f"  valid after all retries:                {m3['valid_after_all_retries_pct']}%")
    print(f"  never valid:                            {m3['failed_pct']}%")

    print("\n[4] ERROR ANALYSIS")
    print(f"  length check, Pearson r (word diff vs bidirectional margin): "
          f"{s['metric4_length_check']['pearson_r_word_diff_vs_bidir_margin']}")
    for title, rows_ in (("Largest order gaps (what swapping corrects)", top_gap),
                         ("Largest bidirectional instability (what remains)", top_inst)):
        print(f"  {title}:")
        for p in rows_:
            print(f"    pair {p['pair_id']:>2} | gap={p['order_gap']} | spa={p['second_speaker_advantage']} "
                  f"| bidir={p['bidir_margin_mean']} range={p['bidir_margin_range']} "
                  f"| words pro/con={p['pro_words']}/{p['con_words']} | {p['topic'][:45]}")
    print("=" * 72)
    print(f"Wrote {paths.summary}, {paths.pair_csv}, {paths.top_csv}")


# ----------------------------------------------------------------------- cli

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["generate", "judge", "report", "all"])
    ap.add_argument("--n", type=int, default=24, help="number of topic pairs (max %d)" % len(TOPICS))
    ap.add_argument("--repeats", type=int, default=4, help="judge repeats per order (>=2; 4 enables the control row)")
    ap.add_argument("--top", type=int, default=5, help="rows in the error-analysis lists")
    ap.add_argument("--sleep", type=float, default=0.5, help="seconds between API calls")
    ap.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "eval_runs")
    ap.add_argument("--valid-only", action="store_true",
                    help="report only on pairs whose Pro and Con are inside the word range")
    ap.add_argument("--min-words", type=int, default=180)
    ap.add_argument("--max-words", type=int, default=220)
    args = ap.parse_args()
    if args.repeats < 2:
        ap.error("--repeats must be at least 2 (sigma needs repeated runs)")
    paths = Paths(args.out_dir)

    if args.phase in ("generate", "all"):
        cmd_generate(args, paths)
    if args.phase in ("judge", "all"):
        cmd_judge(args, paths)
    if args.phase in ("report", "all"):
        cmd_report(args, paths)


if __name__ == "__main__":
    main()