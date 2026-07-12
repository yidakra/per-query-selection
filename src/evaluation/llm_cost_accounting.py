#!/usr/bin/env python
"""What does the Full tier actually cost per query?

The frontier plots use cost = (number of similarity components scored), normalised so Full = 1.0,
giving A = 0.2 and B = 0.4. That treats all five components as unit cost. This script checks that
assumption, because if it is wrong the cost axis of every frontier figure is wrong.

Two facts make the proxy suspect:

 1. Tiers are NESTED (A subset of B subset of Full) and every tier pays `query_vs_video`. Video and
    caption encoding is GALLERY-side: done once, offline, amortised over all queries. It is not a
    per-query cost at all.

 2. The Full tier alone pays LLM event decomposition, which is per-query and ONLINE. From
    text_planner.get_prequel_sequel_during(), each query costs:
        3 extract calls        (temporal, spatial, event)
      + 3 event-list calls     (prequel, during, sequel)
      + |cartesian| refine calls PER event type  (refine_event_query, one per product element)
    and |cartesian| equals the number of final paraphrases for that event type.

So the honest per-query online cost of Full is dominated by ~30 generations on a 70B model, while
A and B issue ZERO. The component count understates the gap by orders of magnitude -- which means
our reported savings are CONSERVATIVE, not inflated.

Token counts here are exact where the dataset stored the generation (prequel/during/sequel), and
estimated from the jinja templates elsewhere (the pipeline did not persist extract/refine calls).
Character counts are exact; token figures use a 4-chars/token approximation and are labelled.

CPU-only.
"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from tracking import track  # noqa: E402
from datasets import load_from_disk  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root, wherever it is checked out
REPO = _ROOT
DS = f"{REPO}/data/MultiVENT/Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_noASR"
TPL = f"{REPO}/external/q2e_official/src/data/prompts"
EVENTS = ["prequel", "during", "sequel"]
CH_PER_TOK = 4.0        # crude; labelled as approximate everywhere it is used
GEN_MODEL = "LLaMA-3.3-70B"


def approx_tok(chars):
    return chars / CH_PER_TOK


def main():
    with track("llm-cost-accounting", gpu_ids=[], tags=["cost-model"],
               config={"gen_model": GEN_MODEL, "chars_per_token": CH_PER_TOK}) as tr:
        ds = load_from_disk(DS)
        seen, rows = set(), []
        for r in ds:
            q = r["query"]
            if q in seen:
                continue
            seen.add(q)
            md = r["metadata"]
            n_para = {e: len(r[e]) for e in EVENTS}
            # exact, from stored generations
            p_chars = sum(len(md[f"{e}_prompt"]) for e in EVENTS)
            g_chars = sum(len(md[f"{e}_raw_generation"]) for e in EVENTS)
            rows.append({"query": q, "n_para": n_para,
                         "stored_prompt_chars": p_chars, "stored_gen_chars": g_chars})

        # template sizes for the calls the pipeline did NOT persist
        tpl_chars = {}
        for t in ["extract_temporal_info", "extract_spatial_info", "extract_event_info",
                  "refine_event_query"]:
            tpl_chars[t] = len(open(f"{TPL}/{t}.jinja").read())

        T = len(rows)
        para = np.array([sum(r["n_para"].values()) for r in rows])           # refine calls / query
        # |cartesian| == number of final paraphrases for that event type, so refine calls == para
        calls = 3 + 3 + para                                                  # extract + lists + refine
        stored_p = np.array([r["stored_prompt_chars"] for r in rows])
        stored_g = np.array([r["stored_gen_chars"] for r in rows])
        # estimated prompt chars for unpersisted calls
        est_extract_p = sum(tpl_chars[t] for t in
                            ["extract_temporal_info", "extract_spatial_info", "extract_event_info"])
        est_refine_p = para * tpl_chars["refine_event_query"]
        total_p = stored_p + est_extract_p + est_refine_p
        # generations: extract ~80 chars each, refine ~60 chars each (short structured outputs)
        total_g = stored_g + 3 * 80 + para * 60

        print(f"\n### Per-query LLM cost of the Full tier ({GEN_MODEL}, MultiVENT noASR, T={T})\n")
        print(f"  paraphrases / query (all 3 event types) : mean {para.mean():.1f}  "
              f"median {np.median(para):.0f}  max {para.max()}")
        print(f"  LLM generation calls / query            : mean {calls.mean():.1f}  "
              f"median {np.median(calls):.0f}  max {calls.max()}")
        print(f"      = 3 extract + 3 event-list + {para.mean():.1f} refine (one per cartesian element)")
        print(f"  prompt chars / query   (exact+est)      : mean {total_p.mean():,.0f}")
        print(f"  generated chars / query (exact+est)     : mean {total_g.mean():,.0f}")
        print(f"  ~prompt tokens / query   (@{CH_PER_TOK:.0f} ch/tok) : mean {approx_tok(total_p.mean()):,.0f}")
        print(f"  ~generated tokens / query (@{CH_PER_TOK:.0f} ch/tok): mean {approx_tok(total_g.mean()):,.0f}")

        print(f"\n### What each tier pays PER QUERY (online)\n")
        print(f"  {'tier':6} {'LLM calls':>10} {'~gen tokens':>12} {'sim components':>16} {'proxy cost':>11}")
        for tier, llm, comps, proxy in [("A", 0, 1, 0.2), ("B", 0, 2, 0.4), ("Full", calls.mean(), 5, 1.0)]:
            gt = approx_tok(total_g.mean()) if llm else 0
            print(f"  {tier:6} {llm:10.1f} {gt:12,.0f} {comps:16d} {proxy:11.1f}")

        print(f"\n  Gallery-side encoding (video frames, VLM captions) is amortised offline and is")
        print(f"  paid IDENTICALLY by every tier, so it does not enter the per-query routing cost.")
        print(f"  The component-count proxy rates Full at 5x tier A. In per-query online terms Full")
        print(f"  additionally issues ~{calls.mean():.0f} generations on a {GEN_MODEL} while A issues zero.")
        print(f"  => the proxy UNDERSTATES the cost of Full. Reported savings are conservative.")

        # ---- ENERGY ESTIMATE (not a measurement): what the LLM decomposition costs in joules ----
        # The 70B ran offline, so unlike the similarity components (cost_model_findings.md, measured
        # on GPU1) this is ESTIMATED. Two independent methods, reported as a band, not a point.
        #
        # Method 1 (FLOPs, primary): a dense decoder-only forward is ~2*P FLOP per token processed
        # (P = params; 1 MAC = 2 FLOP), for BOTH prefill (prompt) and cached decode (generation).
        #   FLOP/query = 2 * P * (prompt_tok + gen_tok)
        # This workload is prompt-heavy (prompt:gen ~ 8.6:1), i.e. prefill-dominated, so effective
        # utilisation sits at the compute-bound (higher) end. Effective delivered efficiency
        # (TFLOP/J) = accelerator peak * MFU; scenarios below are grounded in published peaks:
        #   A100 SXM BF16 peak 312 TFLOP/s @ 400 W = 0.78 TFLOP/J ; H100 SXM BF16 989 @ 700 W = 1.41;
        #   H100 FP8 ~1979 @ 700 W = 2.83.  MFU for prefill-heavy inference ~ 25-40%.
        # Method 2 (per-output-token, cross-check): published measured energy PER GENERATED token,
        # 0.39 J (Llama-3-70B FP8, 8xH100 vLLM, batched; llm-tracker.info / ML.ENERGY) to
        # 3.5 J (Llama-65B, A100/V100, Samsi et al. 2023). These undercount OUR cost because they
        # amortise little prefill, whereas we prefill 8x more tokens than we generate.
        P = 70e9
        p_tok, g_tok = approx_tok(total_p.mean()), approx_tok(total_g.mean())
        flops_per_query = 2.0 * P * (p_tok + g_tok)
        SIM_FULL_J = 1418.36        # measured Full-tier similarity energy (cost_model_findings.md 4)
        SIM_B_J = 22.35             # measured tier-B energy
        flops_scen = {"A100_BF16_25pctMFU": 0.78e12 * 0.25, "H100_BF16_35pctMFU": 1.41e12 * 0.35,
                      "H100_FP8_40pctMFU": 2.83e12 * 0.40}     # effective FLOP/J
        e_flops = {k: flops_per_query / v for k, v in flops_scen.items()}
        e_tok = {"H100_FP8_batched_0.39": g_tok * 0.39, "A100_2023_3.5": g_tok * 3.5}
        e_all = list(e_flops.values()) + list(e_tok.values())
        e_lo, e_hi = min(e_all), max(e_all)
        e_central = e_flops["H100_BF16_35pctMFU"]              # transparent primary point
        full_total_lo, full_total_hi = SIM_FULL_J + e_lo, SIM_FULL_J + e_hi
        full_total_central = SIM_FULL_J + e_central

        print(f"\n### LLM energy of the Full tier (ESTIMATE, not measured; {GEN_MODEL})\n")
        print(f"  FLOPs/query = 2*P*(prompt+gen) = 2*70e9*({p_tok:,.0f}+{g_tok:,.0f}) = {flops_per_query:.3e}")
        print(f"  method 1 (FLOPs / efficiency):")
        for k, v in e_flops.items():
            print(f"    {k:22s}: {v:8,.0f} J")
        print(f"  method 2 (per-generated-token, cross-check; LOWER bound -- prompt-heavy):")
        for k, v in e_tok.items():
            print(f"    {k:22s}: {v:8,.0f} J")
        flo, fhi = min(e_flops.values()), max(e_flops.values())     # the prefill-aware band
        print(f"  => LLM decomposition ~ {flo:,.0f}-{fhi:,.0f} J/query (FLOPs model, central {e_central:,.0f} J);")
        print(f"     the per-output-token cross-check ({min(e_tok.values()):,.0f}-{max(e_tok.values()):,.0f} J) brackets it from")
        print(f"     below, since it under-counts our 8.6:1 prompt:gen prefill.")
        print(f"     Central estimate is {e_central/SIM_FULL_J:.1f}x the MEASURED similarity energy of Full "
              f"({SIM_FULL_J:.0f} J); tiers A and B pay NONE of it.")
        print(f"\n  Full end-to-end ~ {full_total_lo:,.0f}-{full_total_hi:,.0f} J/query (central {full_total_central:,.0f}).")
        print(f"  B->Full price ratio: {SIM_FULL_J/SIM_B_J:.0f}x (similarity only) -> "
              f"{full_total_central/SIM_B_J:.0f}x end-to-end (central; range {full_total_lo/SIM_B_J:.0f}-{full_total_hi/SIM_B_J:.0f}x).")
        print(f"  The estimate is uncertain by ~6x, but the conclusion is not: the LLM decomposition is")
        print(f"  comparable to or larger than Full's entire similarity cost, so every 'savings vs Full'")
        print(f"  is a lower bound, and the router (which never buys Full) avoids all of it.")

        out = {
            "gen_model": GEN_MODEL, "T": T, "chars_per_token_approx": CH_PER_TOK,
            "paraphrases_per_query": {"mean": float(para.mean()), "max": int(para.max())},
            "llm_calls_per_query": {"mean": float(calls.mean()), "median": float(np.median(calls)),
                                    "max": int(calls.max()),
                                    "breakdown": "3 extract + 3 event-list + one refine per cartesian element"},
            "prompt_chars_per_query_mean": float(total_p.mean()),
            "generated_chars_per_query_mean": float(total_g.mean()),
            "approx_prompt_tokens_per_query": float(approx_tok(total_p.mean())),
            "approx_generated_tokens_per_query": float(approx_tok(total_g.mean())),
            "exact_fields": ["prequel/during/sequel prompt+generation chars"],
            "estimated_fields": ["extract_* and refine_event_query prompt/generation chars"],
            "tier_online_cost": {"A": {"llm_calls": 0, "sim_components": 1},
                                 "B": {"llm_calls": 0, "sim_components": 2},
                                 "Full": {"llm_calls": float(calls.mean()), "sim_components": 5}},
            "energy_estimate": {
                "disclaimer": "ESTIMATE, not a measurement -- the 70B ran offline. Similarity "
                              "energy (sim_full_j) IS measured (cost_model_findings.md).",
                "flops_per_query": float(flops_per_query),
                "flops_model": "2 * P(=70e9) * (prompt_tok + gen_tok)",
                "method1_flops_joules": {k: float(v) for k, v in e_flops.items()},
                "method2_per_output_token_joules": {k: float(v) for k, v in e_tok.items()},
                "llm_joules_per_query": {"low": float(e_lo), "central_H100_BF16_35pctMFU": float(e_central),
                                         "high": float(e_hi)},
                "sim_full_j_measured": SIM_FULL_J, "sim_b_j_measured": SIM_B_J,
                "full_end_to_end_joules": {"low": float(full_total_lo), "central": float(full_total_central),
                                           "high": float(full_total_hi)},
                "b_to_full_ratio": {"similarity_only": round(SIM_FULL_J / SIM_B_J, 1),
                                    "end_to_end_central": round(full_total_central / SIM_B_J, 1),
                                    "end_to_end_low": round(full_total_lo / SIM_B_J, 1),
                                    "end_to_end_high": round(full_total_hi / SIM_B_J, 1)},
            },
        }
        of = f"{REPO}/results/ablations/llm_cost_accounting.json"
        json.dump(out, open(of, "w"), indent=2)
        print(f"\nwrote {of}")
        tr.summary({"llm_calls_per_query_mean": float(calls.mean()),
                    "approx_generated_tokens_per_query": float(approx_tok(total_g.mean()))})


if __name__ == "__main__":
    main()
