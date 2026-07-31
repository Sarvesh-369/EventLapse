#!/usr/bin/env python3
import os
import re
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_n_and_f(sample_id: str):
    n_match = re.search(r"_N(\d+)_", str(sample_id))
    f_match = re.search(r"_F([\d\.]+)_", str(sample_id))
    n_val = int(n_match.group(1)) if n_match else 0
    f_val = float(f_match.group(1)) if f_match else 1.0
    return n_val, f_val

def extract_pred_timestamps(text: str):
    if not text:
        return []
    ts = []
    # MM:SS
    matches = re.findall(r"\b(\d{1,2}):(\d{2})(?:\.(\d+))?\b", str(text))
    for m, s, ms in matches:
        val = int(m)*60 + int(s) + (float(f"0.{ms}") if ms else 0.0)
        ts.append(val)
    if not ts:
        sec_matches = re.findall(r"\b(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)\b", str(text), re.IGNORECASE)
        for s in sec_matches:
            val = float(s)
            if 0 <= val <= 25.0:
                ts.append(val)
    return ts

def load_gt_timestamps(task: str, sample_id: str):
    gt_file = Path(f"data/traces/{task}/{sample_id}_trace.json")
    if gt_file.exists():
        with open(gt_file) as f:
            gt_data = json.load(f)
        return [ev["timestamp"] for ev in gt_data.get("events", [])]
    return []

def compute_trace_metrics_for_row(row):
    task = row.get("task", "bounce_ball")
    sample_id = row.get("sample_id", "")
    N, F = parse_n_and_f(sample_id)
    is_exact = row.get("exact_match_result") == True
    raw_resp = row.get("raw_model_response", "")

    gt_ts = load_gt_timestamps(task, sample_id)
    pred_ts = extract_pred_timestamps(raw_resp)

    M = len(pred_ts)
    gt_N = len(gt_ts) if gt_ts else N

    vor = (M / gt_N) if gt_N > 0 else (1.0 if M == 0 else float(M))

    matched = 0
    gt_matched = set()
    for pt in pred_ts:
        for idx_gt, gt_t in enumerate(gt_ts):
            if idx_gt not in gt_matched and abs(pt - gt_t) <= 1.0:
                matched += 1
                gt_matched.add(idx_gt)
                break

    p = (matched / M) if M > 0 else (1.0 if gt_N == 0 else 0.0)
    r = (matched / gt_N) if gt_N > 0 else (1.0 if M == 0 else 0.0)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    is_acr = is_exact and (f1 < 0.80)
    is_rfr = (not is_exact) and (f1 >= 0.80)

    return pd.Series({
        "N": N,
        "F": F,
        "is_exact": is_exact,
        "vor": vor,
        "precision": p,
        "recall": r,
        "f1": f1,
        "is_acr": is_acr,
        "is_rfr": is_rfr
    })

def summarize_metrics(sub_df):
    tot = len(sub_df)
    if tot == 0:
        return {"em": 0.0, "vor": 0.0, "p": 0.0, "r": 0.0, "f1": 0.0, "acr": 0.0, "rfr": 0.0}
    em = sub_df["is_exact"].mean() * 100.0
    vor = sub_df["vor"].mean()
    p = sub_df["precision"].mean() * 100.0
    r = sub_df["recall"].mean() * 100.0
    f1 = sub_df["f1"].mean() * 100.0
    acr = (sub_df["is_acr"].sum() / tot) * 100.0
    rfr = (sub_df["is_rfr"].sum() / tot) * 100.0
    return {"em": em, "vor": vor, "p": p, "r": r, "f1": f1, "acr": acr, "rfr": rfr}

def main():
    root_dir = Path(__file__).resolve().parent.parent
    outputs_dir = root_dir / "outputs"
    paper_dir = outputs_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    print("=== Computing Full Trace-Grounded Metrics for Paper ===")

    baseline_file = outputs_dir / "results_matrix_gemini_gemini-3.6-flash_native_video_structured_trace.jsonl"
    b_rows = []
    with open(baseline_file) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data.get("exact_match_result") is not None and data.get("error_message") is None:
                    b_rows.append(data)

    b_df = pd.DataFrame(b_rows)
    m_metrics = b_df.apply(compute_trace_metrics_for_row, axis=1)
    b_df = pd.concat([b_df, m_metrics], axis=1)

    # -------------------------------------------------------------
    # 1. TABLE 1: Trace-Grounded Evaluation Table
    # -------------------------------------------------------------
    part_a_rows = []
    for domain, d_name in [("bounce_ball", "Bounce Ball"), ("blinking", "Blinking"), ("state_machine", "State Machine")]:
        sub = b_df[b_df["task"] == domain]
        res = summarize_metrics(sub)
        part_a_rows.append(f"  {d_name:<24} & {res['em']:5.1f}\\% & {res['vor']:.2f} & {res['p']:5.1f}\\% & {res['r']:5.1f}\\% & {res['f1']:5.1f}\\% & {res['acr']:4.1f}\\% & {res['rfr']:4.1f}\\% \\\\")

    macro_res = summarize_metrics(b_df)
    part_a_rows.append(f"  \\textbf{{Macro Average}}   & \\textbf{{{macro_res['em']:.1f}\\%}} & \\textbf{{{macro_res['vor']:.2f}}} & \\textbf{{{macro_res['p']:.1f}\\%}} & \\textbf{{{macro_res['r']:.1f}\\%}} & \\textbf{{{macro_res['f1']:.1f}\\%}} & \\textbf{{{macro_res['acr']:.1f}\\%}} & \\textbf{{{macro_res['rfr']:.1f}\\%}} \\\\")

    # Part B: Regional Capability Subsets
    reg_defs = [
        ("Low Count, Low Freq", b_df[(b_df["N"] <= 3) & (b_df["F"] <= 2.0)]),
        ("High Count, Low Freq", b_df[(b_df["N"] >= 5) & (b_df["F"] <= 2.0)]),
        ("Low Count, High Freq", b_df[(b_df["N"] <= 3) & (b_df["F"] >= 2.5)]),
        ("High Count, High Freq", b_df[(b_df["N"] >= 5) & (b_df["F"] >= 2.5)]),
    ]
    part_b_rows = []
    for r_label, r_sub in reg_defs:
        res = summarize_metrics(r_sub)
        part_b_rows.append(f"  {r_label:<24} & {res['em']:5.1f}\\% & {res['vor']:.2f} & {res['p']:5.1f}\\% & {res['r']:5.1f}\\% & {res['f1']:5.1f}\\% & {res['acr']:4.1f}\\% & {res['rfr']:4.1f}\\% \\\\")

    # Part C: Bounce Ball Interventions
    visual_interventions = [
        ("Native + Structured (Baseline)", "native_video", "structured_trace"),
        ("Dense Sampling (4 FPS)", "frames_4fps", "structured_trace"),
        ("Oracle Event Evidence", "oracle_evidence", "structured_trace"),
    ]
    prompt_interventions = [
        ("Direct Answer", "native_video", "direct"),
        ("Structured Trace", "native_video", "structured_trace"),
        ("Multi-Turn Verification", "native_video", "multi_turn_verification"),
        ("Thinking / CoT", "native_video", "thinking"),
        ("Role Prompting", "native_video", "role_prompting"),
    ]

    def format_intervention_rows(inter_list):
        rows = []
        for i_label, i_mode, i_cond in inter_list:
            pattern = f"outputs/results_matrix_gemini_gemini-3.6-flash_{i_mode}_{i_cond}.jsonl"
            i_files = glob.glob(pattern)
            if i_files:
                i_rows = []
                with open(i_files[0]) as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            if d.get("task") == "bounce_ball" and d.get("exact_match_result") is not None and d.get("error_message") is None:
                                i_rows.append(d)
                if i_rows:
                    i_df = pd.DataFrame(i_rows)
                    i_metrics = i_df.apply(compute_trace_metrics_for_row, axis=1)
                    i_df = pd.concat([i_df, i_metrics], axis=1)
                    res = summarize_metrics(i_df)
                    rows.append(f"  {i_label:<32} & {res['em']:5.1f}\\% & {res['vor']:.2f} & {res['p']:5.1f}\\% & {res['r']:5.1f}\\% & {res['f1']:5.1f}\\% & {res['acr']:4.1f}\\% & {res['rfr']:4.1f}\\% \\\\")
        return rows

    visual_rows = format_intervention_rows(visual_interventions)
    prompt_rows = format_intervention_rows(prompt_interventions)

    tab1_tex = """\\begin{table*}[t]
\\centering
\\small
\\begin{tabularx}{\\textwidth}{X c c c c c c c}
\\toprule
\\textbf{Evaluation Domain / Region} & \\textbf{Final EM (\\%)} & \\textbf{VOR} & \\textbf{Trace P (\\%)} & \\textbf{Trace R (\\%)} & \\textbf{Trace F1 (\\%)} & \\textbf{ACR (\\%)} & \\textbf{RFR (\\%)} \\\\
\\midrule
\\multicolumn{8}{l}{\\textbf{Part A. Trace Evaluation by Task Domain}} \\\\
\\midrule
""" + "\n".join(part_a_rows) + """
\\midrule
\\multicolumn{8}{l}{\\textbf{Part B. Trace Evaluation across ($N \\times F$) Capability Regions}} \\\\
\\midrule
""" + "\n".join(part_b_rows) + """
\\midrule
\\multicolumn{8}{l}{\\textbf{Part C. Trace Diagnosis of Bounce Ball Interventions}} \\\\
\\midrule
\\multicolumn{8}{l}{\\textit{Visual Evidence Family}} \\\\
\\midrule
""" + "\n".join(visual_rows) + """
\\midrule
\\multicolumn{8}{l}{\\textit{Prompting and Reasoning Family}} \\\\
\\midrule
""" + "\n".join(prompt_rows) + """
\\bottomrule
\\end{tabularx}
\\caption{\\textbf{Trace-Grounded Evaluation and Diagnostic Breakdown for EventLapse.} \\emph{Part A.} Trace performance across synthetic domains. \\emph{Part B.} Performance across ($N \\times F$) capability regions. \\emph{Part C.} Diagnostic interventions on the Bouncing Ball task.}
\\label{tab:trace_grounded_evaluation}
\\end{table*}
"""
    tab1_path = paper_dir / "tab_1_trace_grounded_evaluation.tex"
    with open(tab1_path, "w") as f:
        f.write(tab1_tex)
    print(f"Saved {tab1_path}")

    # -------------------------------------------------------------
    # 2. FIGURE 3: N x F Heatmaps
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    domains = ["bounce_ball", "blinking", "state_machine"]
    domain_titles = {
        "bounce_ball": "Bounce Ball (Wall Contacts)",
        "blinking": "Blinking (Light Pulses)",
        "state_machine": "State Machine (Transitions)"
    }

    for idx, domain in enumerate(domains):
        sub_df = b_df[b_df["task"] == domain]
        if not sub_df.empty:
            pivot = sub_df.pivot_table(
                index="F",
                columns="N",
                values="is_exact",
                aggfunc="mean"
            )
            if 0 in pivot.columns:
                n0_val = pivot[0].dropna().mean()
                if not np.isnan(n0_val):
                    pivot[0] = pivot[0].fillna(n0_val)

            sns.heatmap(
                pivot,
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                ax=axes[idx],
                vmin=0.0,
                vmax=1.0,
                cbar=(idx == 2),
                cbar_kws={"label": "Exact Match Accuracy"} if idx == 2 else None
            )
            axes[idx].set_title(domain_titles.get(domain, domain), fontsize=13, fontweight="bold", pad=10)
            axes[idx].set_xlabel("Event Count N", fontsize=11, fontweight="bold")
            axes[idx].set_ylabel("Event Frequency F (Hz)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig3_path = paper_dir / "fig_3_n_x_f_heatmaps.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig3_path}")

    # -------------------------------------------------------------
    # 3. FIGURE 4: Intervention Analysis Plots (3 Panels)
    # -------------------------------------------------------------
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # Panel A: Sampling Densities vs N
    fps_files = [
        ("1 FPS", "outputs/results_matrix_gemini_gemini-3.6-flash_frames_1fps_structured_trace.jsonl"),
        ("2 FPS", "outputs/results_matrix_gemini_gemini-3.6-flash_frames_2fps_structured_trace.jsonl"),
        ("4 FPS", "outputs/results_matrix_gemini_gemini-3.6-flash_frames_4fps_structured_trace.jsonl"),
        ("8 FPS", "outputs/results_matrix_gemini_gemini-3.6-flash_frames_8fps_structured_trace.jsonl"),
        ("16 FPS", "outputs/results_matrix_gemini_gemini-3.6-flash_frames_16fps_structured_trace.jsonl"),
    ]
    for lbl, fp in fps_files:
        if os.path.exists(fp):
            f_rows = []
            with open(fp) as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        if d.get("exact_match_result") is not None and d.get("error_message") is None:
                            n_val, _ = parse_n_and_f(d.get("sample_id"))
                            f_rows.append({"N": n_val, "is_exact": d.get("exact_match_result") == True})
            if f_rows:
                f_df = pd.DataFrame(f_rows)
                n_grp = f_df.groupby("N")["is_exact"].mean()
                ax_a.plot(n_grp.index, n_grp.values, marker="o", label=lbl, linewidth=2)

    ax_a.set_title("Panel A: Sampling Densities vs. N", fontsize=12, fontweight="bold")
    ax_a.set_xlabel("Event Count N", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("Exact Match Accuracy", fontsize=11, fontweight="bold")
    ax_a.set_ylim(-0.05, 1.05)
    ax_a.axhline(0.80, color="red", linestyle="--", alpha=0.7, label="τ = 0.80")
    ax_a.legend(fontsize=9)
    ax_a.grid(True, linestyle=":", alpha=0.6)

    # Panel B: Oracle Evidence vs N
    oracle_files = [
        ("Native Video", "outputs/results_matrix_gemini_gemini-3.6-flash_native_video_structured_trace.jsonl"),
        ("16 FPS", "outputs/results_matrix_gemini_gemini-3.6-flash_frames_16fps_structured_trace.jsonl"),
        ("Oracle Evidence", "outputs/results_matrix_gemini_gemini-3.6-flash_oracle_evidence_structured_trace.jsonl"),
    ]
    for lbl, fp in oracle_files:
        if os.path.exists(fp):
            f_rows = []
            with open(fp) as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        if d.get("task") == "bounce_ball" and d.get("exact_match_result") is not None and d.get("error_message") is None:
                            n_val, _ = parse_n_and_f(d.get("sample_id"))
                            f_rows.append({"N": n_val, "is_exact": d.get("exact_match_result") == True})
            if f_rows:
                f_df = pd.DataFrame(f_rows)
                n_grp = f_df.groupby("N")["is_exact"].mean()
                ax_b.plot(n_grp.index, n_grp.values, marker="s", label=lbl, linewidth=2)

    ax_b.set_title("Panel B: Oracle Evidence vs. N", fontsize=12, fontweight="bold")
    ax_b.set_xlabel("Event Count N", fontsize=11, fontweight="bold")
    ax_b.set_ylabel("Exact Match Accuracy", fontsize=11, fontweight="bold")
    ax_b.set_ylim(-0.05, 1.05)
    ax_b.axhline(0.80, color="red", linestyle="--", alpha=0.7, label="τ = 0.80")
    ax_b.legend(fontsize=9)
    ax_b.grid(True, linestyle=":", alpha=0.6)

    # Panel C: Prompting Strategies vs N (Line Graph)
    p_files = [
        ("Direct", "outputs/results_matrix_gemini_gemini-3.6-flash_native_video_direct.jsonl", "#1f77b4"),
        ("Structured Trace", "outputs/results_matrix_gemini_gemini-3.6-flash_native_video_structured_trace.jsonl", "#2ca02c"),
        ("Multi-Turn", "outputs/results_matrix_gemini_gemini-3.6-flash_native_video_multi_turn_verification.jsonl", "#d62728"),
        ("Thinking", "outputs/results_matrix_gemini_gemini-3.6-flash_native_video_thinking.jsonl", "#ff7f0e"),
        ("Role Prompting", "outputs/results_matrix_gemini_gemini-3.6-flash_native_video_role_prompting.jsonl", "#9467bd"),
    ]
    for lbl, fp, color in p_files:
        if os.path.exists(fp):
            f_rows = []
            with open(fp) as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        if d.get("task") == "bounce_ball" and d.get("exact_match_result") is not None and d.get("error_message") is None:
                            n_val, _ = parse_n_and_f(d.get("sample_id"))
                            f_rows.append({"N": n_val, "is_exact": d.get("exact_match_result") == True})
            if f_rows:
                f_df = pd.DataFrame(f_rows)
                n_grp = f_df.groupby("N")["is_exact"].mean()
                ax_c.plot(n_grp.index, n_grp.values, marker="^", label=lbl, color=color, linewidth=2)

    ax_c.set_title("Panel C: Prompting Strategies vs. N", fontsize=12, fontweight="bold")
    ax_c.set_xlabel("Event Count N", fontsize=11, fontweight="bold")
    ax_c.set_ylabel("Exact Match Accuracy", fontsize=11, fontweight="bold")
    ax_c.set_ylim(-0.05, 1.05)
    ax_c.axhline(0.80, color="red", linestyle="--", alpha=0.7, label="τ = 0.80")
    ax_c.legend(fontsize=9)
    ax_c.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig4_path = paper_dir / "fig_4_intervention_analysis.png"
    plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig4_path}")

    # Copy artifacts to AAAI_27 directory
    aaai_figs = root_dir.parent / "morse papers/morse_profile/AAAI_27/figs"
    aaai_tabs = root_dir.parent / "morse papers/morse_profile/AAAI_27/tables-tex"
    if aaai_figs.exists():
        os.system(f"cp {paper_dir}/*.png '{aaai_figs}/'")
    if aaai_tabs.exists():
        os.system(f"cp {paper_dir}/*.tex '{aaai_tabs}/'")

    print("\n=== All Paper Metrics, Tables, and Figures Successfully Extracted & Saved! ===")

if __name__ == "__main__":
    main()
