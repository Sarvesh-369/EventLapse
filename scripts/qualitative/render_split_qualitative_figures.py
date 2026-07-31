#!/usr/bin/env python3
import os
import json
import html
import subprocess
from pathlib import Path
from PIL import Image

QUAL_DIR = Path(__file__).resolve().parent.parent.parent / "qualitative_examples"
JSON_PATH = QUAL_DIR / "qualitative_dataset_30.json"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CATEGORY_CONFIG = {
    "correct": {
        "title": "Faithful Event Recovery (Correct)",
        "badge_class": "badge-correct",
        "card_class": "card-correct",
        "desc": "Reported trace aligns one-to-one with executable ground truth, supporting the correct final answer."
    },
    "missed_event": {
        "title": "Missed Events (Under-Reporting / Perception Failure)",
        "badge_class": "badge-missed",
        "card_class": "card-error",
        "desc": "Model omits intermediate event transitions under high load, under-reporting the true event count."
    },
    "hallucinated_event": {
        "title": "Hallucinated Events (Over-Reporting / Spurious Detection)",
        "badge_class": "badge-hallucinated",
        "card_class": "card-error",
        "desc": "Model over-reports events, detecting spurious boundary collisions during continuous motion."
    },
    "correct_trace_wrong_final_answer": {
        "title": "Wrong Accumulation (Reasoning Failure Ratio / RFR)",
        "badge_class": "badge-rfr",
        "card_class": "card-warning",
        "desc": "Reasoning Failure Ratio: Trace F1 >= 80%, but final integer count is incorrect due to an aggregation error."
    },
    "incorrect_trace_accidental_correct": {
        "title": "Accidental Correctness (Accidental Correctness Ratio / ACR)",
        "badge_class": "badge-acr",
        "card_class": "card-warning",
        "desc": "Accidental Correctness Ratio: Final count matches ground truth, but the reported trace is unfaithful or incomplete."
    },
    "temporally_displaced_event": {
        "title": "Temporally Displaced Events",
        "badge_class": "badge-displaced",
        "card_class": "card-warning",
        "desc": "Event occurrence is identified, but reported timestamps fall outside the 1.0s tolerance window."
    }
}

def clean_model_text(text):
    if not text:
        return ""
    text = html.escape(str(text))
    return text.replace("\n", "<br>")

def build_card_html(sample, sample_idx, cat_key, cat_info):
    badge_class = cat_info.get("badge_class", "badge-correct")
    card_class = cat_info.get("card_class", "card-correct")

    frame_imgs = ""
    for f_meta in sample.get("frame_meta", []):
        f_path = f_meta["rel_path"]
        ts_label = f_meta["timestamp_str"]
        is_evt = f_meta.get("is_event", False)
        evt_num = f_meta.get("event_num", None)
        
        evt_border_cls = "is-event-frame" if is_evt else ""
        evt_badge_html = f'<span class="event-badge-on-frame">Event #{evt_num}</span>' if is_evt and evt_num else ""
        
        frame_imgs += f"""
        <div class="frame-item {evt_border_cls}">
            {evt_badge_html}
            <img src="{f_path}" alt="Frame {ts_label}">
            <span class="frame-label">{ts_label}</span>
        </div>
        """

    gt_ts_str = ", ".join(sample["gt_ts"]) if sample["gt_ts"] else "None"
    pred_ts_str = ", ".join(sample["pred_ts"]) if sample["pred_ts"] else "None"

    return f"""
    <div class="sample-card {card_class}">
        <div class="frame-strip">
            {frame_imgs}
        </div>
        <div class="meta-header">
            <div class="question-text">
                Case {sample_idx}: {html.escape(sample['question'])}
                <span style="font-size: 11px; font-weight: normal; color: #64748b; margin-left: 6px;">
                    [{sample['task']} | N={sample['control_N']}, F={sample['control_F']} | Seed {sample['seed']}]
                </span>
            </div>
            <span class="badge {badge_class}">{cat_key.replace('_', ' ')}</span>
        </div>

        <div class="info-grid">
            <div class="box box-gt">
                <div class="box-title">
                    <span>Ground Truth Executable Trace</span>
                    <span>GT Count: <strong>{sample['gt_ans']}</strong></span>
                </div>
                <div>Total Target Events: {sample['gt_N']}</div>
                <div class="ts-list"><strong>Timestamps:</strong> [{gt_ts_str}]</div>
            </div>

            <div class="box box-pred">
                <div class="box-title">
                    <span>Gemini 3.6 Flash Prediction</span>
                    <span>Pred Answer: <strong>{sample['pred_ans']}</strong></span>
                </div>
                <div class="metrics-row">
                    <span>Trace F1: {sample['f1']}%</span>
                    <span>Precision: {sample['p']}%</span>
                    <span>Recall: {sample['r']}%</span>
                </div>
                <div class="ts-list"><strong>Reported Timestamps:</strong> [{pred_ts_str}]</div>
                <div class="response-text"><strong>Raw Model Output:</strong><br>{clean_model_text(sample['raw_response'])}</div>
            </div>
        </div>

        <div class="justification-box">
            <strong>Diagnostic Justification:</strong> {html.escape(sample['justification'])}
        </div>
    </div>
    """

def render_split_figures():
    with open(JSON_PATH) as f:
        dataset = json.load(f)

    for cat_key, samples in dataset.items():
        cat_info = CATEGORY_CONFIG.get(cat_key, {})
        
        # Split 5 samples into Part 1 (Cards 1..3) and Part 2 (Cards 4..5)
        splits = [
            ("part1", samples[0:3], 1, "Part 1 (Cases 1–3)"),
            ("part2", samples[3:5], 4, "Part 2 (Cases 4–5)")
        ]

        for part_name, part_samples, start_idx, part_title in splits:
            html_filename = f"qualitative_{cat_key}_{part_name}.html"
            pdf_filename = f"qualitative_{cat_key}_{part_name}.pdf"
            temp_png_filename = f"qualitative_{cat_key}_{part_name}_temp.png"

            single_html_path = QUAL_DIR / html_filename
            temp_png_path = QUAL_DIR / temp_png_filename
            single_pdf_path = QUAL_DIR / pdf_filename

            cards_html = ""
            for idx_offset, s in enumerate(part_samples):
                cards_html += build_card_html(s, start_idx + idx_offset, cat_key, cat_info)

            full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-main: #ffffff;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
            --success-bg: #f0fdf4;
            --success-border: #bbf7d0;
            --success-text: #166534;
            --error-bg: #fef2f2;
            --error-border: #fecaca;
            --error-text: #991b1b;
            --warning-bg: #fffbeb;
            --warning-border: #fde68a;
            --warning-text: #92400e;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #ffffff;
            padding: 8px;
            width: 1180px;
        }}
        .cards-container {{
            display: flex;
            flex-direction: column;
            gap: 14px;
        }}
        .sample-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
        }}
        .frame-strip {{
            display: grid;
            grid-template-columns: repeat(8, 1fr);
            gap: 6px;
            margin-bottom: 12px;
            background: #0f172a;
            padding: 6px;
            border-radius: 8px;
        }}
        .frame-item {{
            position: relative;
            aspect-ratio: 16 / 9;
            overflow: hidden;
            border-radius: 5px;
            background: #1e293b;
            border: 2px solid transparent;
        }}
        .frame-item.is-event-frame {{
            border-color: #22c55e;
            box-shadow: 0 0 6px rgba(34, 197, 94, 0.6);
        }}
        .frame-item img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}
        .frame-label {{
            position: absolute;
            bottom: 2px;
            right: 3px;
            background: rgba(0, 0, 0, 0.85);
            color: #ffffff;
            font-size: 9px;
            font-family: 'JetBrains Mono', monospace;
            padding: 1px 3px;
            border-radius: 3px;
        }}
        .event-badge-on-frame {{
            position: absolute;
            top: 2px;
            left: 2px;
            background: #22c55e;
            color: #052e16;
            font-size: 8px;
            font-weight: 700;
            padding: 1px 3px;
            border-radius: 3px;
            text-transform: uppercase;
        }}
        .meta-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .question-text {{
            font-size: 14.5px;
            font-weight: 700;
            color: #0f172a;
        }}
        .badge {{
            padding: 4px 10px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .badge-correct {{ background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }}
        .badge-missed {{ background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }}
        .badge-hallucinated {{ background: #ffedd5; color: #c2410c; border: 1px solid #fed7aa; }}
        .badge-rfr {{ background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }}
        .badge-acr {{ background: #e0e7ff; color: #4338ca; border: 1px solid #c7d2fe; }}
        .badge-displaced {{ background: #f3e8ff; color: #6b21a8; border: 1px solid #e9d5ff; }}
        .info-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 10px;
        }}
        .box {{
            padding: 10px 12px;
            border-radius: 8px;
            font-size: 12px;
        }}
        .box-gt {{ background: #f8fafc; border: 1px solid #e2e8f0; }}
        .box-pred {{ border: 1px solid var(--border-color); }}
        .card-correct .box-pred {{ background: var(--success-bg); border-color: var(--success-border); color: var(--success-text); }}
        .card-error .box-pred {{ background: var(--error-bg); border-color: var(--error-border); color: var(--error-text); }}
        .card-warning .box-pred {{ background: var(--warning-bg); border-color: var(--warning-border); color: var(--warning-text); }}
        .box-title {{
            font-weight: 700;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
            display: flex;
            justify-content: space-between;
        }}
        .ts-list {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 10.5px;
            background: rgba(255, 255, 255, 0.75);
            padding: 4px 6px;
            border-radius: 4px;
            margin-top: 4px;
            word-break: break-all;
        }}
        .response-text {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            max-height: 90px;
            overflow-y: auto;
            background: #ffffff;
            padding: 6px;
            border-radius: 4px;
            border: 1px solid rgba(0, 0, 0, 0.1);
            margin-top: 4px;
        }}
        .metrics-row {{
            display: flex;
            gap: 10px;
            margin-top: 6px;
            font-size: 11px;
            font-weight: 600;
        }}
        .justification-box {{
            margin-top: 10px;
            padding: 8px 12px;
            background: #f1f5f9;
            border-left: 3.5px solid #3b82f6;
            border-radius: 0 6px 6px 0;
            font-size: 11.5px;
            color: #334155;
        }}
    </style>
</head>
<body>
    <div class="cards-container">
        {cards_html}
    </div>
</body>
</html>
"""
            with open(single_html_path, "w") as out_f:
                out_f.write(full_html)

            # Window height depends on part1 (3 cards ~ 1350px) vs part2 (2 cards ~ 900px)
            win_h = 1450 if part_name == "part1" else 980

            cmd = [
                CHROME, "--headless", "--disable-gpu", "--hide-scrollbars",
                "--force-device-scale-factor=2",
                f"--screenshot={temp_png_path}",
                f"--window-size=1200,{win_h}",
                f"file://{single_html_path}"
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if temp_png_path.exists():
                img = Image.open(temp_png_path)
                bbox = img.getbbox()
                if bbox:
                    cropped = img.crop(bbox).convert("RGB")
                    cropped.save(single_pdf_path, "PDF", resolution=300.0)
                else:
                    img.convert("RGB").save(single_pdf_path, "PDF", resolution=300.0)
                temp_png_path.unlink(missing_ok=True)
                print(f"Rendered {single_pdf_path.name}")

if __name__ == "__main__":
    render_split_figures()
