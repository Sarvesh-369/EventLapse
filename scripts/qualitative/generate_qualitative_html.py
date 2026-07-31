#!/usr/bin/env python3
import json
import html
from pathlib import Path

EVENTLAPSE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = EVENTLAPSE_DIR / "qualitative_examples"
JSON_PATH = OUTPUT_DIR / "qualitative_dataset_30.json"
HTML_PATH = OUTPUT_DIR / "qualitative_examples.html"

CATEGORY_CONFIG = {
    "correct": {
        "title": "Faithful Event Recovery (Correct)",
        "badge_class": "badge-correct",
        "card_class": "card-correct",
        "description": "Reported trace aligns one-to-one with executable ground truth, supporting the correct final answer."
    },
    "missed_event": {
        "title": "Missed Events (Under-Reporting)",
        "badge_class": "badge-missed",
        "card_class": "card-error",
        "description": "Model omits intermediate event transitions under high load, under-reporting the true event count."
    },
    "hallucinated_event": {
        "title": "Hallucinated Events (Over-Reporting)",
        "badge_class": "badge-hallucinated",
        "card_class": "card-error",
        "description": "Model over-reports events, detecting spurious boundary collisions during continuous motion."
    },
    "correct_trace_wrong_final_answer": {
        "title": "Wrong Accumulation (RFR)",
        "badge_class": "badge-rfr",
        "card_class": "card-warning",
        "description": "Reasoning Failure Ratio: Trace F1 ≥ 80%, but final integer count is incorrect due to an aggregation error."
    },
    "incorrect_trace_accidental_correct": {
        "title": "Accidental Correctness (ACR)",
        "badge_class": "badge-acr",
        "card_class": "card-warning",
        "description": "Accidental Correctness Ratio: Final count matches ground truth, but the reported trace is unfaithful or incomplete."
    },
    "temporally_displaced_event": {
        "title": "Temporally Displaced Events",
        "badge_class": "badge-displaced",
        "card_class": "card-warning",
        "description": "Event occurrence is identified, but reported timestamps fall outside the 1.0s tolerance window."
    }
}

def clean_model_text(text):
    if not text:
        return ""
    text = html.escape(str(text))
    return text.replace("\n", "<br>")

def generate_html():
    with open(JSON_PATH) as f:
        dataset = json.load(f)
        
    total_samples = sum(len(samples) for samples in dataset.values())
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EventLapse Qualitative Evaluation - Key Event Frame Strips</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-main: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
            --primary: #2563eb;
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

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-main);
            color: var(--text-main);
            line-height: 1.5;
            padding: 40px 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            background: #ffffff;
            padding: 32px;
            border-radius: 16px;
            border: 1px solid var(--border-color);
            margin-bottom: 32px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }}

        h1 {{
            font-size: 28px;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 8px;
        }}

        p.subtitle {{
            font-size: 15px;
            color: var(--text-muted);
            margin-bottom: 24px;
        }}

        .filter-bar {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}

        .filter-btn {{
            padding: 8px 16px;
            border-radius: 20px;
            border: 1px solid var(--border-color);
            background: #f1f5f9;
            color: var(--text-main);
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .filter-btn:hover {{
            background: #e2e8f0;
        }}

        .filter-btn.active {{
            background: #1e293b;
            color: #ffffff;
            border-color: #1e293b;
        }}

        .category-section {{
            margin-bottom: 48px;
        }}

        .category-header {{
            margin-bottom: 20px;
            padding-bottom: 8px;
            border-bottom: 2px solid var(--border-color);
            display: flex;
            align-items: baseline;
            justify-content: space-between;
        }}

        .category-title {{
            font-size: 20px;
            font-weight: 700;
            color: #1e293b;
        }}

        .category-desc {{
            font-size: 14px;
            color: var(--text-muted);
            margin-top: 4px;
        }}

        .sample-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}

        .sample-card:hover {{
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
        }}

        /* Frame Strip Layout */
        .frame-strip {{
            display: grid;
            grid-template-columns: repeat(8, 1fr);
            gap: 6px;
            margin-bottom: 20px;
            background: #0f172a;
            padding: 6px;
            border-radius: 10px;
        }}

        .frame-item {{
            position: relative;
            aspect-ratio: 16 / 9;
            overflow: hidden;
            border-radius: 6px;
            background: #1e293b;
            border: 2px solid transparent;
        }}

        .frame-item.is-event-frame {{
            border-color: #22c55e;
            box-shadow: 0 0 8px rgba(34, 197, 94, 0.5);
        }}

        .frame-item img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}

        .frame-label {{
            position: absolute;
            bottom: 2px;
            right: 4px;
            background: rgba(0, 0, 0, 0.8);
            color: #ffffff;
            font-size: 9.5px;
            font-family: 'JetBrains Mono', monospace;
            padding: 1px 4px;
            border-radius: 3px;
        }}

        .event-badge-on-frame {{
            position: absolute;
            top: 2px;
            left: 2px;
            background: #22c55e;
            color: #052e16;
            font-size: 8.5px;
            font-weight: 700;
            padding: 1px 4px;
            border-radius: 3px;
            text-transform: uppercase;
        }}

        .meta-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}

        .question-text {{
            font-size: 16px;
            font-weight: 700;
            color: #0f172a;
        }}

        .badge {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .badge-correct {{
            background: #dcfce7;
            color: #15803d;
            border: 1px solid #bbf7d0;
        }}

        .badge-missed {{
            background: #fee2e2;
            color: #b91c1c;
            border: 1px solid #fecaca;
        }}

        .badge-hallucinated {{
            background: #ffedd5;
            color: #c2410c;
            border: 1px solid #fed7aa;
        }}

        .badge-rfr {{
            background: #fef3c7;
            color: #b45309;
            border: 1px solid #fde68a;
        }}

        .badge-acr {{
            background: #e0e7ff;
            color: #4338ca;
            border: 1px solid #c7d2fe;
        }}

        .badge-displaced {{
            background: #f3e8ff;
            color: #6b21a8;
            border: 1px solid #e9d5ff;
        }}

        .info-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-top: 16px;
        }}

        .box {{
            padding: 16px;
            border-radius: 10px;
            font-size: 13px;
        }}

        .box-gt {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
        }}

        .box-pred {{
            border: 1px solid var(--border-color);
        }}

        .card-correct .box-pred {{
            background: var(--success-bg);
            border-color: var(--success-border);
            color: var(--success-text);
        }}

        .card-error .box-pred {{
            background: var(--error-bg);
            border-color: var(--error-border);
            color: var(--error-text);
        }}

        .card-warning .box-pred {{
            background: var(--warning-bg);
            border-color: var(--warning-border);
            color: var(--warning-text);
        }}

        .box-title {{
            font-weight: 700;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
        }}

        .ts-list {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            background: rgba(255, 255, 255, 0.6);
            padding: 8px;
            border-radius: 6px;
            margin-top: 6px;
            word-break: break-all;
        }}

        .response-text {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            max-height: 140px;
            overflow-y: auto;
            background: #ffffff;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid rgba(0, 0, 0, 0.1);
            margin-top: 8px;
        }}

        .metrics-row {{
            display: flex;
            gap: 16px;
            margin-top: 12px;
            font-size: 12px;
            font-weight: 600;
        }}

        .justification-box {{
            margin-top: 16px;
            padding: 12px 16px;
            background: #f1f5f9;
            border-left: 4px solid #3b82f6;
            border-radius: 0 8px 8px 0;
            font-size: 13px;
            color: #334155;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>EventLapse Qualitative Profiling</h1>
            <p class="subtitle">Trace-Grounded Evaluation & Taxonomy Analysis with <strong>Key Event Frame Strips</strong> for <strong>Gemini 3.6 Flash</strong></p>
            <div class="filter-bar">
                <button class="filter-btn active" onclick="filterCategory('all')">All Categories ({total_samples})</button>
                <button class="filter-btn" onclick="filterCategory('correct')">Faithful Recovery (5)</button>
                <button class="filter-btn" onclick="filterCategory('missed_event')">Missed Events (5)</button>
                <button class="filter-btn" onclick="filterCategory('hallucinated_event')">Hallucinated Events (5)</button>
                <button class="filter-btn" onclick="filterCategory('correct_trace_wrong_final_answer')">Wrong Accumulation (5)</button>
                <button class="filter-btn" onclick="filterCategory('incorrect_trace_accidental_correct')">Accidental Correct (5)</button>
                <button class="filter-btn" onclick="filterCategory('temporally_displaced_event')">Displaced Events (5)</button>
            </div>
        </header>
"""

    for cat_key, cat_info in CATEGORY_CONFIG.items():
        samples = dataset.get(cat_key, [])
        if not samples:
            continue
            
        html_content += f"""
        <div class="category-section" id="cat-{cat_key}">
            <div class="category-header">
                <div>
                    <span class="category-title">{cat_info['title']}</span>
                    <p class="category-desc">{cat_info['description']}</p>
                </div>
            </div>
"""

        for sample_idx, sample in enumerate(samples, 1):
            frame_imgs = ""
            frame_meta_list = sample.get("frame_meta", [])
            for f_meta in frame_meta_list:
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
            
            card_class = cat_info["card_class"]
            badge_class = cat_info["badge_class"]
            
            html_content += f"""
            <div class="sample-card {card_class}">
                <div class="frame-strip">
                    {frame_imgs}
                </div>
                <div class="meta-header">
                    <div class="question-text">
                        Case {sample_idx}: {html.escape(sample['question'])}
                        <span style="font-size: 12px; font-weight: normal; color: #64748b; margin-left: 8px;">
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

        html_content += "</div>\n"

    html_content += """
    </div>

    <script>
        function filterCategory(cat) {
            const sections = document.querySelectorAll('.category-section');
            const buttons = document.querySelectorAll('.filter-btn');
            
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            sections.forEach(sec => {
                if (cat === 'all' || sec.id === 'cat-' + cat) {
                    sec.style.display = 'block';
                } else {
                    sec.style.display = 'none';
                }
            });
        }
    </script>
</body>
</html>
"""

    with open(HTML_PATH, "w") as f:
        f.write(html_content)
    print(f"Generated qualitative report HTML with Key Event Frames at {HTML_PATH}")

if __name__ == "__main__":
    generate_html()
