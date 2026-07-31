#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

QUAL_DIR = Path(__file__).resolve().parent.parent.parent / "qualitative_examples"
HTML_MAIN = QUAL_DIR / "qualitative_examples.html"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CATEGORIES = [
    "correct",
    "missed_event",
    "hallucinated_event",
    "correct_trace_wrong_final_answer",
    "incorrect_trace_accidental_correct",
    "temporally_displaced_event"
]

def render_category_htmls():
    with open(HTML_MAIN, "r") as f:
        content = f.read()

    for cat in CATEGORIES:
        # Create a single category html
        cat_html_path = QUAL_DIR / f"qualitative_{cat}.html"
        cat_png_path = QUAL_DIR / f"qualitative_{cat}.png"
        
        # Replace filter script or hide header and other sections
        style_override = "<style>\nheader { display: none !important; }\nbody { padding-top: 10px !important; }\n"
        for other_cat in CATEGORIES:
            if other_cat != cat:
                style_override += f"#cat-{other_cat} {{ display: none !important; }}\n"
        style_override += "</style>\n</head>"
        
        cat_content = content.replace("</head>", style_override)
        with open(cat_html_path, "w") as out_f:
            out_f.write(cat_content)

        # Take screenshot
        cmd = [
            CHROME, "--headless", "--disable-gpu", "--hide-scrollbars",
            f"--screenshot={cat_png_path}",
            "--window-size=1280,2400",
            f"file://{cat_html_path}"
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if cat_png_path.exists():
            print(f"Rendered {cat_png_path.name} ({cat_png_path.stat().st_size} bytes)")

if __name__ == "__main__":
    render_category_htmls()
