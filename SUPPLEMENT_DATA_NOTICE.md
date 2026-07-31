# Supplementary Data Package Notice

Due to the conference submission portal's strict **50 MB file size limit** for Code and Data Supplements, this archive contains a curated **2-seed representative video dataset** (Seeds 0 and 1) covering all $N \times F$ parameter cells across all three visual domains (`bounce_ball`, `blinking`, and `state_machine`), along with 100% of executable ground-truth traces, questions, and evaluation schemas.

To reproduce the full 20-seed dataset locally (or any arbitrary number of seeds):

```bash
# Generate complete benchmark dataset (20 seeds per cell)
python3 scripts/generate_dataset.py --num-seeds 20 --tasks all
```
