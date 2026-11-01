# Demonstration

A 10-slide deck covering:
- the problem;
- the formulation and the methods;
- evidence on real Toronto data;
- the ILP + local-search hybrid;
- one number per remaining track;
- negative results;
- AI use and lessons.

Each slide has speaker notes. Every number on the slides comes from a
track's `results/SUMMARY.md`.

**Where the deck is:**
- **To present or download:** the deck lives as a Claude Slides artifact in
  the student's claude.ai account ("Optimize Anything — Demo"). It exports
  to PowerPoint or PDF from its Share menu. Put the exported PDF here as
  `demo.pdf`.
- **Source:** `slides-source/`, one HTML file per slide plus `deck.json`.

**Suggested live demo (≈ 3 min):**

```bash
cd tracks/E_exam_timetabling
./get_data.sh                   # real Toronto data
python solve.py data/yor-f-83.stu   # baseline fails; DSatur 61.66 → DSatur-proximity + hill climbing 46.36
python solve.py data/hec-s-92.stu   # every greedy method fails …
python hybrid_hec.py                # … the ILP repairs it, hill climbing lowers the cost (~20 s)
```
