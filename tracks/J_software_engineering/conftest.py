import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Only our own tests: more-itertools/ (the subject) and work/ (its mutation copies)
# hold the subject's test suite, which run_experiments.py runs separately.
collect_ignore = ["more-itertools", "work"]
