#!/usr/bin/env bash
# Track J: fetch the subject program, more-itertools v11.2.1 (MIT licence),
# ~7,300 lines of library code and ~9,200 lines of tests.
#   pip install pytest pytest-cov coverage pulp==2.9.0 highspy matplotlib
set -euo pipefail
cd "$(dirname "$0")"
TAG=v11.2.1
if [ ! -d more-itertools ]; then
  git clone https://github.com/more-itertools/more-itertools.git more-itertools
fi
git -C more-itertools checkout -q "$TAG"
git -C more-itertools log --oneline -1
echo "OK. Now: python run_experiments.py"
