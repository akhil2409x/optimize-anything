#!/usr/bin/env bash
# Fetch the Toronto (Carter) exam-timetabling benchmark .stu files into data/.
# Source: the copy in github.com/pl-sch/its-exam-timetabling (data/toronto), pinned commit.
# Original: Carter, Laporte & Lee (1996); see https://people.cs.nott.ac.uk/pszrq/data.htm
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=d8eab9b7e91c13ae14b90dafcec588f605e0f67f
TMP=$(mktemp -d)
git clone -q https://github.com/pl-sch/its-exam-timetabling.git "$TMP/src"
git -C "$TMP/src" checkout -q "$COMMIT"
cp "$TMP"/src/data/toronto/*.stu data/
rm -rf "$TMP"
ls data/*.stu
echo "OK. Now: python run_experiments.py   (picks up every data/*.stu)"
