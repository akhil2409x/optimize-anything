#!/usr/bin/env bash
# Track G: fetch upstream xv6-riscv at the exact commit the patch was made
# against, apply the patch, and check that both kernel variants build.
#
#   Ubuntu/Debian/WSL prerequisites:
#     sudo apt-get install -y qemu-system-misc gcc-riscv64-linux-gnu binutils-riscv64-linux-gnu make git
#     pip install pexpect matplotlib
set -euo pipefail
cd "$(dirname "$0")"

UPSTREAM=https://github.com/mit-pdos/xv6-riscv.git
COMMIT=e93dabeb98431235c6e1a101cb96c73c923fdbc4   # upstream HEAD on 2026-10-26

if [ ! -d xv6-riscv ]; then
  git clone "$UPSTREAM" xv6-riscv
fi
cd xv6-riscv
git checkout -q "$COMMIT"
git checkout -q -B optimize
git apply --check ../patches/xv6-optimize.patch && git apply ../patches/xv6-optimize.patch

for cfg in "SCHED=RR COW=0" "SCHED=MLFQ COW=1"; do
  make clean >/dev/null
  make $cfg kernel/kernel fs.img >/dev/null
  echo "built: $cfg"
done
echo "OK. Boot interactively with:  cd xv6-riscv && make SCHED=MLFQ COW=1 CPUS=1 qemu"
echo "Then try:  cowtest   schedbench mixed   cowbench forkexit   usertests -q"
