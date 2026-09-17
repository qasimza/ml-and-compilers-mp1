#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
OUT="${1:-gemm_cpu_results.txt}"

if [ ! -x ./mp1_cpu ]; then
  echo "mp1_cpu not found. Compile first:"
  echo "  g++ -fopenmp -o mp1_cpu gemm_cpu.cpp"
  exit 1
fi

{
  echo "=== ./mp1_cpu 100 100 100 ==="
  ./mp1_cpu 100 100 100
  echo
  echo "=== ./mp1_cpu 1000 1000 1000 ==="
  ./mp1_cpu 1000 1000 1000
} | tee "$OUT"

echo "Wrote $OUT"
