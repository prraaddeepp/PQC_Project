#!/usr/bin/env bash
# run_many.sh  —  run a benchmark several times and compute mean ± stdev
# usage:  ./run_many.sh <benchmark_binary> [repetitions]

set -euo pipefail
LC_ALL=C      # make awk/printf use the C locale (decimal dot)

########################  argument checking  ########################
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <benchmark_binary> [repetitions]" >&2
  exit 1
fi

BIN=$1
REPS=${2:-10}

if [[ ! -x "$BIN" ]]; then
  echo "Error: '$BIN' is not an executable file" >&2
  exit 1
fi
#####################################################################

echo "Running $BIN   ($REPS repetitions) …"
sum=0
sum2=0          # sum of squares
for ((i=1;i<=REPS;i++)); do
    # run the benchmark, grab the *last* floating-point number on the
    # line that contains “avg cycles”
    c=$( "$BIN" | awk 'BEGIN{IGNORECASE=1} /cycles\/call/ {print $NF}' )
    printf '  run %-3d  %s\n'  "$i"  "$c"
    sum=$(  awk -v s="$sum"  -v x="$c" 'BEGIN{printf "%.10f",s+x}')
    sum2=$( awk -v s="$sum2" -v x="$c" 'BEGIN{printf "%.10f",s+x*x}')
done

mean=$( awk -v s="$sum"  -v n="$REPS" 'BEGIN{printf "%.6f",s/n}')
stdev=$(awk -v s="$sum2" -v m="$mean" -v n="$REPS" \
               'BEGIN{printf "%.6f", sqrt((s/n)-(m*m))}')

echo
echo "RESULT  $BIN"
echo "  mean  = $mean  cycles/call"
echo "  stdev = $stdev"
