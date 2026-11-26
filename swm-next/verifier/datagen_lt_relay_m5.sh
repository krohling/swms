#!/bin/bash
# m5 side of datagen_lt_larg.sh: copy finished, verified units of a set from
# LARG scratch (200 GB quota, shared with everything else of ours) to m5, and
# with `delete` remove each unit on LARG once its m5 copy has the same sha256,
# leaving <unit>.relayed there (which the launcher counts as done).
#
# Usage: datagen_lt_relay_m5.sh <larg host alias> <set> [delete]
#   e.g. datagen_lt_relay_m5.sh larg-pogba lt_v1_b0 delete
# One pass, then exit; bash 3.2 compatible. Prints one RELAYED / MISMATCH line
# per unit and a summary line.
set -u
HOST=${1:?host alias}; SET=${2:?set}; MODE=${3:-keep}
REMOTE=/scratch/cluster/krohling/swms-ltgen/experiments/ltgen_larg/$SET/units
HERE=$(cd "$(dirname "$0")/../.." && pwd)
DEST=${LTGEN_DEST:-$(cd "$HERE/.." && pwd)/swms/datasets_swm/lt_v1/$SET/units}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=12"
mkdir -p "$DEST"
LIST=$($SSH $HOST "cd $REMOTE 2>/dev/null && ls | grep -E '\.h5$' | grep -v '\.part\.h5$' | while read -r f; do sha256sum \$f; done" < /dev/null) \
  || { echo "RELAY_UNREACHABLE $HOST"; exit 2; }
n=0; bad=0
while read -r sum f; do
  [ -n "${f:-}" ] || continue
  u=${f%.h5}
  rsync -a -e "$SSH" "$HOST:$REMOTE/$u.h5" "$HOST:$REMOTE/$u.verify.txt" "$DEST/" < /dev/null \
    || { echo "COPY_FAILED $u"; bad=$((bad+1)); continue; }
  got=$(shasum -a 256 "$DEST/$u.h5" | cut -d' ' -f1)
  if [ "$got" != "$sum" ]; then
    echo "MISMATCH $u larg $sum m5 $got"; bad=$((bad+1)); continue
  fi
  grep -q "^VERIFY_PASS" "$DEST/$u.verify.txt" || { echo "NOT_VERIFIED $u"; bad=$((bad+1)); continue; }
  if [ "$MODE" = delete ]; then
    # only the file whose checksum was just compared; a changed file is kept
    $SSH $HOST "cd $REMOTE && [ \"\$(sha256sum $u.h5 | cut -d' ' -f1)\" = $sum ] && rm $u.h5 && touch $u.relayed" < /dev/null \
      || { echo "REMOTE_DELETE_FAILED $u"; bad=$((bad+1)); continue; }
  fi
  n=$((n+1)); echo "RELAYED $u $(du -k "$DEST/$u.h5" | cut -f1) KB"
done <<EOF
$LIST
EOF
tot=$(ls "$DEST" | grep -c '\.h5$')
echo "RELAY_PASS $(date '+%m-%d %H:%M PT') set $SET: $n copied this pass, $bad problems, $tot units on m5 ($(du -sh "$DEST" | cut -f1)), mode $MODE"
[ $bad = 0 ]
