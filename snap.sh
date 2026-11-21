#!/usr/bin/env bash
# snap.sh — quick LOCAL versioning checkpoint of the Gaudi AI project.
# Commits the AI project repo PLUS both plugin repos in one shot (timestamped).
# Local only — the off-drive backup is a separate once-a-day job
# (daily_backup.sh / ai-daily-backup.timer) that keeps ONE copy on /mnt/data2
# and prunes history to a rolling 7-day window.
#
#   ./snap.sh                    # commit everything dirty, msg "snapshot"
#   ./snap.sh "fixed embed NaN"  # commit with your message
#   git quick "msg"              # same thing via the global alias
set -uo pipefail

ROOT="$HOME/AI"
MSG="${*:-snapshot}"
STAMP="$(date '+%Y-%m-%d %H:%M:%S %Z')"
REPOS=("$ROOT" "$ROOT/hpu-moe-ext" "$ROOT/hpu-dsa-ext")

echo "snap @ $STAMP — \"$MSG\""
any=0
for r in "${REPOS[@]}"; do
  name="$(basename "$r")"
  [ -d "$r/.git" ] || { printf '  %-14s skip (not a git repo)\n' "$name"; continue; }
  git -C "$r" config user.name  >/dev/null 2>&1 || git -C "$r" config user.name  "gaudi-lab"
  git -C "$r" config user.email >/dev/null 2>&1 || git -C "$r" config user.email "maintainer@example.com"
  if [ -z "$(git -C "$r" status --porcelain)" ]; then
    printf '  %-14s clean\n' "$name"; continue
  fi
  git -C "$r" add -A
  if git -C "$r" commit --quiet -m "snap: $MSG ($STAMP)"; then
    printf '  %-14s committed %s\n' "$name" "$(git -C "$r" rev-parse --short HEAD)"; any=1
  else
    printf '  %-14s COMMIT FAILED\n' "$name"
  fi
done
[ "$any" = 1 ] || echo "  (nothing to commit)"
