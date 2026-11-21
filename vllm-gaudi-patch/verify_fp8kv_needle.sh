#!/usr/bin/env bash
# Long-context needle recall for the fp8_inc MLA KV 224-cap fix.
# Builds a ~2300-token prompt with a secret code buried mid-context, then checks recall.
# BEFORE fix: garbage / </think> spam / wrong code.  AFTER fix: prints the exact code -> PASS.
set -euo pipefail
HOST="${HOST:-http://localhost:8000}"
MODEL="${MODEL:-glm52}"
NEEDLE="BANZAI-7391-QUOKKA"

# ~2300-token filler: a distinct sentence repeated, with the needle inserted mid-way.
FILLER="The maintenance logbook records routine turbine inspections across the northern facility. "
BODY=""
for i in $(seq 1 120); do
  BODY+="$FILLER"
  if [ "$i" -eq 60 ]; then
    BODY+="IMPORTANT: the vault access code is ${NEEDLE}. Remember it exactly. "
  fi
done

PROMPT="Read the following log carefully.\n\n${BODY}\n\nQuestion: What is the vault access code? Reply with ONLY the code, nothing else."

# token count sanity (optional): rough estimate
echo "[info] prompt chars: ${#PROMPT} (~$(( ${#PROMPT} / 4 )) tokens est.)"

REQ=$(python3 - "$MODEL" "$PROMPT" <<'PY'
import json,sys
model, prompt = sys.argv[1], sys.argv[2]
print(json.dumps({
  "model": model,
  "messages":[{"role":"user","content":prompt}],
  "max_tokens":64,
  "temperature":0
}))
PY
)

RESP=$(curl -s "${HOST}/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ")
OUT=$(printf '%s' "$RESP" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"])' 2>/dev/null || echo "<parse-error> $RESP")

echo "----- model output -----"
echo "$OUT"
echo "------------------------"
if printf '%s' "$OUT" | grep -qF "$NEEDLE"; then
  echo "NEEDLE PASS: recalled ${NEEDLE}"
  exit 0
else
  echo "NEEDLE FAIL: code not recalled (fp8 KV still corrupting long context)"
  exit 1
fi
