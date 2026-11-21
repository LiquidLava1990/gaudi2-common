#!/usr/bin/env python3
"""Verify a served Qwen multimodal model can actually SEE an image.
Generates a synthetic image with unambiguous content (a big magenta circle and
the text 'GAUDI 7' on white), sends it via the OpenAI vision chat API, and prints
the model's description. PASS = the description mentions the shape/color/text.

Usage: python3 test_vision.py [base_url] [api_key] [served_name]
"""
import base64, io, json, sys, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://<SERVER_IP>:8000/v1"
KEY  = sys.argv[2] if len(sys.argv) > 2 else "${VLLM_API_KEY}"
NAME = sys.argv[3] if len(sys.argv) > 3 else "qwen36"

# --- build a synthetic test image ---
from PIL import Image, ImageDraw, ImageFont
img = Image.new("RGB", (512, 512), "white")
d = ImageDraw.Draw(img)
d.ellipse([120, 80, 392, 352], fill=(220, 20, 160))   # magenta circle
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
except Exception:
    font = ImageFont.load_default()
d.text((150, 400), "GAUDI 7", fill="black", font=font)
buf = io.BytesIO(); img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()

payload = {
    "model": NAME,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in one sentence. What shape and color is it, and what text appears?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ],
    }],
    "max_tokens": 512, "temperature": 0,
}
req = urllib.request.Request(
    f"{BASE}/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
)
try:
    resp = json.load(urllib.request.urlopen(req, timeout=120))
    out = resp["choices"][0]["message"]["content"]
    print("VISION OUTPUT:", repr(out))
    low = out.lower()
    hits = [w for w in ("circle", "magenta", "pink", "purple", "gaudi", "7") if w in low]
    print("KEYWORD HITS:", hits)
    print("VISION PASS" if len(hits) >= 2 else "VISION UNCERTAIN — inspect output")
except Exception as e:
    print("VISION REQUEST FAILED:", repr(e))
    sys.exit(1)
