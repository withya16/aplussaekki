# core/check_pages_text.py
import json
from pathlib import Path

p = Path("artifacts/lecture/pages_text.json")
data = json.loads(p.read_text(encoding="utf-8"))

pages = data["pages"]
lens = [pg.get("raw_len", 0) for pg in pages]

print("page_count:", data["page_count"])
print("nonempty_pages:", sum(1 for x in lens if x > 0), "/", len(lens))
print("zero_text_pages:", sum(1 for x in lens if x == 0))
print("max_len:", max(lens), "min_len:", min(lens))
print("sample first 5 lens:", lens[:5])
