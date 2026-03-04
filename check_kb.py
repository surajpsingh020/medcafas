"""Check KB document format and answer lengths."""
import json
from collections import Counter

with open("data/kb_meta.json") as f:
    meta = json.load(f)

sources = Counter(m.get("source","") for m in meta)

print("Sources:")
for src, cnt in sources.most_common():
    src_lengths = [len(m["answer"].split()) for m in meta if m.get("source","") == src]
    avg = sum(src_lengths)/len(src_lengths) if src_lengths else 0
    short = sum(1 for l in src_lengths if l <= 5)
    print(f"  {src}: {cnt} docs, avg answer={avg:.1f} words, <=5 words: {short} ({100*short/len(src_lengths):.0f}%)")

print("\nSample MedQA answers:")
for m in meta[:3]:
    print(f"  [{m['source']}] A: {m['answer'][:200]}")

print("\nSample PubMedQA-Artificial answers:")
count = 0
for m in meta:
    if m.get("source") == "PubMedQA-Artificial":
        print(f"  A: {m['answer'][:250]}")
        count += 1
        if count >= 2:
            break

print("\nSample MedMCQA answers:")
count = 0
for m in meta:
    if m.get("source") == "MedMCQA":
        print(f"  A: {m['answer'][:250]}")
        count += 1
        if count >= 2:
            break
