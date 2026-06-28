"""Local helper: merge chunk top-Ks -> final top-100 submission.csv."""
import csv
import pickle
import sys

sys.path.insert(0, ".")
from redrob_ranker import reasoning as rsn

OUT = sys.argv[1]
chunks = sys.argv[2:]
allc = []
for ch in chunks:
    allc.extend(pickle.load(open(ch, "rb")))
allc.sort(key=lambda x: (-x[0], x[1]))   # score desc, candidate_id asc
top = allc[:100]

rows = []
prev = None
for i, (final, cid, ft, detail) in enumerate(top):
    score = round(final, 6)
    if prev is not None and score > prev:
        score = prev
    prev = score
    reasoning = rsn.make_reasoning(ft, detail, i + 1)
    rows.append((cid, i + 1, "%.6f" % score, reasoning))

with open(OUT, "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
    w.writerow(["candidate_id", "rank", "score", "reasoning"])
    w.writerows(rows)
print("wrote", OUT, len(rows), "rows; score range %.4f..%.4f" % (rows[0][2] and float(rows[0][2]), float(rows[-1][2])))
