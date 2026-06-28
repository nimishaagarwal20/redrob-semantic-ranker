"""Local helper: precompute corpus IDF over career prose and cache to JSON.
Used only to run the pipeline within a constrained sandbox by splitting work
across processes; `rank.py` does the same internally in one command."""
import json
import sys
import time
from collections import Counter

sys.path.insert(0, ".")
from redrob_ranker import text_model as tm, features as feat

CAND = sys.argv[1]
OUT = sys.argv[2]
t0 = time.time()
df = Counter()
n = 0
with open(CAND) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        n += 1
        df.update(set(tm.tokenize(feat.career_prose(c))))
hi = 0.5 * n
kept = {t: d for t, d in df.items() if 4 <= d <= hi}
idf = tm.build_idf(kept, n)
json.dump({"n": n, "idf": idf}, open(OUT, "w"))
print("idf n=%d vocab=%d %.1fs" % (n, len(idf), time.time() - t0))
