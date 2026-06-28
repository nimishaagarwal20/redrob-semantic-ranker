"""Local helper: score one line-range of the pool, keep top-K, pickle it.
Identical scoring to rank.py; only exists to fit a constrained sandbox window."""
import json
import pickle
import sys
import time

sys.path.insert(0, ".")
from redrob_ranker import text_model as tm, features as feat, scoring

CAND, IDFJSON, START, COUNT, OUT = sys.argv[1:6]
START, COUNT = int(START), int(COUNT)
d = json.load(open(IDFJSON))
idf = d["idf"]
qvec, qnorm = tm.query_vector(idf)

t0 = time.time()
kept = []  # (final, candidate_id, f_slim, detail)
i = 0
with open(CAND) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if i < START:
            i += 1
            continue
        if i >= START + COUNT:
            break
        c = json.loads(line)
        ft = feat.extract(c)
        sem = tm.doc_cosine(tm.tokenize(ft["prose"]), idf, qvec, qnorm)
        final, detail = scoring.score_candidate(ft, sem)
        ft.pop("prose", None)        # keep pickles small
        ft.pop("sig", None)
        kept.append((final, c["candidate_id"], ft, detail))
        i += 1
kept.sort(key=lambda x: (-x[0], x[1]))
kept = kept[:250]
pickle.dump(kept, open(OUT, "wb"))
print("chunk %d..%d -> %d kept, top=%.4f %.1fs"
      % (START, START + COUNT, len(kept), kept[0][0] if kept else 0, time.time() - t0))
