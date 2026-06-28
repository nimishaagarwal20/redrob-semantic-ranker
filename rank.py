#!/usr/bin/env python3
"""
Redrob Intelligent Candidate Discovery & Ranking - entry point.

Single command, CPU-only, no network, well under the 5-minute / 16 GB budget:

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Pipeline (two streaming passes over the JSONL, never loads all 100K into RAM):
  Pass 1: accumulate document-frequency over career prose -> corpus IDF.
  Pass 2: per candidate, compute semantic cosine to the JD concept-query +
          structured rubric + disqualifier gates + behavioral/location
          multipliers + honeypot sinking; keep a bounded top-K heap.
  Output: top-100 CSV (candidate_id, rank, score, reasoning), spec-compliant.
"""
import argparse
import csv
import heapq
import json
import sys
import time
from collections import Counter

from redrob_ranker import text_model as tm
from redrob_ranker import features as feat
from redrob_ranker import scoring
from redrob_ranker import reasoning as rsn

MIN_DF = 4          # drop ultra-rare tokens (noise / typos)
MAX_DF_FRAC = 0.5   # drop tokens appearing in >50% of docs (non-discriminative)
TOP_K = 250         # bounded heap; we only ever need the best ~few hundred


def iter_candidates(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def pass1_idf(path):
    df = Counter()
    n = 0
    for c in iter_candidates(path):
        n += 1
        toks = set(tm.tokenize(feat.career_prose(c)))
        df.update(toks)
    # prune vocabulary
    hi = MAX_DF_FRAC * n
    kept = {t: d for t, d in df.items() if MIN_DF <= d <= hi}
    idf = tm.build_idf(kept, n)
    return idf, n


def pass2_rank(path, idf):
    qvec, qnorm = tm.query_vector(idf)
    heap = []  # (final_score, candidate_id, payload)
    seq = 0
    for c in iter_candidates(path):
        f = feat.extract(c)
        sem = tm.doc_cosine(tm.tokenize(f["prose"]), idf, qvec, qnorm)
        final, detail = scoring.score_candidate(f, sem)
        seq += 1
        item = (final, -seq, c["candidate_id"], f, detail)
        if len(heap) < TOP_K:
            heapq.heappush(heap, item)
        elif final > heap[0][0]:
            heapq.heapreplace(heap, item)
    # sort: score desc, then candidate_id asc (spec tie-break)
    ranked = sorted(heap, key=lambda x: (-x[0], x[2]))
    return ranked


def write_submission(ranked, out_path):
    top = ranked[:100]
    rows = []
    prev = None
    for i, (final, _seq, cid, f, detail) in enumerate(top):
        rank = i + 1
        score = round(final, 6)
        # enforce strictly non-increasing scores (spec rule); ties allowed but
        # we nudge down by an epsilon only if a rounding artifact inverts order
        if prev is not None and score > prev:
            score = prev
        prev = score
        reasoning = rsn.make_reasoning(f, detail, rank)
        rows.append((cid, rank, f"{score:.6f}", reasoning))
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--debug", action="store_true",
                    help="print top-20 with score breakdown to stderr")
    args = ap.parse_args()

    t0 = time.time()
    idf, n = pass1_idf(args.candidates)
    print(f"[pass1] {n} candidates, vocab={len(idf)}  ({time.time()-t0:.1f}s)",
          file=sys.stderr)
    ranked = pass2_rank(args.candidates, idf)
    print(f"[pass2] scored, top-{len(ranked)} kept  ({time.time()-t0:.1f}s)",
          file=sys.stderr)
    rows = write_submission(ranked, args.out)
    print(f"[done] wrote {args.out} ({len(rows)} rows) in {time.time()-t0:.1f}s",
          file=sys.stderr)

    if args.debug:
        for (final, _s, cid, f, d) in ranked[:20]:
            print(f"  {final:.4f} {cid} {f['title'][:34]:34s} "
                  f"yoe={f['yoe']:.1f} dom={f['domain_hits']} prod={f['prod_hits']} "
                  f"sem={d['semantic']:.2f} gate={d['gate']:.2f} beh={d['behavioral']:.2f} "
                  f"loc={f['location'][:18]}", file=sys.stderr)


if __name__ == "__main__":
    main()
