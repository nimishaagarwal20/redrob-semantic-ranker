# Redrob — Intelligent Candidate Discovery & Ranking

Ranks the top 100 candidates from a 100,000-candidate pool for the **Senior AI
Engineer (Founding Team)** job description — the way a great recruiter would, by
reasoning about *what people actually did*, not by counting AI keywords.

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

- **CPU-only, no network, no GPU.** The ranking step imports only the Python
  standard library. Nothing is downloaded; no LLM/API is called.
- **~2 minutes for 100K candidates** on a 16 GB CPU machine (two streaming
  passes; never loads the full pool into RAM).
- Output is a spec-compliant CSV: `candidate_id,rank,score,reasoning`, exactly
  100 rows, scores non-increasing, deterministic `candidate_id`-ascending
  tie-break.

---

## The core idea

The JD says the trap is *"find candidates whose skills section contains the most
AI keywords."* A great recruiter ignores the keyword list and reads the career
story. So our score is built from two things and then **modulated by reality**:

```
final = (0.45 * semantic_prose_match  +  0.55 * structured_rubric)
        * disqualifier_gate      # the JD's explicit "do NOT want" rules
        * behavioral_multiplier  # is this person actually reachable / available?
        * location_factor        # India Tier-1 / willing to relocate
        * honeypot_factor        # sink subtly-impossible profiles
        * experience_gate        # soft 5–9y preference (a range, not a cutoff)
```

Every term is in `[0,1]` (multipliers hover around 1), so the final number is
interpretable and each piece can be read straight into the per-candidate
reasoning string.

### 1. Semantic match over *prose*, not tags
`text_model.py` builds a TF-IDF model over each candidate's **summary + job
descriptions** (never the `skills` list) and scores cosine similarity to a JD
*concept* query (retrieval, ranking, recommendation, embeddings, vector search,
evaluation/NDCG, production/scale, NLP/IR). This is what lets a "plain-language
Tier-5" — someone who built a recommendation system but never wrote "RAG" —
surface, while a keyword-stuffer's tag soup contributes nothing here.

### 2. Structured rubric (`scoring.rubric_score`)
Weighted blend of: **domain evidence** in the prose (0.30), **role-fit** (0.26 —
is this a real ML/AI/IR engineering role, or a Marketing Manager with AI tags?),
**production** signals (0.14), **evaluation** literacy (0.10), **trust-weighted
skills** (0.10), and **experience** shape (0.10). The skill-trust term weights
each core skill by `proficiency × √duration × endorsements`, so an "expert"
skill with 0 months and 0 endorsements counts as padding (~0).

### 3. Disqualifier gates (`scoring.disqualifier_gate`)
Direct from the JD's "Things we explicitly do NOT want": pure research with no
production, entire-career-at-consulting (TCS/Infosys/…/Genpact) with no product
company, CV/speech/robotics-only without NLP/IR, recent-LangChain-only AI
experience, senior-who-no-longer-codes, and title-chasers (many <18-month
stints). Each is a multiplicative penalty, not a hard delete.

### 4. Behavioral availability (`scoring.behavioral_multiplier`)
The JD is explicit: *"a perfect-on-paper candidate who hasn't logged in for 6
months and has a 5% response rate is, for hiring purposes, not actually
available. Down-weight them."* We fold in activity recency, recruiter response
rate, open-to-work, interview-completion, and notice period.

### 5. Honeypots (`features.honeypot_flags`)
~80 candidates have *subtly impossible* profiles (forced to tier 0 in the ground
truth; >10% in your top-100 = disqualification). We don't special-case them — we
read the profile for **internal consistency**: total role tenure exceeding stated
years of experience, `duration_months` contradicting the role's own start/end
dates, a single role longer than the whole career, and clusters of "expert"
skills with zero usage and zero endorsements. Flagged profiles are multiplied by
0.02 and sink. (Validated: **0 honeypots and 0 non-engineering titles in the
top 100.**)

### 6. Reasoning (`reasoning.py`)
Each row gets a 1–2 sentence justification built *only* from facts in that
candidate's record — specialization, production/eval evidence, an engagement note
when the signals are strong, and the genuine concern (location, notice period,
experience band, availability) that affected the rank. No hallucinated skills; no
name-insertion template; tone tracks the rank.

---

## Repository layout

```
rank.py                      # single-command entry point (stdlib only)
redrob_ranker/
  text_model.py              # TF-IDF over career prose + JD concept query
  features.py                # structured feature + honeypot extraction
  scoring.py                 # rubric, gates, multipliers, final score
  reasoning.py               # specific, non-templated reasoning strings
scripts/                     # OPTIONAL helpers to run within a tiny sandbox
  _local_idf.py              #   (split IDF + chunk scoring + merge);
  _local_chunk.py            #   produce identical output to rank.py.
  _local_merge.py
submission.csv               # the top-100 ranking for the released JD
submission_metadata.yaml
requirements.txt
```

## Reproduce

```bash
# One command, ~2 min on CPU, 16 GB, no network:
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Validate against the official format checker:
python validate_submission.py submission.csv
```

`rank.py` makes two streaming passes: pass 1 builds corpus IDF; pass 2 scores
each candidate and keeps a bounded top-K heap. Peak RAM is well under 1 GB.

### Running on a very small machine
If you can't run the full pool in one shot, `scripts/` splits the same
computation across processes (build IDF once → score line-ranges → merge top-100).
The result is byte-for-byte equivalent to the single-command path.

## Sandbox
A Google Colab notebook reproduces the ranker on a ≤100-candidate sample
end-to-end within the compute budget (link in `submission_metadata.yaml`).
