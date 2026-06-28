"""
Lightweight TF-IDF semantic model over candidate *career prose*.

Why career prose (summary + job descriptions) and NOT the skills list?
The JD is explicit: "find candidates whose skills section contains the most AI
keywords" is a trap. Real fit shows up in what a person *did* ("built a
recommendation system", "owned search relevance"), often in plain language that
never names "RAG" or "Pinecone". So our dense-ish signal is computed over the
narrative text, which is where plain-language Tier-5 fits live.

Pure-Python + numpy only (no sklearn/scipy) so the ranking step has a tiny,
fully reproducible dependency footprint and runs CPU-only with no network.
"""
import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z][a-z0-9\+\#\.\-]{1,}")

# Light stoplist - generic resume / english filler that carries no fit signal.
_STOP = set("""
a an the and or of to in on for with at by from as is are was were be been being
i we you he she it they them our your their this that these those my me us
have has had do does did will would can could should may might must not no nor
worked working work built build building using used use made make team teams
across into over under within most some many few more less very much also able
year years month months experience role roles responsible responsibility
company companies project projects help helped helping including include various
""".split())


def tokenize(text):
    if not text:
        return []
    toks = _TOKEN_RE.findall(text.lower())
    return [t for t in toks if t not in _STOP and len(t) > 1]


# ---- The JD "query": the substantive concepts the role is really about. ----
# Weights reflect the JD's own emphasis ("things you absolutely need" first).
# These are CONCEPTS, deliberately phrased the way a candidate might describe
# real work, so we reward narrative evidence over keyword tags.
JD_QUERY_WEIGHTS = {
    # Core: retrieval / ranking / search relevance (the heart of the role)
    "retrieval": 3.0, "ranking": 3.0, "rank": 2.0, "ranker": 2.5, "re-ranking": 2.5,
    "rerank": 2.5, "relevance": 2.5, "search": 2.5, "recommendation": 3.0,
    "recommender": 2.5, "recsys": 2.5, "matching": 2.0, "personalization": 2.0,
    # Embeddings / vector infra
    "embedding": 3.0, "embeddings": 3.0, "vector": 2.5, "semantic": 2.0,
    "faiss": 2.0, "pinecone": 2.0, "weaviate": 2.0, "qdrant": 2.0, "milvus": 2.0,
    "opensearch": 2.0, "elasticsearch": 2.0, "bm25": 2.5, "ann": 1.5, "knn": 1.5,
    "two-tower": 2.0, "sentence-transformers": 2.0, "bge": 1.5, "e5": 1.0,
    # Evaluation (a hard requirement in the JD)
    "ndcg": 3.0, "mrr": 2.5, "map": 1.5, "evaluation": 2.0, "eval": 1.5,
    "offline": 1.5, "online": 1.5, "experiment": 1.5, "metrics": 1.2,
    # Production / scale (JD wants shippers, not researchers)
    "production": 2.0, "deployed": 1.8, "latency": 1.8, "serving": 1.8,
    "scale": 1.5, "throughput": 1.5, "users": 1.2, "traffic": 1.5, "pipeline": 1.0,
    # ML / NLP / IR foundations
    "nlp": 2.5, "ir": 1.5, "transformer": 1.8, "transformers": 1.8, "llm": 1.8,
    "fine-tuning": 1.8, "fine-tune": 1.8, "lora": 1.5, "qlora": 1.5, "peft": 1.5,
    "learning-to-rank": 2.5, "xgboost": 1.5, "machine": 1.0, "learning": 1.0,
    "model": 1.0, "models": 1.0, "python": 1.2,
    # Light HR-tech / marketplace bonus terms
    "marketplace": 1.2, "candidate": 0.8, "recruiter": 0.8, "hiring": 0.6,
}


def build_idf(doc_freq, n_docs):
    """Smoothed idf from a {term: document_frequency} map."""
    idf = {}
    for term, df in doc_freq.items():
        idf[term] = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
    return idf


def query_vector(idf):
    """idf-weighted JD concept vector, restricted to terms seen in the corpus."""
    q = {}
    for term, w in JD_QUERY_WEIGHTS.items():
        if term in idf:
            q[term] = w * idf[term]
    norm = math.sqrt(sum(v * v for v in q.values())) or 1.0
    return q, norm


def doc_cosine(tokens, idf, qvec, qnorm):
    """Cosine similarity between a document's tf-idf vector and the JD query."""
    if not tokens:
        return 0.0
    tf = Counter(tokens)
    # sublinear tf scaling damps repetition / keyword spamming
    dot = 0.0
    sq = 0.0
    for term, c in tf.items():
        w = (1.0 + math.log(c)) * idf.get(term, 0.0)
        sq += w * w
        if term in qvec:
            dot += w * qvec[term]
    dnorm = math.sqrt(sq) or 1.0
    return dot / (dnorm * qnorm)
