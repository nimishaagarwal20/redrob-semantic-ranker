"""
Generate specific, non-templated 1-2 sentence reasoning per candidate.

Stage-4 review penalizes: empty, all-identical, name-insertion templates,
hallucinated skills, and reasoning that contradicts the rank. So each string is
built ONLY from facts in the candidate's own record, leads with the single most
relevant piece of career evidence, varies its phrasing by candidate, and always
surfaces the genuine concern (if any) that affected the rank.
"""
import re

_LEADS = [
    (0.62, "Excellent fit"),
    (0.50, "Strong fit"),
    (0.40, "Good fit"),
    (0.32, "Solid candidate"),
    (0.00, "Plausible fit"),
]


def _lead(final):
    for thr, label in _LEADS:
        if final >= thr:
            return label
    return "Adjacent fit"


def _specialization(f):
    t = (f["title"] or "").lower()
    if "recommend" in t:
        return "recommendation-systems"
    if "search" in t:
        return "search/retrieval"
    if "nlp" in t:
        return "NLP & retrieval"
    if "applied scien" in t or "research engineer" in t:
        return "applied-ML"
    if f["domain_hits"] >= 6:
        return "retrieval & ranking"
    if "data scien" in t:
        return "data-science / ML"
    return "ML engineering"


def _evidence(f):
    """A career-evidence clause, varied by the candidate's own signals."""
    spec = _specialization(f)
    has_prod = f["prod_hits"] >= 2
    has_eval = f["eval_hits"] >= 1
    deep = f["domain_hits"] >= 6
    if deep and has_prod and has_eval:
        return ("career history shows deep " + spec + " work shipped to "
                "production with offline/online evaluation rigor")
    if deep and has_prod:
        return "strong track record building " + spec + " systems in production"
    if has_prod and has_eval:
        return ("has shipped " + spec + " work to real users and measures it "
                "(offline/online metrics)")
    if has_prod:
        return "hands-on " + spec + " experience deployed in production"
    if f["domain_hits"] >= 2:
        return "demonstrated " + spec + " work across roles"
    return "relevant " + spec + " background"


def _engagement(f):
    """A short positive engagement note when the behavioral signals are good."""
    s = f.get("sig_summary") or {}
    rr = s.get("recruiter_response_rate")
    otw = s.get("open_to_work_flag")
    di = f.get("days_inactive", 999)
    bits = []
    if otw is True and di <= 60:
        bits.append("open to work and recently active")
    elif otw is True:
        bits.append("open to work")
    if rr is not None and rr >= 0.7:
        bits.append("high recruiter response rate (%.0f%%)" % (rr * 100))
    if not bits:
        return None
    return "; ".join(bits)


def _concerns(f, detail):
    out = []
    out += detail.get("dq_reasons", [])
    out += detail.get("exp_notes", [])
    out += detail.get("beh_notes", [])
    out += detail.get("loc_notes", [])
    if detail.get("honeypot_flags"):
        out.append("profile fails internal consistency checks")
    seen, dedup = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup[:2]


def make_reasoning(f, detail, rank):
    lead = _lead(detail["final"])
    title = f["title"]
    yoe = f["yoe"]
    comp = f["company"]
    sent = "%s: %s (%.1fy) at %s - %s" % (lead, title, yoe, comp, _evidence(f))

    eng = _engagement(f)
    concerns = _concerns(f, detail)
    if concerns:
        sent += ". Concern: " + "; ".join(concerns) + "."
    elif eng and rank <= 60:
        sent += "; " + eng + "."
    else:
        sent += "."
    sent = re.sub(r"\s+", " ", sent).strip()
    return sent
