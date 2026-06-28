"""
Structured feature extraction from a candidate record.

Everything a great recruiter reads "between the lines" of the JD is turned into
explicit, inspectable features here: what the person actually DID (career prose),
whether the role is a real engineering role, whether the AI signal is trustworthy
or just keyword padding, availability, and profile *consistency* (honeypots).
"""
import datetime
import re

TODAY = datetime.date(2026, 6, 1)

STRONG_TITLE = re.compile(
    r"(machine learning|ml engineer|ml scientist|ai engineer|applied scientist|"
    r"applied ml|nlp|search engineer|recommendation|recommender|relevance|"
    r"information retrieval|data scientist|research engineer|deep learning)", re.I)
OK_TITLE = re.compile(
    r"(software engineer|backend|back-end|data engineer|analytics engineer|"
    r"full stack|platform engineer|cloud engineer|devops|sde|staff engineer|"
    r"principal engineer|developer)", re.I)
NONENG_TITLE = re.compile(
    r"(hr |human resource|recruiter|marketing|sales|account(ant|s)?|content writer|"
    r"copywriter|customer support|operations manager|project manager|program manager|"
    r"business analyst|graphic designer|ui/ux designer|civil engineer|"
    r"mechanical engineer|electrical engineer|product manager|finance|"
    r"supply chain|logistics)", re.I)

CONSULTING = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "hcltech", "ltimindtree", "mindtree",
    "mphasis", "deloitte", "ibm services", "dxc", "genpact",
}

DOMAIN_RE = re.compile(
    r"\b(retrieval|ranking|re-?rank|relevance|recommendation|recommender|recsys|"
    r"semantic search|search relevance|vector (?:search|database|db|index)|"
    r"embedding|sentence-?transformer|bm25|elasticsearch|opensearch|faiss|"
    r"pinecone|weaviate|qdrant|milvus|learning.to.rank|ndcg|mrr|"
    r"information retrieval|nearest neighbor|two-tower|matching|personaliz)", re.I)
PROD_RE = re.compile(
    r"\b(production|deployed|deploy|real users|at scale|latency|serving|served|"
    r"throughput|requests per|qps|traffic|a/b test|ab test|rollout|shipped|"
    r"millions of|users|live)", re.I)
EVAL_RE = re.compile(
    r"\b(ndcg|mrr|\bmap\b|precision|recall|offline (?:eval|metric)|online (?:eval|metric)|"
    r"a/b test|ab test|experiment|benchmark|evaluation framework|offline-to-online)", re.I)
RESEARCH_RE = re.compile(
    r"\b(phd|ph\.d|publication|published|paper|neurips|icml|acl|emnlp|cvpr|"
    r"research lab|academic|thesis|postdoc|professor)", re.I)
CVSPEECH_RE = re.compile(
    r"\b(computer vision|image classif|object detection|segmentation|ocr|"
    r"speech recognition|asr|tts|text-to-speech|robotics|slam|lidar|"
    r"point cloud|gans?|image generation|pose estimation)", re.I)
NLP_IR_RE = re.compile(
    r"\b(nlp|natural language|retrieval|ranking|search|recommendation|"
    r"information retrieval|text|language model|embedding|bert|transformer)", re.I)
LANGCHAIN_RE = re.compile(r"\b(langchain|llamaindex|openai api|gpt-?4|prompt engineering|rag demo)", re.I)
LEAD_ONLY_RE = re.compile(r"\b(architect|tech lead|engineering manager|head of|director|vp )", re.I)
CODE_RE = re.compile(r"\b(python|java|c\+\+|golang|scala|wrote|implemented|coded|built|developed|hands-on)", re.I)

CORE_SKILLS = {
    "embeddings", "vector search", "vector databases", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch", "bm25",
    "semantic search", "retrieval", "information retrieval", "ranking",
    "learning to rank", "recommender systems", "recommendation systems",
    "nlp", "transformers", "sentence transformers", "fine-tuning llms", "lora",
    "qlora", "peft", "pytorch", "tensorflow", "xgboost", "python", "spark",
    "rag", "llm", "ndcg", "sentence-transformers", "hugging face", "bert",
}
PROF_W = {"beginner": 0.4, "intermediate": 0.7, "advanced": 1.0, "expert": 1.15}

INDIA_T1 = re.compile(
    r"(pune|noida|hyderabad|mumbai|delhi|gurgaon|gurugram|bangalore|bengaluru|"
    r"new delhi|ncr|chennai)", re.I)


def _date(s):
    if not s:
        return None
    try:
        return datetime.date.fromisoformat(s)
    except Exception:
        return None


def career_prose(c):
    parts = []
    p = c.get("profile", {})
    parts.append(p.get("summary") or "")
    parts.append(p.get("headline") or "")
    for j in c.get("career_history", []):
        parts.append(j.get("title") or "")
        parts.append(j.get("description") or "")
    return " ".join(parts)


def honeypot_flags(c):
    """Detect 'subtly impossible' profiles (forced to tier 0 in ground truth)."""
    flags = []
    p = c.get("profile", {})
    yoe = p.get("years_of_experience") or 0
    ch = c.get("career_history", [])
    sum_months = sum((j.get("duration_months") or 0) for j in ch)
    if sum_months / 12.0 > yoe + 3.5:
        flags.append("tenure_exceeds_yoe")
    for j in ch:
        sd = _date(j.get("start_date"))
        ed = _date(j.get("end_date")) or TODAY
        dur = j.get("duration_months") or 0
        if sd:
            span = (ed.year - sd.year) * 12 + (ed.month - sd.month)
            if abs(span - dur) > 18:
                flags.append("date_duration_mismatch")
                break
    sk = c.get("skills", [])
    ghost_expert = sum(
        1 for s in sk
        if s.get("proficiency") in ("advanced", "expert")
        and (s.get("duration_months") or 0) == 0
        and (s.get("endorsements") or 0) == 0)
    if ghost_expert >= 4:
        flags.append("ghost_expert_skills")
    if ch:
        longest = max((j.get("duration_months") or 0) for j in ch)
        if longest / 12.0 > yoe + 2.5:
            flags.append("single_role_exceeds_career")
    return flags


def extract(c):
    """Return a dict of structured features used by scoring + reasoning."""
    p = c.get("profile", {})
    sig = c.get("redrob_signals", {})
    ch = c.get("career_history", [])
    title = p.get("current_title") or ""
    prose = career_prose(c)

    domain_hits = len(DOMAIN_RE.findall(prose))
    prod_hits = len(PROD_RE.findall(prose))
    eval_hits = len(EVAL_RE.findall(prose))
    research_hits = len(RESEARCH_RE.findall(prose))
    cv_hits = len(CVSPEECH_RE.findall(prose))
    nlp_hits = len(NLP_IR_RE.findall(prose))

    companies = [(j.get("company") or "").lower() for j in ch]
    industries = [(j.get("industry") or "").lower() for j in ch] + [(p.get("current_industry") or "").lower()]
    consulting_only = bool(companies) and all(
        any(k in comp for k in CONSULTING) for comp in companies)
    ever_product = any(
        ind in ("software", "fintech", "e-commerce", "food delivery", "internet",
                "ai/ml", "saas", "edtech", "adtech", "media", "healthtech ai",
                "transportation", "ai services", "insurance tech")
        for ind in industries)

    skill_trust = 0.0
    core_skill_names = []
    for s in c.get("skills", []):
        name = (s.get("name") or "").lower()
        if any(cs in name or name in cs for cs in CORE_SKILLS):
            dur = s.get("duration_months") or 0
            end = s.get("endorsements") or 0
            prof = PROF_W.get(s.get("proficiency"), 0.5)
            trust = prof * (1.0 + (dur / 12.0) ** 0.5) * (1.0 + min(end, 50) / 25.0)
            if dur == 0 and end == 0:
                trust *= 0.1
            skill_trust += trust
            if dur > 0 or end > 0:
                core_skill_names.append(s.get("name"))

    assess = sig.get("skill_assessment_scores") or {}
    assess_avg = (sum(assess.values()) / len(assess)) if assess else None

    short_stints = sum(
        1 for j in ch
        if not j.get("is_current") and 0 < (j.get("duration_months") or 0) < 18)
    n_past = sum(1 for j in ch if not j.get("is_current"))

    cur = next((j for j in ch if j.get("is_current")), None)
    cur_desc = (cur or {}).get("description", "") if cur else ""
    lead_only_current = bool(LEAD_ONLY_RE.search(title)) and not CODE_RE.search(cur_desc)

    last_active = _date(sig.get("last_active_date"))
    days_inactive = (TODAY - last_active).days if last_active else 999

    return {
        "candidate_id": c.get("candidate_id"),
        "name": p.get("anonymized_name"),
        "title": title,
        "yoe": p.get("years_of_experience") or 0,
        "company": p.get("current_company"),
        "industry": p.get("current_industry"),
        "location": p.get("location") or "",
        "country": p.get("country") or "",
        "prose": prose,
        "domain_hits": domain_hits,
        "prod_hits": prod_hits,
        "eval_hits": eval_hits,
        "research_hits": research_hits,
        "cv_hits": cv_hits,
        "nlp_hits": nlp_hits,
        "strong_title": bool(STRONG_TITLE.search(title)),
        "ok_title": bool(OK_TITLE.search(title)),
        "noneng_title": bool(NONENG_TITLE.search(title)) and not STRONG_TITLE.search(title) and not OK_TITLE.search(title),
        "consulting_only": consulting_only,
        "ever_product": ever_product,
        "skill_trust": skill_trust,
        "core_skill_names": core_skill_names[:6],
        "assess_avg": assess_avg,
        "short_stints": short_stints,
        "n_past": n_past,
        "lead_only_current": lead_only_current,
        "langchain_only": bool(LANGCHAIN_RE.search(prose)) and research_hits == 0 and domain_hits <= 1,
        "india_t1": bool(INDIA_T1.search(p.get("location") or "")),
        "in_india": (p.get("country") or "") == "India",
        "days_inactive": days_inactive,
        "honeypot": honeypot_flags(c),
        "sig": sig,
        "sig_summary": {
            "recruiter_response_rate": sig.get("recruiter_response_rate"),
            "open_to_work_flag": sig.get("open_to_work_flag"),
            "notice_period_days": sig.get("notice_period_days"),
        },
    }
