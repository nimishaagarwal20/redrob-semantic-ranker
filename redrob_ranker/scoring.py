"""
Scoring: turn features + semantic similarity into a single fit score.

Design = "what a great recruiter would weigh", made explicit and auditable:

  final = (0.45 * semantic + 0.55 * rubric)   # what they did + structured fit
          * disqualifier_gate                 # JD's hard 'do NOT want' rules
          * behavioral_multiplier             # are they actually reachable/available
          * location_factor                   # India Tier-1 / relocate preference
          * honeypot_factor                   # sink subtly-impossible profiles
          * experience_gate                   # soft 5-9y band preference

Each component is in [0,1] (multipliers slightly above/below 1) so the final
score stays interpretable and the pieces can be read off for the reasoning text.
"""
import math


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def rubric_score(f):
    """Structured fit in [0,1]. Career evidence is weighted most heavily."""
    if f["strong_title"]:
        role = 1.0
    elif f["ok_title"]:
        role = 0.62
    elif f["noneng_title"]:
        role = 0.04
    else:
        role = 0.30
    if f["noneng_title"] and f["domain_hits"] >= 4 and f["prod_hits"] >= 2:
        role = 0.35

    domain = 1.0 - math.exp(-f["domain_hits"] / 4.0)
    prod = 1.0 - math.exp(-f["prod_hits"] / 3.0)
    ev = 1.0 - math.exp(-f["eval_hits"] / 2.0)
    skill = 1.0 - math.exp(-f["skill_trust"] / 6.0)

    y = f["yoe"]
    if 6 <= y <= 8:
        exp = 1.0
    elif 5 <= y < 6 or 8 < y <= 9:
        exp = 0.9
    elif 4 <= y < 5 or 9 < y <= 11:
        exp = 0.72
    elif 3 <= y < 4 or 11 < y <= 13:
        exp = 0.5
    else:
        exp = 0.3

    r = (0.30 * domain + 0.26 * role + 0.14 * prod +
         0.10 * ev + 0.10 * skill + 0.10 * exp)
    return _clamp(r, 0.0, 1.0), {
        "role": role, "domain": domain, "prod": prod,
        "ev": ev, "skill": skill, "exp": exp}


def disqualifier_gate(f):
    """Multiplicative penalties for the JD's explicit 'do NOT want' list."""
    g = 1.0
    reasons = []
    if f["research_hits"] >= 2 and f["prod_hits"] == 0:
        g *= 0.30
        reasons.append("research-leaning with no clear production signal")
    if f["consulting_only"] and not f["ever_product"]:
        g *= 0.28
        reasons.append("entire career at IT-services/consulting firms")
    if f["cv_hits"] >= 3 and f["nlp_hits"] <= 1 and f["domain_hits"] <= 1:
        g *= 0.45
        reasons.append("CV/speech/robotics focus with little NLP/IR")
    if f["langchain_only"]:
        g *= 0.55
        reasons.append("AI experience appears to be recent LLM-API wrapping only")
    if f["lead_only_current"]:
        g *= 0.62
        reasons.append("current role looks lead/architecture-only (JD wants an IC who codes)")
    if f["n_past"] >= 3 and f["short_stints"] >= 3:
        g *= 0.78
        reasons.append("frequent short stints (title-chaser pattern)")
    return g, reasons


def behavioral_multiplier(f):
    """Availability / reachability multiplier in ~[0.45, 1.12]."""
    s = f["sig"]
    m = 1.0
    notes = []
    otw = s.get("open_to_work_flag")
    if otw is True:
        m *= 1.07
    elif otw is False:
        m *= 0.82
        notes.append("not marked open-to-work")
    di = f["days_inactive"]
    if di <= 30:
        m *= 1.05
    elif di <= 90:
        m *= 1.0
    elif di <= 180:
        m *= 0.84
        notes.append("last active ~%dd ago" % di)
    else:
        m *= 0.62
        notes.append("inactive ~%dd" % di)
    rr = s.get("recruiter_response_rate")
    if rr is not None:
        if rr < 0.15:
            m *= 0.70
            notes.append("low recruiter response rate (%.0f%%)" % (rr * 100))
        elif rr < 0.35:
            m *= 0.9
        elif rr >= 0.6:
            m *= 1.05
    ic = s.get("interview_completion_rate")
    if ic is not None and ic < 0.4:
        m *= 0.9
        notes.append("low interview completion rate")
    nd = s.get("notice_period_days")
    if nd is not None:
        if nd <= 30:
            m *= 1.03
        elif nd > 90:
            m *= 0.92
            notes.append("long notice period (%dd)" % nd)
    if s.get("verified_email") and s.get("verified_phone"):
        m *= 1.01
    return _clamp(m, 0.45, 1.12), notes


def location_factor(f):
    relocate = f["sig"].get("willing_to_relocate")
    if f["india_t1"]:
        return 1.05, []
    if f["in_india"]:
        if relocate:
            return 1.0, []
        return 0.95, ["in India but outside Tier-1 hubs"]
    if relocate:
        return 0.9, ["outside India but willing to relocate"]
    return 0.78, ["outside India, relocation not indicated"]


def honeypot_factor(f):
    if f["honeypot"]:
        return 0.02, f["honeypot"]
    return 1.0, []


def experience_gate(f):
    """Soft gate (not a hard cutoff) reflecting the JD's 5-9y 'range, not a
    requirement'."""
    y = f["yoe"]
    notes = []
    if 5 <= y <= 9:
        g = 1.0
    elif 4 <= y < 5 or 9 < y <= 10:
        g = 0.95
    elif 3 <= y < 4:
        g = 0.80
        notes.append("%.1fy experience is below the 5-9y target band" % y)
    elif 10 < y <= 13:
        g = 0.85
        notes.append("%.1fy experience is above the 5-9y target band" % y)
    elif y < 3:
        g = 0.6
        notes.append("only %.1fy experience" % y)
    else:
        g = 0.7
        notes.append("%.1fy experience well above target band" % y)
    return g, notes


def score_candidate(f, semantic):
    rub, parts = rubric_score(f)
    base = 0.45 * semantic + 0.55 * rub
    gate, dq_reasons = disqualifier_gate(f)
    beh, beh_notes = behavioral_multiplier(f)
    loc, loc_notes = location_factor(f)
    hp, hp_flags = honeypot_factor(f)
    exg, exp_notes = experience_gate(f)
    final = base * gate * beh * loc * hp * exg
    detail = {
        "semantic": semantic, "rubric": rub, "rubric_parts": parts,
        "base": base, "gate": gate, "dq_reasons": dq_reasons,
        "behavioral": beh, "beh_notes": beh_notes,
        "location": loc, "loc_notes": loc_notes,
        "honeypot_factor": hp, "honeypot_flags": hp_flags,
        "exp_gate": exg, "exp_notes": exp_notes,
        "final": final,
    }
    return final, detail
