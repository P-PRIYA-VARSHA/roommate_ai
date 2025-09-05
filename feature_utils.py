
import math

SLEEP_COMPAT = {
    ("early","early"): 1.0,
    ("night","night"): 1.0,
    ("irregular","irregular"): 1.0,
    ("early","night"): 0.0,
    ("night","early"): 0.0,
    ("early","irregular"): 0.5,
    ("irregular","early"): 0.5,
    ("night","irregular"): 0.5,
    ("irregular","night"): 0.5,
}

def _to_set(hobbies_str: str):
    if hobbies_str is None:
        return set()
    if isinstance(hobbies_str, list):
        return set([h.strip().lower() for h in hobbies_str if h and str(h).strip()])
    return set([h.strip().lower() for h in str(hobbies_str).split(",") if h and str(h).strip()])

def jaccard(a: set, b: set) -> float:
    if not a and not b: 
        return 0.0
    return len(a & b) / len(a | b)

def budget_similarity(b1, b2) -> float:
    """Returns similarity in [0,1] based on closeness of budgets (single number or 'low-high')."""
    def _parse(b):
        if b is None:
            return None, None, None
        if isinstance(b, (int,float)):
            return float(b), None, None
        s = str(b).replace("₹","").replace("INR","").strip()
        if "-" in s:
            lo, hi = s.split("-", 1)
            try:
                lo = float(lo.strip())
                hi = float(hi.strip())
                return (lo+hi)/2.0, lo, hi
            except:
                pass
        try:
            return float(s), None, None
        except:
            return None, None, None

    m1, lo1, hi1 = _parse(b1)
    m2, lo2, hi2 = _parse(b2)
    if m1 is None or m2 is None:
        return 0.0
    # similarity by relative difference
    denom = max(m1, m2, 1.0)
    sim = max(0.0, 1.0 - abs(m1 - m2)/denom)
    return sim

def sleeping_compat(s1: str, s2: str) -> float:
    s1 = (s1 or "").strip().lower()
    s2 = (s2 or "").strip().lower()
    return SLEEP_COMPAT.get((s1,s2), 0.0)

def compute_pair_features(u1: dict, u2: dict) -> list:
    """
    Returns a fixed-length feature vector for a pair of users.
    All features are in [0,1] where possible.
    """
    # Cleanliness similarity (1-5 scaled)
    c1 = float(u1.get("cleanlinessLevel") or u1.get("cleanliness") or 0.0)
    c2 = float(u2.get("cleanlinessLevel") or u2.get("cleanliness") or 0.0)
    c_sim = max(0.0, 1.0 - abs(c1 - c2)/4.0)  # 0 if far apart, 1 if same

    # Smoking/Drinking exact matches
    smoking_match = 1.0 if str(u1.get("smokingPreference") or u1.get("smoking") or "").strip().lower() \
                           == str(u2.get("smokingPreference") or u2.get("smoking") or "").strip().lower() else 0.0
    drinking_match = 1.0 if str(u1.get("drinkingPreference") or u1.get("drinking") or "").strip().lower() \
                            == str(u2.get("drinkingPreference") or u2.get("drinking") or "").strip().lower() else 0.0

    # Sleeping schedule soft compatibility
    sleep_sim = sleeping_compat(u1.get("sleepingSchedule") or u1.get("sleeping"),
                                u2.get("sleepingSchedule") or u2.get("sleeping"))

    # Hobbies similarity
    h1 = _to_set(u1.get("hobbies"))
    h2 = _to_set(u2.get("hobbies"))
    h_jacc = jaccard(h1, h2)
    h_overlap = (len(h1 & h2) / max(1, min(len(h1), len(h2)))) if (h1 and h2) else 0.0  # overlap over smaller set

    # Budget similarity
    b_sim = budget_similarity(u1.get("budgetRange") or u1.get("budget"),
                              u2.get("budgetRange") or u2.get("budget"))

    # Location match (simple exact string match for now)
    loc1 = str(u1.get("preferredLocation") or u1.get("location") or "").strip().lower()
    loc2 = str(u2.get("preferredLocation") or u2.get("location") or "").strip().lower()
    loc_match = 1.0 if loc1 and loc2 and loc1 == loc2 else 0.0

    # Additional stabilizers
    c_avg = max(0.0, min(1.0, (c1 + c2) / (2.0 * 5.0)))  # average cleanliness normalized
    # Budget ratio (min/max) to capture relative closeness
    def _mid(b):
        if isinstance(b, (int,float)):
            return float(b)
        s = str(b).replace("₹","").replace("INR","").strip()
        if "-" in s:
            lo, hi = s.split("-", 1)
            try:
                return (float(lo) + float(hi))/2.0
            except:
                return None
        try:
            return float(s)
        except:
            return None
    m1 = _mid(u1.get("budgetRange") or u1.get("budget"))
    m2 = _mid(u2.get("budgetRange") or u2.get("budget"))
    if (m1 is None) or (m2 is None) or (max(m1,m2)<=0):
        budget_ratio = 0.0
    else:
        budget_ratio = min(m1,m2)/max(m1,m2)

    return [
        c_sim,            # 0
        smoking_match,    # 1
        drinking_match,   # 2
        sleep_sim,        # 3
        h_jacc,           # 4
        b_sim,            # 5
        loc_match,        # 6
        h_overlap,        # 7
        c_avg,            # 8
        budget_ratio      # 9
    ]

def rule_based_score(u1: dict, u2: dict, weights=None) -> float:
    """A transparent teacher score in [0,100] used for bootstrapping if labels are absent."""
    if weights is None:
        weights = {
            "cleanliness": 0.2,
            "smoking": 0.15,
            "drinking": 0.15,
            "sleeping": 0.1,
            "hobbies": 0.15,
            "budget": 0.15,
            "location": 0.1
        }
    feats = compute_pair_features(u1,u2)
    c_sim, smoking, drinking, sleep, h_jacc, b_sim, loc, h_overlap, c_avg, budget_ratio = feats

    score = (
        weights["cleanliness"] * c_sim +
        weights["smoking"]     * smoking +
        weights["drinking"]    * drinking +
        weights["sleeping"]    * sleep +
        weights["hobbies"]     * h_jacc +
        weights["budget"]      * b_sim +
        weights["location"]    * loc
    )
    return round(100.0 * max(0.0, min(1.0, score)), 2)
