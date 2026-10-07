"""
SOP Engine — Policy matching against weather data.

Loads SOPs from the JSON file, evaluates each SOP's conditions
against the actual weather data, and returns a list of matched SOPs.
"""

from langchain_core.outputs import chat_generation
import json
from pathlib import Path
from datetime import datetime

SEVERITY_RANK = {"low": 1, "moderate": 2, "high": 3, "critical": 4}


def load_sops(sop_file: str = None) -> list[dict]:
    """Load SOPs from JSON file."""
    if sop_file is None:
        sop_file = Path(__file__).parent / "sops" / "sops.json"
    with open(sop_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["sops"]


# ─────────────────────────────────────────────────────────────
# Condition evaluators — one per condition type
# ─────────────────────────────────────────────────────────────

def _compare(actual, operator: str, threshold) -> bool:
    """Apply a comparison operator."""
    ops = {
        ">=": lambda a, b: a >= b,
        ">":  lambda a, b: a > b,
        "<=": lambda a, b: a <= b,
        "<":  lambda a, b: a < b,
        "==": lambda a, b: a == b,
    }
    fn = ops.get(operator)
    return fn(actual, threshold) if fn else False

#Used by: SOP-001 (extreme heat), SOP-003 (high wind), SOP-006 (elderly/child heat)

def _eval_threshold(rule: dict, weather: dict) -> bool:
    """Compare a current weather field against a threshold value."""
    current = weather.get("current", {})
    value = current.get(rule["field"])
    if value is None:
        return False
    return _compare(value, rule["operator"], rule["value"])

#Used by: SOP-009 (thunderstorm), SOP-010 (fog), SOP-004 (heavy rain)

def _eval_weather_code_in(rule: dict, weather: dict) -> bool:
    """Check if current weather code is in a set of WMO codes."""
    current = weather.get("current", {})
    code = current.get("weather_code")
    if code is None:
        return False
    return code in rule["values"]

#Used by: SOP-002 (UV peak hours)

def _eval_hourly_any(rule: dict, weather: dict) -> bool:
    """True if ANY hour in the optional [start_hour, end_hour] window meets the threshold."""
    hourly = weather.get("hourly", {})
    values = hourly.get(rule["field"], [])
    times = hourly.get("time", [])
    start_hour = rule.get("start_hour", 0)
    end_hour = rule.get("end_hour", 23)

    for t, v in zip(times, values):
        if v is None:
            continue
        try:
            hour = datetime.fromisoformat(t).hour
        except (ValueError, TypeError):
            continue
        if start_hour <= hour <= end_hour:
            if _compare(v, rule["operator"], rule["value"]):
                return True
    return False

#Used by: SOP-012 (sustained heavy rainfall)

def _eval_hourly_consecutive(rule: dict, weather: dict) -> bool:
    """True if at least min_hours consecutive hours meet the threshold."""
    hourly = weather.get("hourly", {})
    values = hourly.get(rule["field"], [])
    min_hours = rule["min_hours"]

    consecutive = 0
    for v in values:
        if v is not None and _compare(v, rule["operator"], rule["value"]):
            consecutive += 1
            if consecutive >= min_hours:
                return True
        else:
            consecutive = 0
    return False

#Used by: SOP-012 (total daily rainfall)

def _eval_hourly_total(rule: dict, weather: dict) -> bool:
    """True if the SUM of a field across all hours meets the threshold."""
    hourly = weather.get("hourly", {})
    values = hourly.get(rule["field"], [])
    total = sum(v for v in values if v is not None)
    return _compare(total, rule["operator"], rule["value"])


# Registry of evaluators
_EVALUATORS = {
    "threshold": _eval_threshold,
    "weather_code_in": _eval_weather_code_in,
    "hourly_any": _eval_hourly_any,
    "hourly_consecutive": _eval_hourly_consecutive,
    "hourly_total": _eval_hourly_total,
}


def evaluate_condition(rule: dict, weather: dict) -> bool:
    """Evaluate a single condition rule against weather data."""
    evaluator = _EVALUATORS.get(rule["type"])
    if evaluator is None:
        return False
    return evaluator(rule, weather)


# ─────────────────────────────────────────────────────────────
# Composite / fuzzy evaluation
# ─────────────────────────────────────────────────────────────

#Used for SOP-008 (Picnic Suitability).

def evaluate_composite(sop: dict, weather: dict) -> dict | None:
    """
    Evaluate a composite (fuzzy) SOP that uses multi-factor scoring.
    """
    conditions = sop["conditions"]
    composite_rule = None
    for rule in conditions["rules"]:
        if rule["type"] == "composite":
            composite_rule = rule
            break

    if composite_rule is None:
        return None

    current = weather.get("current", {})
    total_score = 0.0
    total_weight = 0.0
    factor_details = []

    for factor in composite_rule["factors"]:
        field = factor["field"]
        value = current.get(field)
        if value is None:
            continue

        weight = factor["weight"]
        scoring = factor["scoring"]

        if scoring == "lower_better":
            good_below = factor["good_below"]
            bad_above = factor["bad_above"]
            if value <= good_below:
                score = 1.0
            elif value >= bad_above:
                score = 0.0
            else:
                score = 1.0 - (value - good_below) / (bad_above - good_below)

        elif scoring == "range":
            good_min = factor["good_min"]
            good_max = factor["good_max"]
            bad_below = factor["bad_below"]
            bad_above = factor["bad_above"]
            if good_min <= value <= good_max:
                score = 1.0
            elif value < bad_below or value > bad_above:
                score = 0.0
            elif value < good_min:
                score = (value - bad_below) / (good_min - bad_below)
            else:  # value > good_max
                score = 1.0 - (value - good_max) / (bad_above - good_max)
        else:
            score = 0.5  # Fallback for unknown scoring type

        total_score += score * weight
        total_weight += weight
        factor_details.append({
            "field": field,
            "value": value,
            "score": round(score, 2),
        })

    if total_weight == 0:
        return None

    final_score = total_score / total_weight
    thresholds = composite_rule["thresholds"]

    if final_score >= thresholds["good_above"]:
        tier = "good"
        advice = sop.get("advice_good", sop.get("advice", ""))
    elif final_score >= thresholds["caution_above"]:
        tier = "caution"
        advice = sop.get("advice_caution", sop.get("advice", ""))
    else:
        tier = "not_recommended"
        advice = sop.get("advice_bad", sop.get("advice", ""))

    severity_map = {"good": "low", "caution": "moderate", "not_recommended": "high"}

    return {
        "sop_id": sop["id"],
        "sop_name": sop["name"],
        "category": sop["category"],
        "severity": severity_map[tier],
        "severity_rank": SEVERITY_RANK[severity_map[tier]],
        "tier": tier,
        "composite_score": round(final_score, 2),
        "advice": advice,
        "factor_details": factor_details,
    }


# ─────────────────────────────────────────────────────────────
# Activity & vulnerable-group relevance matching
# ─────────────────────────────────────────────────────────────

def _is_relevant(sop: dict, activities: list[str], vulnerable_groups: list[str]) -> bool:
    """
    Check if an SOP is relevant given the user's activities and vulnerable groups.
    """
    sop_acts = sop.get("applies_to_activities", [])
    sop_vuln = sop.get("applies_to_vulnerable_groups", [])

    # Universal SOP — no filters
    if not sop_acts and not sop_vuln:
        return True

    # Check activity match (keyword containment, case-insensitive)
    act_match = False
    if sop_acts and activities:
        for user_act in activities:
            for sop_keyword in sop_acts:
                if (sop_keyword.lower() in user_act.lower() or
                        user_act.lower() in sop_keyword.lower()):
                    act_match = True
                    break
            if act_match:
                break

    # Check vulnerable group match
    vuln_match = False
    if sop_vuln and vulnerable_groups:
        for user_vg in vulnerable_groups:
            for sop_keyword in sop_vuln:
                if (sop_keyword.lower() in user_vg.lower() or
                        user_vg.lower() in sop_keyword.lower()):
                    vuln_match = True
                    break
            if vuln_match:
                break

    # Apply relevance logic
    if sop_acts and sop_vuln:
        return act_match or vuln_match  # Either filter passing is enough
    elif sop_acts:
        return act_match
    elif sop_vuln:
        return vuln_match

    return True  # Should not reach here


# ─────────────────────────────────────────────────────────────
# Main matching function
# ─────────────────────────────────────────────────────────────

def match_sops(
    weather: dict,
    activities: list[str],
    vulnerable_groups: list[str],
    sop_file: str = None,
) -> list[dict]:
    """
    Match all applicable SOPs against weather data and user context.
    Returns a list of matched SOP dicts, ranked by severity (highest first).
    """
    sops = load_sops(sop_file)
    matched = []

    for sop in sops:
        # Step 1: Is this SOP relevant to the user's activity/context?
        if not _is_relevant(sop, activities, vulnerable_groups):
            continue

        # Step 2: Does it have a composite condition? (Special handling)
        has_composite = any(
            r.get("type") == "composite"
            for r in sop.get("conditions", {}).get("rules", [])
        )

        if has_composite:
            result = evaluate_composite(sop, weather)
            if result:
                matched.append(result)
            continue

        # Step 3: Evaluate standard conditions
        logic = sop["conditions"]["logic"]
        rules = sop["conditions"]["rules"]

        if logic == "OR":
            triggered = any(evaluate_condition(r, weather) for r in rules)
        elif logic == "AND":
            triggered = all(evaluate_condition(r, weather) for r in rules)
        else:
            triggered = False

        if triggered:
            matched.append({
                "sop_id": sop["id"],
                "sop_name": sop["name"],
                "category": sop["category"],
                "severity": sop["severity"],
                "severity_rank": SEVERITY_RANK.get(sop["severity"], 0),
                "advice": sop["advice"],
            })

    # Sort by severity rank, highest first (critical → high → moderate → low)
    matched.sort(key=lambda x: x.get("severity_rank", 0), reverse=True)

    return matched
