import pytest

from evals.matching import contains_fact, facts_matched


@pytest.mark.parametrize("text,alts", [
    ("You get 16 days of annual leave.", ["16"]),
    ("The bonus is $3,000.", ["$3000"]),
    ("The bonus is $3000.", ["$3,000"]),
    ("Needs **2 approving reviews**.", ["2"]),
    ("Approval from a VP is required.", ["VP"]),
    ("Within 45 minutes after rollout.", ["45 minutes"]),
    ("Up to 16-character passwords", ["16"]),
    ("Core hours 10:00 to 15:00 Central", ["10:00"]),
    ("A 4% match", ["4 percent", "4%"]),
    ("Core hours are 10:00–15:00 CT", ["15:00"]),
    ("Release 2026.22 on 2026-11-19", ["2026-11-19"]),
    ("Paid PRO RATA", ["pro rata"]),
])
def test_matches(text, alts):
    assert contains_fact(text, alts)


@pytest.mark.parametrize("text,alts", [
    ("Claims above $2,500 need a VP.", ["2"]),
    ("Release on 2026-11-19.", ["2"]),
    ("You get 12 days.", ["2"]),
    ("A 2.5 hour window", ["2"]),
    ("Use the VPN.", ["VP"]),
    ("The freeze started 2026-11-16.", ["16"]),
    ("The freeze started 2026-11-16.", ["11"]),
    ("Paid within 15 days", ["5"]),
])
def test_does_not_match_inside_other_tokens(text, alts):
    assert not contains_fact(text, alts)


def test_facts_matched_one_result_per_group():
    text = "Secondary caregivers get 12 weeks now and 26 weeks from 2027."
    assert facts_matched(text, [["12 weeks"], ["26 weeks"], ["30 weeks"]]) == [True, True, False]
