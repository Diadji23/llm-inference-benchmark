from llmbench.env import machine_canary_ms
from llmbench.runner import schedule

MODELS = [{"name": f"m{i}"} for i in range(5)]


def test_schedule_visits_each_model_once_per_round():
    plan = schedule(MODELS, rounds=4, seed=0)
    assert len(plan) == 20
    for r in range(4):
        names = sorted(m["name"] for rnd, m in plan if rnd == r)
        assert names == [m["name"] for m in MODELS]


def test_schedule_shuffles_between_rounds_and_is_reproducible():
    plan = schedule(MODELS, rounds=4, seed=0)
    orders = {tuple(m["name"] for rnd, m in plan if rnd == r) for r in range(4)}
    assert len(orders) > 1
    assert plan == schedule(MODELS, rounds=4, seed=0)


def test_canary_returns_positive_milliseconds():
    assert 0 < machine_canary_ms(repeats=3) < 1000
