from llmbench.env import collect_env


def test_collect_env_has_required_fields():
    env = collect_env()
    for key in ("timestamp", "cpu", "ram_total_gb", "gpus", "packages", "python"):
        assert key in env
    assert env["cpu"]["logical_cores"] >= 1
    assert env["ram_total_gb"] > 0
