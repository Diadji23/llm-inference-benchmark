import pytest

from llmbench.speculative import greedy_generate, speculative_generate

VOCAB = 50


def target_next(seq: list[int]) -> int:
    return (seq[-1] * 7 + len(seq)) % VOCAB


class FakeModel:
    """Deterministic next-token rule with a KV 'cache' that is just the token list."""

    def __init__(self, rule):
        self.rule = rule
        self.cache: list[int] = []
        self.passes = 0

    @property
    def n_tokens(self) -> int:
        return len(self.cache)

    def forward(self, tokens, all_logits=False):
        self.passes += 1
        preds = []
        for tok in tokens:
            self.cache.append(tok)
            preds.append(self.rule(self.cache))
        return preds if all_logits else preds[-1:]

    def rollback(self, n_tokens):
        assert n_tokens <= len(self.cache)
        del self.cache[n_tokens:]


def draft_rule(every: int):
    """Agrees with the target except at positions that are multiples of `every`."""
    def rule(seq):
        tok = target_next(seq)
        return (tok + 1) % VOCAB if len(seq) % every == 0 else tok
    return rule


PROMPT = [3, 14, 15, 9, 26]


@pytest.mark.parametrize("k", [1, 2, 4, 8])
@pytest.mark.parametrize("every", [2, 3, 5, 1000])
def test_output_is_exactly_the_target_alone(k, every):
    reference = greedy_generate(FakeModel(target_next), PROMPT, 40).tokens
    res = speculative_generate(FakeModel(target_next), FakeModel(draft_rule(every)),
                               PROMPT, 40, k)
    assert res.tokens == reference
    assert res.accepted == sum(res.accepted_per_round)
    assert res.proposed == k * res.rounds


def test_perfect_draft_accepts_everything_in_few_target_passes():
    target = FakeModel(target_next)
    res = speculative_generate(target, FakeModel(target_next), PROMPT, 41, k=4)
    assert res.acceptance_rate == 1.0
    assert res.tokens_per_round == 5.0  # 4 accepted guesses + 1 bonus token per pass
    assert target.passes == 1 + 8      # prefill + 40 tokens / 5 per pass


def test_wrong_draft_still_progresses_one_token_per_round():
    res = speculative_generate(FakeModel(target_next), FakeModel(lambda s: VOCAB - 1),
                               PROMPT, 20, k=3)
    assert res.accepted == 0
    assert res.tokens_per_round == 1.0


def test_caches_hold_only_accepted_tokens():
    target, draft = FakeModel(target_next), FakeModel(draft_rule(3))
    speculative_generate(target, draft, PROMPT, 30, k=4)
    # The last round may overshoot the 30 tokens returned: compare with a longer reference.
    seq = PROMPT + greedy_generate(FakeModel(target_next), PROMPT, 60).tokens
    # Rejected guesses never stay behind: both caches are prefixes of the real sequence.
    assert target.cache == seq[:len(target.cache)]
    assert draft.cache == seq[:len(draft.cache)]


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        speculative_generate(FakeModel(target_next), FakeModel(target_next), PROMPT, 5, k=0)
