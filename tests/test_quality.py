import numpy as np
import pytest

from llmbench.backends.base import Backend
from llmbench.quality import (
    coarse_kl,
    evaluate_mcq,
    evaluate_text,
    format_mcq,
    log_softmax,
    mcnemar_exact_p,
    window_starts,
)

VOCAB = 50


class FakeBackend(Backend):
    """Logits are a fixed function of the input, so results are known in advance."""

    name = "fake"

    def __init__(self, mode: str, noise: float = 0.0, seed: int = 0):
        self.mode, self.noise = mode, noise
        self.rng = np.random.default_rng(seed)

    def load(self): ...
    def tokenize(self, text): ...
    def detokenize(self, tokens): ...
    def generate(self, prompt_tokens, max_new_tokens, ignore_eos=True): ...

    def logits(self, tokens):
        n = len(tokens)
        if self.mode == "uniform":
            return np.zeros((n, VOCAB))
        # "oracle": a huge logit on the true next token (here: token + 1 mod VOCAB).
        out = np.zeros((n, VOCAB))
        out[np.arange(n), (np.asarray(tokens) + 1) % VOCAB] = 50.0
        return out + self.noise * self.rng.standard_normal((n, VOCAB))


TOKENS = [i % VOCAB for i in range(4 * 64)]  # each token is followed by token + 1


def test_log_softmax_normalizes():
    lp = log_softmax(np.random.default_rng(0).standard_normal((3, 10)))
    assert np.allclose(np.exp(lp).sum(axis=-1), 1.0)


def test_uniform_model_has_perplexity_equal_to_vocab():
    res = evaluate_text(FakeBackend("uniform"), TOKENS, ctx=64, n_windows=4)
    assert res["ppl"] == pytest.approx(VOCAB)
    assert res["n_scored"] == 4 * 32  # second half of each window


def test_perfect_model_has_perplexity_one():
    res = evaluate_text(FakeBackend("oracle"), TOKENS, ctx=64, n_windows=4)
    assert res["ppl"] == pytest.approx(1.0)


def test_agreement_with_itself_is_perfect():
    ref = evaluate_text(FakeBackend("oracle"), TOKENS, ctx=64, n_windows=4)["scored"]
    res = evaluate_text(FakeBackend("oracle"), TOKENS, ctx=64, n_windows=4, reference=ref)
    assert res["top1_agreement"] == 1.0
    assert res["kl_coarse_mean"] == pytest.approx(0.0, abs=1e-9)


def test_noisy_model_disagrees_more():
    ref = evaluate_text(FakeBackend("oracle"), TOKENS, ctx=64, n_windows=4)["scored"]
    low = evaluate_text(FakeBackend("oracle", noise=5.0), TOKENS, 64, 4, reference=ref)
    high = evaluate_text(FakeBackend("oracle", noise=30.0), TOKENS, 64, 4, reference=ref)
    assert low["kl_coarse_mean"] < high["kl_coarse_mean"]
    assert high["top1_agreement"] < 1.0


def test_coarse_kl_is_a_lower_bound_of_full_kl():
    rng = np.random.default_rng(1)
    p_lp = log_softmax(2 * rng.standard_normal((200, VOCAB)))
    q_lp = log_softmax(2 * rng.standard_normal((200, VOCAB)))
    full = (np.exp(p_lp) * (p_lp - q_lp)).sum(axis=-1)
    ids = np.argsort(-p_lp, axis=-1)[:, :5]
    coarse = coarse_kl(ids, np.take_along_axis(p_lp, ids, axis=-1), q_lp)
    assert np.all(coarse <= full + 1e-9)
    assert np.all(coarse >= -1e-9)


def test_window_starts_rejects_short_text():
    assert window_starts(300, 64, 4) == [0, 64, 128, 192]
    with pytest.raises(ValueError):
        window_starts(100, 64, 4)


# --- Multiple choice -----------------------------------------------------------------

LETTER_IDS = {" A": 1, " B": 2, " C": 3, " D": 4}


class AlwaysLetter(FakeBackend):
    """Answers the same letter to every question."""

    def __init__(self, letter: str):
        super().__init__("uniform")
        self.letter_id = LETTER_IDS[f" {letter}"]

    def tokenize(self, text, add_bos=True):
        return [LETTER_IDS[text]] if text in LETTER_IDS else [0] * 10

    def logits(self, tokens):
        out = np.zeros((len(tokens), VOCAB))
        out[-1, self.letter_id] = 10.0
        return out


ITEMS = [{"subject": "x", "question": "q", "choices": ["a", "b", "c", "d"], "answer": a}
         for a in (3, 3, 3, 0)]


def test_format_mcq_ends_with_answer_prompt():
    text = format_mcq({"subject": "high_school_physics", "question": "Why?",
                       "choices": ["1", "2", "3", "4"], "answer": 0})
    assert "about high school physics" in text
    assert "\nD. 4\nAnswer:" in text


def test_mcq_accuracy_and_paired_counts():
    ref = evaluate_mcq(AlwaysLetter("D"), ITEMS)
    assert ref["accuracy"] == pytest.approx(0.75)
    other = evaluate_mcq(AlwaysLetter("A"), ITEMS, reference_preds=ref["preds"])
    assert other["accuracy"] == pytest.approx(0.25)
    assert other["answer_agreement"] == 0.0
    assert (other["fixed_vs_ref"], other["broken_vs_ref"]) == (1, 3)


def test_mcnemar_exact_p():
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(0, 6) == pytest.approx(2 / 64)
    assert mcnemar_exact_p(3, 3) == 1.0
