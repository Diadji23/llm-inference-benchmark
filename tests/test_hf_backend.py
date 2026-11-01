import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from llmbench.backends import make_backend


def _tiny_backend(use_cache: bool):
    """An HF backend around a tiny random Llama, without loading anything from disk."""
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    backend = make_backend("hf", model_path="unused", use_cache=use_cache)
    backend._model = transformers.LlamaForCausalLM(config).eval()
    return backend


def test_greedy_is_the_same_with_and_without_cache():
    prompt = [1, 5, 9, 12, 7, 3, 30, 41]
    with_cache, _ = _tiny_backend(True)._greedy(prompt, 20)
    without_cache, _ = _tiny_backend(False)._greedy(prompt, 20)
    assert len(with_cache) == 20
    assert with_cache == without_cache


def test_greedy_stops_at_eos():
    prompt = [1, 5, 9]
    full, _ = _tiny_backend(True)._greedy(prompt, 10)
    stopped, _ = _tiny_backend(True)._greedy(prompt, 10, eos=full[3])
    assert stopped == full[:full.index(full[3])]


def test_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="dtype"):
        make_backend("hf", model_path="unused", dtype="float8")
