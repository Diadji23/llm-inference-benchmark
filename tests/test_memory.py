import pytest

from llmbench.memory import kv_cache_mib, parse_llamacpp_buffers

# Excerpt of the llama.cpp log for Qwen2.5-0.5B Q4_K_M, n_ctx=2048, F16 KV cache.
LOG = """\
load_tensors:   CPU_Mapped model buffer size =   462.96 MiB
load_tensors:   CPU_REPACK model buffer size =    28.05 MiB
llama_context:        CPU  output buffer size =     0.58 MiB
llama_kv_cache: size =   24.00 MiB (  2048 cells,  24 layers,  1/1 seqs), K (f16):   12.00 MiB
sched_reserve:        CPU compute buffer size =   300.25 MiB
~llama_context:        CPU compute buffer size is 300.2520 MiB, matches expectation
"""


@pytest.mark.parametrize("kv_type, expected", [("f16", 24.0), ("q8_0", 12.75), ("q4_0", 6.75)])
def test_kv_cache_formula_matches_llamacpp(kv_type, expected):
    # Qwen2.5-0.5B: 24 layers, 2 KV heads of dimension 64. Expected values read from the
    # llama.cpp log for each cache type.
    assert kv_cache_mib(24, 2, 64, 2048, kv_type) == pytest.approx(expected)


def test_parse_llamacpp_buffers():
    sizes = parse_llamacpp_buffers(LOG)
    assert sizes == {
        "model_buffer_mib": pytest.approx(462.96 + 28.05),
        "kv_cache_mib": 24.0,
        "compute_buffer_mib": 300.25,
        "output_buffer_mib": 0.58,
    }
