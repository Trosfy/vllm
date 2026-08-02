import pytest
import torch

from vllm.model_executor.models.kimi_linear import _restore_merged_output_order


def test_restore_merged_output_order() -> None:
    tp_size = 4
    output_sizes = [8, 12]
    per_rank = []
    expected_q = []
    expected_kv = []
    for rank in range(tp_size):
        q = torch.full((2, output_sizes[0] // tp_size), rank + 1)
        kv = torch.full((2, output_sizes[1] // tp_size), 10 + rank)
        per_rank.append(torch.cat((q, kv), dim=-1))
        expected_q.append(q)
        expected_kv.append(kv)

    rank_major = torch.cat(per_rank, dim=-1)
    expected = torch.cat((*expected_q, *expected_kv), dim=-1)

    actual = _restore_merged_output_order(rank_major, output_sizes, tp_size)

    torch.testing.assert_close(actual, expected)


def test_restore_merged_output_order_rejects_invalid_width() -> None:
    with pytest.raises(ValueError, match="Unexpected gathered"):
        _restore_merged_output_order(torch.empty(2, 79), [32, 48], 4)
