import vllm.model_executor.layers.vocab_parallel_embedding as vocab_module


def test_non_power_of_two_tp_pads_global_vocab(monkeypatch):
    monkeypatch.setattr(
        vocab_module, "get_tensor_model_parallel_world_size", lambda: 12
    )
    monkeypatch.setattr(vocab_module, "get_tensor_model_parallel_rank", lambda: 11)

    embedding = vocab_module.VocabParallelEmbedding(163840, 16)

    assert embedding.padding_size == 192
    assert embedding.org_vocab_size_padded == 163968
    assert embedding.num_embeddings_padded == 163968
    assert embedding.num_embeddings_per_partition == 13664
    assert embedding.shard_indices.org_vocab_start_index == 150304
    assert embedding.shard_indices.org_vocab_end_index == 163840
    assert embedding.shard_indices.num_org_vocab_padding == 128
    mapping = embedding.get_sharded_to_full_mapping()
    assert mapping is not None
    assert mapping[:163840] == list(range(163840))
    assert mapping[163840:] == list(range(163840, 163968))


def test_power_of_two_tp_preserves_default_padding(monkeypatch):
    monkeypatch.setattr(
        vocab_module, "get_tensor_model_parallel_world_size", lambda: 16
    )
    monkeypatch.setattr(vocab_module, "get_tensor_model_parallel_rank", lambda: 15)

    embedding = vocab_module.VocabParallelEmbedding(163840, 16)

    assert embedding.padding_size == vocab_module.DEFAULT_VOCAB_PADDING_SIZE
    assert embedding.num_embeddings_padded == 163840
    assert embedding.num_embeddings_per_partition == 10240
