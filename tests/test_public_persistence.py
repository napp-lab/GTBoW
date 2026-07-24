"""Public persistence round trips, independent of cached private vocabularies."""
import struct

import numpy as np
import pytest
from pyGTBoW import Vocabulary, VocabularyConfig, SizeBinnedVocabulary, Database


@pytest.mark.parametrize("suffix", [".voc", ".yml"])
def test_vocabulary_round_trip_preserves_queries(tmp_path, suffix):
    rng = np.random.default_rng(42)
    features = [rng.integers(0, 256, (40, 32), dtype=np.uint8) for _ in range(3)]
    vocabulary = Vocabulary(K=3, L=2, weight_method="TF", score_method="L2_NORM")
    vocabulary.create(features)
    path = str(tmp_path / ("vocabulary" + suffix))
    vocabulary.save(path)
    restored = Vocabulary(path)
    assert restored.getWordSize() == vocabulary.getWordSize()
    before = Database(vocabulary, use_di=True, di_levels=0)
    after = Database(restored, use_di=True, di_levels=0)
    for descriptors in features:
        before.add(descriptors)
        after.add(descriptors)
        assert vocabulary.transform(descriptors, 0) == restored.transform(descriptors, 0)
    for descriptors in features:
        assert before.query(descriptors, max_results=3) == after.query(descriptors, max_results=3)


def test_standard_size_binned_round_trip(tmp_path):
    rng = np.random.default_rng(43)
    descriptors = rng.integers(0, 256, (120, 32), dtype=np.uint8)
    scales = np.linspace(10, 40, 120, dtype=np.float32)
    config = VocabularyConfig()
    config.setVocabularyParameters(k=3, L=2, weight_method="TF", score_method="L2_NORM")
    vocabulary = SizeBinnedVocabulary(config)
    vocabulary.create(descriptors, scales, ["a"] * 40 + ["b"] * 40 + ["c"] * 40)
    vocabulary.save(str(tmp_path))
    restored = SizeBinnedVocabulary.load(str(tmp_path), config)
    assert restored.getWordSize() == vocabulary.getWordSize()
    original = vocabulary.transform(descriptors, scales)
    loaded = restored.transform(descriptors, scales)
    for before, after in zip(original[:2], loaded[:2]):
        np.testing.assert_array_equal(before, after)
    assert original[2] == loaded[2]


def test_binary_header_matches_uncompressed_upstream_format(tmp_path):
    vocabulary = Vocabulary(K=3, L=2)
    features = np.random.default_rng(44).integers(0, 256, (40, 32), dtype=np.uint8)
    vocabulary.create([features])
    path = tmp_path / "upstream_header.voc"
    vocabulary.save(str(path))
    data = path.read_bytes()
    magic, compressed, node_count = struct.unpack_from("=Q?I", data)
    assert magic == 88877711233
    assert compressed is False
    assert node_count > 1
    assert Vocabulary(str(path)).transform(features, 0) == vocabulary.transform(features, 0)


def test_compressed_upstream_file_is_rejected(tmp_path):
    path = tmp_path / "unsupported.voc"
    # Only the upstream signature and flag are necessary: reject before reading payload.
    path.write_bytes(struct.pack("=Q?", 88877711233, True))
    with pytest.raises(RuntimeError, match="Compressed DBoW3.*does not distribute QuickLZ"):
        Vocabulary(str(path))


def test_uncompressed_upstream_empty_vocabulary_can_be_loaded(tmp_path):
    path = tmp_path / "upstream_empty.voc"
    # Independently construct the upstream zero-node representation.
    path.write_bytes(struct.pack("=Q?I", 88877711233, False, 0))
    vocabulary = Vocabulary(str(path))
    assert vocabulary.getWordSize() == 0


def test_truncated_binary_header_is_rejected(tmp_path):
    path = tmp_path / "truncated.voc"
    path.write_bytes(struct.pack("=Q", 88877711233))
    with pytest.raises(RuntimeError, match="truncated binary header"):
        Vocabulary(str(path))


def test_independent_upstream_binary_fixture_round_trips(tmp_path):
    # Upstream native serialization on supported Linux: root plus one uint8 leaf.
    # k=2, L=1, L2_NORM=1, TF=1; descriptor dimensions/type: 32, 1, CV_8U=0.
    data = (
        struct.pack("=Q?Iiiii", 88877711233, False, 2, 2, 1, 1, 1)
        + struct.pack("=IIdiii", 1, 0, 1.0, 32, 1, 0)
        + bytes(32)
        + struct.pack("=III", 1, 0, 1)
    )
    path = tmp_path / "upstream_fixture.voc"
    path.write_bytes(data)
    vocabulary = Vocabulary(str(path))
    assert vocabulary.getWordSize() == 1
    features = np.zeros((2, 32), dtype=np.uint8)
    assert vocabulary.transform(features, 0)[0] == {0: 1.0}
    saved = tmp_path / "public_saved.voc"
    vocabulary.save(str(saved))
    assert saved.read_bytes() == data
