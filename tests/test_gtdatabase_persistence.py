import struct

import numpy as np
import pytest

from pyGTBoW import GTDatabase


def sample_entry(offset=0):
    assignments = np.array([[0, 1], [1, 2]], dtype=np.int32) + offset
    weights = np.array([[0.75, 0.25], [0.6, 0.4]], dtype=np.float64)
    poses = np.array(
        [[10.0 + offset, 20.0, 0.25], [30.0, 40.0 + offset, 1.5]],
        dtype=np.float64,
    )
    bow = {
        offset: 0.5,
        offset + 1: 0.75,
        offset + 2: 0.25,
    }
    return bow, assignments, weights, poses


def orientation_entry(orientations):
    orientations = np.asarray(orientations, dtype=np.float64)
    assignments = np.zeros((len(orientations), 1), dtype=np.int32)
    weights = np.ones((len(orientations), 1), dtype=np.float64)
    poses = np.zeros((len(orientations), 3), dtype=np.float64)
    poses[:, 0] = np.arange(len(orientations), dtype=np.float64)
    poses[:, 2] = orientations
    return {0: 1.0}, assignments, weights, poses


@pytest.mark.parametrize("use_di", [False, True])
def test_empty_database_round_trip(tmp_path, use_di):
    path = tmp_path / "empty.gtbow"
    db = GTDatabase(use_di=use_di, di_levels=0)
    db.setNumWords(8)
    db.save(str(path))

    loaded = GTDatabase()
    loaded.load(str(path))
    assert loaded.size() == 0


@pytest.mark.parametrize("use_di", [False, True])
def test_round_trip_query_and_continued_enrollment(tmp_path, use_di):
    path = tmp_path / "database.gtbow"
    db = GTDatabase(use_di=use_di, di_levels=0)
    db.setNumWords(8)
    entries = [sample_entry(0), sample_entry(3)]
    for entry in entries:
        db.addBowVec(*entry)
    before = db.queryBowVec(*entries[0], max_results=5, num_orientation_bins=8)
    db.save(str(path))

    loaded = GTDatabase()
    loaded.load(str(path))
    after = loaded.queryBowVec(*entries[0], max_results=5, num_orientation_bins=8)
    assert loaded.size() == db.size() == 2
    assert [(r[0], r[1]) for r in after[0]] == pytest.approx(
        [(r[0], r[1]) for r in before[0]]
    )
    for old, new in zip(before[0], after[0]):
        np.testing.assert_array_equal(old[2], new[2])
        np.testing.assert_allclose(old[3], new[3])
    np.testing.assert_allclose(before[1], after[1])
    if use_di:
        assert loaded.retrieveFeatures(0) == db.retrieveFeatures(0)

    assert loaded.addBowVec(*sample_entry(1)) == 2
    assert loaded.size() == 3


def test_repeated_save_load(tmp_path):
    db = GTDatabase(use_di=True, di_levels=0)
    db.setNumWords(8)
    db.addBowVec(*sample_entry())
    for index in range(3):
        path = tmp_path / f"cycle-{index}.gtbow"
        db.save(str(path))
        replacement = GTDatabase()
        replacement.load(str(path))
        db = replacement
    assert db.size() == 1


def test_load_failures(tmp_path):
    db = GTDatabase(use_di=False)
    db.setNumWords(4)
    valid = tmp_path / "valid.gtbow"
    db.save(str(valid))
    data = valid.read_bytes()

    cases = {
        "truncated": data[:10],
        "magic": b"BADMAGIC" + data[8:],
        "version": data[:8] + struct.pack("I", 999) + data[12:],
    }
    for name, payload in cases.items():
        path = tmp_path / f"{name}.gtbow"
        path.write_bytes(payload)
        with pytest.raises(RuntimeError):
            GTDatabase().load(str(path))

    with pytest.raises(RuntimeError):
        GTDatabase().load(str(tmp_path / "missing.gtbow"))


def test_orientation_response_api_preserves_legacy_return_contract():
    db = GTDatabase(use_di=False, di_levels=0)
    db.setNumWords(1)
    db.addBowVec(*orientation_entry([0.0]))
    query = orientation_entry([0.05])

    legacy = db.queryBowVec(*query, max_results=1, num_orientation_bins=8)
    extended = db.queryBowVecWithOrientationResponse(
        *query, max_results=1, num_orientation_bins=8
    )

    assert len(legacy) == 2
    assert len(extended) == 3
    assert len(legacy[0][0]) == 4
    assert len(extended[0][0]) == 4
    assert legacy[0][0][0:2] == pytest.approx(extended[0][0][0:2])
    np.testing.assert_array_equal(legacy[0][0][2], extended[0][0][2])
    np.testing.assert_allclose(legacy[0][0][3], extended[0][0][3])
    np.testing.assert_allclose(legacy[1], extended[1])


def test_orientation_response_reports_generic_winner_and_votes():
    db = GTDatabase(use_di=False, di_levels=0)
    db.setNumWords(1)
    db.addBowVec(*orientation_entry([0.0]))

    results, _poses, metadata = db.queryBowVecWithOrientationResponse(
        *orientation_entry([0.05]), max_results=1, num_orientation_bins=8
    )

    assert results[0][0] == metadata[0]["id"] == 0
    assert metadata[0]["winning_bin"] == 4
    assert metadata[0]["orientation_enabled"] is True
    np.testing.assert_allclose(
        metadata[0]["bin_centers_rad"],
        np.linspace(-np.pi, np.pi, 8, endpoint=False),
    )
    assert len(metadata[0]["vote_mass"]) == 8
    assert metadata[0]["vote_mass"][4] == pytest.approx(1.0)
    assert sum(metadata[0]["match_counts"]) == 1
    assert metadata[0]["match_counts"][4] == 1


def test_orientation_response_wraps_pi_to_first_bin():
    db = GTDatabase(use_di=False, di_levels=0)
    db.setNumWords(1)
    db.addBowVec(*orientation_entry([0.0]))

    _results, _poses, metadata = db.queryBowVecWithOrientationResponse(
        *orientation_entry([np.pi]), max_results=1, num_orientation_bins=8
    )

    assert metadata[0]["winning_bin"] == 0
    assert metadata[0]["bin_centers_rad"][0] == pytest.approx(-np.pi)


def test_orientation_response_marks_single_bin_as_disabled():
    db = GTDatabase(use_di=False, di_levels=0)
    db.setNumWords(1)
    db.addBowVec(*orientation_entry([0.0]))

    _results, _poses, metadata = db.queryBowVecWithOrientationResponse(
        *orientation_entry([0.0]), max_results=1, num_orientation_bins=1
    )

    assert metadata[0]["winning_bin"] == 0
    assert metadata[0]["orientation_enabled"] is False
    assert len(metadata[0]["vote_mass"]) == 1
