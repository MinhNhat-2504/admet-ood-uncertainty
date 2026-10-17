"""Split constructors must respect the test fraction, keep groups whole, and be deterministic."""

import numpy as np
import pytest

from admetood import splits

# Two tight analogue series plus two isolated molecules, so group structure is known.
BENZENES = ["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "CCCc1ccccc1", "CCCCc1ccccc1"]
PYRIDINES = ["c1ccncc1", "Cc1ccncc1", "CCc1ccncc1", "CCCc1ccncc1"]
ISOLATED = ["C1CC2CCC1C2", "O=S(=O)(N)c1ccc(Cl)cc1"]
MOLECULES = BENZENES + PYRIDINES + ISOLATED


@pytest.mark.parametrize("name", splits.CONSTRUCTORS)
def test_every_split_is_deterministic_and_non_degenerate(name):
    first = splits.build(name, MOLECULES, 0.3, seed=20261017)
    second = splits.build(name, MOLECULES, 0.3, seed=20261017)
    assert np.array_equal(first, second)
    assert 0 < first.sum() < len(MOLECULES)


@pytest.mark.parametrize("name", splits.CONSTRUCTORS)
def test_test_fraction_is_approximately_respected(name):
    mask = splits.build(name, MOLECULES * 4, 0.3, seed=20261017)
    # Group-based splits cannot hit the target exactly; allow a margin for whole groups.
    assert 0.15 <= mask.mean() <= 0.50


def test_scaffold_split_keeps_each_scaffold_on_one_side():
    mask = splits.build("scaffold", MOLECULES, 0.3, seed=20261017)
    benzene_side = {bool(mask[i]) for i in range(len(BENZENES))}
    pyridine_side = {bool(mask[i]) for i in range(len(BENZENES), len(BENZENES) + len(PYRIDINES))}
    assert len(benzene_side) == 1, "a benzene scaffold group was split across the boundary"
    assert len(pyridine_side) == 1, "a pyridine scaffold group was split across the boundary"


def test_size_split_holds_out_the_largest_molecules():
    mask = splits.build("size", MOLECULES, 0.3, seed=20261017)
    from rdkit import Chem
    sizes = np.array([Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in MOLECULES])
    assert sizes[mask].min() >= sizes[~mask].max()


def test_frontier_split_holds_out_the_least_similar_molecules():
    scores = splits.mean_similarity_to_others(MOLECULES)
    mask = splits.build("frontier", MOLECULES, 0.3, seed=20261017)
    assert scores[mask].max() <= scores[~mask].min()


def test_random_split_differs_between_seeds():
    a = splits.build("random", MOLECULES * 4, 0.3, seed=1)
    b = splits.build("random", MOLECULES * 4, 0.3, seed=2)
    assert not np.array_equal(a, b)


def test_unparsable_molecules_do_not_break_any_constructor():
    polluted = MOLECULES + ["not_a_molecule", ""]
    for name in splits.CONSTRUCTORS:
        mask = splits.build(name, polluted, 0.3, seed=20261017)
        assert mask.shape == (len(polluted),)


def test_unknown_split_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown split"):
        splits.build("not_a_split", MOLECULES, 0.3, seed=1)
