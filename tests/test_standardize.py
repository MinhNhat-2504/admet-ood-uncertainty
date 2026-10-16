"""Identity keys must collapse exactly the variants each level is meant to merge."""

from admetood.standardize import LEVELS, keys_for

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_SODIUM = "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]"
ALANINE_R = "C[C@H](N)C(=O)O"
ALANINE_S = "C[C@@H](N)C(=O)O"
ALANINE_FLAT = "CC(N)C(=O)O"
PENTANEDIONE_KETO = "CC(=O)CC(=O)C"
PENTANEDIONE_ENOL = "CC(O)=CC(=O)C"


def test_salt_and_charge_collapse_to_the_parent():
    """A sodium salt is the same compound as its neutral acid for leakage purposes."""
    free, salt = keys_for(ASPIRIN), keys_for(ASPIRIN_SODIUM)
    assert salt.error is None
    assert free.canonical_smiles != salt.canonical_smiles  # Exact SMILES misses it.
    assert free.inchikey != salt.inchikey                  # Plain InChIKey misses it too.
    assert free.parent_inchikey == salt.parent_inchikey    # The parent key catches it.


def test_stereoisomers_collapse_only_at_the_stereo_insensitive_level():
    r, s, flat = keys_for(ALANINE_R), keys_for(ALANINE_S), keys_for(ALANINE_FLAT)
    assert r.inchikey != s.inchikey
    assert r.stereo_insensitive == s.stereo_insensitive == flat.stereo_insensitive


def test_tautomers_collapse_only_at_the_tautomer_level():
    keto, enol = keys_for(PENTANEDIONE_KETO), keys_for(PENTANEDIONE_ENOL)
    assert keto.parent_inchikey != enol.parent_inchikey
    assert keto.stereo_insensitive != enol.stereo_insensitive
    assert keto.tautomer_insensitive == enol.tautomer_insensitive


def test_unparsable_input_is_reported_not_raised():
    bad = keys_for("not_a_molecule")
    assert bad.error == "unparsable"
    assert all(getattr(bad, level) is None for level in LEVELS)
    assert keys_for("").error == "empty"


def test_levels_are_monotone_from_strict_to_permissive():
    """Each level must merge at least as much as the previous one."""
    pairs = [(ASPIRIN, ASPIRIN_SODIUM), (ALANINE_R, ALANINE_S),
             (PENTANEDIONE_KETO, PENTANEDIONE_ENOL)]
    for left, right in pairs:
        a, b = keys_for(left), keys_for(right)
        matches = [getattr(a, level) == getattr(b, level) for level in LEVELS[:-1]]
        assert matches == sorted(matches), f"{left}/{right} merge then split again"
