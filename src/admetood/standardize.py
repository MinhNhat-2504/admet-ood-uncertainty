"""Molecule standardization and identity keys for leakage auditing.

Exact-SMILES deduplication misses salts, charge states, tautomers and stereochemistry,
which is how published benchmarks retain train/test leaks. This module derives a ladder
of increasingly permissive identity keys so an audit can report leakage at each level
rather than a single number.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")  # Parsing failures are reported through `error` instead.

# Identity levels, strictest first. Each later level merges molecules the previous
# one kept apart, so leakage counts are monotone non-decreasing along this list.
LEVELS = ("canonical_smiles", "inchikey", "parent_inchikey",
          "stereo_insensitive", "tautomer_insensitive", "scaffold")


@dataclass(frozen=True)
class Keys:
    """Identity keys for one input SMILES; `error` is set when parsing fails."""

    smiles: str
    canonical_smiles: str | None = None
    inchikey: str | None = None
    parent_inchikey: str | None = None
    stereo_insensitive: str | None = None
    tautomer_insensitive: str | None = None
    scaffold: str | None = None
    error: str | None = None


@lru_cache(maxsize=1)
def _uncharger() -> rdMolStandardize.Uncharger:
    return rdMolStandardize.Uncharger()


@lru_cache(maxsize=1)
def _tautomer_enumerator() -> rdMolStandardize.TautomerEnumerator:
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTransforms(200)  # Bound the search; large molecules can blow up.
    return enumerator


def parent_mol(mol: Chem.Mol) -> Chem.Mol:
    """Largest organic fragment, neutralized: strips salts, solvents and counterions."""
    # Cleanup can return a molecule without ring info, which FragmentParent requires.
    Chem.SanitizeMol(mol)
    fragment = rdMolStandardize.FragmentParent(mol)
    Chem.SanitizeMol(fragment)
    return _uncharger().uncharge(fragment)


def keys_for(smiles: str) -> Keys:
    """Compute every identity key for one SMILES, never raising on bad input."""
    if not isinstance(smiles, str) or not smiles.strip():
        return Keys(smiles=str(smiles), error="empty")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return Keys(smiles=smiles, error="unparsable")
    try:
        cleaned = rdMolStandardize.Cleanup(mol)
        parent = parent_mol(cleaned)
        if parent.GetNumAtoms() == 0:
            return Keys(smiles=smiles, error="empty_parent")

        flat = Chem.Mol(parent)
        Chem.RemoveStereochemistry(flat)
        canonical_tautomer = _tautomer_enumerator().Canonicalize(flat)
        scaffold = MurckoScaffold.GetScaffoldForMol(flat)

        return Keys(
            smiles=smiles,
            canonical_smiles=Chem.MolToSmiles(mol),
            inchikey=inchi.MolToInchiKey(mol) or None,
            parent_inchikey=inchi.MolToInchiKey(parent) or None,
            # First InChIKey block encodes the skeleton; the second encodes stereo
            # and protonation, so truncating merges stereoisomers of one parent.
            stereo_insensitive=(inchi.MolToInchiKey(parent) or "|")[:14] or None,
            tautomer_insensitive=inchi.MolToInchiKey(canonical_tautomer)[:14] or None,
            scaffold=Chem.MolToSmiles(scaffold) if scaffold.GetNumAtoms() else "",
        )
    except (Chem.rdchem.MolSanitizeException, ValueError, RuntimeError) as error:
        return Keys(smiles=smiles, error=f"{type(error).__name__}: {error}"[:200])
