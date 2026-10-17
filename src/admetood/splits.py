"""Construct the prespecified simulated splits from the training set only.

Each constructor partitions the training molecules into `train'` and `test'` at the
prespecified test fraction. The real prospective test set is never read here: these
splits exist to be compared against it, so letting it influence their construction
would void the comparison.

Group-based constructors (scaffold, butina) assign whole groups, so a molecule and its
close analogues cannot straddle the boundary.
"""

from __future__ import annotations

from collections import defaultdict
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

from .standardize import keys_for

CONSTRUCTORS = ("random", "scaffold", "butina", "frontier", "size")
BUTINA_CUTOFF = 0.65  # Tanimoto distance cutoff; 0.35 similarity.


def _generator():
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprints(smiles: list[str]) -> list:
    """Morgan fingerprints; unparsable molecules yield None and are handled by callers."""
    generator = _generator()
    molecules = [Chem.MolFromSmiles(s) if isinstance(s, str) else None for s in smiles]
    return [generator.GetFingerprint(m) if m is not None else None for m in molecules]


def _assign_groups(groups: dict[object, list[int]], n_total: int,
                   test_fraction: float, largest_first: bool = True) -> np.ndarray:
    """Fill training with whole groups until the target size, remainder becomes test.

    Groups are ordered by size so the split is deterministic, matching the
    MoleculeNet/TDC scaffold-split convention.
    """
    target_train = round(n_total * (1.0 - test_fraction))
    ordered = sorted(groups.values(),
                     key=lambda members: (-len(members) if largest_first else len(members),
                                          members[0]))
    is_test = np.ones(n_total, dtype=bool)
    filled = 0
    for members in ordered:
        if filled + len(members) <= target_train:
            is_test[members] = False
            filled += len(members)
    return is_test


def split_random(smiles: list[str], test_fraction: float, seed: int) -> np.ndarray:
    n = len(smiles)
    is_test = np.zeros(n, dtype=bool)
    chosen = np.random.default_rng(seed).choice(n, size=round(n * test_fraction),
                                                replace=False)
    is_test[chosen] = True
    return is_test


def split_scaffold(smiles: list[str], test_fraction: float, seed: int) -> np.ndarray:
    """Bemis-Murcko scaffold sets assigned whole, largest first."""
    groups: dict[object, list[int]] = defaultdict(list)
    for index, smi in enumerate(smiles):
        keys = keys_for(smi)
        # Unparsable molecules and acyclic ones (empty scaffold) each get their own group
        # rather than being pooled into one artificial mega-group.
        scaffold = keys.scaffold if keys.scaffold else f"__singleton_{index}"
        groups[scaffold].append(index)
    return _assign_groups(groups, len(smiles), test_fraction)


def split_butina(smiles: list[str], test_fraction: float, seed: int,
                 cutoff: float = BUTINA_CUTOFF) -> np.ndarray:
    """Butina sphere-exclusion clusters assigned whole, largest first."""
    fps = fingerprints(smiles)
    valid = [i for i, fp in enumerate(fps) if fp is not None]
    valid_fps = [fps[i] for i in valid]

    # Condensed lower-triangle distance list, as ClusterData expects.
    distances: list[float] = []
    for row in range(1, len(valid_fps)):
        similarity = DataStructs.BulkTanimotoSimilarity(valid_fps[row], valid_fps[:row])
        distances.extend(1.0 - s for s in similarity)

    clusters = Butina.ClusterData(distances, len(valid_fps), cutoff, isDistData=True)
    groups: dict[object, list[int]] = {
        cluster_id: [valid[member] for member in cluster]
        for cluster_id, cluster in enumerate(clusters)
    }
    for index, fp in enumerate(fps):          # Unparsable molecules cluster alone.
        if fp is None:
            groups[f"__invalid_{index}"] = [index]
    return _assign_groups(groups, len(smiles), test_fraction)


def mean_similarity_to_others(smiles: list[str]) -> np.ndarray:
    """Mean Tanimoto of each molecule to all others; low values are chemically isolated."""
    fps = fingerprints(smiles)
    valid = [i for i, fp in enumerate(fps) if fp is not None]
    valid_fps = [fps[i] for i in valid]
    scores = np.full(len(smiles), np.nan)
    for position, fp in enumerate(valid_fps):
        similarity = DataStructs.BulkTanimotoSimilarity(fp, valid_fps)
        # Exclude the self-similarity of 1.0 contributed by this molecule.
        scores[valid[position]] = (sum(similarity) - 1.0) / max(len(valid_fps) - 1, 1)
    return scores


def split_frontier(smiles: list[str], test_fraction: float, seed: int) -> np.ndarray:
    """Hold out the chemically most isolated molecules: a deliberately hard split."""
    scores = mean_similarity_to_others(smiles)
    # Unparsable molecules are maximally "isolated"; rank them last so they do not
    # silently dominate the held-out set.
    ranking = np.lexsort((np.arange(len(smiles)), np.nan_to_num(scores, nan=np.inf)))
    is_test = np.zeros(len(smiles), dtype=bool)
    is_test[ranking[:round(len(smiles) * test_fraction)]] = True
    return is_test


def split_size(smiles: list[str], test_fraction: float, seed: int) -> np.ndarray:
    """Hold out the largest molecules by heavy-atom count."""
    counts = np.array([
        (Chem.MolFromSmiles(s).GetNumHeavyAtoms()
         if isinstance(s, str) and Chem.MolFromSmiles(s) is not None else -1)
        for s in smiles
    ], dtype=float)
    ranking = np.lexsort((np.arange(len(smiles)), -counts))
    is_test = np.zeros(len(smiles), dtype=bool)
    is_test[ranking[:round(len(smiles) * test_fraction)]] = True
    return is_test


SPLIT_FUNCTIONS = {
    "random": split_random,
    "scaffold": split_scaffold,
    "butina": split_butina,
    "frontier": split_frontier,
    "size": split_size,
}


def build(name: str, smiles: list[str], test_fraction: float, seed: int) -> np.ndarray:
    """Return a boolean mask that is True for simulated held-out molecules."""
    if name not in SPLIT_FUNCTIONS:
        raise ValueError(f"Unknown split {name!r}; expected one of {CONSTRUCTORS}")
    mask = SPLIT_FUNCTIONS[name](smiles, test_fraction, seed)
    if mask.sum() == 0 or mask.sum() == len(smiles):
        raise ValueError(f"Split {name!r} produced a degenerate partition")
    return mask
