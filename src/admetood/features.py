"""One fixed molecular representation shared by every model and uncertainty method.

Holding the representation constant means any difference between methods is attributable
to the method, not to its features. Morgan count fingerprints carry substructure
identity; the descriptor block carries the bulk physicochemical properties that drive
ADMET behaviour and that a fingerprint represents only indirectly.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

FP_BITS = 2048
FP_RADIUS = 2

# A small, stable descriptor set. Each is cheap, defined for every valid molecule, and
# interpretable, which keeps failures diagnosable.
DESCRIPTORS: dict[str, callable] = {
    "mol_weight": Descriptors.MolWt,
    "clogp": Crippen.MolLogP,
    "tpsa": rdMolDescriptors.CalcTPSA,
    "h_donors": rdMolDescriptors.CalcNumHBD,
    "h_acceptors": rdMolDescriptors.CalcNumHBA,
    "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds,
    "rings": rdMolDescriptors.CalcNumRings,
    "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings,
    "fraction_csp3": rdMolDescriptors.CalcFractionCSP3,
    "heavy_atoms": lambda m: float(m.GetNumHeavyAtoms()),
    "formal_charge": lambda m: float(Chem.GetFormalCharge(m)),
    "molar_refractivity": Crippen.MolMR,
    "heteroatoms": rdMolDescriptors.CalcNumHeteroatoms,
    "stereocentres": lambda m: float(rdMolDescriptors.CalcNumAtomStereoCenters(m)),
    "spiro_atoms": rdMolDescriptors.CalcNumSpiroAtoms,
    "bridgehead_atoms": rdMolDescriptors.CalcNumBridgeheadAtoms,
}
FEATURE_NAMES = [f"fp_{i}" for i in range(FP_BITS)] + list(DESCRIPTORS)


def _descriptor_values(mol: Chem.Mol) -> list[float]:
    values = []
    for function in DESCRIPTORS.values():
        try:
            value = float(function(mol))
        except (ValueError, RuntimeError):
            value = np.nan
        values.append(value)
    return values


def featurize(smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return the feature matrix and a mask of molecules that could be featurized.

    Unparsable molecules get a zero row and a False mask entry; callers drop them rather
    than training on silent zeros.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    matrix = np.zeros((len(smiles), len(FEATURE_NAMES)), dtype=np.float32)
    valid = np.zeros(len(smiles), dtype=bool)

    for row, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        counts = generator.GetCountFingerprintAsNumPy(mol)
        matrix[row, :FP_BITS] = counts
        matrix[row, FP_BITS:] = _descriptor_values(mol)
        valid[row] = True

    # A descriptor that failed on one molecule must not poison the whole column.
    descriptor_block = matrix[:, FP_BITS:]
    column_medians = np.nanmedian(np.where(valid[:, None], descriptor_block, np.nan), axis=0)
    column_medians = np.nan_to_num(column_medians, nan=0.0)
    missing = np.isnan(descriptor_block)
    descriptor_block[missing] = np.take(column_medians, np.where(missing)[1])
    matrix[:, FP_BITS:] = descriptor_block
    return matrix, valid
