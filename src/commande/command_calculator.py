"""Étape 2 — Calcul du montant à précommander par coupure.

Formule doc :
    nb_billets_a_commander[c] = max(0,
        SEUIL_MAX[c] * nb_cassettes[c] - nb_billets_presents[c]
    )
    montant_total = Σ nb_billets[c] * c

Les coupures marquées K7 HS (étape 0) sont forcées à 0.
"""

from dataclasses import dataclass, field
from typing import Dict

from .constants import CommandConfig, COUPURES


@dataclass
class CommandeParCoupure:
    """Résultat du calcul de l'étape 2 pour un automate."""

    nb_billets: Dict[int, int] = field(default_factory=dict)

    @property
    def montant_par_coupure(self) -> Dict[int, float]:
        return {c: float(self.nb_billets.get(c, 0) * c) for c in COUPURES}

    @property
    def montant_total(self) -> float:
        return float(sum(self.nb_billets.get(c, 0) * c for c in COUPURES))

    @property
    def total_billets(self) -> int:
        return int(sum(self.nb_billets.values()))

    def is_empty(self) -> bool:
        return all(self.nb_billets.get(c, 0) == 0 for c in COUPURES)


def compute_command_per_coupure(
    solde_chargement: Dict[int, float],
    nb_cassettes_par_coupure: Dict[int, int],
    k7hs: Dict[int, bool],
) -> CommandeParCoupure:
    """Applique la formule métier de l'étape 2.

    Args:
        solde_chargement: Soldes projetés au jour du chargement (en billets).
        nb_cassettes_par_coupure: Nombre de cassettes par coupure pour l'ATM.
        k7hs: Flags HS par coupure (une cassette HS → 0 billets commandés).

    Returns:
        CommandeParCoupure avec nb_billets par coupure.
    """
    nb_billets: Dict[int, int] = {}

    for coupure in COUPURES:
        if k7hs.get(coupure, False):
            nb_billets[coupure] = 0
            continue

        seuil_max = CommandConfig.seuil_max(coupure)
        nb_cassettes = int(nb_cassettes_par_coupure.get(coupure, 0) or 0)
        present = float(solde_chargement.get(coupure, 0.0) or 0.0)

        capacite_max = seuil_max * nb_cassettes
        a_commander = max(0, capacite_max - int(round(present)))
        nb_billets[coupure] = int(a_commander)

    return CommandeParCoupure(nb_billets=nb_billets)


def compute_command_clic_clac(
    nb_cassettes_par_coupure: Dict[int, int],
    k7hs: Dict[int, bool],
) -> CommandeParCoupure:
    """Étape 5 variante « clic-clac » : remplit au seuil maximum.

    En mode remplacement, on ne tient pas compte du solde restant.
    """
    nb_billets: Dict[int, int] = {}
    for coupure in COUPURES:
        if k7hs.get(coupure, False):
            nb_billets[coupure] = 0
            continue
        nb_billets[coupure] = CommandConfig.seuil_max(coupure) * int(
            nb_cassettes_par_coupure.get(coupure, 0) or 0
        )
    return CommandeParCoupure(nb_billets=nb_billets)


def compute_command_exceptionnelle(
    nb_cassettes_par_coupure: Dict[int, int],
    k7hs: Dict[int, bool],
) -> CommandeParCoupure:
    """Étape 4 — commande exceptionnelle : demi-seuil max par coupure."""
    nb_billets: Dict[int, int] = {}
    for coupure in COUPURES:
        if k7hs.get(coupure, False):
            nb_billets[coupure] = 0
            continue
        seuil = CommandConfig.seuil_max(coupure)
        nb_cassettes = int(nb_cassettes_par_coupure.get(coupure, 0) or 0)
        nb_billets[coupure] = (seuil * nb_cassettes) // 2
    return CommandeParCoupure(nb_billets=nb_billets)
