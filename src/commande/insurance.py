"""Étape 6 — Vérification de l'assurance agence et du cap global.

1) Total (commandes en cours pour l'agence + commande courante) ≤ assurance
   agence. Si dépassement : on réduit la commande puis on revérifie le
   montant minimal.
2) Même contrôle avec ``INSURANCE_GLOBAL_CAP`` (300 000 € par défaut).

Si, après réduction, la commande passe sous ``MIN_COMMAND_AMOUNT``, elle est
supprimée (comme à l'étape 3).
"""

from dataclasses import dataclass
from typing import Tuple

from .command_calculator import CommandeParCoupure
from .constants import CommandConfig, COUPURES
from .verifications import _reduce_proportionally_to_amount


@dataclass
class InsuranceFlags:
    reduite_assurance_agence: bool = False
    reduite_cap_global: bool = False
    supprimee_apres_reduction: bool = False


def apply_insurance_caps(
    commande: CommandeParCoupure,
    commandes_en_cours_agence_eur: float,
    insurance_amount_eur: float,
    global_cap_eur: float = None,
    min_amount: float = None,
) -> Tuple[CommandeParCoupure, InsuranceFlags]:
    """Applique les deux caps (assurance agence + cap global) dans l'ordre."""
    flags = InsuranceFlags()
    global_cap = CommandConfig.INSURANCE_GLOBAL_CAP if global_cap_eur is None else global_cap_eur
    min_threshold = CommandConfig.MIN_COMMAND_AMOUNT if min_amount is None else min_amount

    en_cours = max(0.0, float(commandes_en_cours_agence_eur))
    nb_billets = dict(commande.nb_billets)

    # Cap 1 : assurance agence.
    if insurance_amount_eur is not None and insurance_amount_eur > 0:
        target = max(0.0, float(insurance_amount_eur) - en_cours)
        if commande.montant_total > target:
            nb_billets = _reduce_proportionally_to_amount(nb_billets, target)
            flags.reduite_assurance_agence = True

    # Recalcul intermédiaire
    partial = CommandeParCoupure(nb_billets=nb_billets)

    # Cap 2 : cap global 300k€.
    target_global = max(0.0, float(global_cap) - en_cours)
    if partial.montant_total > target_global:
        nb_billets = _reduce_proportionally_to_amount(nb_billets, target_global)
        flags.reduite_cap_global = True

    reduced = CommandeParCoupure(nb_billets=nb_billets)

    # Si la réduction a fait passer la commande sous le minimum, on supprime.
    if (flags.reduite_assurance_agence or flags.reduite_cap_global) and reduced.montant_total < min_threshold:
        flags.supprimee_apres_reduction = True
        reduced = CommandeParCoupure(nb_billets={c: 0 for c in COUPURES})

    return reduced, flags
