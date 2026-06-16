"""Étape 3 — Vérifications spécifiques sur la commande calculée.

1) Si montant total < MIN_COMMAND_AMOUNT → commande supprimée (0 partout).
2) Si mode de livraison Axytrans :
   - montant total ≤ 75 000 €
   - nb billets total ≤ 2 600 × nb_conteneurs
   → réduction si l'une de ces limites est dépassée.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

from .command_calculator import CommandeParCoupure
from .constants import CommandConfig, COUPURES


@dataclass
class VerificationFlags:
    commande_supprimee: bool = False
    reduite_axytrans_montant: bool = False
    reduite_axytrans_billets: bool = False


def check_min_command(
    commande: CommandeParCoupure,
    min_amount: float = None,
) -> Tuple[CommandeParCoupure, VerificationFlags]:
    """1ère vérification : montant minimal.

    Si le total est inférieur au seuil, on remet toutes les coupures à 0.
    """
    threshold = CommandConfig.MIN_COMMAND_AMOUNT if min_amount is None else min_amount
    flags = VerificationFlags()

    if commande.montant_total < threshold:
        flags.commande_supprimee = True
        return CommandeParCoupure(nb_billets={c: 0 for c in COUPURES}), flags

    return commande, flags


def _reduce_proportionally_to_amount(
    nb_billets: Dict[int, int], target_eur: float
) -> Dict[int, int]:
    """Réduit progressivement la commande (coupures les plus faibles d'abord)
    jusqu'à repasser sous ``target_eur``.

    Retirer les petites coupures en premier minimise la perte de valeur pour
    l'ATM (on conserve les billets de 100 qui contribuent le plus au montant
    avec peu de billets).
    """
    result = dict(nb_billets)
    for coupure in COUPURES:  # ordre croissant 5, 10, 20, 50, 100
        while sum(v * c for c, v in result.items()) > target_eur and result.get(coupure, 0) > 0:
            result[coupure] -= 1
    return result


def _reduce_proportionally_to_billets(
    nb_billets: Dict[int, int], target_billets: int
) -> Dict[int, int]:
    """Réduit le nombre total de billets, coupures faibles en premier."""
    result = dict(nb_billets)
    for coupure in COUPURES:
        while sum(result.values()) > target_billets and result.get(coupure, 0) > 0:
            result[coupure] -= 1
    return result


def reduce_for_axytrans(
    commande: CommandeParCoupure,
    mode_livraison: str,
    nb_conteneurs: int,
    max_eur: float = None,
    max_billets_per_container: int = None,
) -> Tuple[CommandeParCoupure, VerificationFlags]:
    """2e vérification : limites Axytrans (montant et nb billets)."""
    flags = VerificationFlags()

    axytrans_mode = CommandConfig.AXYTRANS_MODE_LIVRAISON
    if (mode_livraison or '').strip().lower() != axytrans_mode:
        return commande, flags

    max_eur = CommandConfig.AXYTRANS_MAX_EUR if max_eur is None else max_eur
    max_per_container = (
        CommandConfig.AXYTRANS_MAX_BILLETS_PER_CONTAINER
        if max_billets_per_container is None
        else max_billets_per_container
    )
    max_billets = max_per_container * max(1, int(nb_conteneurs or 1))

    nb_billets = dict(commande.nb_billets)

    if commande.montant_total > max_eur:
        nb_billets = _reduce_proportionally_to_amount(nb_billets, max_eur)
        flags.reduite_axytrans_montant = True

    # Recalculer sur la version potentiellement réduite.
    total_billets = sum(nb_billets.values())
    if total_billets > max_billets:
        nb_billets = _reduce_proportionally_to_billets(nb_billets, max_billets)
        flags.reduite_axytrans_billets = True

    return CommandeParCoupure(nb_billets=nb_billets), flags
