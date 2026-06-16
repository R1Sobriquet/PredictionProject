"""Étape 4 — Détection et autorisation de commande exceptionnelle.

Règle métier :
- Après avoir calculé la commande (étape 2 + vérifications 3), on projette
  les soldes au **soir du jour de chargement** (3,0 × DMQ) en incluant la
  commande. Si pour au moins une coupure le solde projeté est inférieur au
  DMQ du jour, alors l'ATM risque d'être vide → commande exceptionnelle.
- Une commande exceptionnelle n'est **autorisée** que si un numéro de tournée
  est disponible au jour de livraison (dépendance injectable).
"""

from dataclasses import dataclass
from datetime import date
from typing import Callable, Dict, Optional

from .constants import COUPURES


# Type alias : callback injectable qui retourne True si une tournée est
# disponible au jour de livraison.
TourneeAvailableFn = Callable[[date, int], bool]

# Stub par défaut : toujours disponible.
DEFAULT_TOURNEE_AVAILABLE: TourneeAvailableFn = lambda _d, _atm: True  # noqa: E731


@dataclass
class ExceptionalDecision:
    risque_vide: bool = False
    autorisee: bool = False


def detect_risque_vide(
    solde_soir: Dict[int, float],
    dmq_par_coupure: Dict[int, float],
    k7hs: Optional[Dict[int, bool]] = None,
) -> bool:
    """Retourne True si, pour au moins une coupure active, le solde au soir
    est inférieur à la consommation quotidienne moyenne.

    Les coupures HS sont ignorées (cassette indisponible → pas d'alerte).
    """
    k7hs = k7hs or {}
    for c in COUPURES:
        if k7hs.get(c, False):
            continue
        dmq = float(dmq_par_coupure.get(c, 0.0))
        if dmq <= 0:
            continue
        solde = float(solde_soir.get(c, 0.0))
        if solde < dmq:
            return True
    return False


def evaluate_exceptional(
    solde_soir: Dict[int, float],
    dmq_par_coupure: Dict[int, float],
    day_livraison: date,
    atm_id: int,
    k7hs: Optional[Dict[int, bool]] = None,
    tournee_available: TourneeAvailableFn = DEFAULT_TOURNEE_AVAILABLE,
) -> ExceptionalDecision:
    """Évalue si une commande exceptionnelle doit être déclenchée et si elle
    est autorisée (tournée disponible).
    """
    decision = ExceptionalDecision()
    decision.risque_vide = detect_risque_vide(solde_soir, dmq_par_coupure, k7hs)

    if decision.risque_vide:
        decision.autorisee = bool(tournee_available(day_livraison, atm_id))

    return decision
