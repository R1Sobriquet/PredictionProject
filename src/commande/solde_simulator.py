"""Étapes 1 & 4 — Simulation des soldes de l'automate.

Étape 1 : solde au jour du chargement (2,5 × DMQ hors férié, cf. doc)
Étape 4 : solde au soir du jour de chargement (3,0 × DMQ) après ajout de la
commande calculée — sert à déterminer si une commande exceptionnelle doit être
déclenchée.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional

import pandas as pd

from .constants import CommandConfig, COUPURES


# Type alias : callback injectable pour la gestion des jours fériés.
IsHolidayFn = Callable[[date], bool]

# Stub par défaut : aucun jour férié.
DEFAULT_IS_HOLIDAY: IsHolidayFn = lambda _d: False  # noqa: E731


@dataclass
class PendingCommand:
    """Commande précédente non encore chargée (alerte étape 1)."""

    loading_date: date
    amounts_par_coupure: Dict[int, float] = field(default_factory=dict)


def _as_date(d) -> date:
    """Normalise une date (datetime, pd.Timestamp, str) → `date`."""
    if isinstance(d, date) and not hasattr(d, 'hour'):
        return d
    return pd.Timestamp(d).date()


def simulate_solde_at_loading(
    solde_jour: Dict[int, float],
    dmq_par_coupure: Dict[int, float],
    day_commande: date,
    day_chargement: date,
    pending_commands: Optional[List[PendingCommand]] = None,
    is_holiday: IsHolidayFn = DEFAULT_IS_HOLIDAY,
    conso_factor: Optional[float] = None,
) -> Dict[int, float]:
    """Projette les soldes jusqu'au jour de chargement.

    Règle doc (étape 1) :
    - On part des soldes du jour de commande.
    - On soustrait DMQ pour chaque jour écoulé jusqu'au chargement.
    - Hors férié : on utilise 2,5 × DMQ (chargement anticipé) quand l'écart
      commande→chargement vaut 3 jours normalement.
    - Les commandes non chargées rencontrées dans l'intervalle sont ajoutées.

    Args:
        solde_jour: Soldes du jour par coupure.
        dmq_par_coupure: DMQ (par coupure) estimé pour cet automate.
        day_commande: Jour de commande.
        day_chargement: Jour prévu de chargement.
        pending_commands: Commandes précédentes non encore chargées.
        is_holiday: Callback `date → bool`.
        conso_factor: Nombre de jours de DMQ à consommer (défaut : 2,5).

    Returns:
        Dict[coupure, solde projeté].
    """
    day_commande = _as_date(day_commande)
    day_chargement = _as_date(day_chargement)

    factor = conso_factor if conso_factor is not None else CommandConfig.DMQ_CONSO_JOURS_CHARGEMENT
    pending = pending_commands or []

    soldes = {c: float(solde_jour.get(c, 0.0)) for c in COUPURES}
    dmq = {c: float(dmq_par_coupure.get(c, 0.0)) for c in COUPURES}

    if day_chargement <= day_commande:
        return soldes

    days_total = (day_chargement - day_commande).days

    # Identifier les commandes à charger entre day_commande et day_chargement.
    pending_by_date: Dict[date, Dict[int, float]] = {}
    for pc in pending:
        pcd = _as_date(pc.loading_date)
        if day_commande < pcd <= day_chargement:
            pending_by_date.setdefault(pcd, {})
            for c, v in pc.amounts_par_coupure.items():
                pending_by_date[pcd][c] = pending_by_date[pcd].get(c, 0.0) + float(v)

    # Nombre de jours "consommation" à appliquer.
    # Hors férié, on calcule `factor` consommations (2.5 par défaut).
    current = day_commande
    effective_conso_days = 0.0
    for _ in range(days_total):
        current = current + timedelta(days=1)
        if not is_holiday(current):
            effective_conso_days += 1.0

        # Ajouter les commandes non chargées au bon jour.
        if current in pending_by_date:
            for c, v in pending_by_date[current].items():
                soldes[c] = soldes.get(c, 0.0) + v

    # Appliquer le facteur : par défaut 2.5 jours de DMQ (chargement anticipé).
    # Si l'écart réel est différent de 3 jours ouvrés, on garde `factor` tel quel.
    for c in COUPURES:
        soldes[c] = max(0.0, soldes[c] - dmq[c] * factor)

    return soldes


def simulate_solde_evening_after_loading(
    solde_jour: Dict[int, float],
    dmq_par_coupure: Dict[int, float],
    day_commande: date,
    day_chargement: date,
    commanded_values: Dict[int, float],
    pending_commands: Optional[List[PendingCommand]] = None,
    is_holiday: IsHolidayFn = DEFAULT_IS_HOLIDAY,
) -> Dict[int, float]:
    """Projette les soldes au soir du jour de chargement (étape 4).

    Diffère de `simulate_solde_at_loading` par :
    - le facteur de consommation (3,0 × DMQ au lieu de 2,5)
    - l'ajout de la commande calculée aux soldes projetés.

    Le résultat sert à détecter si l'automate risque d'être vide malgré la
    commande (→ déclenche une commande exceptionnelle).
    """
    soldes = simulate_solde_at_loading(
        solde_jour=solde_jour,
        dmq_par_coupure=dmq_par_coupure,
        day_commande=day_commande,
        day_chargement=day_chargement,
        pending_commands=pending_commands,
        is_holiday=is_holiday,
        conso_factor=CommandConfig.DMQ_CONSO_SOIR,
    )

    for c in COUPURES:
        soldes[c] = soldes.get(c, 0.0) + float(commanded_values.get(c, 0.0))

    return soldes
