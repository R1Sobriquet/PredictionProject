"""Étape 0 — Détection des cassettes hors service (K7 HS).

Règle métier (cf. documentation PredikATM, section 4.1 étape 0) :
« On boucle sur les 15 derniers soldes reçus en vérifiant s'ils ont bougé par
rapport au jour précédent. Si le solde d'une cassette n'a pas bougé depuis au
moins les 3 derniers jours, alors la cassette est considérée comme HS. »
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .constants import CommandConfig, COUPURES, SOLDES_BY_COUPURE


def detect_k7hs(
    historical_soldes: pd.DataFrame,
    window_days: Optional[int] = None,
    stale_threshold: Optional[int] = None,
) -> Dict[int, bool]:
    """Détecte les cassettes hors service pour un automate.

    Args:
        historical_soldes: DataFrame trié par date ASC avec les colonnes
            `soldes_5/10/20/50/100`. Représente l'historique des soldes d'un
            unique automate (pas de regroupement nécessaire).
        window_days: Nombre de derniers soldes à considérer (défaut: 15).
        stale_threshold: Nombre de jours consécutifs sans variation au-delà
            desquels la cassette est déclarée HS (défaut: 3).

    Returns:
        Dict[coupure, bool]: True si la cassette est HS, False sinon.
        Une coupure absente du DataFrame est considérée HS (pas de cassette).
    """
    window = window_days or CommandConfig.K7HS_WINDOW_DAYS
    stale = stale_threshold or CommandConfig.K7HS_STALE_DAYS

    if historical_soldes is None or historical_soldes.empty:
        return {c: True for c in COUPURES}

    tail = historical_soldes.tail(window)

    result: Dict[int, bool] = {}
    for coupure in COUPURES:
        col = SOLDES_BY_COUPURE[coupure]

        if col not in tail.columns:
            # Pas de cassette pour cette coupure → HS par convention
            result[coupure] = True
            continue

        series = tail[col].to_numpy(dtype=float)
        if len(series) < stale + 1:
            # Historique trop court pour conclure → non-HS par défaut
            result[coupure] = False
            continue

        # Variations jour par jour ; np.diff renvoie len-1 valeurs.
        diffs = np.diff(series)
        # Comparaison flottante tolérante (petites oscillations d'arrondi)
        no_move = np.isclose(diffs, 0.0, atol=1e-6)

        # Les `stale` dernières variations toutes nulles → HS
        result[coupure] = bool(no_move[-stale:].all())

    return result


def detect_k7hs_from_row(
    current_row: pd.Series,
    history: pd.DataFrame,
) -> Dict[int, bool]:
    """Variante pratique : utilise `history` et inclut la ligne courante.

    Utile quand on appelle le détecteur pendant l'itération sur un DataFrame.
    """
    frame = history.copy() if history is not None else pd.DataFrame()
    if current_row is not None:
        frame = pd.concat([frame, current_row.to_frame().T], ignore_index=True)
    return detect_k7hs(frame)
