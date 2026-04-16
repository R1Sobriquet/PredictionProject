"""Helpers calendrier FR (jours fériés, veilles, jours de paie).

Calendrier statique — aucune dépendance externe. Les jours fériés français
sont calculés à la volée à partir des règles officielles (fixes + Pâques),
pour toute année demandée.

API :
    - ``is_french_holiday(d)``      : jour férié FR ?
    - ``is_eve_of_holiday(d)``      : veille de jour férié ?
    - ``is_payday(d)``              : jour de paie (fin de mois ou ~5) ?
    - ``french_holidays(year)``     : set de ``date`` fériés pour ``year``.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from typing import FrozenSet


def _easter_sunday(year: int) -> date:
    """Date du dimanche de Pâques (algorithme de Butcher/Meeus anonyme)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


@lru_cache(maxsize=32)
def french_holidays(year: int) -> FrozenSet[date]:
    """Retourne le set des jours fériés français pour ``year``.

    Inclut les 11 jours fériés légaux métropolitains :
    1er janvier, lundi de Pâques, 1er mai, 8 mai, Ascension,
    lundi de Pentecôte, 14 juillet, 15 août, 1er novembre,
    11 novembre, 25 décembre.
    """
    easter = _easter_sunday(year)
    holidays = {
        date(year, 1, 1),            # Jour de l'An
        easter + timedelta(days=1),  # Lundi de Pâques
        date(year, 5, 1),            # Fête du Travail
        date(year, 5, 8),            # Victoire 1945
        easter + timedelta(days=39), # Ascension
        easter + timedelta(days=50), # Lundi de Pentecôte
        date(year, 7, 14),           # Fête Nationale
        date(year, 8, 15),           # Assomption
        date(year, 11, 1),           # Toussaint
        date(year, 11, 11),          # Armistice 1918
        date(year, 12, 25),          # Noël
    }
    return frozenset(holidays)


def is_french_holiday(d: date) -> bool:
    """Vrai si ``d`` est un jour férié français métropolitain."""
    return d in french_holidays(d.year)


def is_eve_of_holiday(d: date) -> bool:
    """Vrai si le lendemain de ``d`` est un jour férié français."""
    return is_french_holiday(d + timedelta(days=1))


def is_payday(d: date) -> bool:
    """Heuristique : jour de paie en France.

    Vrai pour les fins de mois (jour >= 28) et autour du 5 du mois
    (jour 5 : versement typique des pensions / minima sociaux).
    """
    return d.day >= 28 or d.day == 5
