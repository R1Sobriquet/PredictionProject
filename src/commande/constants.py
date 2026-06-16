"""Re-export des constantes métier définies dans `src/utils/config.py`.

Ce module sert de point d'entrée unique pour le moteur de commande : importer
depuis `src.commande.constants` rend les dépendances explicites et évite les
imports croisés entre sous-modules.
"""

from ..utils import (
    CommandConfig,
    ColumnNames,
    COUPURES,
    SOLDES_BY_COUPURE,
    K7HS_BY_COUPURE,
    DMQ_BY_COUPURE,
    PREDICTIF_BY_COUPURE,
    NB_CASSETTES_BY_COUPURE,
    PREDICTIF_COLUMNS,
    DMQ_COLUMNS,
    NB_CASSETTES_COLUMNS,
)

__all__ = [
    'CommandConfig',
    'ColumnNames',
    'COUPURES',
    'SOLDES_BY_COUPURE',
    'K7HS_BY_COUPURE',
    'DMQ_BY_COUPURE',
    'PREDICTIF_BY_COUPURE',
    'NB_CASSETTES_BY_COUPURE',
    'PREDICTIF_COLUMNS',
    'DMQ_COLUMNS',
    'NB_CASSETTES_COLUMNS',
]
