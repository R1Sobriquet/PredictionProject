"""
Tests pour data_processing.py (enrichissement ATM)

3 tests essentiels :
1. Enrichissement fonctionne de bout en bout
2. Variables temporelles et historiques ATM ajoutées
3. Qualité des calculs (days_since_last_order, last_order_amount)

Usage:
    pytest tests/test_data_processing.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing import DataEnrichmentPipeline
from src.utils.config import ColumnNames


def _make_clean_data():
    """Crée des données nettoyées (sortie de data_ingestion) pour les tests."""
    return pd.DataFrame({
        ColumnNames.ORDER_ID: [1, 2, 3, 4, 5, 6],
        ColumnNames.ATM_ID: [101, 101, 101, 102, 102, 102],
        ColumnNames.ORDER_DATE: [
            datetime(2026, 1, 5),   # Lundi
            datetime(2026, 1, 8),   # Jeudi (3 jours après)
            datetime(2026, 1, 12),  # Lundi (4 jours après)
            datetime(2026, 1, 6),   # Mardi
            datetime(2026, 1, 10),  # Samedi (4 jours après)
            datetime(2026, 1, 15),  # Jeudi (5 jours après)
        ],
        ColumnNames.AMOUNT: [10000.0, 15000.0, 12000.0, 8000.0, 9000.0, 11000.0],
        ColumnNames.ORDER_TYPE: ['standard'] * 6,
        ColumnNames.DELIVERY_DATE: pd.to_datetime([
            '2026-01-06', '2026-01-09', '2026-01-13',
            '2026-01-07', '2026-01-12', '2026-01-16',
        ]),
        ColumnNames.LOADING_DATE: pd.to_datetime([
            '2026-01-07', '2026-01-10', '2026-01-14',
            '2026-01-08', '2026-01-13', '2026-01-17',
        ]),
        ColumnNames.CASSETTE_1: [50, 60, 55, 40, 45, 50],
        ColumnNames.CASSETTE_2: [30, 35, 32, 25, 28, 30],
        ColumnNames.CASSETTE_3: [0, 0, 0, 20, 22, 25],
        ColumnNames.CASSETTE_4: [0, 0, 0, 0, 0, 0],
        ColumnNames.CASSETTE_5: [0, 0, 0, 0, 0, 0],
        ColumnNames.SOLDES_5: [100, 200, 150, 80, 90, 110],
        ColumnNames.SOLDES_10: [200, 300, 250, 160, 180, 220],
        ColumnNames.SOLDES_20: [500, 600, 550, 400, 450, 500],
        ColumnNames.SOLDES_50: [1000, 1200, 1100, 800, 900, 1000],
        ColumnNames.SOLDES_100: [2000, 2500, 2200, 1600, 1800, 2000],
        ColumnNames.K7HS_5: [0, 0, 0, 0, 0, 0],
        ColumnNames.K7HS_10: [0, 0, 0, 0, 0, 0],
        ColumnNames.K7HS_20: [0, 0, 0, 0, 0, 0],
        ColumnNames.K7HS_50: [0, 0, 0, 0, 0, 0],
        ColumnNames.K7HS_100: [0, 0, 0, 0, 0, 0],
        ColumnNames.VOLATILITE_DMQ: [0.5, 0.6, 0.55, 0.4, 0.45, 0.5],
        ColumnNames.DMQ_FORTE_DECROISSANCE: [0, 0, 0, 0, 0, 0],
        ColumnNames.DMQ_FORTE_CROISSANCE: [0, 0, 0, 0, 0, 0],
        ColumnNames.EVENEMENT_EN_COURS: [0, 0, 0, 0, 0, 0],
        ColumnNames.RISQUE_ATM_VIDE: [0, 0, 0, 0, 0, 0],
        ColumnNames.CHARGE: [1, 1, 1, 1, 1, 1],
        ColumnNames.ANNULE: [0, 0, 0, 0, 0, 0],
    })


def test_enrichment_works_basic():
    """
    Test 1 : CAS NORMAL

    Vérifie que le pipeline d'enrichissement fonctionne de bout en bout.
    """
    clean_data = _make_clean_data()

    pipeline = DataEnrichmentPipeline()
    pipeline.clean_data = clean_data
    result = pipeline.run_full_enrichment(save_output=False, save_snapshots=False)

    assert result is not None
    assert len(result) == len(clean_data), "Nombre de lignes doit être conservé"

    # Colonnes de base toujours présentes
    assert ColumnNames.ORDER_DATE in result.columns
    assert ColumnNames.ATM_ID in result.columns
    assert ColumnNames.AMOUNT in result.columns

    # Plus de colonnes qu'avant
    assert len(result.columns) > len(clean_data.columns)

    print(f"Test réussi : {len(clean_data.columns)} -> {len(result.columns)} colonnes")


def test_enrichment_adds_variables():
    """
    Test 2 : VARIABLES AJOUTÉES

    Vérifie que toutes les features attendues sont présentes :
    - Temporelles (weekday, is_weekend, etc.)
    - Historiques ATM (days_since_last_order, last_order_amount, etc.)
    - Agrégées (total_soldes, cassettes_actives, etc.)
    - Saisonnières (day_of_year, sin/cos)
    """
    clean_data = _make_clean_data()

    pipeline = DataEnrichmentPipeline()
    pipeline.clean_data = clean_data
    result = pipeline.run_full_enrichment(save_output=False, save_snapshots=False)

    # Variables temporelles
    temporal_cols = [
        ColumnNames.YEAR, ColumnNames.MONTH, ColumnNames.DAY,
        ColumnNames.WEEKDAY, ColumnNames.WEEKDAY_NAME,
        ColumnNames.IS_WEEKEND, ColumnNames.WEEK_NUMBER,
    ]
    for col in temporal_cols:
        assert col in result.columns, f"Colonne temporelle {col} manquante"

    # Variables historiques ATM
    history_cols = [
        ColumnNames.DAYS_SINCE_LAST_ORDER,
        ColumnNames.LAST_ORDER_AMOUNT,
        ColumnNames.AVG_RELOAD_FREQUENCY,
        ColumnNames.AVG_ORDER_AMOUNT,
        ColumnNames.STD_ORDER_AMOUNT,
        ColumnNames.ORDER_COUNT_LAST_30D,
    ]
    for col in history_cols:
        assert col in result.columns, f"Colonne historique {col} manquante"

    # Variables agrégées ATM
    assert 'total_soldes' in result.columns
    assert 'cassettes_actives' in result.columns
    assert 'delivery_delay' in result.columns
    assert 'loading_delay' in result.columns

    # Variables saisonnières
    assert 'day_of_year' in result.columns
    assert 'quarter' in result.columns
    assert 'day_of_year_sin' in result.columns

    print("Test réussi : toutes les features sont présentes")


def test_enrichment_output_quality():
    """
    Test 3 : QUALITÉ DES CALCULS

    Vérifie la précision des calculs historiques ATM :
    - days_since_last_order est correct
    - last_order_amount est correct
    - Les calculs sont faits PAR ATM
    """
    clean_data = _make_clean_data()

    pipeline = DataEnrichmentPipeline()
    pipeline.clean_data = clean_data
    result = pipeline.run_full_enrichment(save_output=False, save_snapshots=False)

    # Trier comme le pipeline le fait
    result = result.sort_values([ColumnNames.ATM_ID, ColumnNames.ORDER_DATE]).reset_index(drop=True)

    # ATM 101 : commandes le 5, 8, 12 janvier
    atm101 = result[result[ColumnNames.ATM_ID] == 101].reset_index(drop=True)

    # Première commande : days_since_last = 0 (pas de précédent)
    assert atm101.iloc[0][ColumnNames.DAYS_SINCE_LAST_ORDER] == 0, \
        "Première commande : days_since_last_order doit être 0"

    # Deuxième commande (8 jan) : 3 jours après le 5 jan
    assert atm101.iloc[1][ColumnNames.DAYS_SINCE_LAST_ORDER] == 3, \
        f"ATM 101, cmd 2 : days_since_last = 3, obtenu {atm101.iloc[1][ColumnNames.DAYS_SINCE_LAST_ORDER]}"

    # Troisième commande (12 jan) : 4 jours après le 8 jan
    assert atm101.iloc[2][ColumnNames.DAYS_SINCE_LAST_ORDER] == 4, \
        f"ATM 101, cmd 3 : days_since_last = 4, obtenu {atm101.iloc[2][ColumnNames.DAYS_SINCE_LAST_ORDER]}"

    # last_order_amount : première commande = 0, deuxième = montant de la première
    assert atm101.iloc[0][ColumnNames.LAST_ORDER_AMOUNT] == 0, \
        "Première commande : last_order_amount doit être 0"

    assert atm101.iloc[1][ColumnNames.LAST_ORDER_AMOUNT] == 10000.0, \
        f"ATM 101, cmd 2 : last_order_amount = 10000, obtenu {atm101.iloc[1][ColumnNames.LAST_ORDER_AMOUNT]}"

    assert atm101.iloc[2][ColumnNames.LAST_ORDER_AMOUNT] == 15000.0, \
        f"ATM 101, cmd 3 : last_order_amount = 15000, obtenu {atm101.iloc[2][ColumnNames.LAST_ORDER_AMOUNT]}"

    # Vérification PAR ATM (ATM 102 séparé)
    atm102 = result[result[ColumnNames.ATM_ID] == 102].reset_index(drop=True)
    assert atm102.iloc[0][ColumnNames.DAYS_SINCE_LAST_ORDER] == 0, \
        "ATM 102 : première commande indépendante de ATM 101"

    # Delivery delay : delivery_date - order_date
    assert atm101.iloc[0]['delivery_delay'] == 1, \
        "ATM 101, cmd 1 : delivery_delay = 1 jour"

    # Cassettes actives : ATM 101 a cassettes 1 et 2 actives (3,4,5 = 0)
    assert atm101.iloc[0]['cassettes_actives'] == 2, \
        f"ATM 101 : 2 cassettes actives, obtenu {atm101.iloc[0]['cassettes_actives']}"

    print("Test réussi : calculs historiques ATM corrects")


if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])
