"""
Tests pour data_ingestion.py (pipeline ATM)

3 tests essentiels :
1. Cas normal (pipeline fonctionne avec données ATM)
2. Cas d'erreur (commandes annulées, montants invalides)
3. Qualité finale (déduplication, colonnes standardisées)

Usage:
    pytest tests/test_data_ingestion.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_ingestion import DataIngestionPipeline
from src.utils.config import ColumnNames, COLUMNS_TO_LOAD


def _create_test_excel(tmp_path, data: pd.DataFrame, filename="test.xlsx") -> Path:
    """Helper : crée un fichier Excel de test avec les colonnes DC_*."""
    filepath = tmp_path / filename
    data.to_excel(filepath, index=False, engine='openpyxl')
    return filepath


def _make_base_data(n_rows=10):
    """Crée des données ATM de base avec les colonnes DC_* requises."""
    data = {
        'DC_Commande_Id': list(range(1, n_rows + 1)),
        'DC_Automate_Id': [101, 102] * (n_rows // 2) + [101] * (n_rows % 2),
        'DC_Date_Cmd': pd.date_range('2026-01-01', periods=n_rows, freq='D'),
        'DC_Type_Cmd': ['standard'] * n_rows,
        'DC_Montant_Cmd': np.random.randint(5000, 50000, n_rows).astype(float),
        'DC_Livraison_Prev_Date': pd.date_range('2026-01-02', periods=n_rows, freq='D'),
        'DC_Chargement_Prev_Date': pd.date_range('2026-01-03', periods=n_rows, freq='D'),
        'DC_Cassette_1': np.random.randint(0, 100, n_rows),
        'DC_Cassette_2': np.random.randint(0, 100, n_rows),
        'DC_Cassette_3': np.random.randint(0, 100, n_rows),
        'DC_Cassette_4': np.random.randint(0, 100, n_rows),
        'DC_Cassette_5': np.random.randint(0, 100, n_rows),
        'DC_Ajustement_5': np.random.randint(0, 500, n_rows),
        'DC_Ajustement_10': np.random.randint(0, 500, n_rows),
        'DC_Ajustement_20': np.random.randint(0, 500, n_rows),
        'DC_Ajustement_50': np.random.randint(0, 500, n_rows),
        'DC_Ajustement_100': np.random.randint(0, 500, n_rows),
        'DC_SoldesDuJour_5': np.random.randint(0, 1000, n_rows),
        'DC_SoldesDuJour_10': np.random.randint(0, 1000, n_rows),
        'DC_SoldesDuJour_20': np.random.randint(0, 1000, n_rows),
        'DC_SoldesDuJour_50': np.random.randint(0, 1000, n_rows),
        'DC_SoldesDuJour_100': np.random.randint(0, 1000, n_rows),
        'DC_K7HS_5': [0] * n_rows,
        'DC_K7HS_10': [0] * n_rows,
        'DC_K7HS_20': [0] * n_rows,
        'DC_K7HS_50': [0] * n_rows,
        'DC_K7HS_100': [0] * n_rows,
        'DC_VolatiliteDmq': np.random.uniform(0, 1, n_rows).round(2),
        'DC_DmqForteDecroissance': [0] * n_rows,
        'DC_DmqForteCroissance': [0] * n_rows,
        'DC_Annule': [0] * n_rows,
        'DC_Chargé': [1] * n_rows,
        'DC_EvenementEnCour': [0] * n_rows,
        'DC_RisqueAutomateVide': [0] * n_rows,
    }
    return pd.DataFrame(data)


def test_pipeline_works_basic(tmp_path):
    """
    Test 1 : CAS NORMAL

    Vérifie que le pipeline fonctionne avec des données ATM valides.
    """
    data = _make_base_data(10)
    filepath = _create_test_excel(tmp_path, data)

    pipeline = DataIngestionPipeline(filepath)
    result = pipeline.run_full_pipeline(save_snapshots=False)

    # Vérifications de base
    assert result is not None
    assert len(result) > 0

    # Colonnes standardisées présentes
    assert ColumnNames.ORDER_ID in result.columns
    assert ColumnNames.ATM_ID in result.columns
    assert ColumnNames.ORDER_DATE in result.columns
    assert ColumnNames.AMOUNT in result.columns

    # Pas de valeurs manquantes sur les colonnes critiques
    assert result[ColumnNames.ORDER_ID].isna().sum() == 0
    assert result[ColumnNames.ATM_ID].isna().sum() == 0
    assert result[ColumnNames.AMOUNT].isna().sum() == 0

    # Types corrects
    assert pd.api.types.is_datetime64_any_dtype(result[ColumnNames.ORDER_DATE])
    assert pd.api.types.is_float_dtype(result[ColumnNames.AMOUNT])

    print(f"Test réussi : {len(result)} commandes ATM ingérées")


def test_pipeline_handles_errors(tmp_path):
    """
    Test 2 : CAS D'ERREUR

    Vérifie que le pipeline gère :
    - Commandes annulées (DC_Annule = 1) → exclues
    - Montants négatifs ou aberrants → exclus
    - Doublons sur DC_Commande_Id → dédupliqués
    """
    data = _make_base_data(8)

    # Commande annulée
    data.loc[0, 'DC_Annule'] = 1

    # Montant négatif
    data.loc[1, 'DC_Montant_Cmd'] = -500.0

    # Montant aberrant (> 500 000)
    data.loc[2, 'DC_Montant_Cmd'] = 999999.0

    # Doublon sur DC_Commande_Id
    data.loc[7, 'DC_Commande_Id'] = data.loc[3, 'DC_Commande_Id']

    filepath = _create_test_excel(tmp_path, data)

    pipeline = DataIngestionPipeline(filepath)
    result = pipeline.run_full_pipeline(save_snapshots=False)

    # Pas de commandes annulées
    if ColumnNames.ANNULE in result.columns:
        assert (result[ColumnNames.ANNULE] == 1).sum() == 0

    # Pas de montants négatifs
    assert all(result[ColumnNames.AMOUNT] >= 0)

    # Pas de montants aberrants
    assert all(result[ColumnNames.AMOUNT] <= 500000)

    # Pas de doublons sur order_id
    assert result.duplicated(subset=[ColumnNames.ORDER_ID]).sum() == 0

    # Au moins 4 lignes valides restantes (8 - 1 annulée - 1 négatif - 1 aberrant - 1 doublon)
    assert len(result) >= 4

    print(f"Test réussi : erreurs gérées, {len(result)} commandes valides")


def test_pipeline_output_quality(tmp_path):
    """
    Test 3 : QUALITÉ FINALE

    Vérifie la structure du résultat :
    - Colonnes standardisées (noms internes, pas DC_*)
    - Commandes individuelles préservées (pas d'agrégation)
    - Tri par date + ATM
    """
    data = _make_base_data(10)
    filepath = _create_test_excel(tmp_path, data)

    pipeline = DataIngestionPipeline(filepath)
    result = pipeline.run_full_pipeline(save_snapshots=False)

    # Les colonnes source DC_* ne doivent plus être là (standardisées)
    dc_columns = [c for c in result.columns if c.startswith('DC_')]
    assert len(dc_columns) == 0, f"Colonnes DC_* non standardisées : {dc_columns}"

    # Commandes individuelles préservées
    assert len(result) == 10, f"Toutes les commandes doivent être préservées, obtenu {len(result)}"

    # Tri par date
    dates = result[ColumnNames.ORDER_DATE].values
    assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1)), \
        "Les données doivent être triées par date"

    # Résumé
    summary = pipeline.get_data_summary()
    assert summary['total_orders'] == 10
    assert summary['unique_atms'] >= 1
    assert summary['total_amount'] > 0

    print(f"Test réussi : structure finale correcte ({len(result)} commandes, "
          f"{summary['unique_atms']} ATMs)")


if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])
