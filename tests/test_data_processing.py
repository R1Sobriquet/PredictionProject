"""
Tests minimaux pour data_processing.py

3 tests essentiels qui couvrent 90% des cas d'usage :
1. Cas normal (l'enrichissement fonctionne)
2. Variables ajoutées (toutes les colonnes temporelles sont là)
3. Qualité des calculs (quantity_prev_day correct)

Usage:
    pytest tests/test_data_processing_minimal.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Ajouter le répertoire racine au path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing import DataEnrichmentPipeline
from src.utils.config import ColumnNames, WEEKDAY_NAMES


def test_enrichment_works_basic(tmp_path):
    """
    Test 1 : CAS NORMAL

    Vérifie que le pipeline d'enrichissement fonctionne de bout en bout.

    Scénario :
    - Données nettoyées (1 ligne par jour/article)
    - Enrichissement avec variables temporelles
    - Le pipeline doit se terminer avec succès
    """

    # ===== DONNÉES D'ENTRÉE NETTOYÉES =====
    # Format : sortie de data_ingestion.py
    dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')

    data = []
    for date in dates:
        for article_id in [1, 2]:
            data.append({
                ColumnNames.DATE: date,
                ColumnNames.ARTICLE_ID: article_id,
                ColumnNames.QUANTITY: np.random.randint(20, 60)
            })

    clean_data = pd.DataFrame(data)

    # Sauvegarder dans un fichier temporaire
    csv_file = tmp_path / "clean_data.csv"
    clean_data.to_csv(csv_file, index=False)

    # ===== EXÉCUTION DU PIPELINE =====
    pipeline = DataEnrichmentPipeline()
    pipeline.clean_data = clean_data  # Injection directe des données
    result = pipeline.run_full_enrichment(save_output=False)

    # ===== VÉRIFICATIONS DE BASE =====
    assert result is not None, "Le pipeline doit retourner des données"
    assert len(result) == len(clean_data), "Le nombre de lignes doit être conservé"

    # Vérifier que les colonnes de base sont toujours là
    assert ColumnNames.DATE in result.columns
    assert ColumnNames.ARTICLE_ID in result.columns
    assert ColumnNames.QUANTITY in result.columns

    # Vérifier qu'il y a plus de colonnes qu'avant (enrichissement)
    original_columns = len(clean_data.columns)
    enriched_columns = len(result.columns)
    assert enriched_columns > original_columns, \
        f"Doit avoir plus de colonnes après enrichissement ({original_columns} → {enriched_columns})"

    # Vérifier qu'il n'y a pas de valeurs manquantes dans les colonnes temporelles
    assert result[ColumnNames.WEEKDAY].isna().sum() == 0
    assert result[ColumnNames.WEEKDAY_NAME].isna().sum() == 0

    print(f"✅ Test réussi : {original_columns} colonnes → {enriched_columns} colonnes enrichies")


def test_enrichment_adds_variables(tmp_path):
    """
    Test 2 : VARIABLES AJOUTÉES

    Vérifie que toutes les variables temporelles et de retard sont ajoutées :
    - Variables temporelles : weekday, weekday_name, is_weekend, month, etc.
    - Variables de retard : quantity_prev_day (quantité du jour précédent)
    - Moyennes mobiles : quantity_rolling_mean_7d, quantity_rolling_mean_30d

    C'est le test le plus important car il valide l'objectif du module.
    """

    # ===== DONNÉES D'ENTRÉE AVEC PATTERNS CONNUS =====
    data = pd.DataFrame({
        ColumnNames.DATE: [
            datetime(2024, 1, 1),   # Lundi
            datetime(2024, 1, 2),   # Mardi
            datetime(2024, 1, 3),   # Mercredi
            datetime(2024, 1, 4),   # Jeudi
            datetime(2024, 1, 5),   # Vendredi
            datetime(2024, 1, 6),   # Samedi (weekend)
            datetime(2024, 1, 7),   # Dimanche (weekend)
        ],
        ColumnNames.ARTICLE_ID: [1, 1, 1, 1, 1, 1, 1],
        ColumnNames.QUANTITY: [50, 45, 60, 55, 50, 30, 20]
    })

    # ===== EXÉCUTION DU PIPELINE =====
    pipeline = DataEnrichmentPipeline()
    pipeline.clean_data = data
    result = pipeline.run_full_enrichment(save_output=False)

    # ===== VÉRIFICATION DES VARIABLES TEMPORELLES =====

    # 1. Variables temporelles de base
    required_temporal_columns = [
        ColumnNames.YEAR,           # year
        ColumnNames.MONTH,          # month
        ColumnNames.DAY,            # day
        ColumnNames.WEEKDAY,        # weekday (0-6)
        ColumnNames.WEEKDAY_NAME,   # weekday_name (Lundi, Mardi...)
        ColumnNames.IS_WEEKEND,     # is_weekend (True/False)
        ColumnNames.WEEK_NUMBER     # week_number
    ]

    for col in required_temporal_columns:
        assert col in result.columns, f"Colonne {col} doit être ajoutée"

    print("✅ Toutes les variables temporelles sont présentes")

    # 2. Vérification des valeurs : jour de la semaine
    # 2024-01-01 = Lundi (weekday=0)
    assert result.iloc[0][ColumnNames.WEEKDAY] == 0, \
        "2024-01-01 doit être un lundi (weekday=0)"
    assert result.iloc[0][ColumnNames.WEEKDAY_NAME] == "Lundi", \
        "2024-01-01 doit afficher 'Lundi'"

    # 3. Vérification des valeurs : weekend
    # 2024-01-06 = Samedi (weekend)
    assert result.iloc[5][ColumnNames.IS_WEEKEND] == True, \
        "2024-01-06 (samedi) doit être marqué comme weekend"
    assert result.iloc[6][ColumnNames.IS_WEEKEND] == True, \
        "2024-01-07 (dimanche) doit être marqué comme weekend"

    # 2024-01-01 = Lundi (pas weekend)
    assert result.iloc[0][ColumnNames.IS_WEEKEND] == False, \
        "2024-01-01 (lundi) ne doit pas être marqué comme weekend"

    print("✅ Valeurs des jours de semaine et weekend correctes")

    # ===== VÉRIFICATION DES VARIABLES DE RETARD =====

    # 4. Variable de retard : quantity_prev_day
    assert ColumnNames.QUANTITY_PREV_DAY in result.columns, \
        "Colonne quantity_prev_day doit être ajoutée"

    # Vérification manuelle des valeurs
    # Jour 1 : quantity_prev_day = 0 (pas de jour précédent)
    assert result.iloc[0][ColumnNames.QUANTITY_PREV_DAY] == 0, \
        "Premier jour : quantity_prev_day doit être 0"

    # Jour 2 : quantity_prev_day = 50 (quantité du jour 1)
    assert result.iloc[1][ColumnNames.QUANTITY_PREV_DAY] == 50, \
        f"Jour 2 : quantity_prev_day doit être 50 (qty jour 1), obtenu {result.iloc[1][ColumnNames.QUANTITY_PREV_DAY]}"

    # Jour 3 : quantity_prev_day = 45 (quantité du jour 2)
    assert result.iloc[2][ColumnNames.QUANTITY_PREV_DAY] == 45, \
        f"Jour 3 : quantity_prev_day doit être 45 (qty jour 2), obtenu {result.iloc[2][ColumnNames.QUANTITY_PREV_DAY]}"

    print("✅ Variable quantity_prev_day calculée correctement")

    # ===== VÉRIFICATION DES MOYENNES MOBILES =====

    # 5. Moyennes mobiles
    rolling_7d = f"{ColumnNames.QUANTITY}_rolling_mean_7d"
    rolling_30d = f"{ColumnNames.QUANTITY}_rolling_mean_30d"

    assert rolling_7d in result.columns, \
        "Colonne rolling_mean_7d doit être ajoutée"
    assert rolling_30d in result.columns, \
        "Colonne rolling_mean_30d doit être ajoutée"

    # Vérification que ce sont des nombres valides
    assert not result[rolling_7d].isna().all(), \
        "rolling_mean_7d ne doit pas être que des NaN"

    print("✅ Moyennes mobiles calculées")

    # ===== VÉRIFICATION DES VARIABLES SAISONNIÈRES =====

    # 6. Variables saisonnières (optionnelles mais utiles)
    seasonal_columns = ['day_of_year', 'quarter', 'day_of_year_sin', 'day_of_year_cos']

    for col in seasonal_columns:
        if col in result.columns:
            print(f"   ✅ Variable saisonnière {col} présente")

    print(f"✅ Test réussi : {len(result.columns)} colonnes au total")


def test_enrichment_output_quality(tmp_path):
    """
    Test 3 : QUALITÉ DES CALCULS

    Vérifie la précision des calculs, notamment quantity_prev_day :
    - Pour chaque article, la quantité du jour précédent est correcte
    - Les calculs sont faits PAR ARTICLE (pas mélangés entre articles)
    - Les cas limites sont gérés (premier jour, changement d'article)
    """

    # ===== DONNÉES AVEC 2 ARTICLES POUR TESTER LA SÉPARATION =====
    data = pd.DataFrame({
        ColumnNames.DATE: [
            datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3),  # Article 1
            datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3),  # Article 2
        ],
        ColumnNames.ARTICLE_ID: [1, 1, 1, 2, 2, 2],
        ColumnNames.QUANTITY: [100, 200, 300, 50, 60, 70]
    })

    # ===== EXÉCUTION DU PIPELINE =====
    pipeline = DataEnrichmentPipeline()
    pipeline.clean_data = data
    result = pipeline.run_full_enrichment(save_output=False)

    # Trier pour avoir les données dans l'ordre
    result = result.sort_values([ColumnNames.ARTICLE_ID, ColumnNames.DATE]).reset_index(drop=True)

    # ===== VÉRIFICATIONS PAR ARTICLE =====

    # 1. ARTICLE 1 - Vérifications détaillées
    article1_data = result[result[ColumnNames.ARTICLE_ID] == 1].reset_index(drop=True)

    # Jour 1 Article 1 : quantity_prev_day = 0 (premier jour)
    assert article1_data.iloc[0][ColumnNames.QUANTITY_PREV_DAY] == 0, \
        "Article 1, Jour 1 : quantity_prev_day doit être 0 (pas de jour précédent)"

    # Jour 2 Article 1 : quantity_prev_day = 100 (qty jour 1 article 1)
    assert article1_data.iloc[1][ColumnNames.QUANTITY_PREV_DAY] == 100, \
        f"Article 1, Jour 2 : quantity_prev_day doit être 100, obtenu {article1_data.iloc[1][ColumnNames.QUANTITY_PREV_DAY]}"

    # Jour 3 Article 1 : quantity_prev_day = 200 (qty jour 2 article 1)
    assert article1_data.iloc[2][ColumnNames.QUANTITY_PREV_DAY] == 200, \
        f"Article 1, Jour 3 : quantity_prev_day doit être 200, obtenu {article1_data.iloc[2][ColumnNames.QUANTITY_PREV_DAY]}"

    print("✅ Article 1 : quantity_prev_day correct sur 3 jours")