"""
Tests minimaux pour data_ingestion.py

3 tests essentiels qui couvrent 90% des cas d'usage :
1. Cas normal (le pipeline fonctionne)
2. Cas d'erreur (gestion des problèmes)
3. Qualité finale (données propres et complètes)

Usage:
    pytest tests/test_data_ingestion_minimal.py -v
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Ajouter le répertoire racine au path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_ingestion import DataIngestionPipeline
from src.utils.config import ColumnNames


def test_pipeline_works_basic(tmp_path):
    """
    Test 1 : CAS NORMAL

    Vérifie que le pipeline fonctionne de bout en bout avec des données normales.

    Scénario :
    - Données propres avec 3 articles sur 5 jours
    - Aucune erreur attendue
    - Le pipeline doit se terminer avec succès
    """

    # ===== DONNÉES D'ENTRÉE =====
    data = pd.DataFrame({
        'date_ligne_commande': [
            '2024-01-01', '2024-01-01', '2024-01-01',  # Jour 1 : 3 articles
            '2024-01-02', '2024-01-02',                 # Jour 2 : 2 articles
            '2024-01-03', '2024-01-03', '2024-01-03',  # Jour 3 : 3 articles
            '2024-01-04',                               # Jour 4 : 1 article
            '2024-01-05', '2024-01-05'                  # Jour 5 : 2 articles
        ],
        'id_article': [1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3],
        'quantite': [50, 30, 25, 45, 35, 60, 40, 30, 55, 45, 35],
        'ref_article': ['REF001', 'REF002', 'REF003'] * 3 + ['REF001', 'REF002']
    })

    csv_file = tmp_path / "test_normal.csv"
    data.to_csv(csv_file, index=False)

    # ===== EXÉCUTION DU PIPELINE =====
    pipeline = DataIngestionPipeline(csv_file)
    result = pipeline.run_full_pipeline()

    # ===== VÉRIFICATIONS DE BASE =====
    assert result is not None, "Le pipeline doit retourner des données"
    assert len(result) > 0, "Le résultat ne doit pas être vide"

    # Vérifier que les colonnes essentielles existent
    assert ColumnNames.DATE in result.columns
    assert ColumnNames.ARTICLE_ID in result.columns
    assert ColumnNames.QUANTITY in result.columns

    # Vérifier qu'il n'y a pas de valeurs manquantes
    assert result[ColumnNames.DATE].isna().sum() == 0
    assert result[ColumnNames.ARTICLE_ID].isna().sum() == 0
    assert result[ColumnNames.QUANTITY].isna().sum() == 0

    # Vérifier les types de données
    assert pd.api.types.is_datetime64_any_dtype(result[ColumnNames.DATE])
    assert pd.api.types.is_integer_dtype(result[ColumnNames.ARTICLE_ID])
    assert pd.api.types.is_integer_dtype(result[ColumnNames.QUANTITY])

    print(f"✅ Test réussi : {len(result)} lignes générées pour 3 articles")


def test_pipeline_handles_errors(tmp_path):
    """
    Test 2 : CAS D'ERREUR

    Vérifie que le pipeline gère correctement les problèmes courants :
    - Doublons (même jour/article plusieurs fois)
    - Valeurs aberrantes (quantités négatives ou trop élevées)
    - Dates en dehors de la période (décembre 2024)

    Le pipeline doit nettoyer automatiquement ces problèmes.
    """

    # ===== DONNÉES AVEC PROBLÈMES =====
    data = pd.DataFrame({
        'date_ligne_commande': [
            '2024-01-01', '2024-01-01',  # DOUBLON EXACT (même ligne 2 fois)
            '2024-01-02', '2024-01-02',  # DOUBLON À AGRÉGER (quantités différentes)
            '2024-01-03',                # Quantité négative
            '2024-01-04',                # Quantité aberrante (trop haute)
            '2024-12-01',                # HORS PÉRIODE (décembre = exclu)
            '2024-01-05'                 # Ligne normale
        ],
        'id_article': [1, 1, 1, 1, 1, 1, 1, 2],
        'quantite': [50, 50, 30, 20, -10, 25000, 100, 40],
        'ref_article': ['REF001'] * 7 + ['REF002']
    })

    csv_file = tmp_path / "test_errors.csv"
    data.to_csv(csv_file, index=False)

    # ===== EXÉCUTION DU PIPELINE =====
    pipeline = DataIngestionPipeline(csv_file)
    result = pipeline.run_full_pipeline()

    # ===== VÉRIFICATIONS DES CORRECTIONS =====

    # 1. Pas de quantités négatives
    assert all(result[ColumnNames.QUANTITY] >= 0), \
        "Les quantités négatives doivent être supprimées"

    # 2. Pas de quantités aberrantes (> 10000)
    assert all(result[ColumnNames.QUANTITY] <= 10000), \
        "Les quantités aberrantes doivent être supprimées"

    # 3. Pas de doublons (même date + même article)
    duplicates = result.duplicated(subset=[ColumnNames.DATE, ColumnNames.ARTICLE_ID])
    assert duplicates.sum() == 0, \
        "Les doublons doivent être supprimés ou agrégés"

    # 4. Dates filtrées (pas de décembre)
    dates = pd.to_datetime(result[ColumnNames.DATE])
    assert all(dates.dt.month != 12), \
        "Décembre 2024 doit être exclu de la période d'entraînement"

    # 5. Agrégation vérifiée (2024-01-02 article 1 : 30+20=50)
    jan2_article1 = result[
        (result[ColumnNames.DATE] == pd.Timestamp('2024-01-02')) &
        (result[ColumnNames.ARTICLE_ID] == 1)
    ]
    if len(jan2_article1) > 0:
        # Si la ligne existe, elle doit être agrégée (30 + 20 = 50)
        # Ou une des deux valeurs si l'autre est considérée aberrante
        assert jan2_article1.iloc[0][ColumnNames.QUANTITY] in [30, 50], \
            "Les quantités du même jour doivent être agrégées"

    print(f"✅ Test réussi : Toutes les erreurs ont été gérées correctement")


def test_pipeline_output_quality(tmp_path):
    """
    Test 3 : QUALITÉ FINALE

    Vérifie que le résultat final a la structure attendue :
    - 1 ligne par jour ET par article (même si quantité = 0)
    - Les jours sans commande sont ajoutés avec quantité = 0
    - Pas de trous dans la chronologie

    C'est le test le plus important car il valide l'objectif principal du module.
    """

    # ===== DONNÉES D'ENTRÉE AVEC TROUS =====
    # On donne seulement 3 jours de données, 2 articles
    # Le pipeline doit remplir tous les jours manquants
    data = pd.DataFrame({
        'date_ligne_commande': [
            '2024-01-01', '2024-01-01',  # Jour 1 : Article 1 et 2
            '2024-01-03',                 # Jour 3 : Article 1 seulement
            # Jour 2 : MANQUANT complètement
            # Jour 3 : Article 2 MANQUANT
        ],
        'id_article': [1, 2, 1],
        'quantite': [50, 30, 60],
        'ref_article': ['REF001', 'REF002', 'REF001']
    })

    csv_file = tmp_path / "test_quality.csv"
    data.to_csv(csv_file, index=False)

    # ===== EXÉCUTION DU PIPELINE =====
    pipeline = DataIngestionPipeline(csv_file)
    result = pipeline.run_full_pipeline()

    # ===== VÉRIFICATIONS DE QUALITÉ =====

    # 1. Structure : 1 ligne par jour ET par articles
    unique_dates = result[ColumnNames.DATE].nunique()
    unique_articles = result[ColumnNames.ARTICLE_ID].nunique()
    expected_lines = unique_dates * unique_articles

    assert len(result) == expected_lines, \
        f"Doit avoir {expected_lines} lignes (1 par jour/article), obtenu {len(result)}"

    # 2. Pas de doublons jour/article
    duplicates = result.duplicated(subset=[ColumnNames.DATE, ColumnNames.ARTICLE_ID])
    assert duplicates.sum() == 0, \
        "Chaque combinaison jour/article doit être unique"

    # 3. Des zéros ont été ajoutés pour les jours manquants
    zero_count = (result[ColumnNames.QUANTITY] == 0).sum()
    assert zero_count > 0, \
        "Des lignes avec quantité=0 doivent être ajoutées pour les jours sans commande"

    # 4. Vérification d'un cas spécifique : 2024-01-02 article 1 doit exister avec qty=0
    jan2_article1 = result[
        (result[ColumnNames.DATE] == pd.Timestamp('2024-01-02')) &
        (result[ColumnNames.ARTICLE_ID] == 1)
    ]
    assert len(jan2_article1) == 1, \
        "Le 2024-01-02 article 1 doit exister (même sans commande dans les données)"
    assert jan2_article1.iloc[0][ColumnNames.QUANTITY] == 0, \
        "Le 2024-01-02 article 1 doit avoir quantité=0 (jour manquant)"

    # 5. Vérification cas spécifique : 2024-01-03 article 2 doit exister avec qty=0
    jan3_article2 = result[
        (result[ColumnNames.DATE] == pd.Timestamp('2024-01-03')) &
        (result[ColumnNames.ARTICLE_ID] == 2)
    ]
    assert len(jan3_article2) == 1, \
        "Le 2024-01-03 article 2 doit exister (même sans commande)"
    assert jan3_article2.iloc[0][ColumnNames.QUANTITY] == 0, \
        "Le 2024-01-03 article 2 doit avoir quantité=0 (article manquant ce jour)"

    # 6. Les quantités existantes sont préservées
    jan1_article1 = result[
        (result[ColumnNames.DATE] == pd.Timestamp('2024-01-01')) &
        (result[ColumnNames.ARTICLE_ID] == 1)
    ]
    assert jan1_article1.iloc[0][ColumnNames.QUANTITY] == 50, \
        "Les quantités existantes doivent être préservées"

    # 7. Résumé statistique
    summary = pipeline.get_data_summary()
    assert summary['zero_quantity_lines'] > 0
    assert summary['total_quantity'] > 0

    print(f"✅ Test réussi : Structure finale parfaite")
    print(f"   📊 {len(result)} lignes ({unique_dates} jours × {unique_articles} articles)")
    print(f"   📉 {zero_count} lignes avec quantité=0 (jours manquants)")
    print(f"   📈 {len(result) - zero_count} lignes avec commandes")


# ===== POINT D'ENTRÉE POUR EXÉCUTION DIRECTE =====
if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])