"""
Exemple d'utilisation du modèle CatBoost.

Ce script montre comment :
1. Entraîner le modèle
2. Faire des prédictions avec différents horizons
3. Évaluer les performances
4. Sauvegarder et charger le modèle

Usage:
    python examples/catboost_example.py
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Imports du projet
from src.models import CatBoostForecaster, train_and_evaluate_catboost
from src.utils import ColumnNames, get_file_path


def main():
    print("=" * 70)
    print("🚀 EXEMPLE D'UTILISATION DU MODÈLE CATBOOST")
    print("=" * 70)

    # 1. Charger les données enrichies
    print("\n📊 Chargement des données enrichies...")
    enriched_file = get_file_path('enriched')

    if not enriched_file.exists():
        print("❌ Données enrichies non trouvées.")
        print("   Exécutez d'abord : python main.py --step enrichment")
        return

    enriched_data = pd.read_csv(
        enriched_file,
        parse_dates=[ColumnNames.DATE],
        date_format='%Y-%m-%d'
    )
    print(f"   ✅ {len(enriched_data)} lignes chargées")

    # 2. Entraînement rapide
    print("\n🎯 Entraînement du modèle CatBoost...")
    model, results_by_horizon = train_and_evaluate_catboost(
        enriched_data,
        max_horizon=90,
        test_ratio=0.2
    )

    # 3. Afficher les résultats par horizon
    print("\n📊 PERFORMANCE PAR HORIZON :")
    print("-" * 50)
    print(results_by_horizon.to_string(index=False))

    # 4. Exemple de prédiction pour un article
    print("\n🔮 EXEMPLE DE PRÉDICTION :")
    print("-" * 50)

    # Prendre le premier article
    article_id = enriched_data[ColumnNames.ARTICLE_ID].iloc[0]
    last_date = enriched_data[ColumnNames.DATE].max()

    print(f"   Article : {article_id}")
    print(f"   Dernière date connue : {last_date.date()}")

    # Prédire les 7 prochains jours
    predictions_7d = model.predict_horizon(
        article_id=article_id,
        start_date=last_date + timedelta(days=1),
        horizon_days=7,
        context_data=enriched_data
    )

    print(f"\n   📅 Prédictions pour les 7 prochains jours :")
    for _, row in predictions_7d.iterrows():
        print(f"      {row['date'].date()} (H+{row['horizon']:2d}) : {row['prediction']:.1f} unités")

    # Prédire le mois prochain
    predictions_30d = model.predict_horizon(
        article_id=article_id,
        start_date=last_date + timedelta(days=1),
        horizon_days=30,
        context_data=enriched_data
    )

    total_30d = predictions_30d['prediction'].sum()
    print(f"\n   📈 Total prévu sur 30 jours : {total_30d:.0f} unités")
    print(f"   📊 Moyenne journalière : {total_30d / 30:.1f} unités")

    # 5. Feature importance
    print("\n📊 IMPORTANCE DES FEATURES :")
    print("-" * 50)
    importance_df = model.get_feature_importance()

    for _, row in importance_df.head(10).iterrows():
        bar = "█" * int(row['importance'] / 5)
        print(f"   {row['feature']:25s} {bar} {row['importance']:.1f}")

    # 6. Sauvegarder le modèle
    print("\n💾 Sauvegarde du modèle...")
    model.save_model("data/output/catboost_model")
    print("   ✅ Modèle sauvegardé dans data/output/catboost_model/")

    # 7. Exemple de chargement
    print("\n📂 Test de chargement du modèle...")
    loaded_model = CatBoostForecaster.load_model("data/output/catboost_model")
    print("   ✅ Modèle rechargé avec succès")

    # Vérifier que les prédictions sont identiques
    pred_original = model.predict(
        article_id=article_id,
        prediction_dates=[last_date + timedelta(days=1)],
        context_data=enriched_data
    )[0]

    pred_loaded = loaded_model.predict(
        article_id=article_id,
        prediction_dates=[last_date + timedelta(days=1)],
        context_data=enriched_data
    )[0]

    print(f"   🔍 Vérification : original={pred_original:.2f}, chargé={pred_loaded:.2f}")

    print("\n" + "=" * 70)
    print("✅ EXEMPLE TERMINÉ AVEC SUCCÈS")
    print("=" * 70)


if __name__ == "__main__":
    main()