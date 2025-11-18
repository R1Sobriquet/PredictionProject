"""
Script pour tester et comparer tous les modèles de baseline.

Ce script :
1. Charge les données enrichies
2. Entraîne les 7 modèles de baseline
3. Compare leurs prédictions pour décembre 2024
4. Affiche les résultats de manière claire

Usage:
    python test_all_models.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Ajouter le répertoire racine au path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.baseline import (
    NaiveBaseline,
    HistoricalMeanBaseline,
    WeekdayMeanBaseline,
    MovingAverageBaseline,
    SeasonalNaiveBaseline,
    TrendBaseline,
    create_baseline_suite
)
from src.utils.config import ColumnNames


def load_data():
    """Charge les données enrichies."""
    print("📊 Chargement des données enrichies...")

    enriched_file = Path('../data/processed/commandes_enriched.csv')

    if not enriched_file.exists():
        print(f"❌ Fichier non trouvé : {enriched_file}")
        print("\n💡 Lancez d'abord :")
        print("   python main.py --step enrichment")
        sys.exit(1)

    data = pd.read_csv(enriched_file, parse_dates=['date'])

    print(f"   ✅ Données chargées: {len(data)} lignes")
    print(f"   📦 Nombre d'articles: {data['article_id'].nunique()}")
    print(f"   📅 Période: {data['date'].min()} à {data['date'].max()}")

    return data


def test_single_model(model, data, article_id=1):
    """
    Teste un modèle spécifique et affiche ses prédictions.

    Args:
        model: Instance du modèle
        data: Données d'entraînement
        article_id: ID de l'article à prédire
    """
    print(f"\n{'='*60}")
    print(f"🎯 Test du modèle : {model.name}")
    print(f"{'='*60}")

    # Entraînement
    print("🎯 Entraînement du modèle...")
    model.fit(data)
    print("   ✅ Modèle entraîné")

    # Statistiques de l'article
    article_data = data[data['article_id'] == article_id]
    print(f"\n📊 Statistiques article {article_id}:")
    print(f"   Total jours : {len(article_data)}")
    print(f"   Jours avec commandes : {(article_data['quantity'] > 0).sum()}")
    print(f"   Quantité moyenne : {article_data['quantity'].mean():.2f}")
    print(f"   Quantité max : {article_data['quantity'].max()}")

    # Dernières valeurs
    print(f"\n📅 Dernières valeurs (5 derniers jours):")
    last_5 = article_data.tail(5)[['date', 'quantity']]
    for _, row in last_5.iterrows():
        print(f"      {row['date'].date()}: {row['quantity']}")

    # Prédictions pour décembre 2024
    prediction_dates = [datetime(2024, 12, d) for d in range(1, 8)]
    predictions = model.predict(article_id=article_id, prediction_dates=prediction_dates)

    print(f"\n🔮 Prédictions (7 premiers jours de décembre):")
    for date, pred in zip(prediction_dates, predictions):
        weekday = date.strftime('%A')  # Nom du jour en anglais
        weekday_fr = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'][date.weekday()]
        print(f"      📅 {date.date()} ({weekday_fr}): {pred:.2f}")

    print(f"\n📈 Résumé prédictions:")
    print(f"   Moyenne : {predictions.mean():.2f}")
    print(f"   Min : {predictions.min():.2f}")
    print(f"   Max : {predictions.max():.2f}")


def compare_all_models(data, article_id=1):
    """
    Compare tous les modèles sur le même article.

    Args:
        data: Données d'entraînement
        article_id: ID de l'article à prédire
    """
    print(f"\n{'='*60}")
    print(f"📊 COMPARAISON DE TOUS LES MODÈLES - Article {article_id}")
    print(f"{'='*60}")

    # Créer tous les modèles
    models = [
        NaiveBaseline(),
        HistoricalMeanBaseline(),
        WeekdayMeanBaseline(),
        MovingAverageBaseline(window=7),
        MovingAverageBaseline(window=30),
        SeasonalNaiveBaseline(),
        TrendBaseline(window_days=30)
    ]

    # Dates de prédiction
    prediction_dates = [datetime(2024, 12, d) for d in range(1, 8)]

    # Stocker les résultats
    results = []

    for model in models:
        print(f"\n🔄 Entraînement {model.name}...")

        try:
            # Entraîner
            model.fit(data)

            # Prédire
            predictions = model.predict(article_id=article_id, prediction_dates=prediction_dates)

            # Calculer statistiques
            result = {
                'model': model.name,
                'moyenne': predictions.mean(),
                'min': predictions.min(),
                'max': predictions.max(),
                'écart_type': predictions.std(),
                'predictions': predictions
            }
            results.append(result)

            print(f"   ✅ Moyenne prédite : {result['moyenne']:.2f}")

        except Exception as e:
            print(f"   ❌ Erreur : {e}")

    # Afficher le tableau comparatif
    print(f"\n{'='*80}")
    print(f"📊 TABLEAU COMPARATIF DES PRÉDICTIONS")
    print(f"{'='*80}")

    # En-tête
    print(f"{'Modèle':<25} {'Moyenne':<12} {'Min':<10} {'Max':<10} {'Écart-type':<12}")
    print(f"{'-'*80}")

    # Trier par moyenne décroissante
    results.sort(key=lambda x: x['moyenne'], reverse=True)

    for result in results:
        print(f"{result['model']:<25} "
              f"{result['moyenne']:>10.2f}   "
              f"{result['min']:>8.2f}   "
              f"{result['max']:>8.2f}   "
              f"{result['écart_type']:>10.2f}")

    # Détail jour par jour
    print(f"\n{'='*80}")
    print(f"📅 PRÉDICTIONS JOUR PAR JOUR")
    print(f"{'='*80}")

    # En-tête des dates
    print(f"{'Modèle':<25}", end='')
    for date in prediction_dates:
        print(f" {date.day:>5}", end='')
    print()
    print(f"{'-'*80}")

    # Prédictions de chaque modèle
    for result in results:
        print(f"{result['model']:<25}", end='')
        for pred in result['predictions']:
            print(f" {pred:>5.1f}", end='')
        print()

    # Recommandation
    print(f"\n{'='*80}")
    print(f"💡 RECOMMANDATION")
    print(f"{'='*80}")

    # Exclure Naive s'il prédit tout à 0
    valid_results = [r for r in results if r['moyenne'] > 0]

    if valid_results:
        best = valid_results[0]
        print(f"✅ Meilleur modèle recommandé : {best['model']}")
        print(f"   Moyenne prédite : {best['moyenne']:.2f} unités/jour")
        print(f"   Plage : {best['min']:.2f} à {best['max']:.2f}")
    else:
        print(f"⚠️  Tous les modèles prédisent 0 - vérifiez vos données !")


def main():
    """Fonction principale."""
    print("="*60)
    print("🚀 TEST ET COMPARAISON DES MODÈLES DE BASELINE")
    print("="*60)

    # Charger les données
    data = load_data()

    # Choix de l'utilisateur
    print("\n🎯 Que voulez-vous faire ?")
    print("   1. Tester UN modèle spécifique (recommandé: WeekdayMean)")
    print("   2. Comparer TOUS les modèles")
    print("   3. Les deux")

    try:
        choice = input("\nVotre choix (1/2/3) [défaut: 3] : ").strip() or "3"
    except:
        choice = "3"

    # Article à analyser
    articles_disponibles = sorted(data['article_id'].unique())
    print(f"\n📦 Articles disponibles : {articles_disponibles}")

    try:
        article_id = int(input(f"Article à analyser [défaut: {articles_disponibles[0]}] : ").strip() or articles_disponibles[0])
    except:
        article_id = articles_disponibles[0]

    # Exécution
    if choice in ["1", "3"]:
        # Test d'un modèle spécifique (WeekdayMean recommandé)
        model = WeekdayMeanBaseline()
        test_single_model(model, data, article_id)

    if choice in ["2", "3"]:
        # Comparaison de tous
        compare_all_models(data, article_id)

    # Conseils finaux
    print(f"\n{'='*60}")
    print(f"💡 PROCHAINES ÉTAPES")
    print(f"{'='*60}")
    print("1. Si vous voyez des prédictions réalistes → Parfait !")
    print("2. Si tout est à 0 → Vérifiez vos données brutes")
    print("3. Pour évaluer sur données de test :")
    print("   python main.py --step baselines")
    print("4. Pour voir tous les graphiques :")
    print("   python main.py --step analysis")


if __name__ == "__main__":
    main()