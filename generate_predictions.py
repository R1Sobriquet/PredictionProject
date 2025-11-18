"""
Script pour générer les prédictions concrètes de commandes.

Ce script :
1. Charge les données enrichies
2. Charge le meilleur modèle baseline (ou celui spécifié)
3. Génère les prédictions pour décembre 2024 et/ou 2025
4. Sauvegarde dans un fichier CSV exploitable

Usage:
    python generate_predictions.py --month 2024-12  # Décembre 2024
    python generate_predictions.py --month 2025-01  # Janvier 2025
    python generate_predictions.py --year 2025      # Toute l'année 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import argparse
import sys

# Ajouter le répertoire racine au path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.baseline import (
    WeekdayMeanBaseline,
    NaiveBaseline,
    HistoricalMeanBaseline,
    MovingAverageBaseline,
    SeasonalNaiveBaseline
)
from src.utils.config import ColumnNames, get_file_path


def load_enriched_data():
    """Charge les données enrichies."""
    print("📊 Chargement des données enrichies...")

    enriched_file = get_file_path('enriched')

    if not enriched_file.exists():
        print(f"❌ Fichier non trouvé : {enriched_file}")
        print("\n💡 Lancez d'abord :")
        print("   python main.py --step enrichment")
        sys.exit(1)

    data = pd.read_csv(
        enriched_file,
        parse_dates=[ColumnNames.DATE],
        date_format='%Y-%m-%d'
    )

    print(f"   ✅ {len(data)} lignes chargées")
    print(f"   📦 {data[ColumnNames.ARTICLE_ID].nunique()} articles")

    return data


def generate_predictions_for_period(
        model,
        data,
        start_date,
        end_date,
        articles=None
):
    """
    Génère les prédictions pour une période donnée.

    Args:
        model: Modèle baseline entraîné
        data: Données d'entraînement
        start_date: Date de début (datetime)
        end_date: Date de fin (datetime)
        articles: Liste d'articles (None = tous)

    Returns:
        DataFrame avec colonnes [date, article_id, prediction, model]
    """
    print(f"\n🔮 Génération des prédictions : {start_date.date()} à {end_date.date()}")

    # Liste des articles
    if articles is None:
        articles = sorted(data[ColumnNames.ARTICLE_ID].unique())

    print(f"   📦 {len(articles)} articles à prédire")

    # Génération des dates
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    print(f"   📅 {len(date_range)} jours à prédire")

    # Prédictions pour chaque article
    all_predictions = []

    for i, article_id in enumerate(articles, 1):
        if i % 10 == 0 or i == len(articles):
            print(f"   ⏳ Progression : {i}/{len(articles)} articles", end='\r')

        try:
            # Générer les prédictions pour cet article
            predictions = model.predict(
                article_id=article_id,
                prediction_dates=date_range.tolist(),
                context_data=data
            )

            # Créer un DataFrame pour cet article
            article_predictions = pd.DataFrame({
                ColumnNames.DATE: date_range,
                ColumnNames.ARTICLE_ID: article_id,
                'quantite_predite': np.round(predictions, 2),  # Arrondi à 2 décimales
                'quantite_predite_entier': np.round(predictions).astype(int),  # Version entière
                'model': model.name
            })

            all_predictions.append(article_predictions)

        except Exception as e:
            print(f"\n   ⚠️  Erreur article {article_id}: {e}")
            continue

    print("\n   ✅ Prédictions terminées")

    # Concaténer toutes les prédictions
    result_df = pd.concat(all_predictions, ignore_index=True)

    return result_df


def add_business_info(predictions_df, data):
    """
    Ajoute des informations métier utiles aux prédictions.

    Args:
        predictions_df: DataFrame des prédictions
        data: Données historiques

    Returns:
        DataFrame enrichi
    """
    print("\n📊 Ajout d'informations métier...")

    # Ajouter le jour de la semaine
    predictions_df['jour_semaine'] = predictions_df[ColumnNames.DATE].dt.day_name()
    predictions_df['est_weekend'] = predictions_df[ColumnNames.DATE].dt.weekday.isin([5, 6])

    # Calculer la moyenne historique par article
    historical_means = data.groupby(ColumnNames.ARTICLE_ID)[ColumnNames.QUANTITY].mean()
    predictions_df['moyenne_historique'] = predictions_df[ColumnNames.ARTICLE_ID].map(historical_means).round(2)

    # Écart par rapport à la moyenne historique
    predictions_df['ecart_vs_historique'] = (
            predictions_df['quantite_predite'] - predictions_df['moyenne_historique']
    ).round(2)

    print("   ✅ Informations ajoutées")

    return predictions_df


def save_predictions(predictions_df, output_file):
    """Sauvegarde les prédictions dans un fichier CSV."""
    print(f"\n💾 Sauvegarde des prédictions...")

    # Réorganiser les colonnes pour plus de clarté
    columns_order = [
        ColumnNames.DATE,
        ColumnNames.ARTICLE_ID,
        'quantite_predite_entier',  # ← LA COLONNE IMPORTANTE !
        'quantite_predite',
        'moyenne_historique',
        'ecart_vs_historique',
        'jour_semaine',
        'est_weekend',
        'model'
    ]

    predictions_df = predictions_df[columns_order]

    # Sauvegarder
    output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_file, index=False)

    print(f"   ✅ Fichier sauvegardé : {output_file}")
    print(f"   📊 {len(predictions_df)} lignes de prédictions")

    return output_file


def display_summary(predictions_df):
    """Affiche un résumé des prédictions."""
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DES PRÉDICTIONS")
    print("=" * 70)

    print(f"\n📅 Période couverte :")
    print(f"   Du {predictions_df[ColumnNames.DATE].min().date()}")
    print(f"   Au {predictions_df[ColumnNames.DATE].max().date()}")
    print(f"   → {predictions_df[ColumnNames.DATE].nunique()} jours")

    print(f"\n📦 Articles :")
    print(f"   {predictions_df[ColumnNames.ARTICLE_ID].nunique()} articles différents")

    print(f"\n📈 Statistiques globales :")
    print(f"   Quantité totale prédite : {predictions_df['quantite_predite_entier'].sum():,}")
    print(f"   Moyenne par jour/article : {predictions_df['quantite_predite'].mean():.2f}")
    print(f"   Min : {predictions_df['quantite_predite'].min():.2f}")
    print(f"   Max : {predictions_df['quantite_predite'].max():.2f}")

    print(f"\n🔝 Top 5 articles (quantité totale prédite) :")
    top_articles = (
        predictions_df
        .groupby(ColumnNames.ARTICLE_ID)['quantite_predite_entier']
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    for article_id, total in top_articles.items():
        print(f"   Article {article_id}: {total:,} unités")

    print(f"\n📋 Aperçu (5 premières lignes) :")
    print(predictions_df[[
        ColumnNames.DATE,
        ColumnNames.ARTICLE_ID,
        'quantite_predite_entier',
        'jour_semaine'
    ]].head(5).to_string(index=False))


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Génère les prédictions de commandes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python generate_predictions.py --month 2024-12          # Décembre 2024
  python generate_predictions.py --month 2025-01          # Janvier 2025
  python generate_predictions.py --year 2025              # Toute l'année 2025
  python generate_predictions.py --start 2024-12-01 --end 2024-12-31
        """
    )

    parser.add_argument('--month', help='Mois à prédire (format: YYYY-MM)')
    parser.add_argument('--year', type=int, help='Année complète à prédire')
    parser.add_argument('--start', help='Date de début (format: YYYY-MM-DD)')
    parser.add_argument('--end', help='Date de fin (format: YYYY-MM-DD)')
    parser.add_argument('--model', default='weekday',
                        choices=['weekday', 'naive', 'historical', 'seasonal', 'moving_average'],
                        help='Modèle à utiliser (défaut: weekday)')
    parser.add_argument('--output', help='Fichier de sortie (optionnel)')

    args = parser.parse_args()

    # Déterminer les dates
    if args.month:
        year, month = map(int, args.month.split('-'))
        start_date = datetime(year, month, 1)
        # Dernier jour du mois
        if month == 12:
            end_date = datetime(year, 12, 31)
        else:
            end_date = datetime(year, month + 1, 1) - pd.Timedelta(days=1)

    elif args.year:
        start_date = datetime(args.year, 1, 1)
        end_date = datetime(args.year, 12, 31)

    elif args.start and args.end:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d')

    else:
        # Par défaut : décembre 2024
        print("⚠️  Aucune période spécifiée, utilisation de décembre 2024 par défaut")
        start_date = datetime(2024, 12, 1)
        end_date = datetime(2024, 12, 31)

    print("=" * 70)
    print("🚀 GÉNÉRATION DES PRÉDICTIONS DE COMMANDES")
    print("=" * 70)
    print(f"📅 Période : {start_date.date()} à {end_date.date()}")
    print(f"🎯 Modèle : {args.model}")

    # Charger les données
    data = load_enriched_data()

    # Créer le modèle
    print(f"\n🎯 Initialisation du modèle : {args.model}")

    if args.model == 'weekday':
        model = WeekdayMeanBaseline()
    elif args.model == 'naive':
        model = NaiveBaseline()
    elif args.model == 'historical':
        model = HistoricalMeanBaseline()
    elif args.model == 'seasonal':
        model = SeasonalNaiveBaseline()
    elif args.model == 'moving_average':
        model = MovingAverageBaseline(window=7)

    # Entraîner le modèle
    print("   🎯 Entraînement du modèle...")
    model.fit(data)
    print("   ✅ Modèle entraîné")

    # Générer les prédictions
    predictions_df = generate_predictions_for_period(
        model=model,
        data=data,
        start_date=start_date,
        end_date=end_date
    )

    # Ajouter des informations métier
    predictions_df = add_business_info(predictions_df, data)

    # Déterminer le fichier de sortie
    if args.output:
        output_file = Path(args.output)
    else:
        # Nom automatique basé sur la période
        period_str = f"{start_date.strftime('%Y%m')}-{end_date.strftime('%Y%m')}"
        output_file = Path(f"data/output/predictions_{period_str}_{args.model}.csv")

    # Sauvegarder
    save_predictions(predictions_df, output_file)

    # Afficher le résumé
    display_summary(predictions_df)

    print("\n" + "=" * 70)
    print("✅ GÉNÉRATION TERMINÉE AVEC SUCCÈS")
    print("=" * 70)
    print(f"\n📄 Fichier de prédictions : {output_file}")
    print(f"\n💡 Utilisez ce fichier pour :")
    print(f"   1. Planifier les commandes fournisseurs")
    print(f"   2. Optimiser les stocks")
    print(f"   3. Anticiper la demande")


if __name__ == "__main__":
    main()