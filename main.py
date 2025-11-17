"""
Script principal du projet de prévision de commandes.

Usage :
    python main.py --help                    # Afficher l'aide
    python main.py --step ingestion         # Étape 1A: Ingestion des données
    python main.py --step enrichment        # Étape 1B: Enrichissement
    python main.py --step analysis          # Analyse et visualisations
    python main.py --step baselines         # Entraînement des baselines
    python main.py --step all               # Pipeline complet
    python main.py --article 123            # Analyse d'un article spécifique

Ce script orchestre l'ensemble du pipeline de prévision.
"""

import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Import des modules du projet
from src import (
    DataIngestionPipeline, DataEnrichmentPipeline, DataVisualization,
    create_article_dashboard, create_global_analysis, analyze_article_pattern
)
from src.models.baseline import create_baseline_suite, evaluate_all_baselines
from src.utils import ColumnNames, get_file_path, Messages

# Configuration du logging principal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forecasting_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def run_ingestion_step() -> bool:
    """
    Exécute l'étape 1A : Ingestion et nettoyage des données.

    Returns:
        bool: Succès de l'opération
    """
    logger.info("🚀 ÉTAPE 1A : INGESTION ET NETTOYAGE DES DONNÉES")
    logger.info("=" * 60)

    try:
        # Pipeline d'ingestion
        pipeline = DataIngestionPipeline()
        clean_data = pipeline.run_full_pipeline()

        # Résumé des résultats
        summary = pipeline.get_data_summary()

        logger.info("📊 RÉSUMÉ DE L'INGESTION :")
        for key, value in summary.items():
            logger.info(f"   {key}: {value}")

        return True

    except Exception as e:
        logger.error(f"❌ ERREUR LORS DE L'INGESTION : {e}")
        return False


def run_enrichment_step() -> bool:
    """
    Exécute l'étape 1B : Enrichissement des données.

    Returns:
        bool: Succès de l'opération
    """
    logger.info("🚀 ÉTAPE 1B : ENRICHISSEMENT DES DONNÉES")
    logger.info("=" * 60)

    try:
        # Pipeline d'enrichissement
        pipeline = DataEnrichmentPipeline()
        enriched_data = pipeline.run_full_enrichment()

        # Résumé des résultats
        summary = pipeline.get_enrichment_summary()

        logger.info("📊 RÉSUMÉ DE L'ENRICHISSEMENT :")
        logger.info(f"   🏆 Jour le plus fort : {summary.get('jour_plus_fort', 'Non calculé')}")

        weekend_analysis = summary.get('analyse_weekend', {})
        ratio = weekend_analysis.get('ratio_weekend_vs_semaine', 'Non calculé')
        logger.info(f"   📈 Ratio weekend/semaine : {ratio}")

        # Vérification des variables de retard
        viz = DataVisualization(enriched_data)
        viz.show_lag_verification(n_rows=10)

        return True

    except Exception as e:
        logger.error(f"❌ ERREUR LORS DE L'ENRICHISSEMENT : {e}")
        return False


def run_analysis_step() -> bool:
    """
    Exécute l'analyse et les visualisations des données.

    Returns:
        bool: Succès de l'opération
    """
    logger.info("🚀 ANALYSE ET VISUALISATIONS")
    logger.info("=" * 60)

    try:
        # Chargement des données enrichies
        viz = DataVisualization()
        viz.load_enriched_data()

        # Création du dossier de sortie pour les graphiques
        output_dir = Path("data/output/charts")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Analyse globale par jour de semaine
        logger.info("📊 Génération de l'analyse par jour de semaine...")
        fig_weekday, stats_weekday = viz.plot_weekday_analysis(
            save_path=output_dir / "weekday_analysis.png"
        )

        # 2. Comparaison weekend vs semaine
        logger.info("📊 Génération de la comparaison weekend/semaine...")
        fig_weekend = viz.plot_weekend_vs_weekday_comparison()
        fig_weekend.savefig(output_dir / "weekend_comparison.png", dpi=300, bbox_inches='tight')

        # 3. Analyse d'un article exemple
        enriched_data = viz.enriched_data
        sample_article = enriched_data[ColumnNames.ARTICLE_ID].iloc[0]

        logger.info(f"📊 Génération du graphique pour l'article exemple {sample_article}...")
        fig_article = viz.plot_daily_sales_by_article(
            article_id=sample_article,
            save_path=output_dir / f"article_{sample_article}_analysis.png"
        )

        # Affichage des graphiques (si environnement le permet)
        try:
            plt.show()
        except:
            logger.info("   ℹ️  Affichage graphique non disponible, fichiers sauvegardés uniquement")

        logger.info(f"💾 Graphiques sauvegardés dans : {output_dir}")

        return True

    except Exception as e:
        logger.error(f"❌ ERREUR LORS DE L'ANALYSE : {e}")
        return False


def run_baselines_step() -> bool:
    """
    Exécute l'entraînement et l'évaluation des modèles de baseline.

    Returns:
        bool: Succès de l'opération
    """
    logger.info("🚀 ENTRAÎNEMENT DES MODÈLES DE BASELINE")
    logger.info("=" * 60)

    try:
        # Chargement des données enrichies
        enriched_file = get_file_path('enriched')
        if not enriched_file.exists():
            logger.error("❌ Fichier de données enrichies non trouvé. Exécutez d'abord les étapes précédentes.")
            return False

        enriched_data = pd.read_csv(
            enriched_file,
            parse_dates=[ColumnNames.DATE],
            date_format='%Y-%m-%d'
        )

        # Division train/test (90% train, 10% test)
        total_days = enriched_data[ColumnNames.DATE].nunique()
        test_days = max(7, total_days // 10)  # Au minimum 7 jours de test

        sorted_dates = sorted(enriched_data[ColumnNames.DATE].unique())
        split_date = sorted_dates[-test_days]

        train_data = enriched_data[enriched_data[ColumnNames.DATE] < split_date].copy()
        test_data = enriched_data[enriched_data[ColumnNames.DATE] >= split_date].copy()

        logger.info(f"   📊 Division des données :")
        logger.info(f"      Train : {len(train_data)} lignes ({len(sorted_dates) - test_days} jours)")
        logger.info(f"      Test  : {len(test_data)} lignes ({test_days} jours)")

        # Création de la suite de baselines
        baselines = create_baseline_suite()

        logger.info(f"   🎯 Modèles à entraîner : {[b.name for b in baselines]}")

        # Évaluation complète
        results_df = evaluate_all_baselines(baselines, train_data, test_data)

        # Sauvegarde des résultats
        output_file = Path("data/output/baseline_results.csv")
        results_df.to_csv(output_file, index=False)

        logger.info(f"💾 Résultats sauvegardés : {output_file}")

        # Affichage du podium
        logger.info("=" * 60)
        logger.info("🏆 PODIUM DES BASELINES :")
        for i in range(min(3, len(results_df))):
            model = results_df.iloc[i]
            logger.info(f"   {i+1}. 🥇 {model['model']}: MAE={model['mae']:.2f}")

        return True

    except Exception as e:
        logger.error(f"❌ ERREUR LORS DE L'ENTRAÎNEMENT DES BASELINES : {e}")
        return False


def analyze_specific_article(article_id: int) -> bool:
    """
    Analyse détaillée d'un article spécifique.

    Args:
        article_id: ID de l'article à analyser

    Returns:
        bool: Succès de l'opération
    """
    logger.info(f"🚀 ANALYSE DÉTAILLÉE DE L'ARTICLE {article_id}")
    logger.info("=" * 60)

    try:
        # Chargement des données enrichies
        enriched_file = get_file_path('enriched')
        if not enriched_file.exists():
            logger.error("❌ Données enrichies non trouvées. Exécutez d'abord les étapes précédentes.")
            return False

        enriched_data = pd.read_csv(
            enriched_file,
            parse_dates=[ColumnNames.DATE],
            date_format='%Y-%m-%d'
        )

        # Vérification que l'article existe
        if article_id not in enriched_data[ColumnNames.ARTICLE_ID].values:
            logger.error(f"❌ Article {article_id} non trouvé dans les données.")
            available_articles = enriched_data[ColumnNames.ARTICLE_ID].unique()[:5]
            logger.info(f"   Articles disponibles (premiers 5): {available_articles.tolist()}")
            return False

        # Analyse des patterns
        analysis = analyze_article_pattern(enriched_data, article_id)

        logger.info("📊 ANALYSE DES PATTERNS :")
        for key, value in analysis.items():
            if key != 'stats_par_jour':
                logger.info(f"   {key}: {value}")

        # Génération des graphiques
        viz = DataVisualization(enriched_data)
        output_dir = Path("data/output/articles")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Dashboard de l'article
        dashboard = create_article_dashboard(
            enriched_data,
            article_id,
            output_dir
        )

        logger.info(f"💾 Dashboard généré pour l'article {article_id}")

        # Affichage si possible
        try:
            plt.show()
        except:
            logger.info("   ℹ️  Graphiques sauvegardés uniquement")

        return True

    except Exception as e:
        logger.error(f"❌ ERREUR LORS DE L'ANALYSE DE L'ARTICLE {article_id} : {e}")
        return False


def run_full_pipeline() -> bool:
    """
    Exécute le pipeline complet de bout en bout.

    Returns:
        bool: Succès de l'opération complète
    """
    logger.info("🚀 PIPELINE COMPLET DE PRÉVISION")
    logger.info("=" * 60)

    start_time = datetime.now()

    steps = [
        ("Ingestion", run_ingestion_step),
        ("Enrichissement", run_enrichment_step),
        ("Analyse", run_analysis_step),
        ("Baselines", run_baselines_step)
    ]

    success_count = 0

    for step_name, step_function in steps:
        logger.info(f"\n{'='*20} {step_name.upper()} {'='*20}")

        if step_function():
            success_count += 1
            logger.info(f"✅ {step_name} terminée avec succès")
        else:
            logger.error(f"❌ {step_name} échouée")
            break

    # Résumé final
    duration = datetime.now() - start_time

    logger.info("\n" + "=" * 60)
    logger.info("📊 RÉSUMÉ DU PIPELINE COMPLET")
    logger.info("=" * 60)
    logger.info(f"✅ Étapes réussies : {success_count}/{len(steps)}")
    logger.info(f"⏱️  Durée totale : {duration}")

    if success_count == len(steps):
        logger.info("🎉 PIPELINE COMPLET TERMINÉ AVEC SUCCÈS !")
        logger.info("   👉 Consultez les fichiers dans data/output/")
        return True
    else:
        logger.error("❌ Pipeline incomplet. Consultez les logs pour les détails.")
        return False


def main():
    """
    Point d'entrée principal du script.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline de prévision de commandes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
  python main.py --step all                  # Pipeline complet
  python main.py --step ingestion           # Seulement l'ingestion
  python main.py --article 123              # Analyse de l'article 123
  python main.py --step analysis            # Seulement les visualisations
        """
    )

    parser.add_argument(
        '--step',
        choices=['ingestion', 'enrichment', 'analysis', 'baselines', 'all'],
        help='Étape du pipeline à exécuter'
    )

    parser.add_argument(
        '--article',
        type=int,
        help='ID d\'article pour une analyse détaillée'
    )

    args = parser.parse_args()

    # Validation des arguments
    if not args.step and not args.article:
        parser.print_help()
        return

    # Log de démarrage
    logger.info("🚀 DÉMARRAGE DU PIPELINE DE PRÉVISION")
    logger.info(f"⏰ Heure de début : {datetime.now()}")
    logger.info(f"📁 Répertoire de travail : {Path.cwd()}")

    success = False

    try:
        if args.article:
            success = analyze_specific_article(args.article)
        elif args.step == 'ingestion':
            success = run_ingestion_step()
        elif args.step == 'enrichment':
            success = run_enrichment_step()
        elif args.step == 'analysis':
            success = run_analysis_step()
        elif args.step == 'baselines':
            success = run_baselines_step()
        elif args.step == 'all':
            success = run_full_pipeline()

    except KeyboardInterrupt:
        logger.info("⚠️  Pipeline interrompu par l'utilisateur")
        success = False
    except Exception as e:
        logger.error(f"❌ Erreur inattendue : {e}")
        success = False

    # Message de fin
    if success:
        logger.info("✅ Exécution terminée avec succès")
        sys.exit(0)
    else:
        logger.error("❌ Exécution terminée avec des erreurs")
        sys.exit(1)


if __name__ == "__main__":
    main()