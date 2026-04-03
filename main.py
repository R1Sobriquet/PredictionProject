"""
Script principal du projet de prévision de commandes ATM.

Usage :
    python main.py --help                    # Afficher l'aide
    python main.py --step ingestion         # Étape 1 : Ingestion des données
    python main.py --step enrichment        # Étape 2 : Enrichissement
    python main.py --step analysis          # Analyse et visualisations
    python main.py --step baselines         # Entraînement des baselines
    python main.py --step catboost          # Entraînement CatBoost
    python main.py --step all               # Pipeline complet
    python main.py --atm 123               # Analyse d'un ATM spécifique
"""

import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src import (
    DataIngestionPipeline, DataEnrichmentPipeline, DataVisualization,
    create_atm_dashboard, create_global_analysis,
)
from src.data_processing import analyze_atm_pattern
from src.models.baseline import create_baseline_suite, evaluate_all_baselines
from src.models.catboost_model import CatBoostForecaster, train_and_evaluate_catboost
from src.utils import ColumnNames, get_file_path, Messages

# Configuration du logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/forecasting_pipeline.log'),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def run_ingestion_step() -> bool:
    """Étape 1 : Ingestion et nettoyage des données ATM."""
    logger.info("ÉTAPE 1 : INGESTION ET NETTOYAGE DES DONNÉES ATM")
    logger.info("=" * 60)

    try:
        pipeline = DataIngestionPipeline()
        clean_data = pipeline.run_full_pipeline()

        summary = pipeline.get_data_summary()
        logger.info("RÉSUMÉ DE L'INGESTION :")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")

        return True

    except Exception as e:
        logger.error(f"ERREUR LORS DE L'INGESTION : {e}")
        return False


def run_enrichment_step() -> bool:
    """Étape 2 : Enrichissement des données avec features ATM."""
    logger.info("ÉTAPE 2 : ENRICHISSEMENT DES DONNÉES ATM")
    logger.info("=" * 60)

    try:
        pipeline = DataEnrichmentPipeline()
        enriched_data = pipeline.run_full_enrichment()

        summary = pipeline.get_enrichment_summary()
        logger.info("RÉSUMÉ DE L'ENRICHISSEMENT :")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")

        # Vérification des features historiques
        viz = DataVisualization(enriched_data)
        viz.show_atm_history_verification(n_rows=10)

        return True

    except Exception as e:
        logger.error(f"ERREUR LORS DE L'ENRICHISSEMENT : {e}")
        return False


def run_analysis_step() -> bool:
    """Analyse et visualisations des données ATM."""
    logger.info("ANALYSE ET VISUALISATIONS ATM")
    logger.info("=" * 60)

    try:
        viz = DataVisualization()
        viz.load_enriched_data()

        output_dir = Path("data/output/charts")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Analyse par jour de semaine
        logger.info("Génération de l'analyse par jour de semaine...")
        fig_weekday, stats_weekday = viz.plot_weekday_analysis(
            save_path=output_dir / "weekday_analysis.png",
        )

        # Comparaison weekend/semaine
        logger.info("Génération de la comparaison weekend/semaine...")
        fig_weekend = viz.plot_weekend_vs_weekday_comparison()
        fig_weekend.savefig(output_dir / "weekend_comparison.png", dpi=300, bbox_inches='tight')

        # Analyse d'un ATM exemple
        enriched_data = viz.enriched_data
        sample_atm = enriched_data[ColumnNames.ATM_ID].iloc[0]

        logger.info(f"Génération du graphique pour l'ATM exemple {sample_atm}...")
        fig_atm = viz.plot_daily_amounts_by_atm(
            atm_id=sample_atm,
            save_path=output_dir / f"atm_{sample_atm}_analysis.png",
        )

        try:
            plt.show()
        except Exception:
            logger.info("  Affichage graphique non disponible, fichiers sauvegardés uniquement")

        logger.info(f"Graphiques sauvegardés dans : {output_dir}")
        return True

    except Exception as e:
        logger.error(f"ERREUR LORS DE L'ANALYSE : {e}")
        return False


def run_baselines_step() -> bool:
    """Entraînement et évaluation des modèles de baseline."""
    logger.info("ENTRAÎNEMENT DES MODÈLES DE BASELINE")
    logger.info("=" * 60)

    try:
        enriched_file = get_file_path('enriched')
        if not enriched_file.exists():
            logger.error("Fichier enrichi non trouvé. Exécutez les étapes précédentes.")
            return False

        enriched_data = pd.read_csv(
            enriched_file,
            parse_dates=[ColumnNames.ORDER_DATE],
            date_format='%Y-%m-%d',
        )

        # Division train/test temporelle (90/10)
        sorted_dates = sorted(enriched_data[ColumnNames.ORDER_DATE].unique())
        test_days = max(7, len(sorted_dates) // 10)
        split_date = sorted_dates[-test_days]

        train_data = enriched_data[enriched_data[ColumnNames.ORDER_DATE] < split_date].copy()
        test_data = enriched_data[enriched_data[ColumnNames.ORDER_DATE] >= split_date].copy()

        logger.info(f"  Train : {len(train_data)} lignes ({len(sorted_dates) - test_days} jours)")
        logger.info(f"  Test  : {len(test_data)} lignes ({test_days} jours)")

        baselines = create_baseline_suite()
        logger.info(f"  Modèles : {[b.name for b in baselines]}")

        results_df = evaluate_all_baselines(baselines, train_data, test_data)

        output_file = Path("data/output/baseline_results.csv")
        results_df.to_csv(output_file, index=False)
        logger.info(f"Résultats sauvegardés : {output_file}")

        logger.info("=" * 60)
        logger.info("PODIUM DES BASELINES :")
        for i in range(min(3, len(results_df))):
            model = results_df.iloc[i]
            logger.info(f"  {i + 1}. {model['model']}: MAE={model['mae']:.2f}")

        return True

    except Exception as e:
        logger.error(f"ERREUR BASELINES : {e}")
        return False


def run_catboost_step(max_horizon: int = 90) -> bool:
    """Entraînement et évaluation du modèle CatBoost."""
    logger.info("ENTRAÎNEMENT DU MODÈLE CATBOOST")
    logger.info("=" * 60)

    try:
        enriched_file = get_file_path('enriched')
        if not enriched_file.exists():
            logger.error("Fichier enrichi non trouvé. Exécutez les étapes précédentes.")
            return False

        enriched_data = pd.read_csv(
            enriched_file,
            parse_dates=[ColumnNames.ORDER_DATE],
            date_format='%Y-%m-%d',
        )

        logger.info(f"  Données chargées : {len(enriched_data)} lignes")

        model, results_by_horizon = train_and_evaluate_catboost(
            enriched_data, max_horizon=max_horizon, test_ratio=0.2,
        )

        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        results_by_horizon.to_csv(output_dir / "catboost_results_by_horizon.csv", index=False)
        model.save_model(output_dir / "catboost_model")

        logger.info("=" * 60)
        logger.info("PERFORMANCE PAR HORIZON :")
        for _, row in results_by_horizon.iterrows():
            logger.info(f"  H+{row['horizon']:2.0f}j : MAE={row['mae']:.2f}, RMSE={row['rmse']:.2f}")

        importance_df = model.get_feature_importance()
        logger.info("\nTOP 5 FEATURES :")
        for _, row in importance_df.head(5).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.1f}")

        return True

    except Exception as e:
        logger.error(f"ERREUR CATBOOST : {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_specific_atm(atm_id: int) -> bool:
    """Analyse détaillée d'un ATM spécifique."""
    logger.info(f"ANALYSE DÉTAILLÉE DE L'ATM {atm_id}")
    logger.info("=" * 60)

    try:
        enriched_file = get_file_path('enriched')
        if not enriched_file.exists():
            logger.error("Données enrichies non trouvées.")
            return False

        enriched_data = pd.read_csv(
            enriched_file,
            parse_dates=[ColumnNames.ORDER_DATE],
            date_format='%Y-%m-%d',
        )

        if atm_id not in enriched_data[ColumnNames.ATM_ID].values:
            logger.error(f"ATM {atm_id} non trouvé dans les données.")
            available = enriched_data[ColumnNames.ATM_ID].unique()[:5]
            logger.info(f"  ATMs disponibles (premiers 5) : {available.tolist()}")
            return False

        analysis = analyze_atm_pattern(enriched_data, atm_id)
        logger.info("ANALYSE DES PATTERNS :")
        for key, value in analysis.items():
            if key != 'stats_par_jour':
                logger.info(f"  {key}: {value}")

        output_dir = Path("data/output/atms")
        output_dir.mkdir(parents=True, exist_ok=True)

        create_atm_dashboard(enriched_data, atm_id, output_dir)
        logger.info(f"Dashboard généré pour l'ATM {atm_id}")

        try:
            plt.show()
        except Exception:
            logger.info("  Graphiques sauvegardés uniquement")

        return True

    except Exception as e:
        logger.error(f"ERREUR ANALYSE ATM {atm_id} : {e}")
        return False


def run_full_pipeline() -> bool:
    """Exécute le pipeline complet."""
    logger.info("PIPELINE COMPLET DE PRÉVISION ATM")
    logger.info("=" * 60)

    start_time = datetime.now()

    steps = [
        ("Ingestion", run_ingestion_step),
        ("Enrichissement", run_enrichment_step),
        ("Analyse", run_analysis_step),
        ("Baselines", run_baselines_step),
        ("CatBoost", run_catboost_step),
    ]

    success_count = 0

    for step_name, step_function in steps:
        logger.info(f"\n{'=' * 20} {step_name.upper()} {'=' * 20}")

        if step_function():
            success_count += 1
            logger.info(f"{step_name} terminée avec succès")
        else:
            logger.error(f"{step_name} échouée")
            break

    duration = datetime.now() - start_time

    logger.info("\n" + "=" * 60)
    logger.info("RÉSUMÉ DU PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Étapes réussies : {success_count}/{len(steps)}")
    logger.info(f"Durée totale : {duration}")

    if success_count == len(steps):
        logger.info("PIPELINE TERMINÉ AVEC SUCCÈS")
        logger.info("  Consultez les fichiers dans data/output/")
        return True
    else:
        logger.error("Pipeline incomplet. Consultez les logs.")
        return False


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Pipeline de prévision de commandes ATM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
  python main.py --step all                  # Pipeline complet
  python main.py --step ingestion           # Seulement l'ingestion
  python main.py --atm 123                  # Analyse de l'ATM 123
  python main.py --step analysis            # Seulement les visualisations
        """,
    )

    parser.add_argument(
        '--step',
        choices=['ingestion', 'enrichment', 'analysis', 'baselines', 'catboost', 'all'],
        help="Étape du pipeline à exécuter",
    )

    parser.add_argument(
        '--atm',
        type=int,
        help="ID d'ATM pour une analyse détaillée",
    )

    args = parser.parse_args()

    if not args.step and not args.atm:
        parser.print_help()
        return

    logger.info("DÉMARRAGE DU PIPELINE DE PRÉVISION ATM")
    logger.info(f"Heure : {datetime.now()}")
    logger.info(f"Répertoire : {Path.cwd()}")

    success = False

    try:
        if args.atm:
            success = analyze_specific_atm(args.atm)
        elif args.step == 'ingestion':
            success = run_ingestion_step()
        elif args.step == 'enrichment':
            success = run_enrichment_step()
        elif args.step == 'analysis':
            success = run_analysis_step()
        elif args.step == 'baselines':
            success = run_baselines_step()
        elif args.step == 'catboost':
            success = run_catboost_step()
        elif args.step == 'all':
            success = run_full_pipeline()

    except KeyboardInterrupt:
        logger.info("Pipeline interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur inattendue : {e}")

    if success:
        logger.info("Exécution terminée avec succès")
        sys.exit(0)
    else:
        logger.error("Exécution terminée avec des erreurs")
        sys.exit(1)


if __name__ == "__main__":
    main()
