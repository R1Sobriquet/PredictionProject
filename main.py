"""
Script principal du projet de prévision de commandes ATM.

Usage :
    python main.py --help                    # Afficher l'aide
    python main.py --step ingestion         # Étape 1 : Ingestion des données
    python main.py --step enrichment        # Étape 2 : Enrichissement
    python main.py --step analysis          # Analyse et visualisations
    python main.py --step baselines         # Entraînement des baselines
    python main.py --step catboost          # Entraînement CatBoost
    python main.py --step command           # Moteur de commande par coupure
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
from src.models.catboost_model import (
    CatBoostForecaster, train_and_evaluate_catboost, MultiCoupureForecaster,
)
from src.models.evaluation import time_series_cv
from src.commande import CommandPipeline
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


def run_baselines_step(cv: str = 'simple') -> bool:
    """Entraînement et évaluation des modèles de baseline.

    Args:
        cv: 'simple' (split 90/10) ou 'timeseries' (3-fold expansif).
    """
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

        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        baselines = create_baseline_suite()
        logger.info(f"  Modèles : {[b.name for b in baselines]}")

        if cv == 'timeseries':
            logger.info("  Stratégie CV : TimeSeriesSplit 3-fold expansif")
            all_rows = []
            # Pour chaque baseline : utilise create_baseline_suite() comme
            # factory — évite d'introspecter les signatures __init__ variables.
            for baseline_idx, baseline in enumerate(baselines):
                factory = lambda i=baseline_idx: create_baseline_suite()[i]
                cv_df = time_series_cv(factory, enriched_data, n_splits=3)
                if cv_df.empty:
                    logger.warning(f"  {baseline.name} : CV vide, skip")
                    continue
                cv_df.insert(0, 'model', baseline.name)
                all_rows.append(cv_df)
                mean_mae = cv_df['mae'].mean()
                std_mae = cv_df['mae'].std()
                logger.info(
                    f"  {baseline.name}: MAE {mean_mae:.2f} ± {std_mae:.2f} "
                    f"(sur {len(cv_df)} folds)"
                )

            if not all_rows:
                logger.error("Aucun fold exploitable.")
                return False
            results_df = pd.concat(all_rows, ignore_index=True)
            output_file = output_dir / "baseline_results_cv.csv"
            results_df.to_csv(output_file, index=False)
            logger.info(f"Résultats CV sauvegardés : {output_file}")

            # Podium basé sur la MAE moyenne
            ranking = (
                results_df.groupby('model')['mae']
                .agg(['mean', 'std'])
                .sort_values('mean')
            )
            logger.info("=" * 60)
            logger.info("PODIUM BASELINES (MAE moyenne sur 3 folds) :")
            for i, (model, row) in enumerate(ranking.head(3).iterrows()):
                logger.info(f"  {i + 1}. {model}: MAE={row['mean']:.2f} ± {row['std']:.2f}")
            return True

        # Split simple (défaut) : 90/10 temporel
        sorted_dates = sorted(enriched_data[ColumnNames.ORDER_DATE].unique())
        test_days = max(7, len(sorted_dates) // 10)
        split_date = sorted_dates[-test_days]

        train_data = enriched_data[enriched_data[ColumnNames.ORDER_DATE] < split_date].copy()
        test_data = enriched_data[enriched_data[ColumnNames.ORDER_DATE] >= split_date].copy()

        logger.info(f"  Train : {len(train_data)} lignes ({len(sorted_dates) - test_days} jours)")
        logger.info(f"  Test  : {len(test_data)} lignes ({test_days} jours)")

        results_df = evaluate_all_baselines(baselines, train_data, test_data)

        output_file = output_dir / "baseline_results.csv"
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


def run_catboost_step(
    max_horizon: int = 90,
    cv: str = 'simple',
    preset: str = 'default',
) -> bool:
    """Entraînement et évaluation du modèle CatBoost.

    Args:
        max_horizon: Horizon max pour la prévision multi-horizon.
        cv: 'simple' (split 90/10) ou 'timeseries' (3-fold expansif).
        preset: Preset d'hyperparamètres ('fast' | 'default' | 'deep').
    """
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

        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        if cv == 'timeseries':
            logger.info(
                f"  Stratégie CV : TimeSeriesSplit 3-fold expansif (preset='{preset}')"
            )
            factory = lambda: CatBoostForecaster(max_horizon=max_horizon, preset=preset)
            cv_df = time_series_cv(factory, enriched_data, n_splits=3, horizon=1)
            if cv_df.empty:
                logger.error("Aucun fold exploitable.")
                return False
            cv_df.insert(0, 'model', f'CatBoost_H{max_horizon}')
            cv_df.to_csv(output_dir / "catboost_results_cv.csv", index=False)
            mean_mae = cv_df['mae'].mean()
            std_mae = cv_df['mae'].std()
            logger.info("=" * 60)
            logger.info(
                f"CatBoost (3-fold TS) : MAE {mean_mae:.2f} ± {std_mae:.2f}"
            )
            for _, row in cv_df.iterrows():
                logger.info(
                    f"  fold {int(row['fold'])}: MAE={row['mae']:.2f}, "
                    f"RMSE={row['rmse']:.2f}"
                )
            return True

        # Split simple (défaut)
        logger.info(f"  Preset CatBoost : '{preset}'")
        model, results_by_horizon = train_and_evaluate_catboost(
            enriched_data, max_horizon=max_horizon, test_ratio=0.2, preset=preset,
        )

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


def run_catboost_dmq_step(preset: str = 'default') -> bool:
    """Entraînement MultiCoupureForecaster (5 modèles CatBoost, un par coupure).

    Chaque modèle prédit le DMQ d'une coupure spécifique (dmq_5..100) au lieu
    du montant total. C'est le mode qui doit battre Weekday_Mean.
    """
    logger.info("ENTRAÎNEMENT CATBOOST PAR COUPURE (MultiCoupureForecaster)")
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
        logger.info(f"  Preset CatBoost : '{preset}'")

        # Vérifier la présence des colonnes cibles
        from src.utils import COUPURES, DMQ_BY_COUPURE
        missing = [DMQ_BY_COUPURE[c] for c in COUPURES if DMQ_BY_COUPURE[c] not in enriched_data.columns]
        if missing:
            logger.error(f"Colonnes DMQ manquantes : {missing}. Re-lancez --step enrichment.")
            return False

        # Split temporel 80/20
        dates = sorted(enriched_data[ColumnNames.ORDER_DATE].unique())
        split_idx = int(len(dates) * 0.8)
        split_date = dates[split_idx]

        train_data = enriched_data[enriched_data[ColumnNames.ORDER_DATE] < split_date].copy()
        test_data = enriched_data[enriched_data[ColumnNames.ORDER_DATE] >= split_date].copy()

        logger.info(f"  Train : {len(train_data)} lignes jusqu'au {split_date.date()}")
        logger.info(f"  Test  : {len(test_data)} lignes à partir du {split_date.date()}")

        # Entraînement des 5 modèles
        mcf = MultiCoupureForecaster(
            max_horizon=14,
            preset=preset,
        )
        mcf.fit(train_data)

        # Évaluation par coupure sur le test set
        from src.models.evaluation import evaluate_per_coupure
        import numpy as np

        predictions = {}
        actuals = {}
        atm_arr = test_data[ColumnNames.ATM_ID].to_numpy()
        date_arr = test_data[ColumnNames.ORDER_DATE].to_numpy()
        for c in COUPURES:
            dmq_col = DMQ_BY_COUPURE[c]
            y_true = test_data[dmq_col].to_numpy(dtype=float)

            # Coupure sans variance : prédicteur constant ; sinon batch vectorisé
            is_constant = c in mcf.constant_predictions
            if is_constant:
                y_pred = np.full(
                    len(test_data), max(0.0, mcf.constant_predictions.get(c, 0.0))
                )
            else:
                y_pred = mcf.models[c].predict_batch(
                    atm_arr, date_arr, context_data=train_data, horizon=1,
                )

            predictions[c] = y_pred
            actuals[c] = y_true
            tag = " (constant)" if is_constant else ""
            logger.info(f"  Coupure {c:>3}€ : {len(y_pred)} prédictions{tag}")

        results_df = evaluate_per_coupure(predictions, actuals)

        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_dir / "catboost_dmq_results.csv", index=False)

        logger.info("=" * 60)
        logger.info("PERFORMANCE PAR COUPURE (CatBoost DMQ) :")
        for _, row in results_df.iterrows():
            logger.info(
                f"  {str(row['coupure']):>5}€ : MAE={row['mae']:.2f}, "
                f"RMSE={row['rmse']:.2f}, MAPE={row['mape']:.1f}%"
            )

        return True

    except Exception as e:
        logger.error(f"ERREUR CATBOOST DMQ : {e}")
        import traceback
        traceback.print_exc()
        return False


def run_command_step(
    day_commande: str = None,
    day_livraison: str = None,
) -> bool:
    """Étape 6 : Moteur de commande déterministe par coupure.

    Applique les 6 étapes documentées (K7 HS → Assurance agence) et produit
    un fichier avec 5 valeurs ``predictif_*`` par automate + le flag
    ``is_command`` (True dès qu'une coupure est > 0).
    """
    logger.info("MOTEUR DE COMMANDE DÉTERMINISTE PAR COUPURE")
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

        # Snapshot = dernière ligne par ATM (= état courant à la date de commande)
        enriched_data = enriched_data.sort_values(
            [ColumnNames.ATM_ID, ColumnNames.ORDER_DATE]
        )
        current_data = (
            enriched_data
            .groupby(ColumnNames.ATM_ID, as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
        logger.info(f"  Snapshot : {len(current_data)} automates")

        # Dates : si non fournies, on prend la date la plus récente + 3 j pour
        # la livraison.
        if day_commande:
            day_cmd = pd.Timestamp(day_commande).date()
        else:
            day_cmd = pd.Timestamp(current_data[ColumnNames.ORDER_DATE].max()).date()
        if day_livraison:
            day_liv = pd.Timestamp(day_livraison).date()
        else:
            from datetime import timedelta as _td
            day_liv = day_cmd + _td(days=3)

        logger.info(f"  Jour commande  : {day_cmd}")
        logger.info(f"  Jour livraison : {day_liv}")

        pipeline = CommandPipeline()
        result = pipeline.run(
            current_data=current_data,
            historical_data=enriched_data,
            day_commande=day_cmd,
            day_livraison=day_liv,
        )

        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "commandes_predictives.csv"
        result.to_csv(output_file, index=False)
        logger.info(f"Résultats sauvegardés : {output_file}")

        logger.info("=" * 60)
        logger.info("RÉSUMÉ DU MOTEUR DE COMMANDE :")
        logger.info(f"  Automates traités       : {len(result)}")
        logger.info(f"  Commandes générées      : {int(result[ColumnNames.IS_COMMAND].sum())}")
        logger.info(
            f"  Commandes exceptionnelles : "
            f"{int(result[ColumnNames.IS_COMMAND_EXCEPTIONNELLE].sum())}"
        )
        logger.info(
            f"  Alertes risque vide     : "
            f"{int(result[ColumnNames.ALERTE_RISQUE_VIDE].sum())}"
        )
        logger.info(
            f"  Commandes supprimées    : "
            f"{int(result[ColumnNames.ALERTE_COMMANDE_SUPPRIMEE].sum())}"
        )
        if 'montant_total' in result.columns:
            logger.info(f"  Montant total global    : {int(result['montant_total'].sum())} €")

        return True

    except Exception as e:
        logger.error(f"ERREUR MOTEUR DE COMMANDE : {e}")
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
        choices=['ingestion', 'enrichment', 'analysis', 'baselines', 'catboost', 'catboost-dmq', 'command', 'all'],
        help="Étape du pipeline à exécuter",
    )

    parser.add_argument(
        '--atm',
        type=int,
        help="ID d'ATM pour une analyse détaillée",
    )

    parser.add_argument(
        '--day-commande',
        type=str,
        default=None,
        help="(étape command) Date de commande, format YYYY-MM-DD",
    )

    parser.add_argument(
        '--day-livraison',
        type=str,
        default=None,
        help="(étape command) Date de livraison/chargement, format YYYY-MM-DD",
    )

    parser.add_argument(
        '--cv',
        choices=['simple', 'timeseries'],
        default='simple',
        help="Stratégie de validation pour baselines/catboost : 'simple' (split "
             "temporel 90/10, défaut) ou 'timeseries' (3-fold expansif).",
    )

    parser.add_argument(
        '--catboost-preset',
        choices=['fast', 'default', 'deep'],
        default='default',
        help="(étape catboost) Preset d'hyperparamètres CatBoost.",
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
            success = run_baselines_step(cv=args.cv)
        elif args.step == 'catboost':
            success = run_catboost_step(cv=args.cv, preset=args.catboost_preset)
        elif args.step == 'catboost-dmq':
            success = run_catboost_dmq_step(preset=args.catboost_preset)
        elif args.step == 'command':
            success = run_command_step(
                day_commande=args.day_commande,
                day_livraison=args.day_livraison,
            )
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
