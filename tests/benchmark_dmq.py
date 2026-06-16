"""Benchmark : précision des prédictions DMQ par coupure.

Compare :
- ``CatBoostForecaster``       (ancien modèle — montant total agrégé)
- ``CatBoostDmqForecaster``    (1 modèle par coupure)
- ``WeekdayMeanBaseline``      (baseline jour-de-semaine)

Métriques : MAE, RMSE, MAPE par coupure + TOTAL, sur le **même split temporel**
(TimeSeriesSplit, cf. ``src/models/evaluation.py``).

Sortie : ``data/output/precision_comparison.csv``.

Critère de succès (cf. plan) : MAE/coupure du ``CatBoostDmqForecaster`` doit
battre la baseline ``WeekdayMeanBaseline`` sur **4/5 coupures**.

Usage :
    python tests/benchmark_dmq.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import (
    COUPURES,
    ColumnNames,
    DMQ_BY_COUPURE,
    get_file_path,
)
from src.models.baseline import WeekdayMeanBaseline
from src.models.evaluation import (
    compare_models_per_coupure,
    dmq_actuals_from_enriched,
    evaluate_per_coupure,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _split_temporal(data: pd.DataFrame, test_ratio: float = 0.2):
    dates = np.sort(data[ColumnNames.ORDER_DATE].unique())
    split_idx = int(len(dates) * (1 - test_ratio))
    if split_idx <= 0 or split_idx >= len(dates):
        raise ValueError("Split temporel impossible (données trop courtes).")
    split_date = dates[split_idx]
    train = data[data[ColumnNames.ORDER_DATE] < split_date].copy()
    test = data[data[ColumnNames.ORDER_DATE] >= split_date].copy()
    return train, test, split_date


def _ensure_dmq_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Si le fichier enrichi ne contient pas les colonnes ``dmq_*``, on les
    reconstruit à partir des soldes (différence jour-à-jour positive).
    """
    from src.utils import SOLDES_BY_COUPURE

    missing = [c for c in COUPURES if DMQ_BY_COUPURE[c] not in data.columns]
    if not missing:
        return data

    logger.info("Reconstruction des colonnes DMQ (manquantes) depuis les soldes…")
    data = data.sort_values([ColumnNames.ATM_ID, ColumnNames.ORDER_DATE]).copy()
    for c in missing:
        col_solde = SOLDES_BY_COUPURE[c]
        if col_solde not in data.columns:
            # Pas de solde → DMQ = 0 (ne bloque pas, mais modèle peu informatif).
            data[DMQ_BY_COUPURE[c]] = 0.0
            continue
        diffs = data.groupby(ColumnNames.ATM_ID)[col_solde].diff()
        dmq = (-diffs).clip(lower=0.0).fillna(0.0)
        data[DMQ_BY_COUPURE[c]] = dmq
    return data


def _predict_baseline_per_coupure(
    baseline_cls, train_df: pd.DataFrame, test_df: pd.DataFrame
):
    """Entraîne une baseline dédiée par coupure : on remplace temporairement
    ``amount`` par la colonne DMQ de la coupure visée pour réutiliser
    l'interface existante.
    """
    preds: dict = {}
    for c in COUPURES:
        dmq_col = DMQ_BY_COUPURE[c]
        if dmq_col not in train_df.columns:
            continue
        train_slice = train_df.copy()
        train_slice[ColumnNames.AMOUNT] = train_slice[dmq_col]
        model = baseline_cls()
        model.fit(train_slice)
        y_pred = model.predict_batch(test_df)
        preds[c] = np.asarray(y_pred, dtype=float)
    return preds


def _predict_multi_coupure(
    train_df: pd.DataFrame, test_df: pd.DataFrame
):
    """Tente d'entraîner un ``MultiCoupureForecaster`` (CatBoost). En l'absence
    de CatBoost, renvoie ``None``.
    """
    try:
        from src.models.catboost_model import CATBOOST_AVAILABLE, MultiCoupureForecaster
    except Exception:
        return None

    if not CATBOOST_AVAILABLE:
        logger.warning("CatBoost non disponible → benchmark ML ignoré.")
        return None

    # Vérif des colonnes DMQ
    missing = [c for c in COUPURES if DMQ_BY_COUPURE[c] not in train_df.columns]
    if missing:
        logger.warning(
            "Colonnes DMQ manquantes (%s) — CatBoostDmqForecaster ignoré.", missing
        )
        return None

    model = MultiCoupureForecaster(verbose=False)
    model.fit(train_df)

    preds: dict = {c: [] for c in COUPURES}
    for _, row in test_df.iterrows():
        atm_id = int(row[ColumnNames.ATM_ID])
        pred_date = row[ColumnNames.ORDER_DATE]
        try:
            dmq = model.predict_dmq_par_coupure(
                atm_id=atm_id,
                prediction_date=pred_date,
                context_data=train_df,
                horizon=1,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Erreur predict ATM %s: %s", atm_id, exc)
            dmq = {c: 0.0 for c in COUPURES}
        for c in COUPURES:
            preds[c].append(dmq.get(c, 0.0))
    return {c: np.asarray(v, dtype=float) for c, v in preds.items()}


def run_benchmark(test_ratio: float = 0.2) -> pd.DataFrame:
    enriched_file = get_file_path('enriched')
    if not enriched_file.exists():
        raise FileNotFoundError(
            f"Fichier enrichi introuvable : {enriched_file}. "
            "Exécutez `python main.py --step enrichment` au préalable."
        )

    data = pd.read_csv(
        enriched_file,
        parse_dates=[ColumnNames.ORDER_DATE],
        date_format='%Y-%m-%d',
    )
    logger.info("Données chargées : %d lignes", len(data))

    data = _ensure_dmq_columns(data)

    train, test, split_date = _split_temporal(data, test_ratio=test_ratio)
    logger.info("Split : train=%d lignes, test=%d lignes (coupure: %s)",
                len(train), len(test), split_date)

    actuals = dmq_actuals_from_enriched(test)
    logger.info("Coupures présentes dans le test : %s", sorted(actuals.keys()))

    results_per_model = {}

    # 1) WeekdayMeanBaseline par coupure
    logger.info("→ Baseline : WeekdayMeanBaseline par coupure")
    baseline_preds = _predict_baseline_per_coupure(WeekdayMeanBaseline, train, test)
    if baseline_preds:
        results_per_model['WeekdayMeanBaseline'] = baseline_preds

    # 2) MultiCoupureForecaster (CatBoost, par coupure)
    logger.info("→ CatBoostDmqForecaster (MultiCoupure)")
    ml_preds = _predict_multi_coupure(train, test)
    if ml_preds:
        results_per_model['CatBoostDmqForecaster'] = ml_preds

    if not results_per_model:
        raise RuntimeError("Aucun modèle évalué — benchmark impossible.")

    comparison = compare_models_per_coupure(results_per_model, actuals)

    output_file = ROOT / "data" / "output" / "precision_comparison.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_file, index=False)
    logger.info("Benchmark écrit : %s", output_file)

    # Résumé console : MAE par coupure et par modèle
    logger.info("=" * 60)
    logger.info("RÉSUMÉ PRÉCISION (MAE par coupure)")
    logger.info("=" * 60)
    pivot = comparison.pivot(index='coupure', columns='model', values='mae')
    logger.info("\n%s", pivot.round(2).to_string())

    # Vérification critère de succès : ML bat baseline sur ≥ 4/5 coupures
    if {'WeekdayMeanBaseline', 'CatBoostDmqForecaster'}.issubset(pivot.columns):
        pivot_coupures = pivot.drop(index='TOTAL', errors='ignore')
        wins = (
            pivot_coupures['CatBoostDmqForecaster'] <= pivot_coupures['WeekdayMeanBaseline']
        ).sum()
        logger.info(
            "CatBoostDmqForecaster bat WeekdayMeanBaseline sur %d/5 coupures", wins
        )
        if wins < 4:
            logger.warning("Critère de succès non atteint (attendu ≥ 4/5).")
        else:
            logger.info("Critère de succès atteint ✅")

    return comparison


if __name__ == "__main__":
    run_benchmark()
