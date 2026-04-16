"""Évaluation des modèles de prévision (par coupure et TimeSeriesSplit).

Outils :
- ``evaluate_per_coupure``      : MAE/RMSE/MAPE par coupure.
- ``time_series_cv``            : 3-fold temporel (pas de fuite futur→passé).
- ``compare_models_per_coupure``: tableau comparatif sur le même split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from ..utils import ColumnNames, COUPURES, DMQ_BY_COUPURE
except ImportError:
    from src.utils import ColumnNames, COUPURES, DMQ_BY_COUPURE


logger = logging.getLogger(__name__)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return float('inf')
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_per_coupure(
    predictions: Dict[int, np.ndarray],
    actuals: Dict[int, np.ndarray],
) -> pd.DataFrame:
    """Calcule MAE/RMSE/MAPE par coupure et la ligne ``TOTAL`` agrégée.

    Args:
        predictions: ``{coupure: np.array}`` — prédictions DMQ (ou montant) par
            coupure. Longueurs alignées avec ``actuals``.
        actuals: même format, valeurs observées.

    Returns:
        DataFrame ``[coupure, mae, rmse, mape, n]``.
    """
    rows: List[Dict[str, Any]] = []

    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    for coupure in COUPURES:
        if coupure not in predictions or coupure not in actuals:
            continue
        y_pred = np.asarray(predictions[coupure], dtype=float)
        y_true = np.asarray(actuals[coupure], dtype=float)
        n = min(len(y_pred), len(y_true))
        if n == 0:
            continue
        y_pred = y_pred[:n]
        y_true = y_true[:n]

        rows.append({
            'coupure': coupure,
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mape': _mape(y_true, y_pred),
            'n': int(n),
        })
        all_true.append(y_true)
        all_pred.append(y_pred)

    if all_true:
        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        rows.append({
            'coupure': 'TOTAL',
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mape': _mape(y_true, y_pred),
            'n': int(len(y_true)),
        })

    return pd.DataFrame(rows)


@dataclass
class CVFold:
    fold_index: int
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp
    metrics: Dict[str, Any]


def time_series_cv(
    model_factory: Callable[[], Any],
    data: pd.DataFrame,
    n_splits: int = 3,
    target_column: str = ColumnNames.AMOUNT,
    horizon: int = 1,
    date_column: str = ColumnNames.ORDER_DATE,
) -> pd.DataFrame:
    """Validation croisée temporelle en 3 folds (par défaut).

    Les folds sont **expansifs** (growing window) :
      - Fold 1: train = [t0, t0+25%], test = [t0+25%, t0+50%]
      - Fold 2: train = [t0, t0+50%], test = [t0+50%, t0+75%]
      - Fold 3: train = [t0, t0+75%], test = [t0+75%, tend]

    Args:
        model_factory: Callable qui retourne une nouvelle instance du modèle
            à entraîner à chaque fold (doit exposer ``fit`` / ``predict``).
        data: DataFrame triable par ``date_column``.
        n_splits: Nombre de folds (3 par défaut).
        target_column: Colonne cible.
        horizon: Horizon de prédiction (en jours).

    Returns:
        DataFrame ``[fold, mae, rmse, mape, n_train, n_test]``.
    """
    if data.empty:
        return pd.DataFrame()

    sorted_dates = np.sort(data[date_column].unique())
    n = len(sorted_dates)
    if n < n_splits + 1:
        logger.warning(
            "time_series_cv : %d dates uniques insuffisant pour %d folds", n, n_splits
        )
        return pd.DataFrame()

    # On découpe en n_splits+1 segments de tailles égales.
    splits = np.linspace(0, n, n_splits + 2, dtype=int)

    rows: List[Dict[str, Any]] = []
    for fold in range(1, n_splits + 1):
        train_end_idx = splits[fold]
        test_end_idx = splits[fold + 1]
        train_end = sorted_dates[train_end_idx - 1]
        test_start = sorted_dates[train_end_idx]
        test_end = sorted_dates[test_end_idx - 1]

        train_mask = data[date_column] <= train_end
        test_mask = (data[date_column] > train_end) & (data[date_column] <= test_end)

        train_df = data[train_mask]
        test_df = data[test_mask]

        if train_df.empty or test_df.empty:
            continue

        model = model_factory()
        model.fit(train_df)

        y_true_list: List[float] = []
        y_pred_list: List[float] = []

        # Détecte si le modèle accepte `horizon` (baselines = non, catboost = oui).
        import inspect
        predict_params = inspect.signature(model.predict).parameters
        accepts_horizon = 'horizon' in predict_params

        # Itération simple : une prédiction par ligne de test.
        for _, row in test_df.iterrows():
            atm_id = int(row[ColumnNames.ATM_ID])
            pred_date = row[date_column]
            try:
                kwargs = dict(
                    atm_id=atm_id,
                    prediction_dates=[pred_date],
                    context_data=train_df,
                )
                if accepts_horizon:
                    kwargs['horizon'] = horizon
                pred = model.predict(**kwargs)[0]
            except Exception as exc:  # pragma: no cover
                logger.warning("CV fold %d: erreur predict ATM %s: %s", fold, atm_id, exc)
                continue

            y_pred_list.append(float(pred))
            y_true_list.append(float(row[target_column]))

        if not y_pred_list:
            continue

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        rows.append({
            'fold': fold,
            'train_end': pd.Timestamp(train_end),
            'test_start': pd.Timestamp(test_start),
            'test_end': pd.Timestamp(test_end),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mape': _mape(y_true, y_pred),
            'n_train': int(train_mask.sum()),
            'n_test': int(len(y_pred_list)),
        })

    return pd.DataFrame(rows)


def compare_models_per_coupure(
    models_per_coupure: Dict[str, Dict[int, np.ndarray]],
    actuals: Dict[int, np.ndarray],
) -> pd.DataFrame:
    """Compare plusieurs modèles par coupure sur le même split.

    Args:
        models_per_coupure: ``{nom_modele: {coupure: preds}}``.
        actuals: ``{coupure: valeurs observées}``.

    Returns:
        DataFrame multi-modèles avec colonnes ``[model, coupure, mae, rmse, mape]``.
    """
    all_rows: List[pd.DataFrame] = []
    for name, preds in models_per_coupure.items():
        df = evaluate_per_coupure(preds, actuals)
        df.insert(0, 'model', name)
        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def dmq_actuals_from_enriched(
    test_data: pd.DataFrame,
) -> Dict[int, np.ndarray]:
    """Extrait ``{coupure: array}`` depuis un DataFrame ayant les colonnes DMQ."""
    out: Dict[int, np.ndarray] = {}
    for c in COUPURES:
        col = DMQ_BY_COUPURE[c]
        if col in test_data.columns:
            out[c] = test_data[col].to_numpy(dtype=float)
    return out
