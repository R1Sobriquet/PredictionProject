"""
Modèle CatBoost pour la prévision de montants de commandes ATM.

Target : amount (DC_Montant_Cmd)
Entité : atm_id (DC_Automate_Id)

Features :
- Horizon de prédiction
- Statistiques historiques par ATM
- Variables temporelles de la date cible
- Features ATM (soldes, cassettes, volatilité, etc.)
- Historique de rechargement (dernières commandes)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import warnings
import pickle

try:
    from catboost import CatBoostRegressor, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost non installé. Installez-le avec: pip install catboost")

from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from ..utils import (
        ColumnNames,
        Messages,
        WEEKDAY_NAMES,
        get_file_path,
    )
    from .baseline import BaselineModel
except ImportError:
    from src.utils import (
        ColumnNames,
        Messages,
        WEEKDAY_NAMES,
        get_file_path,
    )
    from src.models.baseline import BaselineModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CatBoostForecaster(BaselineModel):
    """
    Modèle CatBoost pour la prévision de montants de commandes ATM.

    Caractéristiques :
    - Horizon de prédiction paramétrable (1-90 jours)
    - Feature engineering adaptatif court terme / long terme
    - atm_id traité comme feature catégorielle native
    - Early stopping
    """

    SHORT_TERM_THRESHOLD = 14

    def __init__(
        self,
        max_horizon: int = 90,
        iterations: int = 4000,
        learning_rate: float = 0.03,
        depth: int = 6,
        early_stopping_rounds: int = 100,
        random_state: int = 42,
        verbose: bool = False,
        l2_leaf_reg: float = 3.0,
        subsample: float = 0.85,
        rsm: float = 0.85,
        loss_function: str = 'MAE',
        bootstrap_type: str = 'Bernoulli',
    ):
        super().__init__(f"CatBoost_H{max_horizon}")

        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost n'est pas installé.")

        self.max_horizon = max_horizon
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.verbose = verbose

        self.model = None
        self.feature_columns = []
        self.cat_features = []
        self.atm_historical_stats = {}
        self.global_stats = {}

        # Hyperparamètres tunés :
        # - learning_rate 0.03 (vs 0.85) : convergence beaucoup plus stable
        # - loss_function MAE : robuste aux outliers (queues épaisses des montants)
        # - l2_leaf_reg + subsample/rsm : régularisation + bagging Bernoulli
        self.model_params = {
            'iterations': iterations,
            'learning_rate': learning_rate,
            'depth': depth,
            'loss_function': loss_function,
            'l2_leaf_reg': l2_leaf_reg,
            'subsample': subsample,
            'rsm': rsm,
            'bootstrap_type': bootstrap_type,
            'random_seed': random_state,
            'verbose': verbose,
            'early_stopping_rounds': early_stopping_rounds,
            'use_best_model': True,
        }

    def _compute_historical_stats(self, data: pd.DataFrame) -> None:
        """Calcule les statistiques historiques par ATM."""
        logger.info("Calcul des statistiques historiques par ATM...")

        self.global_stats = {
            'mean': data[ColumnNames.AMOUNT].mean(),
            'std': data[ColumnNames.AMOUNT].std(),
            'median': data[ColumnNames.AMOUNT].median(),
        }

        for atm_id in data[ColumnNames.ATM_ID].unique():
            atm_data = data[data[ColumnNames.ATM_ID] == atm_id].sort_values(ColumnNames.ORDER_DATE)

            weekday_means = {}
            if ColumnNames.WEEKDAY in atm_data.columns:
                weekday_means = atm_data.groupby(ColumnNames.WEEKDAY)[ColumnNames.AMOUNT].mean().to_dict()

            month_means = {}
            if ColumnNames.MONTH in atm_data.columns:
                month_means = atm_data.groupby(ColumnNames.MONTH)[ColumnNames.AMOUNT].mean().to_dict()

            # Fréquence de rechargement
            dates = atm_data[ColumnNames.ORDER_DATE]
            if len(dates) > 1:
                diffs = dates.diff().dt.days.dropna()
                avg_frequency = diffs.mean()
            else:
                avg_frequency = 0

            self.atm_historical_stats[atm_id] = {
                'mean': atm_data[ColumnNames.AMOUNT].mean(),
                'std': atm_data[ColumnNames.AMOUNT].std(),
                'median': atm_data[ColumnNames.AMOUNT].median(),
                'max': atm_data[ColumnNames.AMOUNT].max(),
                'min': atm_data[ColumnNames.AMOUNT].min(),
                'weekday_means': weekday_means,
                'month_means': month_means,
                'total_orders': len(atm_data),
                'avg_frequency': avg_frequency,
            }

        logger.info(f"  Stats calculées pour {len(self.atm_historical_stats)} ATMs")

    def _prepare_features_for_training(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prépare les features pour l'entraînement multi-horizon."""
        logger.info(f"Préparation des features pour horizons 1 à {self.max_horizon}...")

        all_examples = []
        data = data.sort_values([ColumnNames.ATM_ID, ColumnNames.ORDER_DATE]).reset_index(drop=True)

        for atm_id in data[ColumnNames.ATM_ID].unique():
            atm_data = data[data[ColumnNames.ATM_ID] == atm_id].reset_index(drop=True)
            atm_stats = self.atm_historical_stats.get(atm_id, {})

            for idx in range(len(atm_data) - 1):
                base_row = atm_data.iloc[idx]

                # Horizons à échantillonner
                max_forward = min(len(atm_data) - idx - 1, self.max_horizon)
                horizons = list(range(1, min(15, max_forward + 1)))
                horizons += [h for h in [21, 30, 45, 60, 75, 90] if h <= max_forward]

                for horizon in horizons:
                    target_row = atm_data.iloc[idx + horizon]

                    features = self._build_features(
                        base_row=base_row,
                        target_date=target_row[ColumnNames.ORDER_DATE],
                        horizon=horizon,
                        atm_stats=atm_stats,
                        historical_data=atm_data.iloc[:idx + 1],
                    )
                    features['target'] = target_row[ColumnNames.AMOUNT]
                    all_examples.append(features)

        training_df = pd.DataFrame(all_examples)
        logger.info(f"  {len(training_df)} exemples d'entraînement générés")
        return training_df

    def _build_features(
        self,
        base_row: pd.Series,
        target_date: datetime,
        horizon: int,
        atm_stats: Dict,
        historical_data: pd.DataFrame,
    ) -> Dict:
        """Construit les features pour une prédiction."""
        features = {}

        # === HORIZON ===
        features['horizon'] = horizon
        features['horizon_log'] = np.log1p(horizon)
        features['is_short_term'] = 1 if horizon <= self.SHORT_TERM_THRESHOLD else 0

        # === ATM ===
        features['atm_id'] = base_row[ColumnNames.ATM_ID]
        features['atm_mean'] = atm_stats.get('mean', self.global_stats.get('mean', 0))
        features['atm_std'] = atm_stats.get('std', self.global_stats.get('std', 0))
        features['atm_median'] = atm_stats.get('median', self.global_stats.get('median', 0))
        features['atm_avg_frequency'] = atm_stats.get('avg_frequency', 0)

        # === TEMPOREL (date cible) ===
        target_dt = pd.Timestamp(target_date)
        features['target_weekday'] = target_dt.weekday()
        features['target_month'] = target_dt.month
        features['target_day'] = target_dt.day
        features['target_quarter'] = target_dt.quarter
        features['target_is_weekend'] = 1 if target_dt.weekday() >= 5 else 0
        features['target_is_month_start'] = 1 if target_dt.day <= 5 else 0
        features['target_is_month_end'] = 1 if target_dt.day > 25 else 0
        features['target_day_of_year'] = target_dt.dayofyear

        # Cycliques
        features['target_weekday_sin'] = np.sin(2 * np.pi * target_dt.weekday() / 7)
        features['target_weekday_cos'] = np.cos(2 * np.pi * target_dt.weekday() / 7)
        features['target_month_sin'] = np.sin(2 * np.pi * target_dt.month / 12)
        features['target_month_cos'] = np.cos(2 * np.pi * target_dt.month / 12)
        features['target_day_of_year_sin'] = np.sin(2 * np.pi * target_dt.dayofyear / 365.25)
        features['target_day_of_year_cos'] = np.cos(2 * np.pi * target_dt.dayofyear / 365.25)

        # Moyennes historiques par jour/mois
        weekday_means = atm_stats.get('weekday_means', {})
        features['atm_weekday_mean'] = weekday_means.get(target_dt.weekday(), atm_stats.get('mean', 0))
        month_means = atm_stats.get('month_means', {})
        features['atm_month_mean'] = month_means.get(target_dt.month, atm_stats.get('mean', 0))

        # === FEATURES ATM (de la ligne de base) ===
        for col in ['volatilite_dmq', 'evenement_en_cours', 'risque_atm_vide',
                     'total_soldes', 'total_ajustement', 'total_k7hs', 'cassettes_actives']:
            if col in base_row.index:
                features[col] = base_row[col]
            else:
                features[col] = 0

        # === FEATURES DE LAG (historique commandes) ===
        if horizon <= self.SHORT_TERM_THRESHOLD and len(historical_data) > 0:
            features['last_amount'] = base_row[ColumnNames.AMOUNT]

            if len(historical_data) >= 2:
                features['prev_amount'] = historical_data.iloc[-2][ColumnNames.AMOUNT]
            else:
                features['prev_amount'] = atm_stats.get('mean', 0)

            # Moyenne des 5 dernières commandes
            tail5 = historical_data.tail(5)[ColumnNames.AMOUNT]
            features['rolling_mean_5cmd'] = tail5.mean()

            # Moyenne des 10 dernières commandes
            tail10 = historical_data.tail(10)[ColumnNames.AMOUNT]
            features['rolling_mean_10cmd'] = tail10.mean()

            # Écart-type récent
            features['rolling_std_5cmd'] = tail5.std() if len(tail5) > 1 else 0

            # Tendance
            features['recent_trend'] = features['rolling_mean_5cmd'] - features['rolling_mean_10cmd']

            # Jours depuis dernière commande
            if ColumnNames.DAYS_SINCE_LAST_ORDER in base_row.index:
                features['days_since_last'] = base_row[ColumnNames.DAYS_SINCE_LAST_ORDER]
            else:
                features['days_since_last'] = 0
        else:
            mean_val = atm_stats.get('mean', 0)
            features['last_amount'] = mean_val
            features['prev_amount'] = mean_val
            features['rolling_mean_5cmd'] = mean_val
            features['rolling_mean_10cmd'] = mean_val
            features['rolling_std_5cmd'] = atm_stats.get('std', 0)
            features['recent_trend'] = 0
            features['days_since_last'] = atm_stats.get('avg_frequency', 0)

        return features

    def fit(self, data: pd.DataFrame, eval_data: Optional[pd.DataFrame] = None) -> 'CatBoostForecaster':
        """Entraîne le modèle CatBoost."""
        logger.info(f"Entraînement du modèle {self.name}")
        logger.info(f"  Données d'entrée : {len(data)} lignes")

        self.training_data = data.copy()

        # 1. Stats historiques
        self._compute_historical_stats(data)

        # 2. Features
        training_df = self._prepare_features_for_training(data)

        # 3. Séparer features / target
        self.feature_columns = [col for col in training_df.columns if col != 'target']
        X_train = training_df[self.feature_columns]
        y_train = training_df['target']

        # 4. Features catégorielles
        self.cat_features = ['atm_id']
        cat_feature_indices = [self.feature_columns.index(f) for f in self.cat_features if f in self.feature_columns]

        logger.info(f"  Features : {len(self.feature_columns)} colonnes")

        # 5. Validation split
        if eval_data is not None:
            eval_df = self._prepare_features_for_training(eval_data)
            X_eval = eval_df[self.feature_columns]
            y_eval = eval_df['target']
        else:
            split_idx = int(len(X_train) * 0.8)
            X_eval = X_train.iloc[split_idx:]
            y_eval = y_train.iloc[split_idx:]
            X_train = X_train.iloc[:split_idx]
            y_train = y_train.iloc[:split_idx]

        eval_set = Pool(X_eval, y_eval, cat_features=cat_feature_indices)
        train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)

        # 6. Entraînement
        self.model = CatBoostRegressor(**self.model_params)
        logger.info("  Début de l'entraînement...")
        self.model.fit(train_pool, eval_set=eval_set, verbose=self.verbose)
        self.is_fitted = True

        best_iter = self.model.get_best_iteration()
        logger.info(f"  Entraînement terminé (meilleure itération : {best_iter})")

        # Feature importance
        importance = self.model.get_feature_importance()
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance,
        }).sort_values('importance', ascending=False)

        logger.info("  Top 5 features :")
        for _, row in importance_df.head(5).iterrows():
            logger.info(f"    - {row['feature']}: {row['importance']:.2f}")

        return self

    def predict(
        self,
        atm_id: int,
        prediction_dates: List[datetime],
        context_data: Optional[pd.DataFrame] = None,
        horizon: Optional[int] = None,
    ) -> np.ndarray:
        """Génère des prédictions pour un ATM sur des dates données."""
        if not self.is_fitted:
            raise ValueError(f"Modèle {self.name} non entraîné.")

        atm_stats = self.atm_historical_stats.get(atm_id, self.global_stats)

        # Récupérer l'historique
        if context_data is not None:
            atm_history = context_data[context_data[ColumnNames.ATM_ID] == atm_id].copy()
            atm_history = atm_history.sort_values(ColumnNames.ORDER_DATE)
        else:
            atm_history = self.training_data[self.training_data[ColumnNames.ATM_ID] == atm_id].copy()
            atm_history = atm_history.sort_values(ColumnNames.ORDER_DATE)

        base_row = atm_history.iloc[-1] if len(atm_history) > 0 else None
        base_date = base_row[ColumnNames.ORDER_DATE] if base_row is not None else None

        predictions = []
        for pred_date in prediction_dates:
            if horizon is not None:
                h = horizon
            elif base_date is not None:
                h = max(1, (pd.Timestamp(pred_date) - pd.Timestamp(base_date)).days)
            else:
                h = 1
            h = min(h, self.max_horizon)

            if base_row is not None:
                features = self._build_features(
                    base_row=base_row,
                    target_date=pred_date,
                    horizon=h,
                    atm_stats=atm_stats,
                    historical_data=atm_history,
                )
            else:
                features = self._build_default_features(atm_id, pred_date, h, atm_stats)

            feature_values = [features.get(col, 0) for col in self.feature_columns]
            X_pred = pd.DataFrame([feature_values], columns=self.feature_columns)
            pred = max(0, self.model.predict(X_pred)[0])
            predictions.append(pred)

        return np.array(predictions)

    def _build_default_features(self, atm_id, target_date, horizon, atm_stats):
        """Features par défaut quand il n'y a pas d'historique."""
        features = {}

        features['horizon'] = horizon
        features['horizon_log'] = np.log1p(horizon)
        features['is_short_term'] = 1 if horizon <= self.SHORT_TERM_THRESHOLD else 0

        features['atm_id'] = atm_id
        features['atm_mean'] = atm_stats.get('mean', self.global_stats.get('mean', 0))
        features['atm_std'] = atm_stats.get('std', self.global_stats.get('std', 0))
        features['atm_median'] = atm_stats.get('median', self.global_stats.get('median', 0))
        features['atm_avg_frequency'] = atm_stats.get('avg_frequency', 0)

        target_dt = pd.Timestamp(target_date)
        features['target_weekday'] = target_dt.weekday()
        features['target_month'] = target_dt.month
        features['target_day'] = target_dt.day
        features['target_quarter'] = target_dt.quarter
        features['target_is_weekend'] = 1 if target_dt.weekday() >= 5 else 0
        features['target_is_month_start'] = 1 if target_dt.day <= 5 else 0
        features['target_is_month_end'] = 1 if target_dt.day > 25 else 0
        features['target_day_of_year'] = target_dt.dayofyear

        features['target_weekday_sin'] = np.sin(2 * np.pi * target_dt.weekday() / 7)
        features['target_weekday_cos'] = np.cos(2 * np.pi * target_dt.weekday() / 7)
        features['target_month_sin'] = np.sin(2 * np.pi * target_dt.month / 12)
        features['target_month_cos'] = np.cos(2 * np.pi * target_dt.month / 12)
        features['target_day_of_year_sin'] = np.sin(2 * np.pi * target_dt.dayofyear / 365.25)
        features['target_day_of_year_cos'] = np.cos(2 * np.pi * target_dt.dayofyear / 365.25)

        weekday_means = atm_stats.get('weekday_means', {})
        features['atm_weekday_mean'] = weekday_means.get(target_dt.weekday(), atm_stats.get('mean', 0))
        month_means = atm_stats.get('month_means', {})
        features['atm_month_mean'] = month_means.get(target_dt.month, atm_stats.get('mean', 0))

        for col in ['volatilite_dmq', 'evenement_en_cours', 'risque_atm_vide',
                     'total_soldes', 'total_ajustement', 'total_k7hs', 'cassettes_actives']:
            features[col] = 0

        mean_val = atm_stats.get('mean', 0)
        features['last_amount'] = mean_val
        features['prev_amount'] = mean_val
        features['rolling_mean_5cmd'] = mean_val
        features['rolling_mean_10cmd'] = mean_val
        features['rolling_std_5cmd'] = atm_stats.get('std', 0)
        features['recent_trend'] = 0
        features['days_since_last'] = atm_stats.get('avg_frequency', 0)

        return features

    def predict_horizon(
        self,
        atm_id: int,
        start_date: datetime,
        horizon_days: int,
        context_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Prédit pour un ATM sur un horizon donné."""
        if horizon_days > self.max_horizon:
            logger.warning(f"Horizon demandé ({horizon_days}j) > max ({self.max_horizon}j)")
            horizon_days = self.max_horizon

        prediction_dates = pd.date_range(start=start_date, periods=horizon_days, freq='D')

        predictions = []
        for i, pred_date in enumerate(prediction_dates):
            pred = self.predict(atm_id, [pred_date], context_data, horizon=i + 1)[0]
            predictions.append({
                ColumnNames.ORDER_DATE: pred_date,
                ColumnNames.ATM_ID: atm_id,
                'prediction': pred,
                'horizon': i + 1,
            })

        return pd.DataFrame(predictions)

    def evaluate_by_horizon(
        self,
        test_data: pd.DataFrame,
        horizons: List[int] = [1, 7, 14, 30, 60, 90],
    ) -> pd.DataFrame:
        """Évalue le modèle par horizon de prédiction."""
        logger.info("Évaluation par horizon...")
        results = []

        test_data_sorted = test_data.sort_values([ColumnNames.ATM_ID, ColumnNames.ORDER_DATE])

        for horizon in horizons:
            if horizon > self.max_horizon:
                continue

            predictions = []
            actuals = []

            for atm_id in test_data[ColumnNames.ATM_ID].unique():
                atm_test = test_data_sorted[test_data_sorted[ColumnNames.ATM_ID] == atm_id]
                if len(atm_test) <= horizon:
                    continue

                for i in range(len(atm_test) - horizon):
                    base_date = atm_test.iloc[i][ColumnNames.ORDER_DATE]
                    target_date = atm_test.iloc[i + horizon][ColumnNames.ORDER_DATE]
                    actual = atm_test.iloc[i + horizon][ColumnNames.AMOUNT]

                    context = test_data[
                        (test_data[ColumnNames.ATM_ID] == atm_id) &
                        (test_data[ColumnNames.ORDER_DATE] <= base_date)
                    ]

                    pred = self.predict(atm_id, [target_date], context, horizon=horizon)[0]
                    predictions.append(pred)
                    actuals.append(actual)

            if predictions:
                predictions = np.array(predictions)
                actuals = np.array(actuals)

                mae = mean_absolute_error(actuals, predictions)
                rmse = np.sqrt(mean_squared_error(actuals, predictions))

                non_zero_mask = actuals != 0
                mape = (
                    np.mean(np.abs((actuals[non_zero_mask] - predictions[non_zero_mask]) / actuals[non_zero_mask])) * 100
                    if non_zero_mask.any() else np.inf
                )

                results.append({
                    'horizon': horizon, 'mae': mae, 'rmse': rmse,
                    'mape': mape, 'n_samples': len(predictions),
                })
                logger.info(f"  H+{horizon:2d}: MAE={mae:.2f}, RMSE={rmse:.2f}, "
                            f"MAPE={mape:.1f}%, n={len(predictions)}")

        return pd.DataFrame(results)

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Modèle non entraîné")
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.get_feature_importance(),
        }).sort_values('importance', ascending=False)

    def save_model(self, path: Union[str, Path]) -> None:
        if not self.is_fitted:
            raise ValueError("Modèle non entraîné")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save_model(str(path / 'catboost_model.cbm'))

        metadata = {
            'name': self.name,
            'max_horizon': self.max_horizon,
            'feature_columns': self.feature_columns,
            'cat_features': self.cat_features,
            'atm_historical_stats': self.atm_historical_stats,
            'global_stats': self.global_stats,
            'model_params': self.model_params,
        }
        with open(path / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Modèle sauvegardé : {path}")

    @classmethod
    def load_model(cls, path: Union[str, Path]) -> 'CatBoostForecaster':
        path = Path(path)

        with open(path / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        instance = cls(max_horizon=metadata['max_horizon'])
        instance.name = metadata['name']
        instance.feature_columns = metadata['feature_columns']
        instance.cat_features = metadata['cat_features']
        instance.atm_historical_stats = metadata['atm_historical_stats']
        instance.global_stats = metadata['global_stats']
        instance.model_params = metadata['model_params']

        instance.model = CatBoostRegressor()
        instance.model.load_model(str(path / 'catboost_model.cbm'))
        instance.is_fitted = True

        logger.info(f"Modèle chargé : {path}")
        return instance


# =============================================================================
#  CatBoost DMQ par coupure
# =============================================================================


class CatBoostDmqForecaster(CatBoostForecaster):
    """Variante de ``CatBoostForecaster`` qui prédit le **DMQ d'une coupure**
    donnée au lieu du montant total ``amount``.

    On instancie typiquement 5 modèles (un par coupure) via
    :class:`MultiCoupureForecaster`.
    """

    def __init__(self, coupure: int, target_column: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.coupure = int(coupure)
        # Par défaut la colonne cible est ``dmq_{coupure}`` (cf. ColumnNames).
        self.target_column = target_column or f"dmq_{self.coupure}"
        self.name = f"CatBoostDMQ_{self.coupure}_H{self.max_horizon}"

    def _prepare_features_for_training(self, data: pd.DataFrame) -> pd.DataFrame:
        """Même logique que la classe parente mais en lisant la target
        configurée (``dmq_{coupure}``) au lieu de ``amount``."""
        logger.info(
            f"[{self.name}] Préparation features pour la coupure {self.coupure}€"
        )

        if self.target_column not in data.columns:
            raise ValueError(
                f"Colonne cible '{self.target_column}' absente des données. "
                f"Colonnes disponibles : {list(data.columns)[:10]}..."
            )

        all_examples = []
        data = data.sort_values([ColumnNames.ATM_ID, ColumnNames.ORDER_DATE]).reset_index(drop=True)

        for atm_id in data[ColumnNames.ATM_ID].unique():
            atm_data = data[data[ColumnNames.ATM_ID] == atm_id].reset_index(drop=True)
            atm_stats = self.atm_historical_stats.get(atm_id, {})

            for idx in range(len(atm_data) - 1):
                base_row = atm_data.iloc[idx]

                max_forward = min(len(atm_data) - idx - 1, self.max_horizon)
                horizons = list(range(1, min(15, max_forward + 1)))
                horizons += [h for h in [21, 30, 45, 60, 75, 90] if h <= max_forward]

                for horizon in horizons:
                    target_row = atm_data.iloc[idx + horizon]

                    features = self._build_features(
                        base_row=base_row,
                        target_date=target_row[ColumnNames.ORDER_DATE],
                        horizon=horizon,
                        atm_stats=atm_stats,
                        historical_data=atm_data.iloc[:idx + 1],
                    )
                    # Target = DMQ de la coupure (ex: dmq_5)
                    features['target'] = float(target_row[self.target_column])
                    all_examples.append(features)

        training_df = pd.DataFrame(all_examples)
        logger.info(
            f"[{self.name}] {len(training_df)} exemples générés (target={self.target_column})"
        )
        return training_df


class MultiCoupureForecaster:
    """Wrapper qui entraîne un :class:`CatBoostDmqForecaster` par coupure et
    expose une API de prédiction unifiée.

    Produit un ``Dict[coupure, float]`` exploitable directement par
    ``CommandPipeline`` (via un ``DmqProvider``).
    """

    def __init__(self, coupures: Optional[List[int]] = None, **kwargs):
        self.coupures = coupures or [5, 10, 20, 50, 100]
        self.models: Dict[int, CatBoostDmqForecaster] = {
            c: CatBoostDmqForecaster(coupure=c, **kwargs) for c in self.coupures
        }
        self.is_fitted = False

    def fit(self, data: pd.DataFrame, eval_data: Optional[pd.DataFrame] = None) -> 'MultiCoupureForecaster':
        for coupure, model in self.models.items():
            logger.info(f"=== Entraînement modèle coupure {coupure}€ ===")
            model.fit(data, eval_data=eval_data)
        self.is_fitted = True
        return self

    def predict_dmq_par_coupure(
        self,
        atm_id: int,
        prediction_date: datetime,
        context_data: Optional[pd.DataFrame] = None,
        horizon: Optional[int] = None,
    ) -> Dict[int, float]:
        """Retourne un dict ``{coupure: dmq_prédit}`` pour un ATM et une date."""
        if not self.is_fitted:
            raise ValueError("MultiCoupureForecaster non entraîné.")

        out: Dict[int, float] = {}
        for coupure, model in self.models.items():
            pred = model.predict(
                atm_id=atm_id,
                prediction_dates=[prediction_date],
                context_data=context_data,
                horizon=horizon,
            )[0]
            out[coupure] = float(max(0.0, pred))
        return out

    def as_dmq_provider(
        self,
        prediction_date: datetime,
        context_data: Optional[pd.DataFrame] = None,
        horizon: Optional[int] = None,
    ):
        """Adaptateur vers l'interface ``DmqProvider`` de ``CommandPipeline``."""

        def _provider(atm_id: int) -> Dict[int, float]:
            return self.predict_dmq_par_coupure(
                atm_id=atm_id,
                prediction_date=prediction_date,
                context_data=context_data,
                horizon=horizon,
            )

        return _provider


# ===== FONCTIONS UTILITAIRES =====

def train_and_evaluate_catboost(
    enriched_data: pd.DataFrame,
    max_horizon: int = 90,
    test_ratio: float = 0.2,
) -> Tuple[CatBoostForecaster, pd.DataFrame]:
    """Entraîne et évalue un modèle CatBoost."""
    logger.info("ENTRAÎNEMENT ET ÉVALUATION CATBOOST")
    logger.info("=" * 50)

    dates = sorted(enriched_data[ColumnNames.ORDER_DATE].unique())
    split_idx = int(len(dates) * (1 - test_ratio))
    split_date = dates[split_idx]

    train_data = enriched_data[enriched_data[ColumnNames.ORDER_DATE] < split_date].copy()
    test_data = enriched_data[enriched_data[ColumnNames.ORDER_DATE] >= split_date].copy()

    logger.info(f"  Train : {len(train_data)} lignes jusqu'au {split_date.date()}")
    logger.info(f"  Test  : {len(test_data)} lignes à partir du {split_date.date()}")

    model = CatBoostForecaster(max_horizon=max_horizon)
    model.fit(train_data)

    results = model.evaluate_by_horizon(test_data)

    global_metrics = model.evaluate(test_data)
    logger.info(f"\nMÉTRIQUES GLOBALES :")
    logger.info(f"  MAE  : {global_metrics['mae']:.2f}")
    logger.info(f"  RMSE : {global_metrics['rmse']:.2f}")
    logger.info(f"  MAPE : {global_metrics['mape']:.1f}%")

    return model, results


if __name__ == "__main__":
    print("=" * 60)
    print("TEST DU MODÈLE CATBOOST ATM")
    print("=" * 60)

    try:
        enriched_file = get_file_path('enriched')

        if not enriched_file.exists():
            print("Fichier de données enrichies non trouvé.")
            print("Exécutez d'abord : python main.py --step enrichment")
        else:
            enriched_data = pd.read_csv(
                enriched_file,
                parse_dates=[ColumnNames.ORDER_DATE],
                date_format='%Y-%m-%d',
            )

            print(f"\nDonnées chargées : {len(enriched_data)} lignes")

            model, results = train_and_evaluate_catboost(enriched_data, max_horizon=90)

            print("\nRÉSULTATS PAR HORIZON :")
            print(results.to_string(index=False))

            model.save_model("data/output/catboost_model")
            print("\nTest terminé avec succès !")

    except Exception as e:
        print(f"\nErreur : {e}")
        import traceback
        traceback.print_exc()
