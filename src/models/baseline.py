"""
Modèles de baseline pour la prévision de montants de commandes ATM.

Modèles :
- Naïf (dernier montant observé)
- Moyenne historique
- Moyenne mobile (dernières N commandes par ATM)
- Moyenne par jour de semaine
- Saisonnalité naïve (même jour semaine précédente)
- Tendance linéaire
- Ensemble de baselines

Target : amount (DC_Montant_Cmd)
Entité : atm_id (DC_Automate_Id)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from ..utils import (
        ColumnNames,
        Messages,
        WEEKDAY_NAMES,
        get_file_path,
    )
except ImportError:
    from src.utils import (
        ColumnNames,
        Messages,
        WEEKDAY_NAMES,
        get_file_path,
    )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaselineModel(ABC):
    """Classe abstraite de base pour tous les modèles de baseline."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.training_data = None
        self.atm_stats = {}

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BaselineModel':
        pass

    @abstractmethod
    def predict(
        self,
        atm_id: int,
        prediction_dates: List[datetime],
        context_data: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        pass

    def predict_atm_range(
        self,
        atm_id: int,
        start_date: datetime,
        end_date: datetime,
        context_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Prédit pour un ATM sur une plage de dates."""
        if not self.is_fitted:
            raise ValueError(f"Modèle {self.name} non entraîné.")

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        predictions = self.predict(atm_id, date_range.tolist(), context_data)

        return pd.DataFrame({
            ColumnNames.ORDER_DATE: date_range,
            ColumnNames.ATM_ID: atm_id,
            'prediction': predictions,
            'model': self.name,
        })

    def evaluate(
        self,
        test_data: pd.DataFrame,
        metrics: List[str] = ['mae', 'rmse', 'mape'],
    ) -> Dict[str, float]:
        """Évalue la performance sur des données de test."""
        if not self.is_fitted:
            raise ValueError(f"Modèle {self.name} non entraîné.")

        predictions = []
        actuals = []

        for _, row in test_data.iterrows():
            pred = self.predict(
                atm_id=row[ColumnNames.ATM_ID],
                prediction_dates=[row[ColumnNames.ORDER_DATE]],
                context_data=test_data,
            )[0]
            predictions.append(pred)
            actuals.append(row[ColumnNames.AMOUNT])

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        results = {}

        if 'mae' in metrics:
            results['mae'] = mean_absolute_error(actuals, predictions)

        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(mean_squared_error(actuals, predictions))

        if 'mape' in metrics:
            non_zero_mask = actuals != 0
            if non_zero_mask.any():
                results['mape'] = np.mean(
                    np.abs((actuals[non_zero_mask] - predictions[non_zero_mask]) / actuals[non_zero_mask])
                ) * 100
            else:
                results['mape'] = np.inf

        return results


class NaiveBaseline(BaselineModel):
    """Prédit le dernier montant observé pour chaque ATM."""

    def __init__(self):
        super().__init__("Naive")

    def fit(self, data: pd.DataFrame) -> 'NaiveBaseline':
        logger.info(f"Entraînement du modèle {self.name}")
        self.training_data = data.copy()

        latest_values = (
            data.sort_values(ColumnNames.ORDER_DATE)
            .groupby(ColumnNames.ATM_ID)[ColumnNames.AMOUNT]
            .last()
            .to_dict()
        )
        self.atm_stats = latest_values
        self.is_fitted = True
        logger.info(f"  {len(latest_values)} ATMs entraînés")
        return self

    def predict(self, atm_id, prediction_dates, context_data=None):
        last_value = self.atm_stats.get(atm_id, 0)
        return np.full(len(prediction_dates), last_value)


class HistoricalMeanBaseline(BaselineModel):
    """Prédit la moyenne historique des montants pour chaque ATM."""

    def __init__(self):
        super().__init__("Historical_Mean")

    def fit(self, data: pd.DataFrame) -> 'HistoricalMeanBaseline':
        logger.info(f"Entraînement du modèle {self.name}")
        self.training_data = data.copy()

        self.atm_stats = data.groupby(ColumnNames.ATM_ID)[ColumnNames.AMOUNT].mean().to_dict()
        self.is_fitted = True
        logger.info(f"  {len(self.atm_stats)} ATMs entraînés")
        return self

    def predict(self, atm_id, prediction_dates, context_data=None):
        mean_value = self.atm_stats.get(atm_id, 0.0)
        return np.full(len(prediction_dates), mean_value)


class MovingAverageBaseline(BaselineModel):
    """
    Moyenne mobile des N dernières commandes par ATM.

    IMPORTANT : La fenêtre est en nombre de commandes (.tail(N)),
    PAS en jours. Un ATM rechargé rarement aurait 0 commandes
    dans une fenêtre de 7 jours.
    """

    def __init__(self, window: int = 7):
        super().__init__(f"Moving_Average_{window}cmd")
        self.window = window

    def fit(self, data: pd.DataFrame) -> 'MovingAverageBaseline':
        logger.info(f"Entraînement du modèle {self.name}")
        self.training_data = data.sort_values(
            [ColumnNames.ATM_ID, ColumnNames.ORDER_DATE]
        ).copy()
        self.is_fitted = True
        logger.info(f"  {data[ColumnNames.ATM_ID].nunique()} ATMs préparés "
                     f"(fenêtre = {self.window} dernières commandes)")
        return self

    def predict(self, atm_id, prediction_dates, context_data=None):
        atm_data = self.training_data[
            self.training_data[ColumnNames.ATM_ID] == atm_id
        ]

        if atm_data.empty:
            return np.zeros(len(prediction_dates))

        # Dernières N commandes (pas jours !)
        recent_orders = atm_data.tail(self.window)
        mean_amount = recent_orders[ColumnNames.AMOUNT].mean()

        return np.full(len(prediction_dates), mean_amount)


class WeekdayMeanBaseline(BaselineModel):
    """Prédit la moyenne par jour de semaine par ATM."""

    def __init__(self):
        super().__init__("Weekday_Mean")

    def fit(self, data: pd.DataFrame) -> 'WeekdayMeanBaseline':
        logger.info(f"Entraînement du modèle {self.name}")

        if ColumnNames.WEEKDAY not in data.columns:
            raise ValueError("Colonne 'weekday' manquante. Utilisez des données enrichies.")

        self.training_data = data.copy()

        weekday_means = (
            data.groupby([ColumnNames.ATM_ID, ColumnNames.WEEKDAY])[ColumnNames.AMOUNT]
            .mean()
            .reset_index()
        )

        self.atm_stats = {}
        for _, row in weekday_means.iterrows():
            atm_id = row[ColumnNames.ATM_ID]
            weekday = row[ColumnNames.WEEKDAY]
            if atm_id not in self.atm_stats:
                self.atm_stats[atm_id] = {}
            self.atm_stats[atm_id][weekday] = row[ColumnNames.AMOUNT]

        self.is_fitted = True
        logger.info(f"  {len(self.atm_stats)} ATMs entraînés avec patterns hebdomadaires")
        return self

    def predict(self, atm_id, prediction_dates, context_data=None):
        predictions = []
        for pred_date in prediction_dates:
            weekday = pred_date.weekday()
            if atm_id in self.atm_stats and weekday in self.atm_stats[atm_id]:
                predictions.append(self.atm_stats[atm_id][weekday])
            elif atm_id in self.atm_stats:
                predictions.append(np.mean(list(self.atm_stats[atm_id].values())))
            else:
                predictions.append(0)
        return np.array(predictions)


class SeasonalNaiveBaseline(BaselineModel):
    """Prédit le montant du même jour de semaine de la semaine précédente."""

    def __init__(self):
        super().__init__("Seasonal_Naive")

    def fit(self, data: pd.DataFrame) -> 'SeasonalNaiveBaseline':
        logger.info(f"Entraînement du modèle {self.name}")
        self.training_data = data.sort_values(
            [ColumnNames.ATM_ID, ColumnNames.ORDER_DATE]
        ).copy()
        self.is_fitted = True
        logger.info(f"  {data[ColumnNames.ATM_ID].nunique()} ATMs préparés")
        return self

    def predict(self, atm_id, prediction_dates, context_data=None):
        atm_data = self.training_data[
            self.training_data[ColumnNames.ATM_ID] == atm_id
        ].copy()

        if atm_data.empty:
            return np.zeros(len(prediction_dates))

        predictions = []
        for pred_date in prediction_dates:
            previous_week_date = pred_date - timedelta(days=7)
            matching = atm_data[atm_data[ColumnNames.ORDER_DATE] == previous_week_date]

            if not matching.empty:
                predictions.append(matching.iloc[0][ColumnNames.AMOUNT])
            else:
                # Fallback: fenêtre de +/- 3 jours
                window_start = previous_week_date - timedelta(days=3)
                window_end = previous_week_date + timedelta(days=3)
                window_data = atm_data[
                    (atm_data[ColumnNames.ORDER_DATE] >= window_start) &
                    (atm_data[ColumnNames.ORDER_DATE] <= window_end)
                ]
                if not window_data.empty:
                    predictions.append(window_data[ColumnNames.AMOUNT].mean())
                else:
                    predictions.append(0)

        return np.array(predictions)


class TrendBaseline(BaselineModel):
    """Tendance linéaire sur les dernières commandes et extrapolation."""

    def __init__(self, window_orders: int = 30):
        super().__init__(f"Trend_{window_orders}cmd")
        self.window_orders = window_orders

    def fit(self, data: pd.DataFrame) -> 'TrendBaseline':
        logger.info(f"Entraînement du modèle {self.name}")
        self.training_data = data.copy()
        self.atm_stats = {}

        for atm_id in data[ColumnNames.ATM_ID].unique():
            atm_data = data[data[ColumnNames.ATM_ID] == atm_id].sort_values(ColumnNames.ORDER_DATE)
            recent = atm_data.tail(self.window_orders)

            if len(recent) >= 2:
                x = np.arange(len(recent))
                y = recent[ColumnNames.AMOUNT].values
                slope = np.polyfit(x, y, 1)[0]

                self.atm_stats[atm_id] = {
                    'slope': slope,
                    'last_value': y[-1],
                    'last_date': recent.iloc[-1][ColumnNames.ORDER_DATE],
                }
            elif len(recent) == 1:
                self.atm_stats[atm_id] = {
                    'slope': 0,
                    'last_value': recent.iloc[0][ColumnNames.AMOUNT],
                    'last_date': recent.iloc[0][ColumnNames.ORDER_DATE],
                }

        self.is_fitted = True
        logger.info(f"  {len(self.atm_stats)} ATMs avec tendances calculées")
        return self

    def predict(self, atm_id, prediction_dates, context_data=None):
        if atm_id not in self.atm_stats:
            return np.zeros(len(prediction_dates))

        stats = self.atm_stats[atm_id]
        slope = stats['slope']
        last_value = stats['last_value']
        last_date = stats['last_date']

        predictions = []
        for pred_date in prediction_dates:
            days_ahead = (pred_date - last_date).days
            prediction = max(0, last_value + slope * days_ahead)
            predictions.append(prediction)

        return np.array(predictions)


class BaselineEnsemble(BaselineModel):
    """Combine plusieurs modèles de baseline avec pondération."""

    def __init__(self, models: List[BaselineModel], weights: Optional[List[float]] = None):
        super().__init__("Ensemble_Baseline")
        self.models = models
        self.weights = weights or [1.0] * len(models)

        if len(self.weights) != len(self.models):
            raise ValueError("Le nombre de poids doit égaler le nombre de modèles")

        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

    def fit(self, data: pd.DataFrame) -> 'BaselineEnsemble':
        logger.info(f"Entraînement de l'ensemble de {len(self.models)} modèles")
        for model in self.models:
            model.fit(data)
        self.is_fitted = True
        logger.info("Ensemble complet entraîné")
        return self

    def predict(self, atm_id, prediction_dates, context_data=None):
        all_predictions = [
            model.predict(atm_id, prediction_dates, context_data)
            for model in self.models
        ]
        return np.average(all_predictions, axis=0, weights=self.weights)


# ===== FONCTIONS UTILITAIRES =====

def create_baseline_suite() -> List[BaselineModel]:
    """Crée une suite complète de modèles de baseline."""
    return [
        NaiveBaseline(),
        HistoricalMeanBaseline(),
        MovingAverageBaseline(window=5),
        MovingAverageBaseline(window=10),
        WeekdayMeanBaseline(),
        SeasonalNaiveBaseline(),
        TrendBaseline(window_orders=30),
    ]


def evaluate_all_baselines(
    baselines: List[BaselineModel],
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> pd.DataFrame:
    """Entraîne et évalue tous les modèles de baseline."""
    logger.info("ÉVALUATION COMPLÈTE DES BASELINES")
    logger.info("=" * 50)

    results = []

    for baseline in baselines:
        try:
            baseline.fit(train_data)
            metrics = baseline.evaluate(test_data)

            results.append({'model': baseline.name, **metrics})
            logger.info(f"{baseline.name}: MAE={metrics.get('mae', 'N/A'):.2f}, "
                        f"RMSE={metrics.get('rmse', 'N/A'):.2f}")

        except Exception as e:
            logger.error(f"Erreur avec {baseline.name}: {e}")
            results.append({
                'model': baseline.name,
                'mae': np.inf, 'rmse': np.inf, 'mape': np.inf,
                'error': str(e),
            })

    results_df = pd.DataFrame(results).sort_values('mae')

    logger.info("=" * 50)
    logger.info("CLASSEMENT FINAL DES BASELINES:")
    for idx, (_, row) in enumerate(results_df.iterrows(), 1):
        logger.info(f"  {idx}. {row['model']}: MAE={row['mae']:.2f}")

    return results_df


if __name__ == "__main__":
    try:
        from src.data_processing import DataEnrichmentPipeline

        pipeline = DataEnrichmentPipeline()
        data = pipeline.load_clean_data()

        naive = NaiveBaseline()
        naive.fit(data)

        test_atm = data[ColumnNames.ATM_ID].iloc[0]
        test_dates = [datetime(2026, 12, 1), datetime(2026, 12, 2)]
        predictions = naive.predict(test_atm, test_dates)

        print(f"Test réussi - Prédictions: {predictions}")

    except Exception as e:
        print(f"Erreur lors du test : {e}")
