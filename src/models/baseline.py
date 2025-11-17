"""
Modèles de baseline pour la prévision de commandes.

Ce module implémente des modèles de référence simples mais robustes :
- Modèle naïf (dernière valeur observée)
- Moyenne historique
- Moyenne mobile
- Moyenne par jour de semaine
- Saisonnalité naive (même jour semaine précédente)
- Tendance linéaire simple
- Ensemble de baselines

Ces modèles servent de références que les modèles avancés doivent battre.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from ..utils import (
        ColumnNames,
        Messages,
        WEEKDAY_NAMES,
        get_file_path
    )
except ImportError:
    from src.utils import (
        ColumnNames,
        Messages,
        WEEKDAY_NAMES,
        get_file_path
    )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaselineModel(ABC):
    """
    Classe abstraite de base pour tous les modèles de baseline.

    Définit l'interface commune que tous les modèles doivent implémenter.
    """

    def __init__(self, name: str):
        """
        Initialise le modèle de baseline.

        Args:
            name: Nom du modèle pour l'identification
        """
        self.name = name
        self.is_fitted = False
        self.training_data = None
        self.article_stats = {}

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BaselineModel':
        """
        Entraîne le modèle sur les données historiques.

        Args:
            data: Données d'entraînement enrichies

        Returns:
            Self: Instance du modèle entraîné
        """
        pass

    @abstractmethod
    def predict(self,
                article_id: int,
                prediction_dates: List[datetime],
                context_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Génère des prédictions pour un article sur des dates données.

        Args:
            article_id: ID de l'article à prédire
            prediction_dates: Liste des dates pour lesquelles prédire
            context_data: Données contextuelles si nécessaire

        Returns:
            Array: Prédictions pour chaque date
        """
        pass

    def predict_article_range(self,
                              article_id: int,
                              start_date: datetime,
                              end_date: datetime,
                              context_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prédit pour un article sur une plage de dates.

        Args:
            article_id: ID de l'article
            start_date: Date de début
            end_date: Date de fin
            context_data: Données contextuelles

        Returns:
            DataFrame: Prédictions avec colonnes [date, article_id, prediction]
        """
        if not self.is_fitted:
            raise ValueError(f"Modèle {self.name} non entraîné. Appelez fit() d'abord.")

        # Génération des dates
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Prédictions
        predictions = self.predict(article_id, date_range.tolist(), context_data)

        # Formatage des résultats
        result_df = pd.DataFrame({
            ColumnNames.DATE: date_range,
            ColumnNames.ARTICLE_ID: article_id,
            'prediction': predictions,
            'model': self.name
        })

        return result_df

    def evaluate(self,
                 test_data: pd.DataFrame,
                 metrics: List[str] = ['mae', 'rmse', 'mape']) -> Dict[str, float]:
        """
        Évalue la performance du modèle sur des données de test.

        Args:
            test_data: Données de test avec colonnes [date, article_id, quantity]
            metrics: Liste des métriques à calculer

        Returns:
            Dict: Dictionnaire des métriques calculées
        """
        if not self.is_fitted:
            raise ValueError(f"Modèle {self.name} non entraîné.")

        predictions = []
        actuals = []

        # Génération des prédictions pour chaque article/date
        for _, row in test_data.iterrows():
            pred = self.predict(
                article_id=row[ColumnNames.ARTICLE_ID],
                prediction_dates=[row[ColumnNames.DATE]],
                context_data=test_data
            )[0]

            predictions.append(pred)
            actuals.append(row[ColumnNames.QUANTITY])

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Calcul des métriques
        results = {}

        if 'mae' in metrics:
            results['mae'] = mean_absolute_error(actuals, predictions)

        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(mean_squared_error(actuals, predictions))

        if 'mape' in metrics:
            # MAPE avec protection contre division par zéro
            non_zero_mask = actuals != 0
            if non_zero_mask.any():
                mape = np.mean(np.abs((actuals[non_zero_mask] - predictions[non_zero_mask]) / actuals[non_zero_mask])) * 100
                results['mape'] = mape
            else:
                results['mape'] = np.inf

        return results


class NaiveBaseline(BaselineModel):
    """
    Modèle naïf : prédit la dernière valeur observée.

    Simple mais souvent efficace, surtout pour des séries avec peu de variation.
    """

    def __init__(self):
        super().__init__("Naive")

    def fit(self, data: pd.DataFrame) -> 'NaiveBaseline':
        """
        Entraîne le modèle naïf en stockant la dernière valeur de chaque article.
        """
        logger.info(f"🎯 Entraînement du modèle {self.name}")

        self.training_data = data.copy()

        # Pour chaque article, stocker la dernière valeur observée
        latest_values = (
            data.groupby(ColumnNames.ARTICLE_ID)
            .apply(lambda x: x.loc[x[ColumnNames.DATE].idxmax(), ColumnNames.QUANTITY])
            .to_dict()
        )

        self.article_stats = latest_values
        self.is_fitted = True

        logger.info(f"   ✅ {len(latest_values)} articles entraînés")
        return self

    def predict(self,
                article_id: int,
                prediction_dates: List[datetime],
                context_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Prédit la dernière valeur observée pour toutes les dates.
        """
        if article_id not in self.article_stats:
            # Si article non vu, prédire 0
            last_value = 0
        else:
            last_value = self.article_stats[article_id]

        return np.full(len(prediction_dates), last_value)


class HistoricalMeanBaseline(BaselineModel):
    """
    Modèle moyenne historique : prédit la moyenne de toutes les valeurs passées.

    Lisse les variations mais ignore les tendances récentes.
    """

    def __init__(self):
        super().__init__("Historical_Mean")

    def fit(self, data: pd.DataFrame) -> 'HistoricalMeanBaseline':
        """
        Calcule la moyenne historique de chaque article.
        """
        logger.info(f"🎯 Entraînement du modèle {self.name}")

        self.training_data = data.copy()

        # Moyenne de chaque article
        article_means = data.groupby(ColumnNames.ARTICLE_ID)[ColumnNames.QUANTITY].mean().to_dict()

        self.article_stats = article_means
        self.is_fitted = True

        logger.info(f"   ✅ {len(article_means)} articles entraînés")
        return self

    def predict(self,
                article_id: int,
                prediction_dates: List[datetime],
                context_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Prédit la moyenne historique pour toutes les dates.
        """
        if article_id not in self.article_stats:
            mean_value = 0.0
        else:
            mean_value = self.article_stats[article_id]

        return np.full(len(prediction_dates), mean_value)


class MovingAverageBaseline(BaselineModel):
    """
    Modèle moyenne mobile : prédit basé sur la moyenne des N derniers jours.

    Capture mieux les tendances récentes que la moyenne historique.
    """

    def __init__(self, window: int = 7):
        super().__init__(f"Moving_Average_{window}d")
        self.window = window

    def fit(self, data: pd.DataFrame) -> 'MovingAverageBaseline':
        """
        Prépare les données pour le calcul de moyennes mobiles.
        """
        logger.info(f"🎯 Entraînement du modèle {self.name}")

        # Tri des données par article et date
        self.training_data = data.sort_values([ColumnNames.ARTICLE_ID, ColumnNames.DATE]).copy()

        self.is_fitted = True

        unique_articles = data[ColumnNames.ARTICLE_ID].nunique()
        logger.info(f"   ✅ {unique_articles} articles préparés pour moyenne mobile {self.window} jours")

        return self

    def predict(self,
                article_id: int,
                prediction_dates: List[datetime],
                context_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Calcule la moyenne mobile des derniers N jours pour chaque prédiction.
        """
        # Données de l'article
        article_data = self.training_data[
            self.training_data[ColumnNames.ARTICLE_ID] == article_id
        ].copy()

        if article_data.empty:
            return np.zeros(len(prediction_dates))

        predictions = []

        for pred_date in prediction_dates:
            # Date de début pour la moyenne mobile
            start_date = pred_date - timedelta(days=self.window)

            # Filtrage des données dans la fenêtre
            window_data = article_data[
                (article_data[ColumnNames.DATE] >= start_date) &
                (article_data[ColumnNames.DATE] < pred_date)
            ]

            if len(window_data) == 0:
                # Pas d'historique, utiliser 0 ou moyenne globale
                prediction = 0
            else:
                prediction = window_data[ColumnNames.QUANTITY].mean()

            predictions.append(prediction)

        return np.array(predictions)


class WeekdayMeanBaseline(BaselineModel):
    """
    Modèle moyenne par jour de semaine : prédit basé sur la moyenne historique du même jour de la semaine.

    Capture la saisonnalité hebdomadaire (ex: plus de commandes le lundi).
    """

    def __init__(self):
        super().__init__("Weekday_Mean")

    def fit(self, data: pd.DataFrame) -> 'WeekdayMeanBaseline':
        """
        Calcule la moyenne par article et par jour de semaine.
        """
        logger.info(f"🎯 Entraînement du modèle {self.name}")

        if ColumnNames.WEEKDAY not in data.columns:
            raise ValueError("Colonne 'weekday' manquante. Assurez-vous d'avoir des données enrichies.")

        self.training_data = data.copy()

        # Moyenne par article et jour de semaine
        weekday_means = (
            data.groupby([ColumnNames.ARTICLE_ID, ColumnNames.WEEKDAY])[ColumnNames.QUANTITY]
            .mean()
            .reset_index()
        )

        # Conversion en dictionnaire imbriqué {article_id: {weekday: mean}}
        self.article_stats = {}
        for _, row in weekday_means.iterrows():
            article_id = row[ColumnNames.ARTICLE_ID]
            weekday = row[ColumnNames.WEEKDAY]
            mean_qty = row[ColumnNames.QUANTITY]

            if article_id not in self.article_stats:
                self.article_stats[article_id] = {}

            self.article_stats[article_id][weekday] = mean_qty

        self.is_fitted = True

        logger.info(f"   ✅ {len(self.article_stats)} articles entraînés avec patterns hebdomadaires")
        return self

    def predict(self,
                article_id: int,
                prediction_dates: List[datetime],
                context_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Prédit basé sur la moyenne du même jour de semaine.
        """
        predictions = []

        for pred_date in prediction_dates:
            weekday = pred_date.weekday()  # 0=Lundi, 6=Dimanche

            if (article_id in self.article_stats and
                weekday in self.article_stats[article_id]):
                prediction = self.article_stats[article_id][weekday]
            else:
                # Fallback: moyenne globale de l'article ou 0
                if article_id in self.article_stats:
                    # Moyenne de tous les jours de semaine pour cet article
                    prediction = np.mean(list(self.article_stats[article_id].values()))
                else:
                    prediction = 0

            predictions.append(prediction)

        return np.array(predictions)


class SeasonalNaiveBaseline(BaselineModel):
    """
    Modèle saisonnier naïf : prédit la valeur du même jour de semaine de la semaine précédente.

    Combine récence (semaine précédente) et saisonnalité (même jour).
    """

    def __init__(self):
        super().__init__("Seasonal_Naive")

    def fit(self, data: pd.DataFrame) -> 'SeasonalNaiveBaseline':
        """
        Prépare les données pour les prédictions saisonnières.
        """
        logger.info(f"🎯 Entraînement du modèle {self.name}")

        self.training_data = data.sort_values([ColumnNames.ARTICLE_ID, ColumnNames.DATE]).copy()
        self.is_fitted = True

        unique_articles = data[ColumnNames.ARTICLE_ID].nunique()
        logger.info(f"   ✅ {unique_articles} articles préparés pour saisonnalité naive")

        return self

    def predict(self,
                article_id: int,
                prediction_dates: List[datetime],
                context_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Prédit la valeur du même jour de semaine de la semaine précédente.
        """
        article_data = self.training_data[
            self.training_data[ColumnNames.ARTICLE_ID] == article_id
        ].copy()

        if article_data.empty:
            return np.zeros(len(prediction_dates))

        predictions = []

        for pred_date in prediction_dates:
            # Date de la semaine précédente (même jour de semaine)
            previous_week_date = pred_date - timedelta(days=7)

            # Chercher la valeur de cette date
            matching_date = article_data[article_data[ColumnNames.DATE] == previous_week_date]

            if not matching_date.empty:
                prediction = matching_date.iloc[0][ColumnNames.QUANTITY]
            else:
                # Fallback: chercher la date la plus proche dans une fenêtre
                window_start = previous_week_date - timedelta(days=3)
                window_end = previous_week_date + timedelta(days=3)

                window_data = article_data[
                    (article_data[ColumnNames.DATE] >= window_start) &
                    (article_data[ColumnNames.DATE] <= window_end)
                ]

                if not window_data.empty:
                    prediction = window_data[ColumnNames.QUANTITY].mean()
                else:
                    prediction = 0

            predictions.append(prediction)

        return np.array(predictions)


class TrendBaseline(BaselineModel):
    """
    Modèle tendance linéaire : ajuste une tendance linéaire simple et extrapole.

    Capture les tendances à long terme mais ignore la saisonnalité.
    """

    def __init__(self, window_days: int = 30):
        super().__init__(f"Trend_{window_days}d")
        self.window_days = window_days

    def fit(self, data: pd.DataFrame) -> 'TrendBaseline':
        """
        Calcule les paramètres de tendance pour chaque article.
        """
        logger.info(f"🎯 Entraînement du modèle {self.name}")

        self.training_data = data.copy()
        self.article_stats = {}

        for article_id in data[ColumnNames.ARTICLE_ID].unique():
            article_data = data[data[ColumnNames.ARTICLE_ID] == article_id].copy()
            article_data = article_data.sort_values(ColumnNames.DATE)

            # Prendre les N derniers jours pour calculer la tendance
            recent_data = article_data.tail(self.window_days)

            if len(recent_data) >= 2:
                # Régression linéaire simple
                x = np.arange(len(recent_data))
                y = recent_data[ColumnNames.QUANTITY].values

                # Calcul des coefficients (pente et ordonnée)
                slope = np.polyfit(x, y, 1)[0]
                intercept = y[-1] - slope * (len(recent_data) - 1)

                self.article_stats[article_id] = {
                    'slope': slope,
                    'intercept': intercept,
                    'last_value': y[-1],
                    'last_date': recent_data.iloc[-1][ColumnNames.DATE]
                }
            else:
                # Pas assez de données pour une tendance
                self.article_stats[article_id] = {
                    'slope': 0,
                    'intercept': recent_data.iloc[0][ColumnNames.QUANTITY] if len(recent_data) > 0 else 0,
                    'last_value': recent_data.iloc[0][ColumnNames.QUANTITY] if len(recent_data) > 0 else 0,
                    'last_date': recent_data.iloc[0][ColumnNames.DATE] if len(recent_data) > 0 else None
                }

        self.is_fitted = True

        logger.info(f"   ✅ {len(self.article_stats)} articles avec tendances calculées")
        return self

    def predict(self,
                article_id: int,
                prediction_dates: List[datetime],
                context_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Extrapole la tendance linéaire pour les dates futures.
        """
        if article_id not in self.article_stats:
            return np.zeros(len(prediction_dates))

        stats = self.article_stats[article_id]
        slope = stats['slope']
        last_value = stats['last_value']
        last_date = stats['last_date']

        if last_date is None:
            return np.zeros(len(prediction_dates))

        predictions = []

        for pred_date in prediction_dates:
            # Nombre de jours depuis la dernière observation
            days_ahead = (pred_date - last_date).days

            # Extrapolation linéaire
            prediction = last_value + slope * days_ahead

            # Contrainte : pas de valeurs négatives
            prediction = max(0, prediction)

            predictions.append(prediction)

        return np.array(predictions)


class BaselineEnsemble(BaselineModel):
    """
    Ensemble de baselines : combine plusieurs modèles simples avec des poids.

    Souvent plus robuste qu'un modèle unique.
    """

    def __init__(self, models: List[BaselineModel], weights: Optional[List[float]] = None):
        super().__init__("Ensemble_Baseline")
        self.models = models
        self.weights = weights or [1.0] * len(models)

        if len(self.weights) != len(self.models):
            raise ValueError("Le nombre de poids doit égaler le nombre de modèles")

        # Normalisation des poids
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

    def fit(self, data: pd.DataFrame) -> 'BaselineEnsemble':
        """
        Entraîne tous les modèles de l'ensemble.
        """
        logger.info(f"🎯 Entraînement de l'ensemble de {len(self.models)} modèles")

        for model in self.models:
            model.fit(data)
            logger.info(f"   ✅ {model.name} entraîné")

        self.is_fitted = True

        logger.info(f"✅ Ensemble complet entraîné")
        return self

    def predict(self,
                article_id: int,
                prediction_dates: List[datetime],
                context_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Combine les prédictions de tous les modèles avec pondération.
        """
        if not all(model.is_fitted for model in self.models):
            raise ValueError("Tous les modèles de l'ensemble doivent être entraînés")

        # Prédictions de chaque modèle
        all_predictions = []
        for model in self.models:
            pred = model.predict(article_id, prediction_dates, context_data)
            all_predictions.append(pred)

        # Moyenne pondérée
        weighted_predictions = np.average(all_predictions, axis=0, weights=self.weights)

        return weighted_predictions


# ===== FONCTIONS UTILITAIRES =====

def create_baseline_suite() -> List[BaselineModel]:
    """
    Crée une suite complète de modèles de baseline recommandés.

    Returns:
        List: Liste des modèles de baseline prêts à utiliser
    """
    baselines = [
        NaiveBaseline(),
        HistoricalMeanBaseline(),
        MovingAverageBaseline(window=7),
        MovingAverageBaseline(window=30),
        WeekdayMeanBaseline(),
        SeasonalNaiveBaseline(),
        TrendBaseline(window_days=30)
    ]

    return baselines


def evaluate_all_baselines(
    baselines: List[BaselineModel],
    train_data: pd.DataFrame,
    test_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Entraîne et évalue tous les modèles de baseline.

    Args:
        baselines: Liste des modèles de baseline
        train_data: Données d'entraînement
        test_data: Données de test

    Returns:
        DataFrame: Résultats de l'évaluation pour chaque modèle
    """
    logger.info("🏁 ÉVALUATION COMPLÈTE DES BASELINES")
    logger.info("=" * 50)

    results = []

    for baseline in baselines:
        try:
            # Entraînement
            baseline.fit(train_data)

            # Évaluation
            metrics = baseline.evaluate(test_data)

            result = {
                'model': baseline.name,
                **metrics
            }
            results.append(result)

            logger.info(f"✅ {baseline.name}: MAE={metrics.get('mae', 'N/A'):.2f}, RMSE={metrics.get('rmse', 'N/A'):.2f}")

        except Exception as e:
            logger.error(f"❌ Erreur avec {baseline.name}: {e}")

            result = {
                'model': baseline.name,
                'mae': np.inf,
                'rmse': np.inf,
                'mape': np.inf,
                'error': str(e)
            }
            results.append(result)

    results_df = pd.DataFrame(results)

    # Classement par MAE
    results_df = results_df.sort_values('mae')

    logger.info("=" * 50)
    logger.info("🏆 CLASSEMENT FINAL DES BASELINES:")
    for idx, (_, row) in enumerate(results_df.iterrows(), 1):
        logger.info(f"   {idx}. {row['model']}: MAE={row['mae']:.2f}")

    return results_df


if __name__ == "__main__":
    # Test des baselines si exécuté directement
    try:
        try:
            from ..data_processing import DataEnrichmentPipeline
        except ImportError:
            from src.data_processing import DataEnrichmentPipeline

        # Chargement des données enrichies
        pipeline = DataEnrichmentPipeline()
        data = pipeline.load_clean_data()

        # Test d'un modèle simple
        naive = NaiveBaseline()
        naive.fit(data)

        # Test de prédiction
        test_article = data[ColumnNames.ARTICLE_ID].iloc[0]
        test_dates = [datetime(2024, 12, 1), datetime(2024, 12, 2)]

        predictions = naive.predict(test_article, test_dates)

        print(f"✅ Test réussi - Prédictions: {predictions}")

    except Exception as e:
        print(f"Erreur lors du test : {e}")
        print("Assurez-vous que les données enrichies existent.")