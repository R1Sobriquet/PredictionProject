"""
Modèle CatBoost pour la prévision de commandes.

Ce module implémente un modèle de prévision basé sur CatBoost avec :
- Horizon de prédiction paramétrable (1 à 90 jours)
- Feature engineering adaptatif selon l'horizon
- Intégration avec l'architecture existante (hérite de BaselineModel)

Usage:
    from src.models.catboost_model import CatBoostForecaster

    model = CatBoostForecaster(max_horizon=90)
    model.fit(train_data)
    predictions = model.predict(article_id=1, prediction_dates=[...], horizon=7)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
from abc import ABC
import warnings
import pickle

# CatBoost
try:
    from catboost import CatBoostRegressor, Pool

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost non installé. Installez-le avec: pip install catboost")

# Scikit-learn pour les métriques
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Imports du projet
try:
    from ..utils import (
        ColumnNames,
        Messages,
        WEEKDAY_NAMES,
        get_file_path
    )
    from .baseline import BaselineModel
except ImportError:
    from src.utils import (
        ColumnNames,
        Messages,
        WEEKDAY_NAMES,
        get_file_path
    )
    from src.models.baseline import BaselineModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CatBoostForecaster(BaselineModel):
    """
    Modèle de prévision CatBoost avec horizon paramétrable.

    Caractéristiques :
    - Horizon de prédiction de 1 à 90 jours (paramétrable)
    - Feature engineering adaptatif :
        - Court terme (1-14 jours) : features riches (lags, rolling means)
        - Moyen/Long terme (15+ jours) : features stables uniquement
    - Article_id traité comme feature catégorielle native
    - Early stopping pour éviter l'overfitting

    Attributes:
        max_horizon: Horizon maximum de prédiction en jours
        model: Instance CatBoostRegressor entraînée
        feature_columns: Liste des colonnes de features utilisées
        cat_features: Liste des features catégorielles
        article_historical_stats: Statistiques historiques par article
    """

    # Seuil pour basculer vers les features stables uniquement
    SHORT_TERM_THRESHOLD = 14

    def __init__(
            self,
            max_horizon: int = 90,
            iterations: int = 4444,
            learning_rate: float = 0.85,
            depth: int = 6,
            early_stopping_rounds: int = 66,
            random_state: int = 42,
            verbose: bool = False
    ):
        """
        Initialise le modèle CatBoost.

        Args:
            max_horizon: Horizon maximum de prédiction en jours (défaut: 90)
            iterations: Nombre maximum d'itérations (défaut: 1000)
            learning_rate: Taux d'apprentissage (défaut: 0.1)
            depth: Profondeur maximale des arbres (défaut: 6)
            early_stopping_rounds: Rounds pour early stopping (défaut: 50)
            random_state: Graine aléatoire pour reproductibilité
            verbose: Afficher les logs d'entraînement
        """
        super().__init__(f"CatBoost_H{max_horizon}")

        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost n'est pas installé. Installez-le avec: pip install catboost")

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
        self.article_historical_stats = {}
        self.global_stats = {}

        # Configuration CatBoost
        self.model_params = {
            'iterations': iterations,
            'learning_rate': learning_rate,
            'depth': depth,
            'loss_function': 'RMSE',
            'random_seed': random_state,
            'verbose': verbose,
            'early_stopping_rounds': early_stopping_rounds,
            'use_best_model': True
        }

    def _compute_historical_stats(self, data: pd.DataFrame) -> None:
        """
        Calcule les statistiques historiques par article.

        Ces stats sont utilisées pour les prédictions à long terme
        où les features de lag ne sont pas disponibles.

        Args:
            data: Données d'entraînement
        """
        logger.info("📊 Calcul des statistiques historiques par article...")

        # Stats globales
        self.global_stats = {
            'mean': data[ColumnNames.QUANTITY].mean(),
            'std': data[ColumnNames.QUANTITY].std(),
            'median': data[ColumnNames.QUANTITY].median()
        }

        # Stats par article
        for article_id in data[ColumnNames.ARTICLE_ID].unique():
            article_data = data[data[ColumnNames.ARTICLE_ID] == article_id]

            # Moyenne par jour de semaine pour cet article
            weekday_means = article_data.groupby(ColumnNames.WEEKDAY)[ColumnNames.QUANTITY].mean().to_dict()

            # Moyenne par mois pour cet article
            month_means = article_data.groupby(ColumnNames.MONTH)[ColumnNames.QUANTITY].mean().to_dict()

            # Stats générales de l'article
            self.article_historical_stats[article_id] = {
                'mean': article_data[ColumnNames.QUANTITY].mean(),
                'std': article_data[ColumnNames.QUANTITY].std(),
                'median': article_data[ColumnNames.QUANTITY].median(),
                'max': article_data[ColumnNames.QUANTITY].max(),
                'min': article_data[ColumnNames.QUANTITY].min(),
                'weekday_means': weekday_means,
                'month_means': month_means,
                'total_orders': article_data[ColumnNames.QUANTITY].sum(),
                'days_with_orders': (article_data[ColumnNames.QUANTITY] > 0).sum(),
                'order_frequency': (article_data[ColumnNames.QUANTITY] > 0).mean()
            }

        logger.info(f"   ✅ Stats calculées pour {len(self.article_historical_stats)} articles")

    def _prepare_features_for_training(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare les features pour l'entraînement avec différents horizons.

        Génère des exemples d'entraînement pour tous les horizons (1 à max_horizon).

        Args:
            data: Données enrichies

        Returns:
            DataFrame: Données avec features et target pour tous les horizons
        """
        logger.info(f"🔧 Préparation des features pour horizons 1 à {self.max_horizon} jours...")

        all_training_examples = []

        # Trier les données
        data = data.sort_values([ColumnNames.ARTICLE_ID, ColumnNames.DATE]).reset_index(drop=True)

        # Pour chaque article
        for article_id in data[ColumnNames.ARTICLE_ID].unique():
            article_data = data[data[ColumnNames.ARTICLE_ID] == article_id].copy()
            article_data = article_data.sort_values(ColumnNames.DATE).reset_index(drop=True)

            # Stats historiques de l'article
            article_stats = self.article_historical_stats.get(article_id, {})

            # Pour chaque ligne (date de référence)
            for idx in range(len(article_data) - self.max_horizon):
                base_row = article_data.iloc[idx]
                base_date = base_row[ColumnNames.DATE]

                # Générer des exemples pour différents horizons
                # On ne génère pas TOUS les horizons pour éviter un dataset trop grand
                # On échantillonne : 1-14 (tous), puis 21, 30, 45, 60, 75, 90
                horizons_to_sample = list(range(1, min(15, self.max_horizon + 1)))
                horizons_to_sample += [h for h in [21, 30, 45, 60, 75, 90] if h <= self.max_horizon]

                for horizon in horizons_to_sample:
                    if idx + horizon >= len(article_data):
                        continue

                    target_row = article_data.iloc[idx + horizon]
                    target_quantity = target_row[ColumnNames.QUANTITY]
                    target_date = target_row[ColumnNames.DATE]

                    # Construction des features
                    features = self._build_features(
                        base_row=base_row,
                        target_date=target_date,
                        horizon=horizon,
                        article_stats=article_stats,
                        historical_data=article_data.iloc[:idx + 1]  # Données jusqu'à la date de base
                    )

                    features['target'] = target_quantity
                    all_training_examples.append(features)

        training_df = pd.DataFrame(all_training_examples)

        logger.info(f"   ✅ {len(training_df)} exemples d'entraînement générés")
        logger.info(f"   📊 Distribution des horizons : {training_df['horizon'].value_counts().head(10).to_dict()}")

        return training_df

    def _build_features(
            self,
            base_row: pd.Series,
            target_date: datetime,
            horizon: int,
            article_stats: Dict,
            historical_data: pd.DataFrame
    ) -> Dict:
        """
        Construit les features pour une prédiction donnée.

        Args:
            base_row: Ligne de données de référence (dernière date connue)
            target_date: Date à prédire
            horizon: Nombre de jours dans le futur
            article_stats: Statistiques historiques de l'article
            historical_data: Données historiques disponibles

        Returns:
            Dict: Features pour cet exemple
        """
        features = {}

        # === FEATURE : HORIZON ===
        features['horizon'] = horizon
        features['horizon_log'] = np.log1p(horizon)
        features['is_short_term'] = 1 if horizon <= self.SHORT_TERM_THRESHOLD else 0

        # === FEATURES ARTICLE ===
        features['article_id'] = base_row[ColumnNames.ARTICLE_ID]

        # Stats historiques de l'article (toujours disponibles)
        features['article_mean'] = article_stats.get('mean', self.global_stats['mean'])
        features['article_std'] = article_stats.get('std', self.global_stats['std'])
        features['article_median'] = article_stats.get('median', self.global_stats['median'])
        features['article_order_frequency'] = article_stats.get('order_frequency', 0.5)

        # === FEATURES TEMPORELLES DE LA DATE CIBLE ===
        if isinstance(target_date, pd.Timestamp):
            target_dt = target_date
        else:
            target_dt = pd.Timestamp(target_date)

        features['target_weekday'] = target_dt.weekday()
        features['target_month'] = target_dt.month
        features['target_day'] = target_dt.day
        features['target_quarter'] = target_dt.quarter
        features['target_is_weekend'] = 1 if target_dt.weekday() >= 5 else 0
        features['target_is_month_start'] = 1 if target_dt.day <= 5 else 0
        features['target_is_month_end'] = 1 if target_dt.day > 25 else 0
        features['target_day_of_year'] = target_dt.dayofyear

        # Variables cycliques pour la saisonnalité
        features['target_weekday_sin'] = np.sin(2 * np.pi * target_dt.weekday() / 7)
        features['target_weekday_cos'] = np.cos(2 * np.pi * target_dt.weekday() / 7)
        features['target_month_sin'] = np.sin(2 * np.pi * target_dt.month / 12)
        features['target_month_cos'] = np.cos(2 * np.pi * target_dt.month / 12)
        features['target_day_of_year_sin'] = np.sin(2 * np.pi * target_dt.dayofyear / 365.25)
        features['target_day_of_year_cos'] = np.cos(2 * np.pi * target_dt.dayofyear / 365.25)

        # Moyenne historique pour ce jour de semaine
        weekday_means = article_stats.get('weekday_means', {})
        features['article_weekday_mean'] = weekday_means.get(target_dt.weekday(), article_stats.get('mean', 0))

        # Moyenne historique pour ce mois
        month_means = article_stats.get('month_means', {})
        features['article_month_mean'] = month_means.get(target_dt.month, article_stats.get('mean', 0))

        # === FEATURES DE LAG (court terme uniquement) ===
        if horizon <= self.SHORT_TERM_THRESHOLD and len(historical_data) > 0:
            # Quantité du dernier jour connu
            features['last_quantity'] = base_row[ColumnNames.QUANTITY]

            # Quantité il y a 7 jours (si disponible)
            if len(historical_data) >= 7:
                features['quantity_lag_7'] = historical_data.iloc[-7][ColumnNames.QUANTITY]
            else:
                features['quantity_lag_7'] = article_stats.get('mean', 0)

            # Moyenne des 7 derniers jours
            if len(historical_data) >= 7:
                features['rolling_mean_7d'] = historical_data.tail(7)[ColumnNames.QUANTITY].mean()
            else:
                features['rolling_mean_7d'] = historical_data[ColumnNames.QUANTITY].mean()

            # Moyenne des 30 derniers jours
            if len(historical_data) >= 30:
                features['rolling_mean_30d'] = historical_data.tail(30)[ColumnNames.QUANTITY].mean()
            else:
                features['rolling_mean_30d'] = historical_data[ColumnNames.QUANTITY].mean()

            # Écart-type récent (volatilité)
            if len(historical_data) >= 7:
                features['rolling_std_7d'] = historical_data.tail(7)[ColumnNames.QUANTITY].std()
            else:
                features['rolling_std_7d'] = 0

            # Tendance récente (différence moyenne 7j vs 30j)
            features['recent_trend'] = features['rolling_mean_7d'] - features.get('rolling_mean_30d',
                                                                                  features['rolling_mean_7d'])

        else:
            # Long terme : utiliser les moyennes historiques
            features['last_quantity'] = article_stats.get('mean', 0)
            features['quantity_lag_7'] = article_stats.get('mean', 0)
            features['rolling_mean_7d'] = article_stats.get('mean', 0)
            features['rolling_mean_30d'] = article_stats.get('mean', 0)
            features['rolling_std_7d'] = article_stats.get('std', 0)
            features['recent_trend'] = 0

        return features

    def fit(self, data: pd.DataFrame, eval_data: Optional[pd.DataFrame] = None) -> 'CatBoostForecaster':
        """
        Entraîne le modèle CatBoost sur les données.

        Args:
            data: Données d'entraînement enrichies
            eval_data: Données de validation pour early stopping (optionnel)

        Returns:
            Self: Instance du modèle entraîné
        """
        logger.info(f"🎯 Entraînement du modèle {self.name}")
        logger.info(f"   📊 Données d'entrée : {len(data)} lignes")

        # Stocker les données d'entraînement
        self.training_data = data.copy()

        # 1. Calculer les statistiques historiques
        self._compute_historical_stats(data)

        # 2. Préparer les features d'entraînement
        training_df = self._prepare_features_for_training(data)

        # 3. Séparer features et target
        self.feature_columns = [col for col in training_df.columns if col != 'target']
        X_train = training_df[self.feature_columns]
        y_train = training_df['target']

        # 4. Identifier les features catégorielles
        self.cat_features = ['article_id']
        cat_feature_indices = [self.feature_columns.index(f) for f in self.cat_features if f in self.feature_columns]

        logger.info(f"   📋 Features : {len(self.feature_columns)} colonnes")
        logger.info(f"   🏷️  Features catégorielles : {self.cat_features}")

        # 5. Préparer les données de validation si fournies
        if eval_data is not None:
            eval_df = self._prepare_features_for_training(eval_data)
            X_eval = eval_df[self.feature_columns]
            y_eval = eval_df['target']
            eval_set = Pool(X_eval, y_eval, cat_features=cat_feature_indices)
        else:
            # Split automatique 80/20 pour validation
            split_idx = int(len(X_train) * 0.8)
            X_eval = X_train.iloc[split_idx:]
            y_eval = y_train.iloc[split_idx:]
            X_train = X_train.iloc[:split_idx]
            y_train = y_train.iloc[:split_idx]
            eval_set = Pool(X_eval, y_eval, cat_features=cat_feature_indices)

        # 6. Créer et entraîner le modèle
        self.model = CatBoostRegressor(**self.model_params)

        train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)

        logger.info(f"   🚀 Début de l'entraînement...")

        self.model.fit(
            train_pool,
            eval_set=eval_set,
            verbose=self.verbose
        )

        self.is_fitted = True

        # 7. Afficher les résultats
        best_iteration = self.model.get_best_iteration()
        logger.info(f"   ✅ Entraînement terminé")
        logger.info(f"   🎯 Meilleure itération : {best_iteration}")

        # Feature importance
        feature_importance = self.model.get_feature_importance()
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        logger.info(f"   📊 Top 5 features importantes :")
        for _, row in importance_df.head(5).iterrows():
            logger.info(f"      - {row['feature']}: {row['importance']:.2f}")

        return self

    def predict(
            self,
            article_id: int,
            prediction_dates: List[datetime],
            context_data: Optional[pd.DataFrame] = None,
            horizon: Optional[int] = None
    ) -> np.ndarray:
        """
        Génère des prédictions pour un article sur des dates données.

        Args:
            article_id: ID de l'article à prédire
            prediction_dates: Liste des dates pour lesquelles prédire
            context_data: Données contextuelles (dernières valeurs connues)
            horizon: Horizon de prédiction (si None, calculé automatiquement)

        Returns:
            Array: Prédictions pour chaque date
        """
        if not self.is_fitted:
            raise ValueError(f"Modèle {self.name} non entraîné. Appelez fit() d'abord.")

        predictions = []

        # Récupérer les stats de l'article
        article_stats = self.article_historical_stats.get(article_id, self.global_stats)

        # Dernière date connue et données historiques
        if context_data is not None:
            article_history = context_data[context_data[ColumnNames.ARTICLE_ID] == article_id].copy()
            article_history = article_history.sort_values(ColumnNames.DATE)
            if len(article_history) > 0:
                base_row = article_history.iloc[-1]
                base_date = base_row[ColumnNames.DATE]
            else:
                base_row = None
                base_date = None
        else:
            article_history = self.training_data[self.training_data[ColumnNames.ARTICLE_ID] == article_id].copy()
            article_history = article_history.sort_values(ColumnNames.DATE)
            base_row = article_history.iloc[-1] if len(article_history) > 0 else None
            base_date = base_row[ColumnNames.DATE] if base_row is not None else None

        for pred_date in prediction_dates:
            # Calculer l'horizon
            if horizon is not None:
                h = horizon
            elif base_date is not None:
                h = (pd.Timestamp(pred_date) - pd.Timestamp(base_date)).days
                h = max(1, h)  # Minimum 1 jour
            else:
                h = 1

            # Limiter à max_horizon
            h = min(h, self.max_horizon)

            # Construire les features
            if base_row is not None:
                features = self._build_features(
                    base_row=base_row,
                    target_date=pred_date,
                    horizon=h,
                    article_stats=article_stats,
                    historical_data=article_history
                )
            else:
                # Pas d'historique, utiliser les moyennes
                features = self._build_default_features(article_id, pred_date, h, article_stats)

            # Préparer pour la prédiction
            feature_values = [features.get(col, 0) for col in self.feature_columns]
            X_pred = pd.DataFrame([feature_values], columns=self.feature_columns)

            # Prédiction
            pred = self.model.predict(X_pred)[0]

            # Contrainte : pas de valeurs négatives
            pred = max(0, pred)

            predictions.append(pred)

        return np.array(predictions)

    def _build_default_features(
            self,
            article_id: int,
            target_date: datetime,
            horizon: int,
            article_stats: Dict
    ) -> Dict:
        """
        Construit des features par défaut quand il n'y a pas d'historique.

        Args:
            article_id: ID de l'article
            target_date: Date à prédire
            horizon: Horizon de prédiction
            article_stats: Statistiques de l'article

        Returns:
            Dict: Features par défaut
        """
        features = {}

        # Horizon
        features['horizon'] = horizon
        features['horizon_log'] = np.log1p(horizon)
        features['is_short_term'] = 1 if horizon <= self.SHORT_TERM_THRESHOLD else 0

        # Article
        features['article_id'] = article_id
        features['article_mean'] = article_stats.get('mean', self.global_stats['mean'])
        features['article_std'] = article_stats.get('std', self.global_stats['std'])
        features['article_median'] = article_stats.get('median', self.global_stats['median'])
        features['article_order_frequency'] = article_stats.get('order_frequency', 0.5)

        # Temporelles
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

        # Moyennes historiques
        weekday_means = article_stats.get('weekday_means', {})
        features['article_weekday_mean'] = weekday_means.get(target_dt.weekday(), article_stats.get('mean', 0))

        month_means = article_stats.get('month_means', {})
        features['article_month_mean'] = month_means.get(target_dt.month, article_stats.get('mean', 0))

        # Lags (valeurs par défaut)
        mean_val = article_stats.get('mean', 0)
        features['last_quantity'] = mean_val
        features['quantity_lag_7'] = mean_val
        features['rolling_mean_7d'] = mean_val
        features['rolling_mean_30d'] = mean_val
        features['rolling_std_7d'] = article_stats.get('std', 0)
        features['recent_trend'] = 0

        return features

    def predict_horizon(
            self,
            article_id: int,
            start_date: datetime,
            horizon_days: int,
            context_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Prédit pour un article sur un horizon donné à partir d'une date.

        Méthode pratique pour prédire N jours dans le futur.

        Args:
            article_id: ID de l'article
            start_date: Date de début (dernière date connue + 1)
            horizon_days: Nombre de jours à prédire
            context_data: Données contextuelles

        Returns:
            DataFrame: Prédictions avec colonnes [date, article_id, prediction, horizon]
        """
        if horizon_days > self.max_horizon:
            logger.warning(
                f"⚠️  Horizon demandé ({horizon_days}j) > max ({self.max_horizon}j). Limité à {self.max_horizon}j.")
            horizon_days = self.max_horizon

        # Générer les dates de prédiction
        prediction_dates = pd.date_range(start=start_date, periods=horizon_days, freq='D')

        # Prédictions
        predictions = []
        for i, pred_date in enumerate(prediction_dates):
            horizon = i + 1  # Horizon croissant : J+1, J+2, ..., J+N
            pred = self.predict(
                article_id=article_id,
                prediction_dates=[pred_date],
                context_data=context_data,
                horizon=horizon
            )[0]

            predictions.append({
                ColumnNames.DATE: pred_date,
                ColumnNames.ARTICLE_ID: article_id,
                'prediction': pred,
                'horizon': horizon
            })

        return pd.DataFrame(predictions)

    def evaluate_by_horizon(
            self,
            test_data: pd.DataFrame,
            horizons: List[int] = [1, 7, 14, 30, 60, 90]
    ) -> pd.DataFrame:
        """
        Évalue le modèle par horizon de prédiction.

        Args:
            test_data: Données de test
            horizons: Liste des horizons à évaluer

        Returns:
            DataFrame: Métriques par horizon
        """
        logger.info("📊 Évaluation par horizon de prédiction...")

        results = []

        for horizon in horizons:
            if horizon > self.max_horizon:
                continue

            predictions = []
            actuals = []

            # Pour chaque article et date de test
            test_data_sorted = test_data.sort_values([ColumnNames.ARTICLE_ID, ColumnNames.DATE])

            for article_id in test_data[ColumnNames.ARTICLE_ID].unique():
                article_test = test_data_sorted[test_data_sorted[ColumnNames.ARTICLE_ID] == article_id]

                # On a besoin d'au moins horizon+1 jours
                if len(article_test) <= horizon:
                    continue

                for i in range(len(article_test) - horizon):
                    base_date = article_test.iloc[i][ColumnNames.DATE]
                    target_date = article_test.iloc[i + horizon][ColumnNames.DATE]
                    actual = article_test.iloc[i + horizon][ColumnNames.QUANTITY]

                    # Contexte : données jusqu'à base_date
                    context = test_data[
                        (test_data[ColumnNames.ARTICLE_ID] == article_id) &
                        (test_data[ColumnNames.DATE] <= base_date)
                        ]

                    pred = self.predict(
                        article_id=article_id,
                        prediction_dates=[target_date],
                        context_data=context,
                        horizon=horizon
                    )[0]

                    predictions.append(pred)
                    actuals.append(actual)

            if len(predictions) > 0:
                predictions = np.array(predictions)
                actuals = np.array(actuals)

                mae = mean_absolute_error(actuals, predictions)
                rmse = np.sqrt(mean_squared_error(actuals, predictions))

                # MAPE avec protection
                non_zero_mask = actuals != 0
                if non_zero_mask.any():
                    mape = np.mean(
                        np.abs((actuals[non_zero_mask] - predictions[non_zero_mask]) / actuals[non_zero_mask])) * 100
                else:
                    mape = np.inf

                results.append({
                    'horizon': horizon,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'n_samples': len(predictions)
                })

                logger.info(
                    f"   H+{horizon:2d}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%, n={len(predictions)}")

        return pd.DataFrame(results)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Retourne l'importance des features.

        Returns:
            DataFrame: Features triées par importance
        """
        if not self.is_fitted:
            raise ValueError("Modèle non entraîné")

        importance = self.model.get_feature_importance()

        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def save_model(self, path: Union[str, Path]) -> None:
        """
        Sauvegarde le modèle entraîné.

        Args:
            path: Chemin de sauvegarde (dossier)
        """
        if not self.is_fitted:
            raise ValueError("Modèle non entraîné")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Sauvegarder le modèle CatBoost
        self.model.save_model(str(path / 'catboost_model.cbm'))

        # Sauvegarder les métadonnées
        metadata = {
            'name': self.name,
            'max_horizon': self.max_horizon,
            'feature_columns': self.feature_columns,
            'cat_features': self.cat_features,
            'article_historical_stats': self.article_historical_stats,
            'global_stats': self.global_stats,
            'model_params': self.model_params
        }

        with open(path / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"💾 Modèle sauvegardé : {path}")

    @classmethod
    def load_model(cls, path: Union[str, Path]) -> 'CatBoostForecaster':
        """
        Charge un modèle sauvegardé.

        Args:
            path: Chemin du dossier contenant le modèle

        Returns:
            CatBoostForecaster: Instance du modèle chargé
        """
        path = Path(path)

        # Charger les métadonnées
        with open(path / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        # Créer l'instance
        instance = cls(max_horizon=metadata['max_horizon'])
        instance.name = metadata['name']
        instance.feature_columns = metadata['feature_columns']
        instance.cat_features = metadata['cat_features']
        instance.article_historical_stats = metadata['article_historical_stats']
        instance.global_stats = metadata['global_stats']
        instance.model_params = metadata['model_params']

        # Charger le modèle CatBoost
        instance.model = CatBoostRegressor()
        instance.model.load_model(str(path / 'catboost_model.cbm'))
        instance.is_fitted = True

        logger.info(f"📂 Modèle chargé : {path}")

        return instance


# ===== FONCTIONS UTILITAIRES =====

def train_and_evaluate_catboost(
        enriched_data: pd.DataFrame,
        max_horizon: int = 90,
        test_ratio: float = 0.2
) -> Tuple[CatBoostForecaster, pd.DataFrame]:
    """
    Fonction rapide pour entraîner et évaluer un modèle CatBoost.

    Args:
        enriched_data: Données enrichies
        max_horizon: Horizon maximum
        test_ratio: Ratio de données de test

    Returns:
        Tuple: (modèle entraîné, DataFrame des résultats)
    """
    logger.info("🚀 ENTRAÎNEMENT ET ÉVALUATION CATBOOST")
    logger.info("=" * 50)

    # Split temporel
    dates = sorted(enriched_data[ColumnNames.DATE].unique())
    split_idx = int(len(dates) * (1 - test_ratio))
    split_date = dates[split_idx]

    train_data = enriched_data[enriched_data[ColumnNames.DATE] < split_date].copy()
    test_data = enriched_data[enriched_data[ColumnNames.DATE] >= split_date].copy()

    logger.info(f"   📊 Train : {len(train_data)} lignes jusqu'au {split_date.date()}")
    logger.info(f"   📊 Test  : {len(test_data)} lignes à partir du {split_date.date()}")

    # Entraînement
    model = CatBoostForecaster(max_horizon=max_horizon)
    model.fit(train_data)

    # Évaluation par horizon
    results = model.evaluate_by_horizon(test_data)

    # Évaluation globale
    global_metrics = model.evaluate(test_data)
    logger.info(f"\n📊 MÉTRIQUES GLOBALES :")
    logger.info(f"   MAE  : {global_metrics['mae']:.2f}")
    logger.info(f"   RMSE : {global_metrics['rmse']:.2f}")
    logger.info(f"   MAPE : {global_metrics['mape']:.1f}%")

    return model, results


if __name__ == "__main__":
    """Test du module si exécuté directement."""

    print("=" * 60)
    print("🧪 TEST DU MODÈLE CATBOOST")
    print("=" * 60)

    try:
        # Charger les données enrichies
        enriched_file = get_file_path('enriched')

        if not enriched_file.exists():
            print("❌ Fichier de données enrichies non trouvé.")
            print("   Exécutez d'abord : python main.py --step enrichment")
        else:
            enriched_data = pd.read_csv(
                enriched_file,
                parse_dates=[ColumnNames.DATE],
                date_format='%Y-%m-%d'
            )

            print(f"\n📊 Données chargées : {len(enriched_data)} lignes")

            # Entraînement et évaluation
            model, results = train_and_evaluate_catboost(enriched_data, max_horizon=90)

            # Afficher les résultats
            print("\n📊 RÉSULTATS PAR HORIZON :")
            print(results.to_string(index=False))

            # Sauvegarder le modèle
            model.save_model("data/output/catboost_model")

            print("\n✅ Test terminé avec succès !")

    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback

        traceback.print_exc()