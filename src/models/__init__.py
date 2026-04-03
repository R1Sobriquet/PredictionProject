"""
Module de modélisation pour la prévision de commandes ATM.

Contient :
- Modèles de baseline (référence)
- Modèle CatBoost (prédiction avancée avec horizon paramétrable)
"""

from .baseline import (
    BaselineModel,
    NaiveBaseline,
    HistoricalMeanBaseline,
    MovingAverageBaseline,
    WeekdayMeanBaseline,
    SeasonalNaiveBaseline,
    TrendBaseline,
    BaselineEnsemble,
    create_baseline_suite,
    evaluate_all_baselines,
)

from .catboost_model import (
    CatBoostForecaster,
    train_and_evaluate_catboost,
)

__all__ = [
    'BaselineModel',
    'NaiveBaseline',
    'HistoricalMeanBaseline',
    'MovingAverageBaseline',
    'WeekdayMeanBaseline',
    'SeasonalNaiveBaseline',
    'TrendBaseline',
    'BaselineEnsemble',
    'create_baseline_suite',
    'evaluate_all_baselines',
    'CatBoostForecaster',
    'train_and_evaluate_catboost',
]
