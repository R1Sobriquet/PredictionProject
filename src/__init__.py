"""
Module principal du projet de prévision de commandes ATM.

Architecture :
- data_ingestion : Chargement et nettoyage des données (.xlsx ou BDD)
- data_processing : Enrichissement avec features ATM
- visualization : Graphiques et analyses
- models/ : Modèles de prévision (baseline et CatBoost)
- utils/ : Configuration et utilitaires
"""

try:
    from .data_ingestion import DataIngestionPipeline, quick_data_ingestion
    from .data_processing import DataEnrichmentPipeline, quick_enrichment, analyze_atm_pattern
    from .visualization import DataVisualization, create_atm_dashboard, create_global_analysis
    from .database_connector import DatabaseConnector
    from .models import (
        BaselineModel, NaiveBaseline, HistoricalMeanBaseline,
        MovingAverageBaseline, WeekdayMeanBaseline, SeasonalNaiveBaseline,
        TrendBaseline, BaselineEnsemble,
    )
except ImportError:
    from src.data_ingestion import DataIngestionPipeline, quick_data_ingestion
    from src.data_processing import DataEnrichmentPipeline, quick_enrichment, analyze_atm_pattern
    from src.visualization import DataVisualization, create_atm_dashboard, create_global_analysis
    from src.database_connector import DatabaseConnector
    from src.models.baseline import (
        BaselineModel, NaiveBaseline, HistoricalMeanBaseline,
        MovingAverageBaseline, WeekdayMeanBaseline, SeasonalNaiveBaseline,
        TrendBaseline, BaselineEnsemble,
    )

__version__ = "2.0.0"
__author__ = "Forecasting Team"

__all__ = [
    'DataIngestionPipeline',
    'DataEnrichmentPipeline',
    'DataVisualization',
    'DatabaseConnector',
    'quick_data_ingestion',
    'quick_enrichment',
    'analyze_atm_pattern',
    'create_atm_dashboard',
    'create_global_analysis',
    'BaselineModel',
    'NaiveBaseline',
    'HistoricalMeanBaseline',
    'MovingAverageBaseline',
    'WeekdayMeanBaseline',
    'SeasonalNaiveBaseline',
    'TrendBaseline',
    'BaselineEnsemble',
]
