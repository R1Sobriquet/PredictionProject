"""
Module principal du projet de prévision de commandes.

Architecture :
- data_ingestion : Chargement et nettoyage des données
- data_processing : Enrichissement des données
- visualization : Graphiques et analyses
- models/ : Modèles de prévision (baseline et avancés)
- utils/ : Configuration et utilitaires
"""

try:
    # Import relatif (quand utilisé comme package)
    from .data_ingestion import DataIngestionPipeline, quick_data_ingestion, preview_raw_data
    from .data_processing import DataEnrichmentPipeline, quick_enrichment, analyze_article_pattern
    from .visualization import DataVisualization, create_article_dashboard, create_global_analysis
    from .models import (
        BaselineModel, NaiveBaseline, HistoricalMeanBaseline,
        MovingAverageBaseline, WeekdayMeanBaseline, SeasonalNaiveBaseline,
        TrendBaseline, BaselineEnsemble
    )
except ImportError:
    # Import absolu (quand exécuté directement)
    from src.data_ingestion import DataIngestionPipeline, quick_data_ingestion, preview_raw_data
    from src.data_processing import DataEnrichmentPipeline, quick_enrichment, analyze_article_pattern
    from src.visualization import DataVisualization, create_article_dashboard, create_global_analysis
    from src.models.baseline import (
        BaselineModel, NaiveBaseline, HistoricalMeanBaseline,
        MovingAverageBaseline, WeekdayMeanBaseline, SeasonalNaiveBaseline,
        TrendBaseline, BaselineEnsemble
    )

__version__ = "1.0.0"
__author__ = "Forecasting Team"

__all__ = [
    # Pipeline principal
    'DataIngestionPipeline',
    'DataEnrichmentPipeline',
    'DataVisualization',

    # Fonctions rapides
    'quick_data_ingestion',
    'quick_enrichment',
    'preview_raw_data',

    # Analyse et visualisation
    'analyze_article_pattern',
    'create_article_dashboard',
    'create_global_analysis',

    # Modèles de baseline
    'BaselineModel',
    'NaiveBaseline',
    'HistoricalMeanBaseline',
    'MovingAverageBaseline',
    'WeekdayMeanBaseline',
    'SeasonalNaiveBaseline',
    'TrendBaseline',
    'BaselineEnsemble'
]


