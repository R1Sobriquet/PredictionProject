"""
Module d'ingestion et de nettoyage des données de commandes.

NOUVEAU : Support hybride pour charger depuis :
- CSV (mode legacy)
- SQL Server (mode recommandé)

La source est configurée dans le fichier .env (DATA_SOURCE=csv ou sqlserver)

Ce module implémente l'étape 1 du projet :
- Récupération des données 2024 (sauf décembre)
- Nettoyage des données (doublons, valeurs aberrantes)
- Retravail pour avoir 1 ligne par jour/article avec quantité (même 0)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
import logging
from pathlib import Path

try:
    from .utils import (
        ColumnNames,
        ValidationRules,
        Messages,
        DataSourceConfig,
        get_training_date_range,
        get_file_path
    )
    from .database_connector import SQLServerConnector
except ImportError:
    from src.utils import (
        ColumnNames,
        ValidationRules,
        Messages,
        DataSourceConfig,
        get_training_date_range,
        get_file_path
    )
    from src.database_connector import SQLServerConnector

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """
    Pipeline d'ingestion et de nettoyage des données de commandes.

    Support hybride :
    - CSV : charge depuis un fichier CSV local
    - SQL Server : charge directement depuis la base de données

    Processus :
    1. Chargement des données brutes (CSV ou SQL Server)
    2. Validation et nettoyage
    3. Agrégation par jour/article
    4. Ajout des jours/articles manquants avec quantité 0
    """

    def __init__(
        self,
        source_file_path: Optional[Path] = None,
        data_source: Optional[str] = None,
        db_connector: Optional[SQLServerConnector] = None
    ):
        """
        Initialise le pipeline d'ingestion.

        Args:
            source_file_path: Chemin vers le fichier CSV (si mode CSV)
            data_source: Source de données ('csv' ou 'sqlserver'). Si None, utilise la config.
            db_connector: Connecteur SQL Server (optionnel, créé automatiquement si besoin)
        """
        # Déterminer la source de données
        self.data_source = data_source or DataSourceConfig.DEFAULT_SOURCE

        # Configuration selon la source
        if self.data_source == 'sqlserver':
            self.db_connector = db_connector or SQLServerConnector()
            self.source_file_path = None
            logger.info(f"📊 Mode SQL Server : {DataSourceConfig.DB_SERVER} / {DataSourceConfig.DB_NAME}")
        else:
            self.source_file_path = source_file_path or get_file_path('raw')
            self.db_connector = None
            logger.info(f"📁 Mode CSV : {self.source_file_path}")

        self.training_start, self.training_end = get_training_date_range()
        self.raw_data = None
        self.clean_data = None
        self.final_data = None

    def load_raw_data(self) -> pd.DataFrame:
        """
        Charge les données brutes depuis la source configurée (CSV ou SQL Server).

        Returns:
            DataFrame: Données brutes chargées

        Raises:
            FileNotFoundError: Si le fichier source n'existe pas (mode CSV)
            ConnectionError: Si la connexion SQL Server échoue (mode SQL Server)
            ValueError: Si le fichier est vide ou mal formaté
        """
        if self.data_source == 'sqlserver':
            return self._load_from_sqlserver()
        else:
            return self._load_from_csv()

    def _load_from_csv(self) -> pd.DataFrame:
        """
        Charge les données depuis un fichier CSV.

        Returns:
            DataFrame: Données brutes depuis CSV
        """
        logger.info(f"📁 Chargement des données CSV depuis : {self.source_file_path}")

        if not self.source_file_path.exists():
            raise FileNotFoundError(f"{Messages.ERROR_FILE_NOT_FOUND}: {self.source_file_path}")

        try:
            # Chargement avec gestion des erreurs d'encodage
            self.raw_data = pd.read_csv(
                self.source_file_path,
                encoding='utf-8',
                parse_dates=[ColumnNames.SOURCE_DATE],
                date_format='%Y-%m-%d'
            )

            if self.raw_data.empty:
                raise ValueError(Messages.ERROR_NO_DATA)

            logger.info(f"{Messages.DATA_LOADED} - {len(self.raw_data)} lignes (CSV)")
            return self.raw_data

        except Exception as e:
            logger.error(f"Erreur lors du chargement CSV : {e}")
            raise

    def _load_from_sqlserver(self) -> pd.DataFrame:
        """
        Charge les données depuis SQL Server.

        Returns:
            DataFrame: Données brutes depuis SQL Server
        """
        logger.info(f"🗄️  Chargement des données depuis SQL Server...")

        try:
            # Connexion et récupération des données
            # On filtre déjà par date au niveau SQL pour optimiser
            self.raw_data = self.db_connector.fetch_commandes_data(
                start_date=None,  # On prend tout, on filtrera après
                end_date=None
            )

            if self.raw_data.empty:
                raise ValueError(Messages.ERROR_NO_DATA)

            logger.info(f"{Messages.DATA_LOADED} - {len(self.raw_data)} lignes (SQL Server)")
            return self.raw_data

        except Exception as e:
            logger.error(f"{Messages.ERROR_DB_CONNECTION}: {e}")
            raise ConnectionError(f"Impossible de charger depuis SQL Server : {e}")

    def standardize_columns(self) -> pd.DataFrame:
        """
        Standardise les noms de colonnes selon notre configuration.

        Returns:
            DataFrame: Données avec colonnes standardisées
        """
        column_mapping = {
            ColumnNames.SOURCE_DATE: ColumnNames.DATE,
            ColumnNames.SOURCE_ARTICLE_ID: ColumnNames.ARTICLE_ID,
            ColumnNames.SOURCE_ARTICLE_REF: ColumnNames.ARTICLE_REF,
            ColumnNames.SOURCE_QUANTITY: ColumnNames.QUANTITY
        }

        self.raw_data = self.raw_data.rename(columns=column_mapping)

        # Vérification que les colonnes essentielles sont présentes
        required_columns = [ColumnNames.DATE, ColumnNames.ARTICLE_ID, ColumnNames.QUANTITY]
        missing_columns = [col for col in required_columns if col not in self.raw_data.columns]

        if missing_columns:
            raise ValueError(f"Colonnes manquantes : {missing_columns}")

        logger.info("✅ Colonnes standardisées")
        return self.raw_data

    def filter_training_period(self) -> pd.DataFrame:
        """
        Filtre les données pour la période d'entraînement (2024 sauf décembre).

        Returns:
            DataFrame: Données filtrées sur la période d'entraînement
        """
        initial_count = len(self.raw_data)

        # Assurer que la colonne DATE est en datetime64 (sans heure)
        # Ceci évite les problèmes de comparaison datetime.datetime vs datetime.date
        self.raw_data[ColumnNames.DATE] = pd.to_datetime(self.raw_data[ColumnNames.DATE]).dt.normalize()

        # Convertir les dates de training en Timestamp pandas pour comparaison cohérente
        training_start_ts = pd.Timestamp(self.training_start.date())
        training_end_ts = pd.Timestamp(self.training_end.date())

        # Filtrage par date
        # Normaliser les dates (enlever heures/minutes)
        self.raw_data[ColumnNames.DATE] = pd.to_datetime(self.raw_data[ColumnNames.DATE]).dt.normalize()

        # Convertir en Timestamp pandas pour comparaison cohérente
        training_start_ts = pd.Timestamp(self.training_start.date())
        training_end_ts = pd.Timestamp(self.training_end.date())

        mask = (
                (self.raw_data[ColumnNames.DATE] >= training_start_ts) &
                (self.raw_data[ColumnNames.DATE] <= training_end_ts)
        )
        self.raw_data = self.raw_data[mask].copy()
        filtered_count = len(self.raw_data)

        logger.info(f"✅ Période filtrée : {initial_count} -> {filtered_count} lignes")
        logger.info(f"   Période : {self.training_start.date()} à {self.training_end.date()}")

        return self.raw_data

    def validate_and_clean_data(self) -> pd.DataFrame:
        """
        Valide et nettoie les données selon les règles métier.

        Returns:
            DataFrame: Données nettoyées et validées
        """
        logger.info("🧹 Début du nettoyage des données...")
        initial_count = len(self.raw_data)

        # 1. Suppression des lignes avec des valeurs manquantes critiques
        self.raw_data = self.raw_data.dropna(subset=[
            ColumnNames.DATE,
            ColumnNames.ARTICLE_ID,
            ColumnNames.QUANTITY
        ])

        # 2. Validation des quantités
        invalid_qty_mask = (
            (self.raw_data[ColumnNames.QUANTITY] < ValidationRules.MIN_QUANTITY) |
            (self.raw_data[ColumnNames.QUANTITY] > ValidationRules.MAX_QUANTITY) |
            (~np.isfinite(self.raw_data[ColumnNames.QUANTITY]))
        )

        invalid_count = invalid_qty_mask.sum()
        if invalid_count > 0:
            logger.warning(f"⚠️  {invalid_count} lignes avec quantités invalides supprimées")
            self.raw_data = self.raw_data[~invalid_qty_mask]

        # 3. Conversion des types de données
        self.raw_data[ColumnNames.ARTICLE_ID] = self.raw_data[ColumnNames.ARTICLE_ID].astype(int)
        self.raw_data[ColumnNames.QUANTITY] = self.raw_data[ColumnNames.QUANTITY].astype(int)

        # 4. Suppression des doublons exacts
        duplicates_count = self.raw_data.duplicated().sum()
        if duplicates_count > 0:
            logger.warning(f"⚠️  {duplicates_count} doublons exacts supprimés")
            self.raw_data = self.raw_data.drop_duplicates()

        clean_count = len(self.raw_data)
        logger.info(f"{Messages.DATA_CLEANED} - {initial_count} -> {clean_count} lignes")

        return self.raw_data

    def aggregate_daily_data(self) -> pd.DataFrame:
        """
        Agrège les données par jour et par article.
        Plusieurs lignes le même jour pour le même article sont sommées.

        Returns:
            DataFrame: Données agrégées par jour/article
        """
        logger.info("📊 Agrégation des données par jour/article...")

        # Agrégation : somme des quantités par date et article
        self.clean_data = self.raw_data.groupby([
            ColumnNames.DATE,
            ColumnNames.ARTICLE_ID
        ])[ColumnNames.QUANTITY].sum().reset_index()

        # Tri par date puis par article_id
        self.clean_data = self.clean_data.sort_values([
            ColumnNames.DATE,
            ColumnNames.ARTICLE_ID
        ]).reset_index(drop=True)

        unique_combinations = len(self.clean_data)
        unique_dates = self.clean_data[ColumnNames.DATE].nunique()
        unique_articles = self.clean_data[ColumnNames.ARTICLE_ID].nunique()

        logger.info(f"✅ Données agrégées :")
        logger.info(f"   - {unique_combinations} combinaisons jour/article")
        logger.info(f"   - {unique_dates} jours uniques")
        logger.info(f"   - {unique_articles} articles uniques")

        return self.clean_data

    def fill_missing_combinations(self) -> pd.DataFrame:
        """
        Ajoute les combinaisons jour/article manquantes avec quantité = 0.

        OBJECTIF : Avoir 1 ligne par jour ET par article, même si quantité = 0

        Returns:
            DataFrame: Données complètes avec toutes les combinaisons
        """
        logger.info("🔄 Ajout des combinaisons manquantes (quantité = 0)...")

        # Récupération de toutes les dates et articles uniques
        all_dates = pd.date_range(
            start=self.training_start,
            end=self.training_end,
            freq='D'
        )
        all_articles = self.clean_data[ColumnNames.ARTICLE_ID].unique()

        logger.info(f"   - Période : {len(all_dates)} jours")
        logger.info(f"   - Articles : {len(all_articles)} articles")
        logger.info(f"   - Combinaisons théoriques : {len(all_dates) * len(all_articles)}")

        # Création de toutes les combinaisons possibles
        all_combinations = pd.MultiIndex.from_product(
            [all_dates, all_articles],
            names=[ColumnNames.DATE, ColumnNames.ARTICLE_ID]
        ).to_frame(index=False)

        # Fusion avec les données existantes (LEFT JOIN)
        self.final_data = pd.merge(
            all_combinations,
            self.clean_data,
            on=[ColumnNames.DATE, ColumnNames.ARTICLE_ID],
            how='left'
        )

        # Remplacement des valeurs manquantes par 0
        self.final_data[ColumnNames.QUANTITY] = self.final_data[ColumnNames.QUANTITY].fillna(0).astype(int)

        # Statistiques finales
        total_combinations = len(self.final_data)
        zero_quantity_count = (self.final_data[ColumnNames.QUANTITY] == 0).sum()

        logger.info(f"✅ Données complétées :")
        logger.info(f"   - {total_combinations} lignes finales")
        logger.info(f"   - {zero_quantity_count} lignes avec quantité = 0")
        logger.info(f"   - {total_combinations - zero_quantity_count} lignes avec commandes")

        return self.final_data

    def save_clean_data(self, output_path: Optional[Path] = None) -> Path:
        """
        Sauvegarde les données nettoyées.

        Args:
            output_path: Chemin de sortie (optionnel)

        Returns:
            Path: Chemin du fichier sauvegardé
        """
        if self.final_data is None:
            raise ValueError("Aucune donnée à sauvegarder. Exécutez le pipeline d'abord.")

        output_path = output_path or get_file_path('clean')

        self.final_data.to_csv(output_path, index=False, date_format='%Y-%m-%d')
        logger.info(f"💾 Données sauvegardées : {output_path}")

        return output_path

    def run_full_pipeline(self) -> pd.DataFrame:
        """
        Exécute le pipeline complet d'ingestion et de nettoyage.

        Returns:
            DataFrame: Données finales nettoyées et complétées
        """
        logger.info("🚀 DÉBUT DU PIPELINE D'INGESTION")
        logger.info("=" * 50)
        logger.info(f"📊 Source de données : {self.data_source.upper()}")

        try:
            # Étapes du pipeline
            self.load_raw_data()
            self.standardize_columns()
            self.filter_training_period()
            self.validate_and_clean_data()
            self.aggregate_daily_data()
            self.fill_missing_combinations()

            # Sauvegarde
            self.save_clean_data()

            logger.info("=" * 50)
            logger.info("✅ PIPELINE D'INGESTION TERMINÉ AVEC SUCCÈS")

            return self.final_data

        except Exception as e:
            logger.error(f"❌ ERREUR DANS LE PIPELINE : {e}")
            raise
        finally:
            # Fermer la connexion SQL Server si elle existe
            if self.db_connector and self.db_connector.connection:
                self.db_connector.disconnect()

    def get_data_summary(self) -> dict:
        """
        Retourne un résumé statistique des données finales.

        Returns:
            dict: Statistiques des données
        """
        if self.final_data is None:
            return {"error": "Aucune donnée disponible"}

        return {
            "data_source": self.data_source,
            "total_lines": len(self.final_data),
            "unique_dates": self.final_data[ColumnNames.DATE].nunique(),
            "unique_articles": self.final_data[ColumnNames.ARTICLE_ID].nunique(),
            "total_quantity": self.final_data[ColumnNames.QUANTITY].sum(),
            "zero_quantity_lines": (self.final_data[ColumnNames.QUANTITY] == 0).sum(),
            "date_range": {
                "start": self.final_data[ColumnNames.DATE].min(),
                "end": self.final_data[ColumnNames.DATE].max()
            }
        }


# ===== FONCTIONS UTILITAIRES =====

def quick_data_ingestion(
    source: str = None,
    source_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Fonction rapide pour l'ingestion complète des données.

    Args:
        source: 'csv' ou 'sqlserver' (si None, utilise la config)
        source_file: Chemin vers le fichier source (si mode CSV)

    Returns:
        DataFrame: Données prêtes pour l'analyse
    """
    source_path = Path(source_file) if source_file else None
    pipeline = DataIngestionPipeline(source_path, data_source=source)
    return pipeline.run_full_pipeline()


def preview_raw_data(file_path: Path, n_rows: int = 10) -> None:
    """
    Affiche un aperçu des données brutes (CSV uniquement).

    Args:
        file_path: Chemin vers le fichier
        n_rows: Nombre de lignes à afficher
    """
    try:
        data = pd.read_csv(file_path, nrows=n_rows)
        print(f"📋 Aperçu des {n_rows} premières lignes :")
        print("=" * 60)
        print(data.to_string())
        print("=" * 60)
        print(f"Colonnes disponibles : {list(data.columns)}")
        print(f"Forme du fichier complet : {pd.read_csv(file_path, usecols=[0]).shape[0]} lignes")

    except Exception as e:
        print(f"❌ Erreur lors de l'aperçu : {e}")


if __name__ == "__main__":
    # Test du pipeline si exécuté directement
    print("="*60)
    print("🧪 TEST DU PIPELINE D'INGESTION")
    print("="*60)

    # Afficher la source configurée
    from src.utils.config import print_data_source_info
    print_data_source_info()

    print("\n🚀 Lancement du pipeline...")

    pipeline = DataIngestionPipeline()
    try:
        data = pipeline.run_full_pipeline()
        summary = pipeline.get_data_summary()

        print("\n📊 RÉSUMÉ DES DONNÉES :")
        for key, value in summary.items():
            print(f"   {key}: {value}")

    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        print("\n💡 Vérifications :")
        print("   1. Le fichier .env existe avec les bonnes valeurs")
        print("   2. DATA_SOURCE est 'csv' ou 'sqlserver'")
        print("   3. Si SQL Server : les credentials sont corrects")
        print("   4. Si CSV : le fichier existe dans data/raw/")