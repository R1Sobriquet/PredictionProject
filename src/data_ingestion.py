"""
Module d'ingestion et de nettoyage des données de commandes ATM.

Support hybride :
- Mode CSV : lecture directe de fichiers Excel (.xlsx) via openpyxl
- Mode Database : requête SQL (MySQL ou SQL Server)

Source unique : d_CommandesDetailCalcul.xlsx (ne pas utiliser d_Commandes.xlsx)

Pipeline simplifié (commandes individuelles, pas de remplissage de zéros) :
1. Chargement des données brutes
2. Filtrage des commandes annulées
3. Standardisation des colonnes
4. Filtrage de la période
5. Validation et nettoyage
6. Sauvegarde
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging
from pathlib import Path

try:
    from .utils import (
        ColumnNames,
        ValidationRules,
        Messages,
        DataSourceConfig,
        COLUMNS_TO_LOAD,
        COLUMN_MAPPING,
        get_training_date_range,
        get_file_path,
    )
    from .database_connector import DatabaseConnector
except ImportError:
    from src.utils import (
        ColumnNames,
        ValidationRules,
        Messages,
        DataSourceConfig,
        COLUMNS_TO_LOAD,
        COLUMN_MAPPING,
        get_training_date_range,
        get_file_path,
    )
    from src.database_connector import DatabaseConnector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """
    Pipeline d'ingestion et de nettoyage des données de commandes ATM.

    Support hybride :
    - CSV/Excel : charge depuis un fichier .xlsx avec openpyxl
    - Database : charge depuis MySQL ou SQL Server

    Les commandes annulées (DC_Annule == 1) sont exclues.
    Les commandes individuelles sont conservées (pas d'agrégation ni de zero-fill).
    """

    def __init__(
        self,
        source_file_path: Optional[Path] = None,
        data_source: Optional[str] = None,
        db_connector: Optional[DatabaseConnector] = None,
    ):
        self.data_source = data_source or DataSourceConfig.DEFAULT_SOURCE

        if self.data_source == 'database':
            self.db_connector = db_connector or DatabaseConnector()
            self.source_file_path = None
            logger.info(f"Mode Database ({DataSourceConfig.DB_TYPE}) : "
                        f"{DataSourceConfig.DB_SERVER} / {DataSourceConfig.DB_NAME}")
        else:
            self.source_file_path = Path(source_file_path) if source_file_path else Path(DataSourceConfig.CSV_FILE_PATH)
            self.db_connector = None
            logger.info(f"Mode Excel : {self.source_file_path}")

        self.training_start, self.training_end = get_training_date_range()
        self.raw_data = None
        self.clean_data = None

    def load_raw_data(self) -> pd.DataFrame:
        """
        Charge les données brutes depuis la source configurée.

        Returns:
            DataFrame: Données brutes chargées
        """
        if self.data_source == 'database':
            return self._load_from_database()
        else:
            return self._load_from_excel()

    def _load_from_excel(self) -> pd.DataFrame:
        """Charge les données depuis un fichier Excel (.xlsx) avec openpyxl."""
        logger.info(f"Chargement Excel depuis : {self.source_file_path}")

        if not self.source_file_path.exists():
            raise FileNotFoundError(f"{Messages.ERROR_FILE_NOT_FOUND}: {self.source_file_path}")

        try:
            self.raw_data = pd.read_excel(
                self.source_file_path,
                engine='openpyxl',
                usecols=COLUMNS_TO_LOAD,
            )

            if self.raw_data.empty:
                raise ValueError(Messages.ERROR_NO_DATA)

            logger.info(f"{Messages.DATA_LOADED} - {len(self.raw_data)} lignes (Excel)")
            return self.raw_data

        except Exception as e:
            logger.error(f"Erreur lors du chargement Excel : {e}")
            raise

    def _load_from_database(self) -> pd.DataFrame:
        """Charge les données depuis la base de données (MySQL ou SQL Server)."""
        logger.info("Chargement depuis la base de données...")

        try:
            # Le connecteur filtre déjà DC_Annule != 1 et sélectionne COLUMNS_TO_LOAD
            self.raw_data = self.db_connector.fetch_commandes_data()

            if self.raw_data.empty:
                raise ValueError(Messages.ERROR_NO_DATA)

            logger.info(f"{Messages.DATA_LOADED} - {len(self.raw_data)} lignes (Database)")
            return self.raw_data

        except Exception as e:
            logger.error(f"{Messages.ERROR_DB_CONNECTION}: {e}")
            raise ConnectionError(f"Impossible de charger depuis la base : {e}")

    def filter_cancelled_orders(self) -> pd.DataFrame:
        """
        Filtre les commandes annulées (DC_Annule == 1).
        Nécessaire pour le mode Excel car le filtre SQL n'est pas appliqué.

        Returns:
            DataFrame: Données sans commandes annulées
        """
        if ColumnNames.SOURCE_ANNULE in self.raw_data.columns:
            col = ColumnNames.SOURCE_ANNULE
        elif ColumnNames.ANNULE in self.raw_data.columns:
            col = ColumnNames.ANNULE
        else:
            logger.info("Colonne annulation non trouvée, aucun filtrage")
            return self.raw_data

        initial_count = len(self.raw_data)
        cancelled_mask = self.raw_data[col] == 1
        cancelled_count = cancelled_mask.sum()

        if cancelled_count > 0:
            self.raw_data = self.raw_data[~cancelled_mask].copy()
            logger.info(f"Commandes annulées filtrées : {cancelled_count} supprimées "
                        f"({initial_count} -> {len(self.raw_data)})")
        else:
            logger.info("Aucune commande annulée trouvée")

        return self.raw_data

    def standardize_columns(self) -> pd.DataFrame:
        """
        Standardise les noms de colonnes DC_* vers les noms internes.

        Returns:
            DataFrame: Données avec colonnes standardisées
        """
        # Appliquer le mapping source → standardisé
        rename_map = {k: v for k, v in COLUMN_MAPPING.items() if k in self.raw_data.columns}
        self.raw_data = self.raw_data.rename(columns=rename_map)

        # Vérifier les colonnes essentielles
        required = [ColumnNames.ORDER_ID, ColumnNames.ATM_ID, ColumnNames.ORDER_DATE, ColumnNames.AMOUNT]
        missing = [col for col in required if col not in self.raw_data.columns]

        if missing:
            raise ValueError(f"Colonnes manquantes après standardisation : {missing}")

        logger.info(f"Colonnes standardisées ({len(rename_map)} renommées)")
        return self.raw_data

    def filter_training_period(self) -> pd.DataFrame:
        """
        Filtre les données pour la période d'entraînement.

        Returns:
            DataFrame: Données filtrées
        """
        initial_count = len(self.raw_data)

        # Convertir en datetime
        self.raw_data[ColumnNames.ORDER_DATE] = pd.to_datetime(
            self.raw_data[ColumnNames.ORDER_DATE]
        ).dt.normalize()

        # Convertir aussi les dates prévisionnelles si présentes
        for date_col in [ColumnNames.DELIVERY_DATE, ColumnNames.LOADING_DATE]:
            if date_col in self.raw_data.columns:
                self.raw_data[date_col] = pd.to_datetime(
                    self.raw_data[date_col], errors='coerce'
                )

        # Filtrage
        training_start_ts = pd.Timestamp(self.training_start.date())
        training_end_ts = pd.Timestamp(self.training_end.date())

        mask = (
            (self.raw_data[ColumnNames.ORDER_DATE] >= training_start_ts) &
            (self.raw_data[ColumnNames.ORDER_DATE] <= training_end_ts)
        )
        self.raw_data = self.raw_data[mask].copy()

        logger.info(f"Période filtrée : {initial_count} -> {len(self.raw_data)} lignes")
        logger.info(f"  Période : {self.training_start.date()} à {self.training_end.date()}")

        return self.raw_data

    def validate_and_clean_data(self) -> pd.DataFrame:
        """
        Valide et nettoie les données.

        Returns:
            DataFrame: Données nettoyées
        """
        logger.info("Début du nettoyage des données...")
        initial_count = len(self.raw_data)

        # 1. Suppression des lignes avec valeurs manquantes critiques
        self.raw_data = self.raw_data.dropna(subset=[
            ColumnNames.ORDER_ID,
            ColumnNames.ATM_ID,
            ColumnNames.ORDER_DATE,
            ColumnNames.AMOUNT,
        ])

        # 2. Déduplication sur DC_Commande_Id (clé unique)
        duplicates_count = self.raw_data.duplicated(subset=[ColumnNames.ORDER_ID]).sum()
        if duplicates_count > 0:
            logger.warning(f"{duplicates_count} doublons sur order_id supprimés")
            self.raw_data = self.raw_data.drop_duplicates(subset=[ColumnNames.ORDER_ID], keep='first')

        # 3. Validation des montants
        self.raw_data[ColumnNames.AMOUNT] = pd.to_numeric(self.raw_data[ColumnNames.AMOUNT], errors='coerce')

        invalid_amount_mask = (
            self.raw_data[ColumnNames.AMOUNT].isna() |
            (self.raw_data[ColumnNames.AMOUNT] < ValidationRules.MIN_AMOUNT) |
            (self.raw_data[ColumnNames.AMOUNT] > ValidationRules.MAX_AMOUNT)
        )

        invalid_count = invalid_amount_mask.sum()
        if invalid_count > 0:
            logger.warning(f"{invalid_count} lignes avec montants invalides supprimées")
            self.raw_data = self.raw_data[~invalid_amount_mask]

        # 4. Conversion des types
        self.raw_data[ColumnNames.ATM_ID] = self.raw_data[ColumnNames.ATM_ID].astype(int)
        self.raw_data[ColumnNames.AMOUNT] = self.raw_data[ColumnNames.AMOUNT].astype(float)

        # 5. Remplir les NaN numériques dans les colonnes de données (cassettes, soldes, etc.)
        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        self.raw_data[numeric_cols] = self.raw_data[numeric_cols].fillna(0)

        # Tri par date puis par ATM
        self.raw_data = self.raw_data.sort_values(
            [ColumnNames.ORDER_DATE, ColumnNames.ATM_ID]
        ).reset_index(drop=True)

        self.clean_data = self.raw_data
        clean_count = len(self.clean_data)
        logger.info(f"{Messages.DATA_CLEANED} - {initial_count} -> {clean_count} lignes")

        return self.clean_data

    def save_clean_data(self, output_path: Optional[Path] = None) -> Path:
        """Sauvegarde les données nettoyées."""
        if self.clean_data is None:
            raise ValueError("Aucune donnée à sauvegarder. Exécutez le pipeline d'abord.")

        output_path = output_path or get_file_path('clean')
        self.clean_data.to_csv(output_path, index=False, date_format='%Y-%m-%d')
        logger.info(f"Données sauvegardées : {output_path}")
        return output_path

    def save_intermediate_snapshot(self, data: pd.DataFrame, stage_name: str) -> Path:
        """Sauvegarde un snapshot intermédiaire."""
        snapshot_dir = Path("data/snapshots")
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        snapshot_file = snapshot_dir / f"snapshot_{stage_name}.csv"
        data.to_csv(snapshot_file, index=False, date_format='%Y-%m-%d')
        logger.info(f"  Snapshot sauvegardé : {snapshot_file} ({len(data)} lignes)")
        return snapshot_file

    def run_full_pipeline(self, save_snapshots: bool = True) -> pd.DataFrame:
        """
        Exécute le pipeline complet d'ingestion et de nettoyage.

        Args:
            save_snapshots: Si True, sauvegarde des CSV intermédiaires

        Returns:
            DataFrame: Données nettoyées
        """
        logger.info("DÉBUT DU PIPELINE D'INGESTION ATM")
        logger.info("=" * 50)
        logger.info(f"Source de données : {self.data_source.upper()}")

        try:
            # ÉTAPE 1 : Chargement
            logger.info("\nÉTAPE 1 : Chargement des données brutes...")
            self.load_raw_data()
            if save_snapshots:
                self.save_intermediate_snapshot(self.raw_data, "01_raw_loaded")

            # ÉTAPE 2 : Filtrage des commandes annulées
            logger.info("\nÉTAPE 2 : Filtrage des commandes annulées...")
            self.filter_cancelled_orders()
            if save_snapshots:
                self.save_intermediate_snapshot(self.raw_data, "02_filtered_cancelled")

            # ÉTAPE 3 : Standardisation des colonnes
            logger.info("\nÉTAPE 3 : Standardisation des colonnes...")
            self.standardize_columns()
            if save_snapshots:
                self.save_intermediate_snapshot(self.raw_data, "03_standardized")

            # ÉTAPE 4 : Filtrage de la période
            logger.info("\nÉTAPE 4 : Filtrage de la période d'entraînement...")
            self.filter_training_period()
            if save_snapshots:
                self.save_intermediate_snapshot(self.raw_data, "04_filtered_period")

            # ÉTAPE 5 : Validation et nettoyage
            logger.info("\nÉTAPE 5 : Validation et nettoyage...")
            self.validate_and_clean_data()
            if save_snapshots:
                self.save_intermediate_snapshot(self.clean_data, "05_cleaned")

            # ÉTAPE 6 : Sauvegarde finale
            logger.info("\nÉTAPE 6 : Sauvegarde finale...")
            self.save_clean_data()

            logger.info("=" * 50)
            logger.info("PIPELINE D'INGESTION TERMINÉ AVEC SUCCÈS")

            return self.clean_data

        except Exception as e:
            logger.error(f"ERREUR DANS LE PIPELINE : {e}")
            raise
        finally:
            if self.db_connector and self.db_connector.connection:
                self.db_connector.disconnect()

    def get_data_summary(self) -> dict:
        """Retourne un résumé statistique des données."""
        if self.clean_data is None:
            return {"error": "Aucune donnée disponible"}

        return {
            "data_source": self.data_source,
            "total_orders": len(self.clean_data),
            "unique_dates": self.clean_data[ColumnNames.ORDER_DATE].nunique(),
            "unique_atms": self.clean_data[ColumnNames.ATM_ID].nunique(),
            "total_amount": self.clean_data[ColumnNames.AMOUNT].sum(),
            "avg_amount": self.clean_data[ColumnNames.AMOUNT].mean(),
            "date_range": {
                "start": self.clean_data[ColumnNames.ORDER_DATE].min(),
                "end": self.clean_data[ColumnNames.ORDER_DATE].max(),
            },
        }


# ===== FONCTIONS UTILITAIRES =====

def quick_data_ingestion(
    source: str = None,
    source_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fonction rapide pour l'ingestion complète.

    Args:
        source: 'csv' ou 'database'
        source_file: Chemin vers le fichier source (mode CSV)

    Returns:
        DataFrame: Données prêtes pour l'analyse
    """
    source_path = Path(source_file) if source_file else None
    pipeline = DataIngestionPipeline(source_path, data_source=source)
    return pipeline.run_full_pipeline()


if __name__ == "__main__":
    print("=" * 60)
    print("TEST DU PIPELINE D'INGESTION ATM")
    print("=" * 60)

    from src.utils.config import print_data_source_info
    print_data_source_info()

    print("\nLancement du pipeline...")

    pipeline = DataIngestionPipeline()
    try:
        data = pipeline.run_full_pipeline()
        summary = pipeline.get_data_summary()

        print("\nRÉSUMÉ DES DONNÉES :")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"\nErreur : {e}")
        print("\nVérifications :")
        print("  1. Le fichier .env existe avec les bonnes valeurs")
        print("  2. DATA_SOURCE est 'csv' ou 'database'")
        print("  3. Si database : les credentials sont corrects")
        print("  4. Si csv : le fichier .xlsx existe dans data/raw/")
