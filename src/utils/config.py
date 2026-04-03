"""
Configuration centrale du projet de prévision de commandes ATM.
Ce fichier centralise tous les paramètres pour faciliter la maintenance.

Support hybride :
- Lecture directe de fichiers Excel (.xlsx) exportés depuis HFSQL
- Connexion base de données (MySQL ou SQL Server)

La source est configurée via le fichier .env (DATA_SOURCE=csv ou database)
"""

from datetime import datetime
from pathlib import Path
from typing import List, Dict
import os

# Charger les variables d'environnement depuis .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ===== CHEMINS DE FICHIERS =====
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
DOCS_DIR = PROJECT_ROOT / "docs"

# Création automatique des dossiers
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_OUTPUT_DIR, DOCS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ===== NOMS DE FICHIERS =====
RAW_DATA_FILE = "d_CommandesDetailCalcul.xlsx"
CLEAN_DATA_FILE = "commandes_clean.csv"
ENRICHED_DATA_FILE = "commandes_enriched.csv"
PREDICTIONS_FILE = "predictions_2026.csv"


# ===== COLONNES SOURCE HFSQL (DC_*) =====
# Ces noms correspondent aux colonnes du fichier Excel exporté depuis HFSQL (WinDev)

class ColumnNames:
    """Noms des colonnes source HFSQL et leurs équivalents standardisés."""

    # --- Colonnes source (fichier Excel / table BDD) ---

    # Identifiants
    SOURCE_ORDER_ID = "DC_Commande_Id"
    SOURCE_ATM_ID = "DC_Automate_Id"
    SOURCE_ORDER_DATE = "DC_Date_Cmd"
    SOURCE_ORDER_TYPE = "DC_Type_Cmd"

    # Target principale
    SOURCE_AMOUNT = "DC_Montant_Cmd"

    # Dates prévisionnelles
    SOURCE_DELIVERY_DATE = "DC_Livraison_Prev_Date"
    SOURCE_LOADING_DATE = "DC_Chargement_Prev_Date"

    # Cassettes (quantités par cassette physique)
    SOURCE_CASSETTE_1 = "DC_Cassette_1"
    SOURCE_CASSETTE_2 = "DC_Cassette_2"
    SOURCE_CASSETTE_3 = "DC_Cassette_3"
    SOURCE_CASSETTE_4 = "DC_Cassette_4"
    SOURCE_CASSETTE_5 = "DC_Cassette_5"

    # Ajustements par coupure (euros)
    SOURCE_AJUSTEMENT_5 = "DC_Ajustement_5"
    SOURCE_AJUSTEMENT_10 = "DC_Ajustement_10"
    SOURCE_AJUSTEMENT_20 = "DC_Ajustement_20"
    SOURCE_AJUSTEMENT_50 = "DC_Ajustement_50"
    SOURCE_AJUSTEMENT_100 = "DC_Ajustement_100"

    # Soldes du jour par coupure
    SOURCE_SOLDES_5 = "DC_SoldesDuJour_5"
    SOURCE_SOLDES_10 = "DC_SoldesDuJour_10"
    SOURCE_SOLDES_20 = "DC_SoldesDuJour_20"
    SOURCE_SOLDES_50 = "DC_SoldesDuJour_50"
    SOURCE_SOLDES_100 = "DC_SoldesDuJour_100"

    # Cassettes hors service par coupure
    SOURCE_K7HS_5 = "DC_K7HS_5"
    SOURCE_K7HS_10 = "DC_K7HS_10"
    SOURCE_K7HS_20 = "DC_K7HS_20"
    SOURCE_K7HS_50 = "DC_K7HS_50"
    SOURCE_K7HS_100 = "DC_K7HS_100"

    # Volatilité et flags DMQ
    SOURCE_VOLATILITE_DMQ = "DC_VolatiliteDmq"
    SOURCE_DMQ_FORTE_DECROISSANCE = "DC_DmqForteDecroissance"
    SOURCE_DMQ_FORTE_CROISSANCE = "DC_DmqForteCroissance"

    # Flags de statut
    SOURCE_ANNULE = "DC_Annule"
    SOURCE_CHARGE = "DC_Chargé"
    SOURCE_EVENEMENT_EN_COURS = "DC_EvenementEnCour"
    SOURCE_RISQUE_ATM_VIDE = "DC_RisqueAutomateVide"

    # NOTE : DC_Predictif_5/10/20/50/100 sont EXCLUS volontairement.
    # Ce sont des prédictions du système source — les utiliser comme features
    # constituerait une fuite de données (data leakage).

    # --- Colonnes standardisées (utilisées dans le code interne) ---

    # Identifiants
    ORDER_ID = "order_id"
    ATM_ID = "atm_id"
    ORDER_DATE = "order_date"
    ORDER_TYPE = "order_type"

    # Target
    AMOUNT = "amount"

    # Dates
    DELIVERY_DATE = "delivery_date"
    LOADING_DATE = "loading_date"

    # Cassettes
    CASSETTE_1 = "cassette_1"
    CASSETTE_2 = "cassette_2"
    CASSETTE_3 = "cassette_3"
    CASSETTE_4 = "cassette_4"
    CASSETTE_5 = "cassette_5"

    # Ajustements par coupure
    AJUSTEMENT_5 = "ajustement_5"
    AJUSTEMENT_10 = "ajustement_10"
    AJUSTEMENT_20 = "ajustement_20"
    AJUSTEMENT_50 = "ajustement_50"
    AJUSTEMENT_100 = "ajustement_100"

    # Soldes du jour
    SOLDES_5 = "soldes_5"
    SOLDES_10 = "soldes_10"
    SOLDES_20 = "soldes_20"
    SOLDES_50 = "soldes_50"
    SOLDES_100 = "soldes_100"

    # Cassettes hors service
    K7HS_5 = "k7hs_5"
    K7HS_10 = "k7hs_10"
    K7HS_20 = "k7hs_20"
    K7HS_50 = "k7hs_50"
    K7HS_100 = "k7hs_100"

    # Volatilité et flags
    VOLATILITE_DMQ = "volatilite_dmq"
    DMQ_FORTE_DECROISSANCE = "dmq_forte_decroissance"
    DMQ_FORTE_CROISSANCE = "dmq_forte_croissance"

    # Statuts
    ANNULE = "annule"
    CHARGE = "charge"
    EVENEMENT_EN_COURS = "evenement_en_cours"
    RISQUE_ATM_VIDE = "risque_atm_vide"

    # --- Colonnes enrichies (ajoutées par le pipeline) ---
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    WEEKDAY = "weekday"  # 0=Lundi, 6=Dimanche
    WEEKDAY_NAME = "weekday_name"
    IS_WEEKEND = "is_weekend"
    WEEK_NUMBER = "week_number"

    # Features ATM historiques
    DAYS_SINCE_LAST_ORDER = "days_since_last_order"
    LAST_ORDER_AMOUNT = "last_order_amount"
    AVG_RELOAD_FREQUENCY = "avg_reload_frequency"
    AVG_ORDER_AMOUNT = "avg_order_amount"
    STD_ORDER_AMOUNT = "std_order_amount"
    ORDER_COUNT_LAST_30D = "order_count_last_30d"


# ===== LISTE DES COLONNES À CHARGER =====
# Uniquement les colonnes pertinentes (~25 sur 51)
# EXCLUT DC_Predictif_* (fuite de données)

COLUMNS_TO_LOAD = [
    # Identifiants
    ColumnNames.SOURCE_ORDER_ID,
    ColumnNames.SOURCE_ATM_ID,
    ColumnNames.SOURCE_ORDER_DATE,
    ColumnNames.SOURCE_ORDER_TYPE,
    # Target
    ColumnNames.SOURCE_AMOUNT,
    # Dates
    ColumnNames.SOURCE_DELIVERY_DATE,
    ColumnNames.SOURCE_LOADING_DATE,
    # Cassettes
    ColumnNames.SOURCE_CASSETTE_1,
    ColumnNames.SOURCE_CASSETTE_2,
    ColumnNames.SOURCE_CASSETTE_3,
    ColumnNames.SOURCE_CASSETTE_4,
    ColumnNames.SOURCE_CASSETTE_5,
    # Ajustements
    ColumnNames.SOURCE_AJUSTEMENT_5,
    ColumnNames.SOURCE_AJUSTEMENT_10,
    ColumnNames.SOURCE_AJUSTEMENT_20,
    ColumnNames.SOURCE_AJUSTEMENT_50,
    ColumnNames.SOURCE_AJUSTEMENT_100,
    # Soldes du jour
    ColumnNames.SOURCE_SOLDES_5,
    ColumnNames.SOURCE_SOLDES_10,
    ColumnNames.SOURCE_SOLDES_20,
    ColumnNames.SOURCE_SOLDES_50,
    ColumnNames.SOURCE_SOLDES_100,
    # Cassettes hors service
    ColumnNames.SOURCE_K7HS_5,
    ColumnNames.SOURCE_K7HS_10,
    ColumnNames.SOURCE_K7HS_20,
    ColumnNames.SOURCE_K7HS_50,
    ColumnNames.SOURCE_K7HS_100,
    # Volatilité et flags
    ColumnNames.SOURCE_VOLATILITE_DMQ,
    ColumnNames.SOURCE_DMQ_FORTE_DECROISSANCE,
    ColumnNames.SOURCE_DMQ_FORTE_CROISSANCE,
    # Statuts
    ColumnNames.SOURCE_ANNULE,
    ColumnNames.SOURCE_CHARGE,
    ColumnNames.SOURCE_EVENEMENT_EN_COURS,
    ColumnNames.SOURCE_RISQUE_ATM_VIDE,
]

# ===== MAPPING SOURCE → STANDARDISÉ =====

COLUMN_MAPPING: Dict[str, str] = {
    ColumnNames.SOURCE_ORDER_ID: ColumnNames.ORDER_ID,
    ColumnNames.SOURCE_ATM_ID: ColumnNames.ATM_ID,
    ColumnNames.SOURCE_ORDER_DATE: ColumnNames.ORDER_DATE,
    ColumnNames.SOURCE_ORDER_TYPE: ColumnNames.ORDER_TYPE,
    ColumnNames.SOURCE_AMOUNT: ColumnNames.AMOUNT,
    ColumnNames.SOURCE_DELIVERY_DATE: ColumnNames.DELIVERY_DATE,
    ColumnNames.SOURCE_LOADING_DATE: ColumnNames.LOADING_DATE,
    ColumnNames.SOURCE_CASSETTE_1: ColumnNames.CASSETTE_1,
    ColumnNames.SOURCE_CASSETTE_2: ColumnNames.CASSETTE_2,
    ColumnNames.SOURCE_CASSETTE_3: ColumnNames.CASSETTE_3,
    ColumnNames.SOURCE_CASSETTE_4: ColumnNames.CASSETTE_4,
    ColumnNames.SOURCE_CASSETTE_5: ColumnNames.CASSETTE_5,
    ColumnNames.SOURCE_AJUSTEMENT_5: ColumnNames.AJUSTEMENT_5,
    ColumnNames.SOURCE_AJUSTEMENT_10: ColumnNames.AJUSTEMENT_10,
    ColumnNames.SOURCE_AJUSTEMENT_20: ColumnNames.AJUSTEMENT_20,
    ColumnNames.SOURCE_AJUSTEMENT_50: ColumnNames.AJUSTEMENT_50,
    ColumnNames.SOURCE_AJUSTEMENT_100: ColumnNames.AJUSTEMENT_100,
    ColumnNames.SOURCE_SOLDES_5: ColumnNames.SOLDES_5,
    ColumnNames.SOURCE_SOLDES_10: ColumnNames.SOLDES_10,
    ColumnNames.SOURCE_SOLDES_20: ColumnNames.SOLDES_20,
    ColumnNames.SOURCE_SOLDES_50: ColumnNames.SOLDES_50,
    ColumnNames.SOURCE_SOLDES_100: ColumnNames.SOLDES_100,
    ColumnNames.SOURCE_K7HS_5: ColumnNames.K7HS_5,
    ColumnNames.SOURCE_K7HS_10: ColumnNames.K7HS_10,
    ColumnNames.SOURCE_K7HS_20: ColumnNames.K7HS_20,
    ColumnNames.SOURCE_K7HS_50: ColumnNames.K7HS_50,
    ColumnNames.SOURCE_K7HS_100: ColumnNames.K7HS_100,
    ColumnNames.SOURCE_VOLATILITE_DMQ: ColumnNames.VOLATILITE_DMQ,
    ColumnNames.SOURCE_DMQ_FORTE_DECROISSANCE: ColumnNames.DMQ_FORTE_DECROISSANCE,
    ColumnNames.SOURCE_DMQ_FORTE_CROISSANCE: ColumnNames.DMQ_FORTE_CROISSANCE,
    ColumnNames.SOURCE_ANNULE: ColumnNames.ANNULE,
    ColumnNames.SOURCE_CHARGE: ColumnNames.CHARGE,
    ColumnNames.SOURCE_EVENEMENT_EN_COURS: ColumnNames.EVENEMENT_EN_COURS,
    ColumnNames.SOURCE_RISQUE_ATM_VIDE: ColumnNames.RISQUE_ATM_VIDE,
}

# Listes de colonnes groupées (noms standardisés)
CASSETTE_COLUMNS = [
    ColumnNames.CASSETTE_1, ColumnNames.CASSETTE_2, ColumnNames.CASSETTE_3,
    ColumnNames.CASSETTE_4, ColumnNames.CASSETTE_5,
]
AJUSTEMENT_COLUMNS = [
    ColumnNames.AJUSTEMENT_5, ColumnNames.AJUSTEMENT_10, ColumnNames.AJUSTEMENT_20,
    ColumnNames.AJUSTEMENT_50, ColumnNames.AJUSTEMENT_100,
]
SOLDES_COLUMNS = [
    ColumnNames.SOLDES_5, ColumnNames.SOLDES_10, ColumnNames.SOLDES_20,
    ColumnNames.SOLDES_50, ColumnNames.SOLDES_100,
]
K7HS_COLUMNS = [
    ColumnNames.K7HS_5, ColumnNames.K7HS_10, ColumnNames.K7HS_20,
    ColumnNames.K7HS_50, ColumnNames.K7HS_100,
]


# ===== CONFIGURATION SOURCE DE DONNÉES =====
class DataSourceConfig:
    """Configuration de la source de données (CSV/Excel ou base de données)."""

    # Source : 'csv' (fichier Excel .xlsx) ou 'database'
    DEFAULT_SOURCE = os.getenv('DATA_SOURCE', 'csv').lower()

    # Pour CSV/Excel
    CSV_FILE_PATH = os.getenv('CSV_FILE_PATH', str(DATA_RAW_DIR / RAW_DATA_FILE))

    # Type de base de données : 'mysql' ou 'sqlserver'
    DB_TYPE = os.getenv('DB_TYPE', 'mysql').lower()

    # Paramètres de connexion BDD
    DB_SERVER = os.getenv('DB_SERVER', 'localhost')
    DB_NAME = os.getenv('DB_NAME', 'prediction_db')
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_TABLE = os.getenv('DB_TABLE', 'd_CommandesDetailCalcul')
    DB_DRIVER = os.getenv('DB_DRIVER', 'ODBC Driver 17 for SQL Server')
    DB_TIMEOUT = int(os.getenv('DB_TIMEOUT', '30'))
    DB_DEBUG = os.getenv('DB_DEBUG', 'False').lower() == 'true'

    @classmethod
    def is_database(cls) -> bool:
        return cls.DEFAULT_SOURCE == 'database'

    @classmethod
    def is_csv(cls) -> bool:
        return cls.DEFAULT_SOURCE == 'csv'

    @classmethod
    def is_mysql(cls) -> bool:
        return cls.DB_TYPE == 'mysql'

    @classmethod
    def is_sqlserver(cls) -> bool:
        return cls.DB_TYPE == 'sqlserver'

    @classmethod
    def get_source_info(cls) -> dict:
        if cls.is_database():
            return {
                'type': f'Database ({cls.DB_TYPE})',
                'server': cls.DB_SERVER,
                'database': cls.DB_NAME,
                'table': cls.DB_TABLE,
                'user': cls.DB_USER,
            }
        else:
            return {
                'type': 'Excel (.xlsx)',
                'path': cls.CSV_FILE_PATH,
            }


# ===== PARAMÈTRES TEMPORELS =====
TRAINING_YEAR = 2026
DATE_FORMAT = "%Y-%m-%d"


# ===== PARAMÈTRES DE VALIDATION =====
class ValidationRules:
    """Règles de validation des données ATM."""

    # Montant minimum/maximum acceptable (euros)
    MIN_AMOUNT = 0
    MAX_AMOUNT = 500_000  # Montant max raisonnable pour un rechargement ATM

    # Années acceptables
    MIN_YEAR = 2020
    MAX_YEAR = 2030

    # Nombre minimum d'ATMs distincts attendus
    MIN_ATM_COUNT = 1


# ===== PARAMÈTRES D'AFFICHAGE =====
class DisplayConfig:
    """Configuration pour les graphiques et rapports."""

    FIGURE_SIZE = (12, 6)
    PRIMARY_COLOR = "#1f77b4"
    SECONDARY_COLOR = "#ff7f0e"
    PREVIEW_ROWS = 10
    DECIMAL_PLACES = 2


# ===== JOURS DE LA SEMAINE =====
WEEKDAY_NAMES = {
    0: "Lundi",
    1: "Mardi",
    2: "Mercredi",
    3: "Jeudi",
    4: "Vendredi",
    5: "Samedi",
    6: "Dimanche"
}

WEEKEND_DAYS = [5, 6]


# ===== MESSAGES ET LOGS =====
class Messages:
    """Messages standardisés pour les logs."""

    DATA_LOADED = "Données chargées avec succès"
    DATA_CLEANED = "Nettoyage des données terminé"
    DATA_ENRICHED = "Enrichissement des données terminé"
    VALIDATION_OK = "Validation des données réussie"

    ERROR_FILE_NOT_FOUND = "Fichier non trouvé"
    ERROR_INVALID_DATE = "Format de date invalide"
    ERROR_INVALID_AMOUNT = "Montant invalide détecté"
    ERROR_NO_DATA = "Aucune donnée trouvée"
    ERROR_DB_CONNECTION = "Erreur de connexion à la base de données"


# ===== FONCTIONS UTILITAIRES =====
def get_training_date_range():
    """
    Retourne la plage de dates pour l'entraînement (année 2026 complète).

    Returns:
        tuple: (date_debut, date_fin)
    """
    start_date = datetime(TRAINING_YEAR, 1, 1)
    end_date = datetime(TRAINING_YEAR, 12, 31)
    return start_date, end_date


def get_file_path(file_type: str) -> Path:
    """
    Retourne le chemin complet d'un fichier selon son type.

    Args:
        file_type: Type de fichier ('raw', 'clean', 'enriched', 'output')

    Returns:
        Path: Chemin complet du fichier
    """
    file_mapping = {
        'raw': DATA_RAW_DIR / RAW_DATA_FILE,
        'clean': DATA_PROCESSED_DIR / CLEAN_DATA_FILE,
        'enriched': DATA_PROCESSED_DIR / ENRICHED_DATA_FILE,
        'output': DATA_OUTPUT_DIR / PREDICTIONS_FILE,
    }
    return file_mapping.get(file_type, DATA_RAW_DIR / RAW_DATA_FILE)


def print_data_source_info():
    """Affiche les informations sur la source de données configurée."""
    info = DataSourceConfig.get_source_info()

    print("=" * 60)
    print("CONFIGURATION DE LA SOURCE DE DONNÉES")
    print("=" * 60)

    for key, value in info.items():
        print(f"  {key}: {value}")

    print("=" * 60)


if __name__ == "__main__":
    print("Test de la configuration")
    print_data_source_info()

    print("\nChemins des fichiers :")
    for file_type in ['raw', 'clean', 'enriched', 'output']:
        print(f"  {file_type}: {get_file_path(file_type)}")

    print(f"\nColonnes à charger : {len(COLUMNS_TO_LOAD)}")
    print(f"Mapping : {len(COLUMN_MAPPING)} colonnes")
    print("\nConfiguration chargée avec succès")
