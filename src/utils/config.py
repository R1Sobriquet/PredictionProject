"""
Configuration centrale du projet de prévision de commandes.
Ce fichier centralise tous les paramètres pour faciliter la maintenance.

NOUVEAU : Support de SQL Server pour charger les données directement depuis la base.
"""

from datetime import datetime
from pathlib import Path
from typing import List
import os

# Charger les variables d'environnement depuis .env
try:
    from dotenv import load_dotenv
    load_dotenv()  # Charge automatiquement le fichier .env
except ImportError:
    pass  # python-dotenv optionnel

# ===== CHEMINS DE FICHIERS =====
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
DOCS_DIR = PROJECT_ROOT / "docs"

# Création automatique des dossiers s'ils n'existent pas
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_OUTPUT_DIR, DOCS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ===== NOMS DE FICHIERS =====
RAW_DATA_FILE = "commandes_2024.csv"
CLEAN_DATA_FILE = "commandes_clean.csv"
ENRICHED_DATA_FILE = "commandes_enriched.csv"
PREDICTIONS_FILE = "predictions_2025.csv"

# ===== CONFIGURATION SOURCE DE DONNÉES =====
class DataSourceConfig:
    """Configuration de la source de données (CSV ou SQL Server)"""

    # Source par défaut (peut être surchargée par .env)
    # Valeurs possibles : 'csv' ou 'sqlserver'
    DEFAULT_SOURCE = os.getenv('DATA_SOURCE', 'csv').lower()

    # Pour CSV
    CSV_FILE_PATH = os.getenv('CSV_FILE_PATH', str(DATA_RAW_DIR / RAW_DATA_FILE))

    # Pour SQL Server
    DB_SERVER = os.getenv('DB_SERVER', '10.147.18.196')
    DB_NAME = os.getenv('DB_NAME', 'Python')
    DB_USER = os.getenv('DB_USER', 'sa')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_DRIVER = os.getenv('DB_DRIVER', 'ODBC Driver 17 for SQL Server')
    DB_TIMEOUT = int(os.getenv('DB_TIMEOUT', '30'))
    DB_DEBUG = os.getenv('DB_DEBUG', 'False').lower() == 'true'

    # Table SQL Server
    DB_TABLE = 'dbo.ligne_commande'

    @classmethod
    def is_sqlserver(cls) -> bool:
        """Vérifie si on utilise SQL Server comme source."""
        return cls.DEFAULT_SOURCE == 'sqlserver'

    @classmethod
    def is_csv(cls) -> bool:
        """Vérifie si on utilise CSV comme source."""
        return cls.DEFAULT_SOURCE == 'csv'

    @classmethod
    def get_source_info(cls) -> dict:
        """Retourne les infos sur la source de données configurée."""
        if cls.is_sqlserver():
            return {
                'type': 'SQL Server',
                'server': cls.DB_SERVER,
                'database': cls.DB_NAME,
                'table': cls.DB_TABLE,
                'user': cls.DB_USER
            }
        else:
            return {
                'type': 'CSV',
                'path': cls.CSV_FILE_PATH
            }

# ===== PARAMÈTRES TEMPORELS =====
# Année de référence pour l'entraînement
TRAINING_YEAR = 2024

# Mois exclus de l'entraînement (décembre pour la validation)
EXCLUDED_MONTHS = [12]  # Décembre

# Année cible pour les prédictions
PREDICTION_YEAR = 2025

# Format de date dans les fichiers source
DATE_FORMAT = "%Y-%m-%d"


# ===== COLONNES DE LA TABLE SOURCE =====
class ColumnNames:
    """Noms des colonnes dans la table source et leurs équivalents standardisés"""

    # Colonnes source SQL Server (table dbo.ligne_commande)
    SOURCE_DATE = "date_ligne_commande"
    SOURCE_ARTICLE_ID = "id_article"
    SOURCE_ARTICLE_REF = "ref_article"
    SOURCE_QUANTITY = "quantite"

    # Colonnes additionnelles dans SQL Server (non utilisées pour l'instant)
    SOURCE_CLIENT_ID = "id_client"
    SOURCE_MONTANT_UNITAIRE = "montant_unitaire"
    SOURCE_TVA = "tva"
    SOURCE_TOTAL_HT = "total_ht"
    SOURCE_TOTAL_TTC = "total_ttc"

    # Colonnes standardisées (utilisées dans le code)
    DATE = "date"
    ARTICLE_ID = "article_id"
    ARTICLE_REF = "article_ref"
    QUANTITY = "quantity"

    # Colonnes enrichies (ajoutées par nos traitements)
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    WEEKDAY = "weekday"  # 0=Lundi, 6=Dimanche
    WEEKDAY_NAME = "weekday_name"
    IS_WEEKEND = "is_weekend"
    WEEK_NUMBER = "week_number"
    QUANTITY_PREV_DAY = "quantity_prev_day"


# ===== PARAMÈTRES DE VALIDATION =====
class ValidationRules:
    """Règles de validation des données"""

    # Quantité minimum/maximum acceptable
    MIN_QUANTITY = 0
    MAX_QUANTITY = 10000  # À ajuster selon votre métier

    # Années acceptables
    MIN_YEAR = 2020
    MAX_YEAR = 2030

    # Nombre minimum d'articles distincts attendus
    MIN_ARTICLES_COUNT = 1


# ===== PARAMÈTRES D'AFFICHAGE =====
class DisplayConfig:
    """Configuration pour les graphiques et rapports"""

    # Taille des graphiques
    FIGURE_SIZE = (12, 6)

    # Couleurs pour les graphiques
    PRIMARY_COLOR = "#1f77b4"
    SECONDARY_COLOR = "#ff7f0e"

    # Nombre de lignes à afficher dans les aperçus
    PREVIEW_ROWS = 10

    # Format d'affichage des nombres
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

WEEKEND_DAYS = [5, 6]  # Samedi, Dimanche


# ===== MESSAGES ET LOGS =====
class Messages:
    """Messages standardisés pour les logs et retours utilisateur"""

    DATA_LOADED = "✅ Données chargées avec succès"
    DATA_CLEANED = "✅ Nettoyage des données terminé"
    DATA_ENRICHED = "✅ Enrichissement des données terminé"
    VALIDATION_OK = "✅ Validation des données réussie"

    ERROR_FILE_NOT_FOUND = "❌ Fichier non trouvé"
    ERROR_INVALID_DATE = "❌ Format de date invalide"
    ERROR_INVALID_QUANTITY = "❌ Quantité invalide détectée"
    ERROR_NO_DATA = "❌ Aucune donnée trouvée"
    ERROR_DB_CONNECTION = "❌ Erreur de connexion à la base de données"


# ===== FONCTION UTILITAIRE =====
def get_training_date_range():
    """
    Retourne la plage de dates pour l'entraînement (2024 sans décembre)

    Returns:
        tuple: (date_debut, date_fin)
    """
    start_date = datetime(TRAINING_YEAR, 1, 1)
    end_date = datetime(TRAINING_YEAR, 11, 30)  # 30 novembre
    return start_date, end_date


def get_file_path(file_type: str) -> Path:
    """
    Retourne le chemin complet d'un fichier selon son type

    Args:
        file_type: Type de fichier ('raw', 'clean', 'enriched', 'output')

    Returns:
        Path: Chemin complet du fichier
    """
    file_mapping = {
        'raw': DATA_RAW_DIR / RAW_DATA_FILE,
        'clean': DATA_PROCESSED_DIR / CLEAN_DATA_FILE,
        'enriched': DATA_PROCESSED_DIR / ENRICHED_DATA_FILE,
        'output': DATA_OUTPUT_DIR / PREDICTIONS_FILE
    }

    return file_mapping.get(file_type, DATA_RAW_DIR / RAW_DATA_FILE)


def print_data_source_info():
    """Affiche les informations sur la source de données configurée."""
    info = DataSourceConfig.get_source_info()

    print("=" * 60)
    print("📊 CONFIGURATION DE LA SOURCE DE DONNÉES")
    print("=" * 60)

    if info['type'] == 'SQL Server':
        print(f"Type : {info['type']}")
        print(f"Serveur : {info['server']}")
        print(f"Base : {info['database']}")
        print(f"Table : {info['table']}")
        print(f"Utilisateur : {info['user']}")
    else:
        print(f"Type : {info['type']}")
        print(f"Chemin : {info['path']}")

    print("=" * 60)


# Test du module si exécuté directement
if __name__ == "__main__":
    print("🧪 Test de la configuration")
    print_data_source_info()

    print("\n📁 Chemins des fichiers :")
    for file_type in ['raw', 'clean', 'enriched', 'output']:
        print(f"   {file_type}: {get_file_path(file_type)}")

    print("\n✅ Configuration chargée avec succès")