"""
Module de connexion à SQL Server pour le projet de prévision de commandes.

Ce module gère :
- La connexion sécurisée à SQL Server via pyodbc
- Le chargement des credentials depuis .env
- L'exécution de requêtes SQL
- La conversion automatique en DataFrame pandas

Usage:
    from src.database_connector import SQLServerConnector

    connector = SQLServerConnector()
    df = connector.fetch_commandes_data()
"""

import pyodbc
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import os

# Pour charger les variables d'environnement depuis .env
try:
    from dotenv import load_dotenv
    load_dotenv()  # Charge automatiquement le fichier .env
except ImportError:
    logging.warning("python-dotenv non installé. Installez-le avec: pip install python-dotenv")

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SQLServerConnector:
    """
    Connecteur pour SQL Server avec gestion automatique des connexions.

    Charge les credentials depuis les variables d'environnement (.env).
    """

    def __init__(
        self,
        server: Optional[str] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        driver: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialise le connecteur SQL Server.

        Args:
            server: Adresse du serveur SQL (par défaut depuis .env)
            database: Nom de la base de données (par défaut depuis .env)
            username: Nom d'utilisateur (par défaut depuis .env)
            password: Mot de passe (par défaut depuis .env)
            driver: Driver ODBC (par défaut depuis .env)
            timeout: Timeout de connexion en secondes
        """
        # Charger depuis .env si non fourni
        self.server = server or os.getenv('DB_SERVER', '10.147.18.196')
        self.database = database or os.getenv('DB_NAME', 'Python')
        self.username = username or os.getenv('DB_USER', 'sa')
        self.password = password or os.getenv('DB_PASSWORD', '')
        self.driver = driver or os.getenv('DB_DRIVER', 'ODBC Driver 17 for SQL Server')
        self.timeout = int(os.getenv('DB_TIMEOUT', str(timeout)))
        self.debug = os.getenv('DB_DEBUG', 'False').lower() == 'true'

        self.connection = None
        self.cursor = None

        # Validation
        if not self.password:
            logger.warning("⚠️  Mot de passe vide ! Vérifiez votre fichier .env")

    def get_connection_string(self) -> str:
        """
        Construit la chaîne de connexion ODBC.

        Returns:
            str: Chaîne de connexion formatée
        """
        conn_string = (
            f"DRIVER={{{self.driver}}};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            f"UID={self.username};"
            f"PWD={self.password};"
            f"Connection Timeout={self.timeout};"
        )

        if self.debug:
            # Afficher la chaîne sans le mot de passe
            safe_string = conn_string.replace(f"PWD={self.password}", "PWD=*****")
            logger.debug(f"Connection string: {safe_string}")

        return conn_string

    def connect(self) -> bool:
        """
        Établit la connexion à SQL Server.

        Returns:
            bool: True si connexion réussie, False sinon
        """
        try:
            logger.info(f"🔌 Connexion à SQL Server : {self.server} / {self.database}")

            conn_string = self.get_connection_string()
            self.connection = pyodbc.connect(conn_string)
            self.cursor = self.connection.cursor()

            logger.info("✅ Connexion SQL Server établie avec succès")

            # Test de la connexion
            self.cursor.execute("SELECT @@VERSION")
            version = self.cursor.fetchone()[0]
            if self.debug:
                logger.debug(f"SQL Server Version: {version[:50]}...")

            return True

        except pyodbc.Error as e:
            logger.error(f"❌ Erreur de connexion SQL Server : {e}")
            self._suggest_solutions(e)
            return False
        except Exception as e:
            logger.error(f"❌ Erreur inattendue : {e}")
            return False

    def _suggest_solutions(self, error: Exception) -> None:
        """
        Suggère des solutions selon le type d'erreur.

        Args:
            error: Exception capturée
        """
        error_msg = str(error).lower()

        suggestions = []

        if "driver" in error_msg or "odbc" in error_msg:
            suggestions.append("📦 Installez le driver ODBC pour SQL Server :")
            suggestions.append("   Windows : https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server")
            suggestions.append("   Linux : sudo apt-get install unixodbc-dev")

        if "login failed" in error_msg or "authentication" in error_msg:
            suggestions.append("🔑 Vérifiez vos credentials dans le fichier .env")
            suggestions.append("   - DB_USER correct ?")
            suggestions.append("   - DB_PASSWORD correct ?")

        if "network" in error_msg or "timeout" in error_msg or "cannot open" in error_msg:
            suggestions.append("🌐 Vérifiez la connectivité réseau :")
            suggestions.append(f"   - Le serveur {self.server} est-il accessible ?")
            suggestions.append("   - Pas de firewall bloquant ?")
            suggestions.append("   - SQL Server accepte-t-il les connexions TCP/IP ?")

        if suggestions:
            logger.info("\n💡 SUGGESTIONS :")
            for suggestion in suggestions:
                logger.info(suggestion)

    def disconnect(self) -> None:
        """Ferme proprement la connexion."""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            logger.info("🔌 Connexion SQL Server fermée")
        except Exception as e:
            logger.warning(f"⚠️  Erreur lors de la fermeture : {e}")

    def execute_query(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """
        Exécute une requête SQL et retourne un DataFrame.

        Args:
            query: Requête SQL à exécuter.
            params: Paramètres de la requête (optionnel).

        Returns:
            DataFrame: Résultats de la requête.
        """

        if not self.connection:
            if not self.connect():
                raise ConnectionError("Impossible de se connecter à SQL Server")

        try:
            if self.debug:
                logger.debug(f"Exécution de la requête : {query[:100]}...")

            # Essayez d'utiliser SQLAlchemy en priorité (recommandé)
            try:
                from sqlalchemy import create_engine
                from urllib.parse import quote_plus

                conn_str = (
                    f"mssql+pyodbc://{self.username}:{quote_plus(self.password)}"
                    f"@{self.server}/{self.database}?driver={quote_plus(self.driver)}"
                )

                engine = create_engine(conn_str)
                try:
                    df = pd.read_sql(query, con=engine, params=params)
                finally:
                    engine.dispose()

            except ImportError:
                # Fallback sur pyodbc (avec suppression du warning inutile)
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*SQLAlchemy.*")
                    df = pd.read_sql(query, con=self.connection, params=params)

            logger.info(f"✅ Requête exécutée : {len(df)} lignes récupérées")
            return df

        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exécution de la requête : {e}")
            raise

    def fetch_commandes_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        article_ids: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Récupère les données de commandes depuis la table dbo.ligne_commande.

        Args:
            start_date: Date de début (optionnel)
            end_date: Date de fin (optionnel)
            article_ids: Liste d'IDs d'articles à filtrer (optionnel)

        Returns:
            DataFrame: Données de commandes
        """
        logger.info("📊 Récupération des données de commandes depuis SQL Server...")

        # Construction de la requête de base
        query = """
        SELECT 
            [date_ligne_commande],
            [id_article],
            [ref_article],
            [quantite]
        FROM [dbo].[ligne_commande]
        WHERE 1=1
        """

        # Filtres conditionnels
        params = []

        if start_date:
            query += " AND [date_ligne_commande] >= ?"
            params.append(start_date)
            logger.info(f"   📅 Filtré depuis : {start_date.date()}")

        if end_date:
            query += " AND [date_ligne_commande] <= ?"
            params.append(end_date)
            logger.info(f"   📅 Filtré jusqu'à : {end_date.date()}")

        if article_ids:
            placeholders = ','.join('?' * len(article_ids))
            query += f" AND [id_article] IN ({placeholders})"
            params.extend(article_ids)
            logger.info(f"   📦 Filtré sur {len(article_ids)} articles")

        # Tri par date et article
        query += " ORDER BY [date_ligne_commande], [id_article]"

        # Exécution
        df = self.execute_query(query, tuple(params) if params else None)

        # Statistiques
        logger.info(f"   📊 Lignes récupérées : {len(df)}")
        if not df.empty:
            logger.info(f"   📅 Période : {df['date_ligne_commande'].min()} à {df['date_ligne_commande'].max()}")
            logger.info(f"   📦 Articles uniques : {df['id_article'].nunique()}")

        return df

    def test_connection(self) -> Dict[str, Any]:
        """
        Teste la connexion et retourne des informations de diagnostic.

        Returns:
            dict: Informations de diagnostic
        """
        result = {
            'connected': False,
            'server': self.server,
            'database': self.database,
            'username': self.username,
            'driver': self.driver,
            'errors': []
        }

        # Vérifier si une connexion existe déjà
        connection_was_open = self.connection is not None

        try:
            # Ouvrir une connexion seulement si nécessaire
            if not connection_was_open:
                if not self.connect():
                    result['errors'].append("Impossible d'établir la connexion")
                    return result

            result['connected'] = True

            # Informations sur la base
            self.cursor.execute("SELECT COUNT(*) FROM [dbo].[ligne_commande]")
            result['total_rows'] = self.cursor.fetchone()[0]

            self.cursor.execute("""
                SELECT 
                    MIN([date_ligne_commande]) as min_date,
                    MAX([date_ligne_commande]) as max_date,
                    COUNT(DISTINCT [id_article]) as unique_articles
                FROM [dbo].[ligne_commande]
            """)
            row = self.cursor.fetchone()
            result['date_range'] = {
                'min': row[0],
                'max': row[1]
            }
            result['unique_articles'] = row[2]

        except Exception as e:
            result['errors'].append(str(e))
        finally:
            # Fermer seulement si c'est nous qui avons ouvert la connexion
            if not connection_was_open:
                self.disconnect()

        return result

    def __enter__(self):
        """Support du context manager (with statement)."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Fermeture automatique avec context manager."""
        self.disconnect()


# ===== FONCTION UTILITAIRE =====

def quick_fetch_commandes(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Fonction rapide pour récupérer les données de commandes.

    Args:
        start_date: Date de début (optionnel)
        end_date: Date de fin (optionnel)

    Returns:
        DataFrame: Données de commandes

    Example:
        >>> from src.database_connector import quick_fetch_commandes
        >>> df = quick_fetch_commandes(start_date=datetime(2024, 1, 1))
    """
    with SQLServerConnector() as connector:
        return connector.fetch_commandes_data(start_date, end_date)


# ===== TEST DU MODULE =====

if __name__ == "__main__":
    """Test du connecteur si exécuté directement."""

    print("="*60)
    print("🧪 TEST DU CONNECTEUR SQL SERVER")
    print("="*60)

    # Test 1 : Connexion
    print("\n1️⃣ Test de connexion...")
    connector = SQLServerConnector()

    if connector.connect():
        print("   ✅ Connexion réussie !")

        # Test 2 : Informations de diagnostic
        print("\n2️⃣ Informations de la base...")
        diag = connector.test_connection()

        if diag['connected']:
            print(f"   📊 Nombre total de lignes : {diag.get('total_rows', 'N/A')}")
            print(f"   📅 Période : {diag['date_range']['min']} à {diag['date_range']['max']}")
            print(f"   📦 Articles uniques : {diag.get('unique_articles', 'N/A')}")

        # Test 3 : Récupération des données
        print("\n3️⃣ Récupération d'un échantillon...")
        try:
            df = connector.fetch_commandes_data()
            print(f"   ✅ {len(df)} lignes récupérées")
            print("\n   📋 Aperçu des 5 premières lignes :")
            print(df.head().to_string(index=False))
        except Exception as e:
            print(f"   ❌ Erreur : {e}")

        connector.disconnect()
    else:
        print("   ❌ Connexion échouée")
        print("\n💡 Vérifiez :")
        print("   1. Le fichier .env existe avec les bonnes credentials")
        print("   2. Le driver ODBC est installé")
        print("   3. Le serveur SQL Server est accessible")

    print("\n" + "="*60)
    print("🎉 Tests terminés")
    print("="*60)