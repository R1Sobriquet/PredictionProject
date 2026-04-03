"""
Module de connexion aux bases de données pour le projet de prévision ATM.

Supporte :
- MySQL (via pymysql + SQLAlchemy)
- SQL Server (via pyodbc)

Le type de base est configuré via DB_TYPE dans le .env.

Usage:
    from src.database_connector import DatabaseConnector

    with DatabaseConnector() as connector:
        df = connector.fetch_commandes_data()
"""

import pandas as pd
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatabaseConnector:
    """
    Connecteur universel pour MySQL et SQL Server.

    Charge les credentials depuis les variables d'environnement (.env).
    Le type de base est déterminé par DB_TYPE ('mysql' ou 'sqlserver').
    """

    def __init__(
        self,
        db_type: Optional[str] = None,
        server: Optional[str] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        table: Optional[str] = None,
        driver: Optional[str] = None,
        timeout: int = 30,
    ):
        self.db_type = (db_type or os.getenv('DB_TYPE', 'mysql')).lower()
        self.server = server or os.getenv('DB_SERVER', 'localhost')
        self.database = database or os.getenv('DB_NAME', 'prediction_db')
        self.username = username or os.getenv('DB_USER', 'root')
        self.password = password or os.getenv('DB_PASSWORD', '')
        self.table = table or os.getenv('DB_TABLE', 'd_CommandesDetailCalcul')
        self.driver = driver or os.getenv('DB_DRIVER', 'ODBC Driver 17 for SQL Server')
        self.timeout = int(os.getenv('DB_TIMEOUT', str(timeout)))
        self.debug = os.getenv('DB_DEBUG', 'False').lower() == 'true'

        self.connection = None
        self.engine = None

        if not self.password:
            logger.warning("Mot de passe vide ! Vérifiez votre fichier .env")

    @property
    def is_mysql(self) -> bool:
        return self.db_type == 'mysql'

    @property
    def is_sqlserver(self) -> bool:
        return self.db_type == 'sqlserver'

    def connect(self) -> bool:
        """Établit la connexion à la base de données."""
        try:
            logger.info(f"Connexion à {self.db_type} : {self.server} / {self.database}")

            if self.is_mysql:
                return self._connect_mysql()
            elif self.is_sqlserver:
                return self._connect_sqlserver()
            else:
                raise ValueError(f"Type de base non supporté : {self.db_type}. Utilisez 'mysql' ou 'sqlserver'.")

        except Exception as e:
            logger.error(f"Erreur de connexion {self.db_type} : {e}")
            self._suggest_solutions(e)
            return False

    def _connect_mysql(self) -> bool:
        """Connexion MySQL via pymysql."""
        try:
            import pymysql
            self.connection = pymysql.connect(
                host=self.server,
                user=self.username,
                password=self.password,
                database=self.database,
                connect_timeout=self.timeout,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor,
            )
            logger.info("Connexion MySQL établie avec succès")
            return True
        except ImportError:
            logger.error("pymysql non installé. Installez-le avec : pip install pymysql")
            return False

    def _connect_sqlserver(self) -> bool:
        """Connexion SQL Server via pyodbc."""
        import pyodbc
        conn_string = (
            f"DRIVER={{{self.driver}}};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            f"UID={self.username};"
            f"PWD={self.password};"
            f"Connection Timeout={self.timeout};"
        )

        self.connection = pyodbc.connect(conn_string)
        logger.info("Connexion SQL Server établie avec succès")

        if self.debug:
            cursor = self.connection.cursor()
            cursor.execute("SELECT @@VERSION")
            version = cursor.fetchone()[0]
            logger.debug(f"SQL Server Version: {version[:80]}...")
            cursor.close()

        return True

    def disconnect(self) -> None:
        """Ferme proprement la connexion."""
        try:
            if self.engine:
                self.engine.dispose()
                self.engine = None
            if self.connection:
                self.connection.close()
                self.connection = None
            logger.info(f"Connexion {self.db_type} fermée")
        except Exception as e:
            logger.warning(f"Erreur lors de la fermeture : {e}")

    def _suggest_solutions(self, error: Exception) -> None:
        """Suggère des solutions selon le type d'erreur."""
        error_msg = str(error).lower()
        suggestions = []

        if self.is_sqlserver and ("driver" in error_msg or "odbc" in error_msg):
            suggestions.append("Installez le driver ODBC pour SQL Server")
            suggestions.append("  Linux : sudo apt-get install unixodbc-dev")
        elif self.is_mysql and "pymysql" in error_msg:
            suggestions.append("Installez pymysql : pip install pymysql")

        if "login failed" in error_msg or "access denied" in error_msg:
            suggestions.append("Vérifiez vos credentials dans le fichier .env (DB_USER, DB_PASSWORD)")

        if "network" in error_msg or "timeout" in error_msg or "cannot" in error_msg:
            suggestions.append(f"Le serveur {self.server} est-il accessible ?")
            suggestions.append("Vérifiez qu'il n'y a pas de firewall bloquant")

        if suggestions:
            logger.info("SUGGESTIONS :")
            for s in suggestions:
                logger.info(f"  {s}")

    def execute_query(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """
        Exécute une requête SQL et retourne un DataFrame.

        Args:
            query: Requête SQL
            params: Paramètres de la requête (optionnel)

        Returns:
            DataFrame: Résultats de la requête
        """
        if not self.connection:
            if not self.connect():
                raise ConnectionError(f"Impossible de se connecter à {self.db_type}")

        try:
            if self.debug:
                logger.debug(f"Requête : {query[:100]}...")

            if self.is_mysql:
                return self._execute_mysql(query, params)
            else:
                return self._execute_sqlserver(query, params)

        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de la requête : {e}")
            raise

    def _execute_mysql(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """Exécution de requête MySQL."""
        try:
            from sqlalchemy import create_engine
            from urllib.parse import quote_plus

            if not self.engine:
                conn_str = (
                    f"mysql+pymysql://{self.username}:{quote_plus(self.password)}"
                    f"@{self.server}/{self.database}?charset=utf8mb4"
                )
                self.engine = create_engine(conn_str)

            df = pd.read_sql(query, con=self.engine, params=params)
        except ImportError:
            # Fallback sans SQLAlchemy
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                df = pd.read_sql(query, con=self.connection, params=params)

        logger.info(f"Requête exécutée : {len(df)} lignes récupérées")
        return df

    def _execute_sqlserver(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """Exécution de requête SQL Server."""
        try:
            from sqlalchemy import create_engine
            from urllib.parse import quote_plus

            if not self.engine:
                conn_str = (
                    f"mssql+pyodbc://{self.username}:{quote_plus(self.password)}"
                    f"@{self.server}/{self.database}?driver={quote_plus(self.driver)}"
                )
                self.engine = create_engine(conn_str)

            df = pd.read_sql(query, con=self.engine, params=params)
        except ImportError:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                df = pd.read_sql(query, con=self.connection, params=params)

        logger.info(f"Requête exécutée : {len(df)} lignes récupérées")
        return df

    def fetch_commandes_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        atm_ids: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Récupère les données de commandes ATM depuis la base de données.

        Charge uniquement les colonnes pertinentes (exclut DC_Predictif_*).
        Filtre automatiquement les commandes annulées (DC_Annule = 1).

        Args:
            start_date: Date de début (optionnel)
            end_date: Date de fin (optionnel)
            atm_ids: Liste d'IDs d'ATMs à filtrer (optionnel)

        Returns:
            DataFrame: Données de commandes ATM
        """
        logger.info("Récupération des données de commandes ATM...")

        # Construction de la liste des colonnes SQL
        from src.utils.config import COLUMNS_TO_LOAD
        columns_sql = ", ".join(f"[{col}]" for col in COLUMNS_TO_LOAD)

        # Requête de base avec filtre sur les annulations
        query = f"""
        SELECT {columns_sql}
        FROM [{self.table}]
        WHERE [DC_Annule] != 1
        """

        params = []

        if start_date:
            if self.is_mysql:
                query += " AND [DC_Date_Cmd] >= %s"
            else:
                query += " AND [DC_Date_Cmd] >= ?"
            params.append(start_date)
            logger.info(f"  Filtré depuis : {start_date.date()}")

        if end_date:
            if self.is_mysql:
                query += " AND [DC_Date_Cmd] <= %s"
            else:
                query += " AND [DC_Date_Cmd] <= ?"
            params.append(end_date)
            logger.info(f"  Filtré jusqu'à : {end_date.date()}")

        if atm_ids:
            if self.is_mysql:
                placeholders = ','.join(['%s'] * len(atm_ids))
            else:
                placeholders = ','.join(['?'] * len(atm_ids))
            query += f" AND [DC_Automate_Id] IN ({placeholders})"
            params.extend(atm_ids)
            logger.info(f"  Filtré sur {len(atm_ids)} ATMs")

        query += " ORDER BY [DC_Date_Cmd], [DC_Automate_Id]"

        df = self.execute_query(query, tuple(params) if params else None)

        if not df.empty:
            logger.info(f"  Lignes récupérées : {len(df)}")
            logger.info(f"  Période : {df['DC_Date_Cmd'].min()} à {df['DC_Date_Cmd'].max()}")
            logger.info(f"  ATMs uniques : {df['DC_Automate_Id'].nunique()}")

        return df

    def test_connection(self) -> Dict[str, Any]:
        """Teste la connexion et retourne des informations de diagnostic."""
        result = {
            'connected': False,
            'db_type': self.db_type,
            'server': self.server,
            'database': self.database,
            'table': self.table,
            'errors': [],
        }

        connection_was_open = self.connection is not None

        try:
            if not connection_was_open:
                if not self.connect():
                    result['errors'].append("Impossible d'établir la connexion")
                    return result

            result['connected'] = True

            if self.is_mysql:
                count_query = f"SELECT COUNT(*) as cnt FROM `{self.table}` WHERE `DC_Annule` != 1"
                range_query = f"""
                    SELECT MIN(`DC_Date_Cmd`) as min_date, MAX(`DC_Date_Cmd`) as max_date,
                           COUNT(DISTINCT `DC_Automate_Id`) as unique_atms
                    FROM `{self.table}` WHERE `DC_Annule` != 1
                """
            else:
                count_query = f"SELECT COUNT(*) as cnt FROM [{self.table}] WHERE [DC_Annule] != 1"
                range_query = f"""
                    SELECT MIN([DC_Date_Cmd]) as min_date, MAX([DC_Date_Cmd]) as max_date,
                           COUNT(DISTINCT [DC_Automate_Id]) as unique_atms
                    FROM [{self.table}] WHERE [DC_Annule] != 1
                """

            count_df = self.execute_query(count_query)
            result['total_rows'] = int(count_df.iloc[0]['cnt'])

            range_df = self.execute_query(range_query)
            result['date_range'] = {
                'min': range_df.iloc[0]['min_date'],
                'max': range_df.iloc[0]['max_date'],
            }
            result['unique_atms'] = int(range_df.iloc[0]['unique_atms'])

        except Exception as e:
            result['errors'].append(str(e))
        finally:
            if not connection_was_open:
                self.disconnect()

        return result

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# ===== FONCTION UTILITAIRE =====

def quick_fetch_commandes(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fonction rapide pour récupérer les données de commandes ATM.

    Args:
        start_date: Date de début (optionnel)
        end_date: Date de fin (optionnel)

    Returns:
        DataFrame: Données de commandes
    """
    with DatabaseConnector() as connector:
        return connector.fetch_commandes_data(start_date, end_date)


if __name__ == "__main__":
    print("=" * 60)
    print("TEST DU CONNECTEUR BASE DE DONNÉES")
    print("=" * 60)

    connector = DatabaseConnector()
    print(f"\nType : {connector.db_type}")
    print(f"Serveur : {connector.server}")
    print(f"Base : {connector.database}")
    print(f"Table : {connector.table}")

    print("\n1. Test de connexion...")
    if connector.connect():
        print("  Connexion réussie !")

        print("\n2. Informations de diagnostic...")
        diag = connector.test_connection()
        if diag['connected']:
            print(f"  Lignes : {diag.get('total_rows', 'N/A')}")
            print(f"  Période : {diag.get('date_range', {}).get('min')} à {diag.get('date_range', {}).get('max')}")
            print(f"  ATMs uniques : {diag.get('unique_atms', 'N/A')}")

        print("\n3. Récupération d'un échantillon...")
        try:
            df = connector.fetch_commandes_data()
            print(f"  {len(df)} lignes récupérées")
            print(f"\n  Aperçu :\n{df.head().to_string(index=False)}")
        except Exception as e:
            print(f"  Erreur : {e}")

        connector.disconnect()
    else:
        print("  Connexion échouée. Vérifiez votre .env")

    print("\n" + "=" * 60)
