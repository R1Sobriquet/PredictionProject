"""
Script pour rafraîchir le fichier CSV depuis SQL Server.

Ce script :
1. Se connecte à SQL Server
2. Exporte les données dans data/raw/commandes_2024.csv
3. Le fichier CSV est maintenant à jour

Usage:
    python refresh_csv_from_sql.py

Ou automatiquement avant le pipeline :
    python refresh_csv_from_sql.py && python main.py --step all
"""

import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

# Ajouter le répertoire racine au path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.database_connector import SQLServerConnector
from src.utils.config import ColumnNames, get_file_path


def export_sql_to_csv():
    """
    Exporte les données de SQL Server vers le fichier CSV.
    """
    print("=" * 70)
    print("🔄 RAFRAÎCHISSEMENT DU CSV DEPUIS SQL SERVER")
    print("=" * 70)

    # 1. Connexion à SQL Server
    print("\n🔌 Connexion à SQL Server...")
    connector = SQLServerConnector()

    if not connector.connect():
        print("❌ Impossible de se connecter à SQL Server")
        print("\n💡 Vérifications :")
        print("   1. Le fichier .env contient les bons credentials")
        print("   2. Le serveur SQL est accessible")
        print("   3. Le driver ODBC est installé")
        sys.exit(1)

    print("   ✅ Connexion réussie")

    # 2. Récupération des données
    print("\n📊 Récupération des données...")
    try:
        df = connector.fetch_commandes_data(
            start_date=None,  # Toutes les dates
            end_date=None
        )

        if df.empty:
            print("❌ Aucune donnée récupérée")
            connector.disconnect()
            sys.exit(1)

        print(f"   ✅ {len(df)} lignes récupérées")
        print(f"   📅 Période : {df['date_ligne_commande'].min()} à {df['date_ligne_commande'].max()}")
        print(f"   📦 Articles uniques : {df['id_article'].nunique()}")

    except Exception as e:
        print(f"❌ Erreur lors de la récupération : {e}")
        connector.disconnect()
        sys.exit(1)

    # 3. Sauvegarde dans le fichier CSV
    print("\n💾 Sauvegarde dans le fichier CSV...")

    output_file = get_file_path('raw')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Sauvegarder avec le bon format
    df.to_csv(output_file, index=False, date_format='%Y-%m-%d')

    print(f"   ✅ Fichier sauvegardé : {output_file}")
    print(f"   📊 {len(df)} lignes exportées")

    # 4. Vérification du fichier
    print("\n🔍 Vérification du fichier...")

    # Charger le fichier pour vérifier
    verification_df = pd.read_csv(output_file, nrows=5)

    print(f"   ✅ Fichier lisible")
    print(f"   📋 Colonnes : {list(verification_df.columns)}")
    print("\n   📋 Aperçu (5 premières lignes) :")
    print(verification_df.to_string(index=False))

    # 5. Créer une copie de backup avec timestamp
    print("\n💾 Création d'une copie de backup...")

    backup_dir = Path("../../data/backups")
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"commandes_2024_backup_{timestamp}.csv"

    df.to_csv(backup_file, index=False, date_format='%Y-%m-%d')
    print(f"   ✅ Backup créé : {backup_file}")

    # 6. Fermeture de la connexion
    connector.disconnect()

    # 7. Résumé
    print("\n" + "=" * 70)
    print("✅ RAFRAÎCHISSEMENT TERMINÉ AVEC SUCCÈS")
    print("=" * 70)
    print(f"\n📄 Fichier mis à jour : {output_file}")
    print(f"📊 {len(df)} lignes | {df['id_article'].nunique()} articles")
    print(f"📅 Du {df['date_ligne_commande'].min()} au {df['date_ligne_commande'].max()}")
    print(f"\n💾 Backup disponible : {backup_file}")
    print("\n💡 Vous pouvez maintenant lancer le pipeline en mode CSV :")
    print("   python main.py --step all")
    print("\n   Ou configurer DATA_SOURCE=sqlserver dans .env pour")
    print("   toujours avoir les données à jour automatiquement.")


if __name__ == "__main__":
    try:
        export_sql_to_csv()
    except KeyboardInterrupt:
        print("\n⚠️  Export interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue : {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)