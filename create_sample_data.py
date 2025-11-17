"""
Script pour créer automatiquement des données d'exemple réalistes.

Usage:
    python create_sample_data.py

Génère un fichier commandes_2024.csv avec des données de test complètes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import random


def generate_realistic_sample_data():
    """
    Génère des données d'exemple réalistes pour tester le pipeline.

    Caractéristiques :
    - 5 articles différents
    - Période : janvier à novembre 2024
    - Saisonnalité : plus de commandes en semaine qu'en weekend
    - Tendances : certains articles croissants, d'autres stables
    - Quelques jours sans commandes (pour tester les 0)
    """

    print("🎯 Génération de données d'exemple réalistes...")

    # Configuration
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 11, 30)  # Pas décembre comme requis

    articles = [
        {'id': 1, 'ref': 'REF001', 'base_qty': 50, 'trend': 0.1, 'weekend_factor': 0.7},
        {'id': 2, 'ref': 'REF002', 'base_qty': 30, 'trend': 0.05, 'weekend_factor': 0.8},
        {'id': 3, 'ref': 'REF003', 'base_qty': 40, 'trend': -0.02, 'weekend_factor': 0.6},
        {'id': 4, 'ref': 'REF004', 'base_qty': 25, 'trend': 0.15, 'weekend_factor': 0.9},
        {'id': 5, 'ref': 'REF005', 'base_qty': 35, 'trend': 0.0, 'weekend_factor': 0.5}
    ]

    # Génération des dates
    current_date = start_date
    all_data = []

    day_count = 0

    while current_date <= end_date:
        # Facteur saisonnier (plus fort en fin d'année)
        month_factor = 1.0 + (current_date.month - 1) * 0.05

        # Facteur jour de semaine (0=lundi, 6=dimanche)
        weekday = current_date.weekday()
        is_weekend = weekday in [5, 6]  # samedi, dimanche

        for article in articles:
            # Calcul de la quantité de base avec tendance
            days_since_start = (current_date - start_date).days
            trend_factor = 1.0 + (article['trend'] * days_since_start / 365)

            base_qty = article['base_qty'] * month_factor * trend_factor

            # Ajustement weekend
            if is_weekend:
                base_qty *= article['weekend_factor']

            # Ajout de bruit aléatoire
            noise = np.random.normal(0, base_qty * 0.2)
            final_qty = max(0, int(base_qty + noise))

            # Parfois pas de commande (5% de chance)
            if random.random() < 0.05:
                final_qty = 0

            # Ajouter les données
            if final_qty > 0 or random.random() < 0.7:  # Garder quelques zéros
                all_data.append({
                    'date_ligne_commande': current_date.strftime('%Y-%m-%d'),
                    'id_article': article['id'],
                    'quantite': final_qty,
                    'ref_article': article['ref']
                })

        current_date += timedelta(days=1)
        day_count += 1

        # Affichage du progrès
        if day_count % 30 == 0:
            print(f"   📅 {day_count} jours générés...")

    # Création du DataFrame
    df = pd.DataFrame(all_data)

    # Ajout de quelques doublons pour tester le nettoyage
    duplicate_rows = df.sample(n=10).copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    # Mélange des lignes
    df = df.sample(frac=1).reset_index(drop=True)

    print(f"✅ {len(df)} lignes générées")
    print(f"   📊 Période : {df['date_ligne_commande'].min()} à {df['date_ligne_commande'].max()}")
    print(f"   📦 Articles : {sorted(df['id_article'].unique())}")
    print(f"   📈 Lignes avec quantité > 0 : {(df['quantite'] > 0).sum()}")
    print(f"   📉 Lignes avec quantité = 0 : {(df['quantite'] == 0).sum()}")

    return df


def save_sample_data(df, output_path):
    """Sauvegarde les données dans le fichier CSV."""

    # Créer le dossier si nécessaire
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sauvegarde
    df.to_csv(output_path, index=False)

    print(f"💾 Données sauvegardées : {output_path}")
    print(f"   📊 Taille du fichier : {output_path.stat().st_size // 1024} KB")


def verify_sample_data(file_path):
    """Vérifie que les données sont correctes."""

    print("🔍 Vérification des données générées...")

    try:
        df = pd.read_csv(file_path)

        print(f"   ✅ Fichier lisible : {len(df)} lignes")
        print(f"   ✅ Colonnes : {list(df.columns)}")
        print(f"   ✅ Période : {df['date_ligne_commande'].min()} à {df['date_ligne_commande'].max()}")
        print(f"   ✅ Articles uniques : {sorted(df['id_article'].unique())}")

        # Aperçu
        print("\n📋 Aperçu des premières lignes :")
        print(df.head().to_string(index=False))

        return True

    except Exception as e:
        print(f"   ❌ Erreur lors de la vérification : {e}")
        return False


def main():
    """Fonction principale."""

    print("🚀 CRÉATION DE DONNÉES D'EXEMPLE POUR LE PIPELINE")
    print("=" * 60)

    # Chemin de sortie
    output_file = Path("data/raw/commandes_2024.csv")

    # Vérification si le fichier existe déjà
    if output_file.exists():
        response = input(f"⚠️  Le fichier {output_file} existe déjà. Le remplacer ? (y/n): ")
        if response.lower() != 'y':
            print("❌ Opération annulée.")
            return False

    try:
        # Génération
        sample_data = generate_realistic_sample_data()

        # Sauvegarde
        save_sample_data(sample_data, output_file)

        # Vérification
        if verify_sample_data(output_file):
            print("\n" + "=" * 60)
            print("🎉 DONNÉES D'EXEMPLE CRÉÉES AVEC SUCCÈS !")
            print("=" * 60)
            print("\n💡 Vous pouvez maintenant lancer :")
            print("   python main.py --step all")
            print("\nOu tester étape par étape :")
            print("   python main.py --step ingestion")
            print("   python main.py --step enrichment")
            print("   python main.py --step analysis")
            print("   python main.py --step baselines")

            return True
        else:
            print("❌ Erreur lors de la vérification des données")
            return False

    except Exception as e:
        print(f"❌ Erreur lors de la génération : {e}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)