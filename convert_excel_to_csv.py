import pandas as pd
from pathlib import Path

# ===== CHARGE LES FICHIERS EXCEL =====
print("Chargement des fichiers Excel...")

commandes = pd.read_excel("data/raw/d_Commandes.xlsx")
details   = pd.read_excel("data/raw/d_CommandesDetailCalcul.xlsx")

# Affiche les colonnes pour qu'on puisse faire le mapping
print("\n=== Colonnes d_Commandes ===")
print(commandes.columns.tolist())

print("\n=== Colonnes d_CommandesDetailCalcul ===")
print(details.columns.tolist())

print("\n=== Aperçu d_Commandes ===")
print(commandes.head(3))

print("\n=== Aperçu d_CommandesDetailCalcul ===")
print(details.head(3))