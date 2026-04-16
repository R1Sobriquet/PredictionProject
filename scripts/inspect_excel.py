"""Inspection rapide des fichiers Excel Commande.xlsx et CommandeDetail.xlsx.

Affiche pour chaque fichier :
- nom de la feuille
- shape (lignes × colonnes)
- liste complète des colonnes
- 3 premières lignes
- types de données

Usage :
    python scripts/inspect_excel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

FILES = [
    RAW_DIR / "Commande.xlsx",
    RAW_DIR / "CommandeDetail.xlsx",
]


def inspect(path: Path) -> None:
    if not path.exists():
        print(f"[KO] Fichier introuvable : {path}")
        return

    print("=" * 80)
    print(f"FICHIER : {path.name}")
    print("=" * 80)

    # Liste des feuilles
    xl = pd.ExcelFile(path, engine="openpyxl")
    print(f"Feuilles : {xl.sheet_names}")

    for sheet in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        print(f"\n--- Feuille: {sheet} ---")
        print(f"Shape : {df.shape[0]} lignes × {df.shape[1]} colonnes")
        print(f"\nColonnes ({len(df.columns)}) :")
        for i, col in enumerate(df.columns):
            dtype = df[col].dtype
            n_null = df[col].isna().sum()
            print(f"  {i+1:3d}. {col!r:<40} [{dtype}]  nulls={n_null}")

        print("\n3 premières lignes :")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(df.head(3).to_string())

        # Détection d'une clé de jointure candidate
        id_candidates = [c for c in df.columns if "commande" in c.lower() and "id" in c.lower()]
        if id_candidates:
            print(f"\nClé(s) candidate(s) de jointure : {id_candidates}")
            for c in id_candidates:
                print(f"  {c} : {df[c].nunique()} valeurs uniques / {len(df)} lignes")


def main() -> int:
    print(f"Répertoire raw : {RAW_DIR}\n")
    for f in FILES:
        inspect(f)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
