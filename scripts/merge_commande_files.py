"""Fusionne Commande.xlsx + CommandeDetail.xlsx → d_CommandesDetailCalcul.xlsx.

Ce script prend en entrée :
- ``data/raw/Commande.xlsx``       : en-tête commande (1 ligne par commande)
- ``data/raw/CommandeDetail.xlsx`` : détail par commande (soldes/cassettes/coupures)

Et produit ``data/raw/d_CommandesDetailCalcul.xlsx``, le fichier unique attendu
par le pipeline (``src.utils.config.RAW_DATA_FILE``).

Stratégie de jointure
---------------------

1. Détection automatique de la clé de jointure (colonne dont le nom contient à
   la fois ``commande`` et ``id``, ou ``DC_Commande_Id`` en priorité).
2. Jointure ``LEFT`` de Commande (entête) avec CommandeDetail (détails).
3. Si les 2 fichiers partagent des colonnes (hors clé), celles de
   ``CommandeDetail`` l'emportent (suffixe ``_detail`` pour les autres).
4. Écriture du résultat au format attendu.

Si certaines colonnes de ``COLUMNS_TO_LOAD`` sont absentes, elles sont créées
vides (0 ou NaN) avec un avertissement — le pipeline est tolérant aux NaN
(cf. ``data_ingestion._clean_data``).

Usage :
    python scripts/merge_commande_files.py
    python scripts/merge_commande_files.py --dry-run      # Aperçu sans écrire
    python scripts/merge_commande_files.py --key DC_Commande_Id  # Forcer la clé
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import COLUMNS_TO_LOAD, RAW_DATA_FILE  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
SRC_HEADER = RAW_DIR / "Commande.xlsx"
SRC_DETAIL = RAW_DIR / "CommandeDetail.xlsx"
DST = RAW_DIR / RAW_DATA_FILE


def _detect_join_key(df_h: pd.DataFrame, df_d: pd.DataFrame) -> str:
    """Détecte une clé commune entre les deux DataFrames."""
    # Priorité : DC_Commande_Id exact
    if "DC_Commande_Id" in df_h.columns and "DC_Commande_Id" in df_d.columns:
        return "DC_Commande_Id"

    common = [c for c in df_h.columns if c in df_d.columns]
    # Filtre : doit contenir 'commande' + 'id'
    candidates = [
        c for c in common
        if "commande" in c.lower() and "id" in c.lower()
    ]
    if candidates:
        return candidates[0]

    # Dernier recours : première colonne commune
    if common:
        print(
            f"[WARN] Pas de clé 'commande_id' évidente. Utilisation de : {common[0]}",
            file=sys.stderr,
        )
        return common[0]

    raise RuntimeError(
        "Impossible de détecter une clé de jointure commune aux 2 fichiers. "
        "Utilisez --key NOM_COLONNE pour forcer."
    )


def _load_excel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    print(f"Chargement : {path.name}")
    df = pd.read_excel(path, engine="openpyxl")
    print(f"  → {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df


def merge(
    header_path: Path = SRC_HEADER,
    detail_path: Path = SRC_DETAIL,
    output_path: Path = DST,
    join_key: str | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    df_h = _load_excel(header_path)
    df_d = _load_excel(detail_path)

    if join_key is None:
        join_key = _detect_join_key(df_h, df_d)
    print(f"Clé de jointure : {join_key!r}")

    if join_key not in df_h.columns or join_key not in df_d.columns:
        raise KeyError(
            f"La clé {join_key!r} doit exister dans les 2 fichiers. "
            f"Header: {join_key in df_h.columns}, Detail: {join_key in df_d.columns}"
        )

    # Colonnes communes (hors clé) : on garde celles du DETAIL (priorité doc HFSQL)
    common_non_key = [
        c for c in df_h.columns if c in df_d.columns and c != join_key
    ]
    if common_non_key:
        print(f"  Colonnes dupliquées (on garde celles de CommandeDetail) : {common_non_key}")
        df_h = df_h.drop(columns=common_non_key)

    # Merge LEFT : on part de l'entête pour garder 1 ligne par commande
    merged = df_h.merge(df_d, on=join_key, how="left", validate="one_to_many")
    print(f"Après merge : {merged.shape[0]} lignes × {merged.shape[1]} colonnes")

    # Vérif : colonnes attendues par le pipeline
    missing = [c for c in COLUMNS_TO_LOAD if c not in merged.columns]
    if missing:
        print(f"\n[WARN] {len(missing)} colonne(s) attendue(s) manquante(s) — seront créées NaN :")
        for c in missing:
            print(f"    - {c}")
            merged[c] = pd.NA

    present = [c for c in COLUMNS_TO_LOAD if c in merged.columns]
    print(f"\nColonnes présentes requises : {len(present)} / {len(COLUMNS_TO_LOAD)}")

    # On conserve uniquement les colonnes du pipeline (ordre canonique)
    final = merged[COLUMNS_TO_LOAD].copy()

    if dry_run:
        print("\n[DRY-RUN] Pas d'écriture. Aperçu :")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(final.head(3).to_string())
        return final

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_excel(output_path, engine="openpyxl", index=False)
    print(f"\n[OK] Fichier écrit : {output_path}")
    print(f"  Lignes : {len(final)}")
    print(f"  Colonnes : {len(final.columns)}")
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", default=None, help="Forcer la colonne de jointure")
    parser.add_argument("--dry-run", action="store_true", help="Aperçu sans écrire")
    parser.add_argument(
        "--header", type=Path, default=SRC_HEADER, help=f"Défaut: {SRC_HEADER}"
    )
    parser.add_argument(
        "--detail", type=Path, default=SRC_DETAIL, help=f"Défaut: {SRC_DETAIL}"
    )
    parser.add_argument(
        "--output", type=Path, default=DST, help=f"Défaut: {DST}"
    )
    args = parser.parse_args()

    try:
        merge(
            header_path=args.header,
            detail_path=args.detail,
            output_path=args.output,
            join_key=args.key,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"\n[ERREUR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
