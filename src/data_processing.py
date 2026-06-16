"""
Module d'enrichissement des données de commandes ATM.

Features ajoutées :
1. Features temporelles (year, month, weekday, etc.)
2. Features ATM historiques (days_since_last_order, avg_reload_frequency, etc.)
3. Features ATM agrégées (total_soldes, total_k7hs, cassettes_actives, etc.)
4. Features saisonnières (sin/cos cycliques, quarter, etc.)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging
from pathlib import Path

try:
    from .utils import (
        ColumnNames,
        DisplayConfig,
        Messages,
        WEEKDAY_NAMES,
        WEEKEND_DAYS,
        CASSETTE_COLUMNS,
        AJUSTEMENT_COLUMNS,
        SOLDES_COLUMNS,
        K7HS_COLUMNS,
        COUPURES,
        SOLDES_BY_COUPURE,
        DMQ_BY_COUPURE,
        is_french_holiday,
        is_eve_of_holiday,
        is_payday,
        get_file_path,
    )
except ImportError:
    from src.utils import (
        ColumnNames,
        DisplayConfig,
        Messages,
        WEEKDAY_NAMES,
        WEEKEND_DAYS,
        CASSETTE_COLUMNS,
        AJUSTEMENT_COLUMNS,
        SOLDES_COLUMNS,
        K7HS_COLUMNS,
        COUPURES,
        SOLDES_BY_COUPURE,
        DMQ_BY_COUPURE,
        is_french_holiday,
        is_eve_of_holiday,
        is_payday,
        get_file_path,
    )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataEnrichmentPipeline:
    """
    Pipeline d'enrichissement des données ATM.

    Transforme les données nettoyées en dataset enrichi avec :
    - Features temporelles
    - Historique de rechargement par ATM
    - Agrégats par coupure (soldes, ajustements, cassettes HS)
    - Délais livraison/chargement
    - Features saisonnières cycliques
    """

    def __init__(self, clean_data: Optional[pd.DataFrame] = None):
        self.clean_data = clean_data
        self.enriched_data = None

    def load_clean_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """Charge les données nettoyées depuis un fichier."""
        if self.clean_data is not None:
            return self.clean_data

        file_path = file_path or get_file_path('clean')

        if not file_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé : {file_path}")

        logger.info(f"Chargement des données nettoyées : {file_path}")

        self.clean_data = pd.read_csv(
            file_path,
            parse_dates=[ColumnNames.ORDER_DATE],
            date_format='%Y-%m-%d',
        )

        # Convertir les dates optionnelles
        for date_col in [ColumnNames.DELIVERY_DATE, ColumnNames.LOADING_DATE]:
            if date_col in self.clean_data.columns:
                self.clean_data[date_col] = pd.to_datetime(
                    self.clean_data[date_col], errors='coerce'
                )

        logger.info(f"{len(self.clean_data)} lignes chargées")
        return self.clean_data

    def add_temporal_features(self) -> pd.DataFrame:
        """
        Ajoute les variables temporelles basées sur order_date.

        Variables : year, month, day, weekday, weekday_name, is_weekend, week_number
        """
        logger.info("Ajout des variables temporelles...")

        if self.clean_data is None:
            self.load_clean_data()

        self.enriched_data = self.clean_data.copy()

        dt = self.enriched_data[ColumnNames.ORDER_DATE]
        self.enriched_data[ColumnNames.YEAR] = dt.dt.year
        self.enriched_data[ColumnNames.MONTH] = dt.dt.month
        self.enriched_data[ColumnNames.DAY] = dt.dt.day
        self.enriched_data[ColumnNames.WEEKDAY] = dt.dt.weekday
        self.enriched_data[ColumnNames.WEEKDAY_NAME] = self.enriched_data[ColumnNames.WEEKDAY].map(WEEKDAY_NAMES)
        self.enriched_data[ColumnNames.IS_WEEKEND] = self.enriched_data[ColumnNames.WEEKDAY].isin(WEEKEND_DAYS)
        self.enriched_data[ColumnNames.WEEK_NUMBER] = dt.dt.isocalendar().week

        # Features calendrier FR (jours fériés + paie) — signaux forts pour la
        # prévision (veille de férié = souvent pic, fins de mois = pics paie).
        dates_py = dt.dt.date
        self.enriched_data['is_holiday'] = dates_py.map(is_french_holiday).astype(bool)
        self.enriched_data['is_eve_holiday'] = dates_py.map(is_eve_of_holiday).astype(bool)
        self.enriched_data['is_payday'] = dates_py.map(is_payday).astype(bool)

        logger.info("Variables temporelles ajoutées")
        return self.enriched_data

    def add_atm_history_features(self) -> pd.DataFrame:
        """
        Ajoute les features basées sur l'historique de rechargement de chaque ATM.

        Features :
        - days_since_last_order : jours depuis le dernier rechargement
        - last_order_amount : montant du dernier rechargement
        - avg_reload_frequency : fréquence moyenne de rechargement (jours)
        - avg_order_amount : montant moyen historique
        - std_order_amount : écart-type des montants
        - order_count_last_30d : commandes sur les 30 derniers jours calendaires
        """
        logger.info("Ajout des features historiques ATM...")

        if self.enriched_data is None:
            raise ValueError("Exécutez add_temporal_features() d'abord")

        # Trier par ATM et date
        self.enriched_data = self.enriched_data.sort_values(
            [ColumnNames.ATM_ID, ColumnNames.ORDER_DATE]
        ).reset_index(drop=True)

        # --- days_since_last_order ---
        self.enriched_data[ColumnNames.DAYS_SINCE_LAST_ORDER] = (
            self.enriched_data.groupby(ColumnNames.ATM_ID)[ColumnNames.ORDER_DATE]
            .diff()
            .dt.days
            .fillna(0)
            .astype(int)
        )

        # --- last_order_amount ---
        self.enriched_data[ColumnNames.LAST_ORDER_AMOUNT] = (
            self.enriched_data.groupby(ColumnNames.ATM_ID)[ColumnNames.AMOUNT]
            .shift(1)
            .fillna(0)
        )

        # --- avg_reload_frequency (expanding mean of days between orders) ---
        self.enriched_data[ColumnNames.AVG_RELOAD_FREQUENCY] = (
            self.enriched_data.groupby(ColumnNames.ATM_ID)[ColumnNames.DAYS_SINCE_LAST_ORDER]
            .transform(lambda x: x.expanding().mean())
            .round(2)
        )

        # --- avg_order_amount (expanding mean of amounts, excluding current) ---
        self.enriched_data[ColumnNames.AVG_ORDER_AMOUNT] = (
            self.enriched_data.groupby(ColumnNames.ATM_ID)[ColumnNames.AMOUNT]
            .transform(lambda x: x.shift(1).expanding().mean())
            .fillna(0)
            .round(2)
        )

        # --- std_order_amount (expanding std, excluding current) ---
        self.enriched_data[ColumnNames.STD_ORDER_AMOUNT] = (
            self.enriched_data.groupby(ColumnNames.ATM_ID)[ColumnNames.AMOUNT]
            .transform(lambda x: x.shift(1).expanding().std())
            .fillna(0)
            .round(2)
        )

        # --- order_count_last_30d ---
        # Nombre de commandes pour cet ATM dans les 30 jours calendaires précédents
        self.enriched_data[ColumnNames.ORDER_COUNT_LAST_30D] = 0

        for atm_id in self.enriched_data[ColumnNames.ATM_ID].unique():
            atm_mask = self.enriched_data[ColumnNames.ATM_ID] == atm_id
            atm_dates = self.enriched_data.loc[atm_mask, ColumnNames.ORDER_DATE]

            counts = []
            for date in atm_dates:
                window_start = date - pd.Timedelta(days=30)
                count = ((atm_dates >= window_start) & (atm_dates < date)).sum()
                counts.append(count)

            self.enriched_data.loc[atm_mask, ColumnNames.ORDER_COUNT_LAST_30D] = counts

        logger.info("Features historiques ATM ajoutées")
        return self.enriched_data

    def add_atm_data_features(self) -> pd.DataFrame:
        """
        Ajoute les features agrégées à partir des données ATM.

        Features :
        - total_soldes : somme des soldes par coupure
        - total_ajustement : somme des ajustements par coupure
        - total_k7hs : somme des cassettes hors service
        - cassettes_actives : nombre de cassettes non nulles (1-5)
        - delivery_delay : délai livraison en jours
        - loading_delay : délai chargement en jours
        """
        logger.info("Ajout des features agrégées ATM...")

        if self.enriched_data is None:
            raise ValueError("Exécutez add_temporal_features() d'abord")

        # Somme des soldes par coupure
        existing_soldes = [c for c in SOLDES_COLUMNS if c in self.enriched_data.columns]
        if existing_soldes:
            self.enriched_data['total_soldes'] = self.enriched_data[existing_soldes].sum(axis=1)

        # Somme des ajustements par coupure
        existing_ajust = [c for c in AJUSTEMENT_COLUMNS if c in self.enriched_data.columns]
        if existing_ajust:
            self.enriched_data['total_ajustement'] = self.enriched_data[existing_ajust].sum(axis=1)

        # Somme des cassettes hors service
        existing_k7hs = [c for c in K7HS_COLUMNS if c in self.enriched_data.columns]
        if existing_k7hs:
            self.enriched_data['total_k7hs'] = self.enriched_data[existing_k7hs].sum(axis=1)

        # Nombre de cassettes actives (non nulles) parmi les 5 cassettes
        existing_cassettes = [c for c in CASSETTE_COLUMNS if c in self.enriched_data.columns]
        if existing_cassettes:
            self.enriched_data['cassettes_actives'] = (
                self.enriched_data[existing_cassettes].gt(0).sum(axis=1)
            )

        # Délai de livraison (delivery_date - order_date)
        if ColumnNames.DELIVERY_DATE in self.enriched_data.columns:
            self.enriched_data['delivery_delay'] = (
                (self.enriched_data[ColumnNames.DELIVERY_DATE] - self.enriched_data[ColumnNames.ORDER_DATE])
                .dt.days
                .fillna(0)
                .astype(int)
            )

        # Délai de chargement (loading_date - order_date)
        if ColumnNames.LOADING_DATE in self.enriched_data.columns:
            self.enriched_data['loading_delay'] = (
                (self.enriched_data[ColumnNames.LOADING_DATE] - self.enriched_data[ColumnNames.ORDER_DATE])
                .dt.days
                .fillna(0)
                .astype(int)
            )

        logger.info("Features agrégées ATM ajoutées")
        return self.enriched_data

    def add_seasonal_features(self) -> pd.DataFrame:
        """
        Ajoute des variables saisonnières avancées.

        Variables : day_of_year, quarter, is_month_start/middle/end,
                    sin/cos cycliques (day_of_year, weekday)
        """
        logger.info("Ajout des variables saisonnières...")

        if self.enriched_data is None:
            raise ValueError("Exécutez add_temporal_features() d'abord")

        self.enriched_data['day_of_year'] = self.enriched_data[ColumnNames.ORDER_DATE].dt.dayofyear
        self.enriched_data['quarter'] = self.enriched_data[ColumnNames.ORDER_DATE].dt.quarter

        # Position dans le mois
        day = self.enriched_data[ColumnNames.DAY]
        self.enriched_data['is_month_start'] = day <= 5
        self.enriched_data['is_month_middle'] = (day > 10) & (day <= 20)
        self.enriched_data['is_month_end'] = day > 25

        # Encodage cyclique
        self.enriched_data['day_of_year_sin'] = np.sin(2 * np.pi * self.enriched_data['day_of_year'] / 365.25)
        self.enriched_data['day_of_year_cos'] = np.cos(2 * np.pi * self.enriched_data['day_of_year'] / 365.25)
        self.enriched_data['weekday_sin'] = np.sin(2 * np.pi * self.enriched_data[ColumnNames.WEEKDAY] / 7)
        self.enriched_data['weekday_cos'] = np.cos(2 * np.pi * self.enriched_data[ColumnNames.WEEKDAY] / 7)

        logger.info("Variables saisonnières ajoutées")
        return self.enriched_data

    def add_dmq_features(self) -> pd.DataFrame:
        """Ajoute des features issues du DMQ (signal de consommation).

        Features :
        - ``dmq_volatilite``           : écart-type glissant 28 j du montant.
        - ``dmq_trend_7j``             : pente d'une régression linéaire sur les
                                         7 dernières observations (par ATM).
        - ``dmq_trend_28j``            : pente sur 28 j.
        - ``dmq_debut_mois_ratio``     : amount_jour / moyenne_28j (via shift(1))
        - ``soldes_ratio_total``       : total_soldes / montant_assurance si dispo

        Note : ne nécessite pas l'ajout de colonnes cibles DMQ par coupure — ces
        features sont des **signaux d'entrée** pour le modèle. Les DMQ par
        coupure (``dmq_5/10/...``) sont produits séparément (voir config +
        pipeline).
        """
        logger.info("Ajout des features DMQ (volatilité + tendances)...")

        if self.enriched_data is None:
            raise ValueError("Exécutez add_temporal_features() d'abord")

        df = self.enriched_data
        gb = df.groupby(ColumnNames.ATM_ID)[ColumnNames.AMOUNT]

        # Volatilité (écart-type) des montants sur les 28 dernières commandes
        df['dmq_volatilite'] = gb.transform(
            lambda x: x.shift(1).rolling(window=28, min_periods=2).std()
        ).fillna(0).round(2)

        # Pente (tendance) sur 7 et 28 jours via une régression linéaire simple
        def _slope(series: pd.Series) -> float:
            y = series.dropna().to_numpy(dtype=float)
            if len(y) < 2:
                return 0.0
            x = np.arange(len(y), dtype=float)
            # polyfit degré 1 : retourne [pente, intercept]
            try:
                slope = np.polyfit(x, y, 1)[0]
            except (np.linalg.LinAlgError, ValueError):
                return 0.0
            return float(slope)

        df['dmq_trend_7j'] = gb.transform(
            lambda x: x.shift(1).rolling(window=7, min_periods=2).apply(_slope, raw=False)
        ).fillna(0).round(2)

        df['dmq_trend_28j'] = gb.transform(
            lambda x: x.shift(1).rolling(window=28, min_periods=2).apply(_slope, raw=False)
        ).fillna(0).round(2)

        # Ratio "amount du jour vs moyenne glissante" — signal début de mois
        rolling_mean_28 = gb.transform(
            lambda x: x.shift(1).rolling(window=28, min_periods=1).mean()
        )
        safe_mean = rolling_mean_28.replace(0, np.nan)
        df['dmq_debut_mois_ratio'] = (df[ColumnNames.AMOUNT] / safe_mean).fillna(1.0).round(3)

        # Ratio de remplissage des soldes (si colonnes dispo)
        if 'total_soldes' in df.columns and ColumnNames.INSURANCE_AMOUNT in df.columns:
            denom = df[ColumnNames.INSURANCE_AMOUNT].replace(0, np.nan)
            df['soldes_ratio_assurance'] = (df['total_soldes'] / denom).fillna(0).round(3)

        self.enriched_data = df
        logger.info("Features DMQ ajoutées")
        return self.enriched_data

    def add_dmq_per_coupure_features(self) -> pd.DataFrame:
        """Crée les colonnes ``dmq_<c>`` pour chaque coupure.

        Définition : ``DMQ_<c>(t)`` = moyenne sur 28 jours glissants des
        **baisses** quotidiennes de ``solde_<c>`` (consommation observée),
        calculée par ATM. Un ``shift(1)`` est appliqué pour que la valeur à
        la date t ne dépende que du passé strict (pas de leakage).

        Ces colonnes sont produites comme **signaux d'entrée** pour le
        ``CommandPipeline`` (cf. ``pipeline.py`` / ``_default_dmq_provider``)
        et comme targets possibles pour ``CatBoostDmqForecaster``.
        """
        logger.info("Ajout des features DMQ par coupure (dmq_5..100)...")

        if self.enriched_data is None:
            raise ValueError("Exécutez add_temporal_features() d'abord")

        df = self.enriched_data.sort_values(
            [ColumnNames.ATM_ID, ColumnNames.ORDER_DATE]
        ).reset_index(drop=True)

        for c in COUPURES:
            sol_col = SOLDES_BY_COUPURE[c]
            dmq_col = DMQ_BY_COUPURE[c]

            if sol_col not in df.columns:
                logger.warning(
                    f"  Colonne {sol_col} absente : {dmq_col} mis à 0.0"
                )
                df[dmq_col] = 0.0
                continue

            # Baisses quotidiennes = diff négative (clippée à 0)
            diffs = df.groupby(ColumnNames.ATM_ID)[sol_col].diff()
            baisses = (-diffs).clip(lower=0)

            # Moyenne glissante 28j, shift(1) pour éviter toute fuite
            df[dmq_col] = (
                baisses.groupby(df[ColumnNames.ATM_ID])
                .transform(
                    lambda s: s.shift(1).rolling(window=28, min_periods=3).mean()
                )
                .fillna(0.0)
                .round(2)
            )

        self.enriched_data = df
        logger.info(f"  5 colonnes DMQ par coupure ajoutées : {list(DMQ_BY_COUPURE.values())}")
        return self.enriched_data

    def save_enriched_data(self, output_path: Optional[Path] = None) -> Path:
        """Sauvegarde les données enrichies."""
        if self.enriched_data is None:
            raise ValueError("Aucune donnée enrichie à sauvegarder")

        output_path = output_path or get_file_path('enriched')
        self.enriched_data.to_csv(output_path, index=False, date_format='%Y-%m-%d')
        logger.info(f"Données enrichies sauvegardées : {output_path}")
        return output_path

    def save_intermediate_snapshot(self, data: pd.DataFrame, stage_name: str) -> Path:
        """Sauvegarde un snapshot intermédiaire."""
        snapshot_dir = Path("data/snapshots")
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        snapshot_file = snapshot_dir / f"snapshot_enrichment_{stage_name}.csv"
        data.to_csv(snapshot_file, index=False, date_format='%Y-%m-%d')
        logger.info(f"  Snapshot sauvegardé : {snapshot_file} ({len(data)} lignes)")
        return snapshot_file

    def run_full_enrichment(self, save_output: bool = True, save_snapshots: bool = True) -> pd.DataFrame:
        """
        Exécute le pipeline complet d'enrichissement.

        Args:
            save_output: Sauvegarde du résultat final
            save_snapshots: Sauvegarde des snapshots intermédiaires

        Returns:
            DataFrame: Données enrichies
        """
        logger.info("DÉBUT DU PIPELINE D'ENRICHISSEMENT ATM")
        logger.info("=" * 50)

        try:
            # ÉTAPE 1 : Chargement
            logger.info("\nÉTAPE 1 : Chargement des données nettoyées...")
            self.load_clean_data()
            if save_snapshots:
                self.save_intermediate_snapshot(self.clean_data, "01_loaded")

            # ÉTAPE 2 : Variables temporelles
            logger.info("\nÉTAPE 2 : Variables temporelles...")
            self.add_temporal_features()
            if save_snapshots:
                self.save_intermediate_snapshot(self.enriched_data, "02_temporal")

            # ÉTAPE 3 : Historique ATM
            logger.info("\nÉTAPE 3 : Features historiques ATM...")
            self.add_atm_history_features()
            if save_snapshots:
                self.save_intermediate_snapshot(self.enriched_data, "03_atm_history")

            # ÉTAPE 4 : Features agrégées ATM
            logger.info("\nÉTAPE 4 : Features agrégées ATM...")
            self.add_atm_data_features()
            if save_snapshots:
                self.save_intermediate_snapshot(self.enriched_data, "04_atm_data")

            # ÉTAPE 5 : Variables saisonnières
            logger.info("\nÉTAPE 5 : Variables saisonnières...")
            self.add_seasonal_features()
            if save_snapshots:
                self.save_intermediate_snapshot(self.enriched_data, "05_seasonal")

            # ÉTAPE 5b : Features DMQ (volatilité, tendance, ratios)
            logger.info("\nÉTAPE 5b : Features DMQ...")
            self.add_dmq_features()
            if save_snapshots:
                self.save_intermediate_snapshot(self.enriched_data, "05b_dmq")

            # ÉTAPE 5c : DMQ par coupure (dmq_5..100)
            logger.info("\nÉTAPE 5c : Features DMQ par coupure...")
            self.add_dmq_per_coupure_features()
            if save_snapshots:
                self.save_intermediate_snapshot(self.enriched_data, "05c_dmq_per_coupure")

            # ÉTAPE 6 : Sauvegarde
            if save_output:
                logger.info("\nÉTAPE 6 : Sauvegarde finale...")
                self.save_enriched_data()

            logger.info("=" * 50)
            logger.info("PIPELINE D'ENRICHISSEMENT TERMINÉ")
            logger.info(f"  {len(self.enriched_data)} lignes enrichies")
            logger.info(f"  {len(self.enriched_data.columns)} colonnes au total")

            return self.enriched_data

        except Exception as e:
            logger.error(f"ERREUR DANS LE PIPELINE : {e}")
            raise

    def get_enrichment_summary(self) -> Dict:
        """Retourne un résumé des données enrichies."""
        if self.enriched_data is None:
            return {"error": "Aucune donnée enrichie disponible"}

        return {
            "total_lignes": len(self.enriched_data),
            "total_colonnes": len(self.enriched_data.columns),
            "atms_uniques": self.enriched_data[ColumnNames.ATM_ID].nunique(),
            "periode": {
                "debut": self.enriched_data[ColumnNames.ORDER_DATE].min(),
                "fin": self.enriched_data[ColumnNames.ORDER_DATE].max(),
                "nb_jours": self.enriched_data[ColumnNames.ORDER_DATE].nunique(),
            },
            "montant_moyen": self.enriched_data[ColumnNames.AMOUNT].mean(),
            "montant_total": self.enriched_data[ColumnNames.AMOUNT].sum(),
        }


# ===== FONCTIONS UTILITAIRES =====

def quick_enrichment(clean_data_path: str = None) -> pd.DataFrame:
    """Fonction rapide pour l'enrichissement complet."""
    pipeline = DataEnrichmentPipeline()
    return pipeline.run_full_enrichment()


def analyze_atm_pattern(enriched_data: pd.DataFrame, atm_id: int) -> Dict:
    """
    Analyse les patterns d'un ATM spécifique.

    Args:
        enriched_data: Données enrichies
        atm_id: ID de l'ATM à analyser

    Returns:
        dict: Analyse des patterns de l'ATM
    """
    atm_data = enriched_data[enriched_data[ColumnNames.ATM_ID] == atm_id].copy()

    if atm_data.empty:
        return {"error": f"ATM {atm_id} non trouvé"}

    weekday_stats = atm_data.groupby(ColumnNames.WEEKDAY_NAME)[ColumnNames.AMOUNT].agg(
        ['mean', 'sum', 'std']
    ).round(2)

    return {
        "atm_id": atm_id,
        "total_commandes": len(atm_data),
        "montant_total": atm_data[ColumnNames.AMOUNT].sum(),
        "montant_moyen": atm_data[ColumnNames.AMOUNT].mean(),
        "frequence_moyenne_jours": atm_data[ColumnNames.DAYS_SINCE_LAST_ORDER].mean()
            if ColumnNames.DAYS_SINCE_LAST_ORDER in atm_data.columns else None,
        "jour_plus_fort": weekday_stats['mean'].idxmax() if not weekday_stats.empty else None,
        "stats_par_jour": weekday_stats.to_dict('index'),
    }


if __name__ == "__main__":
    pipeline = DataEnrichmentPipeline()
    try:
        data = pipeline.run_full_enrichment()
        summary = pipeline.get_enrichment_summary()

        print("\nRÉSUMÉ DE L'ENRICHISSEMENT :")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"Erreur : {e}")
        print("Assurez-vous que les données nettoyées existent dans data/processed/")
