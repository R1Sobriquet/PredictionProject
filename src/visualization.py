"""
Module de visualisation des données de commandes ATM.

- Graphique des montants quotidiens par ATM
- Analyse des patterns par jour de semaine
- Comparaison weekend/semaine
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import logging
from pathlib import Path

try:
    from .utils import (
        ColumnNames,
        DisplayConfig,
        WEEKDAY_NAMES,
        get_file_path,
    )
except ImportError:
    from src.utils import (
        ColumnNames,
        DisplayConfig,
        WEEKDAY_NAMES,
        WEEKEND_DAYS,
        get_file_path,
    )

plt.style.use('default')
sns.set_palette("husl")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataVisualization:
    """Visualisation et analyse des patterns dans les données de commandes ATM."""

    def __init__(self, enriched_data: Optional[pd.DataFrame] = None):
        self.enriched_data = enriched_data
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10

    def load_enriched_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """Charge les données enrichies depuis un fichier."""
        if self.enriched_data is not None:
            return self.enriched_data

        file_path = file_path or get_file_path('enriched')

        if not file_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé : {file_path}")

        logger.info(f"Chargement des données enrichies : {file_path}")

        self.enriched_data = pd.read_csv(
            file_path,
            parse_dates=[ColumnNames.ORDER_DATE],
            date_format='%Y-%m-%d',
        )

        logger.info(f"{len(self.enriched_data)} lignes chargées pour visualisation")
        return self.enriched_data

    def plot_daily_amounts_by_atm(
        self,
        atm_id: int,
        figsize: Tuple[int, int] = (14, 8),
        show_weekend: bool = True,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Trace les montants de commande quotidiens pour un ATM donné.

        Args:
            atm_id: ID de l'ATM à analyser
            figsize: Taille de la figure
            show_weekend: Mettre en évidence les weekends
            save_path: Chemin de sauvegarde (optionnel)
        """
        if self.enriched_data is None:
            self.load_enriched_data()

        atm_data = self.enriched_data[
            self.enriched_data[ColumnNames.ATM_ID] == atm_id
        ].copy().sort_values(ColumnNames.ORDER_DATE)

        if atm_data.empty:
            raise ValueError(f"Aucune donnée trouvée pour l'ATM {atm_id}")

        logger.info(f"Graphique pour l'ATM {atm_id}")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        fig.suptitle(f'Analyse des Commandes - ATM {atm_id}', fontsize=14, fontweight='bold')

        dates = atm_data[ColumnNames.ORDER_DATE]
        amounts = atm_data[ColumnNames.AMOUNT]

        if show_weekend and ColumnNames.IS_WEEKEND in atm_data.columns:
            weekend_mask = atm_data[ColumnNames.IS_WEEKEND]
            ax1.scatter(dates[~weekend_mask], amounts[~weekend_mask],
                        alpha=0.6, color='steelblue', label='Semaine', s=30)
            ax1.scatter(dates[weekend_mask], amounts[weekend_mask],
                        alpha=0.8, color='orange', label='Weekend', s=30)
        else:
            ax1.scatter(dates, amounts, alpha=0.6, color='steelblue', s=30)

        ax1.plot(dates, amounts, alpha=0.3, color='gray', linewidth=0.5)

        ax1.set_title('Montants des Commandes')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Montant (EUR)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

        # Graphique par jour de semaine
        if ColumnNames.WEEKDAY_NAME in atm_data.columns:
            weekday_means = atm_data.groupby(ColumnNames.WEEKDAY_NAME)[ColumnNames.AMOUNT].mean()
            weekday_order = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
            weekday_means = weekday_means.reindex([d for d in weekday_order if d in weekday_means.index])

            colors = ['lightcoral' if d in ['Samedi', 'Dimanche'] else 'lightblue' for d in weekday_means.index]
            bars = ax2.bar(weekday_means.index, weekday_means.values, color=colors, alpha=0.8)

            for bar, value in zip(bars, weekday_means.values):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         f'{value:.0f}', ha='center', va='bottom', fontsize=9)

            ax2.set_title('Montant Moyen par Jour de Semaine')
            ax2.set_ylabel('Montant Moyen (EUR)')
            ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        logger.info(f"  Total commandes : {len(atm_data)}")
        logger.info(f"  Montant total : {amounts.sum():.0f} EUR")
        logger.info(f"  Montant moyen : {amounts.mean():.0f} EUR")

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  Graphique sauvegardé : {save_path}")

        return fig

    def plot_weekday_analysis(
        self,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[Path] = None,
    ) -> Tuple[plt.Figure, pd.DataFrame]:
        """Analyse des montants par jour de semaine pour tous les ATMs."""
        if self.enriched_data is None:
            self.load_enriched_data()

        logger.info("Analyse des montants par jour de semaine")

        weekday_stats = self.enriched_data.groupby(
            [ColumnNames.WEEKDAY_NAME, ColumnNames.WEEKDAY]
        )[ColumnNames.AMOUNT].agg(['mean', 'median', 'sum', 'std', 'count']).round(2).reset_index()
        weekday_stats = weekday_stats.sort_values(ColumnNames.WEEKDAY)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Analyse des Commandes ATM par Jour de Semaine', fontsize=14, fontweight='bold')

        days = weekday_stats[ColumnNames.WEEKDAY_NAME]
        colors = ['lightcoral' if d in ['Samedi', 'Dimanche'] else 'lightblue' for d in days]

        ax1.bar(days, weekday_stats['mean'], color=colors, alpha=0.8)
        ax1.set_title('Montant Moyen par Jour')
        ax1.set_ylabel('Montant Moyen (EUR)')
        ax1.tick_params(axis='x', rotation=45)

        ax2.bar(days, weekday_stats['sum'], color=colors, alpha=0.8)
        ax2.set_title('Montant Total par Jour')
        ax2.set_ylabel('Montant Total (EUR)')
        ax2.tick_params(axis='x', rotation=45)

        ax3.bar(days, weekday_stats['count'], color=colors, alpha=0.8)
        ax3.set_title('Nombre de Commandes par Jour')
        ax3.set_ylabel('Nombre de Commandes')
        ax3.tick_params(axis='x', rotation=45)

        ax4.bar(days, weekday_stats['std'], color=colors, alpha=0.8)
        ax4.set_title('Variabilité des Montants (Écart-type)')
        ax4.set_ylabel('Écart-type (EUR)')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  Graphique sauvegardé : {save_path}")

        return fig, weekday_stats

    def plot_weekend_vs_weekday_comparison(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """Compare les montants weekend vs semaine."""
        if self.enriched_data is None:
            self.load_enriched_data()

        if ColumnNames.IS_WEEKEND not in self.enriched_data.columns:
            raise ValueError("Colonne 'is_weekend' manquante")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Comparaison Weekend vs Semaine (Commandes ATM)', fontsize=14, fontweight='bold')

        weekend_data = self.enriched_data[self.enriched_data[ColumnNames.IS_WEEKEND]][ColumnNames.AMOUNT]
        weekday_data = self.enriched_data[~self.enriched_data[ColumnNames.IS_WEEKEND]][ColumnNames.AMOUNT]

        categories = ['Semaine', 'Weekend']
        means = [weekday_data.mean(), weekend_data.mean()]
        colors = ['lightblue', 'lightcoral']

        bars = ax1.bar(categories, means, color=colors, alpha=0.8)
        ax1.set_title('Montant Moyen')
        ax1.set_ylabel('Montant Moyen (EUR)')
        for bar, value in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(means) * 0.02,
                     f'{value:.0f}', ha='center', va='bottom', fontweight='bold')

        ax2.hist(weekday_data, bins=30, alpha=0.6, label='Semaine', color='lightblue', density=True)
        ax2.hist(weekend_data, bins=30, alpha=0.6, label='Weekend', color='lightcoral', density=True)
        ax2.set_title('Distribution des Montants')
        ax2.set_xlabel('Montant (EUR)')
        ax2.set_ylabel('Densité')
        ax2.legend()

        plt.tight_layout()

        if weekday_data.mean() > 0:
            ratio = weekend_data.mean() / weekday_data.mean()
            logger.info(f"  Ratio Weekend/Semaine : {ratio:.2f}")

        return fig

    def show_atm_history_verification(self, n_rows: int = 10, atm_ids: Optional[List[int]] = None) -> None:
        """Vérifie les features historiques ATM."""
        if self.enriched_data is None:
            self.load_enriched_data()

        logger.info(f"Vérification des features historiques ATM (premières {n_rows} lignes)")

        display_data = self.enriched_data.copy()

        if atm_ids:
            display_data = display_data[display_data[ColumnNames.ATM_ID].isin(atm_ids)]

        display_data = display_data.sort_values([ColumnNames.ATM_ID, ColumnNames.ORDER_DATE])

        columns_to_show = [
            ColumnNames.ORDER_DATE,
            ColumnNames.ATM_ID,
            ColumnNames.AMOUNT,
        ]
        # Add history columns if present
        for col in [ColumnNames.DAYS_SINCE_LAST_ORDER, ColumnNames.LAST_ORDER_AMOUNT,
                     ColumnNames.AVG_ORDER_AMOUNT, ColumnNames.WEEKDAY_NAME]:
            if col in display_data.columns:
                columns_to_show.append(col)

        preview = display_data[columns_to_show].head(n_rows)

        print("\n" + "=" * 100)
        print("VÉRIFICATION DES FEATURES HISTORIQUES ATM")
        print("=" * 100)
        print(preview.to_string(index=False))
        print("=" * 100)

        logger.info("Vérification terminée")


# ===== FONCTIONS UTILITAIRES =====

def create_atm_dashboard(
    enriched_data: pd.DataFrame,
    atm_id: int,
    output_dir: Optional[Path] = None,
) -> Dict[str, plt.Figure]:
    """Crée un dashboard complet pour un ATM."""
    viz = DataVisualization(enriched_data)
    figures = {}

    try:
        figures['daily_amounts'] = viz.plot_daily_amounts_by_atm(
            atm_id=atm_id,
            save_path=output_dir / f"atm_{atm_id}_daily_amounts.png" if output_dir else None,
        )
        logger.info(f"Dashboard créé pour l'ATM {atm_id}")
    except Exception as e:
        logger.error(f"Erreur dashboard ATM {atm_id}: {e}")

    return figures


def create_global_analysis(
    enriched_data: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Crée une analyse globale de tous les ATMs."""
    viz = DataVisualization(enriched_data)
    results = {'figures': {}, 'stats': {}}

    try:
        fig_weekday, stats_weekday = viz.plot_weekday_analysis(
            save_path=output_dir / "weekday_analysis.png" if output_dir else None,
        )
        results['figures']['weekday_analysis'] = fig_weekday
        results['stats']['weekday_stats'] = stats_weekday
        results['figures']['weekend_comparison'] = viz.plot_weekend_vs_weekday_comparison()
        logger.info("Analyse globale terminée")
    except Exception as e:
        logger.error(f"Erreur analyse globale : {e}")

    return results


if __name__ == "__main__":
    try:
        viz = DataVisualization()
        viz.load_enriched_data()
        viz.show_atm_history_verification(n_rows=15)
        fig_weekday, stats = viz.plot_weekday_analysis()
        plt.show()
    except Exception as e:
        print(f"Erreur : {e}")
        print("Assurez-vous que les données enrichies existent dans data/processed/")
