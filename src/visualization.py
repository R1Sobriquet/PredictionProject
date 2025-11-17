"""
Module de visualisation des données de commandes.

- Graphique des ventes quotidiennes par article
- Analyse des patterns par jour de semaine
- Visualisations pour identifier les tendances cachées
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
        get_file_path
    )
except ImportError:
    from src.utils import (
        ColumnNames,
        DisplayConfig,
        WEEKDAY_NAMES,
        WEEKEND_DAYS,
        get_file_path
    )

# Configuration du style des graphiques
plt.style.use('default')
sns.set_palette("husl")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataVisualization:
    """
    Classe pour la visualisation et l'analyse des patterns dans les données de commandes.

    Génère les graphiques demandés dans l'étape 1B du projet.
    """

    def __init__(self, enriched_data: Optional[pd.DataFrame] = None):
        """
        Initialise la classe de visualisation.

        Args:
            enriched_data: Données enrichies (si None, charge depuis le fichier)
        """
        self.enriched_data = enriched_data

        # Configuration matplotlib pour les graphiques en français
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9

    def load_enriched_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Charge les données enrichies depuis un fichier.

        Args:
            file_path: Chemin du fichier(Optionnel)
        Returns:
            DataFrame: Données enrichies chargées
        """
        if self.enriched_data is not None:
            return self.enriched_data

        file_path = file_path or get_file_path('enriched')

        if not file_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé : {file_path}")

        logger.info(f"Chargement des données enrichies : {file_path}")

        self.enriched_data = pd.read_csv(
            file_path,
            parse_dates=[ColumnNames.DATE],
            date_format='%Y-%m-%d'
        )

        logger.info(f"✅ {len(self.enriched_data)} lignes chargées pour visualisation")
        return self.enriched_data

    def plot_daily_sales_by_article(
        self,
        article_id: int,
        figsize: Tuple[int, int] = (14, 8),
        show_weekend: bool = True,
        show_trend: bool = True,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Trace un graphique des ventes quotidiennes pour un article donné.

        OBJECTIF : Identifier les motifs répétitifs (ex: plus de commandes le lundi)

        Args:
            article_id: ID de l'article à analyser
            figsize: Taille de la figure
            show_weekend: Afficher les weekends en couleur différente
            show_trend: Afficher la tendance (moyenne mobile)
            save_path: Chemin de sauvegarde (optionnel)

        Returns:
            Figure: Objet matplotlib Figure
        """
        if self.enriched_data is None:
            self.load_enriched_data()

        # Filtrage des données pour l'article spécifié
        article_data = self.enriched_data[
            self.enriched_data[ColumnNames.ARTICLE_ID] == article_id
        ].copy().sort_values(ColumnNames.DATE)

        if article_data.empty:
            raise ValueError(f"Aucune donnée trouvée pour l'article {article_id}")

        logger.info(f"📊 Génération du graphique pour l'article {article_id}")

        # Création de la figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        fig.suptitle(f'Analyse des Ventes Quotidiennes - Article {article_id}', fontsize=14, fontweight='bold')

        # === GRAPHIQUE PRINCIPAL ===
        dates = article_data[ColumnNames.DATE]
        quantities = article_data[ColumnNames.QUANTITY]

        # Points de données
        if show_weekend and ColumnNames.IS_WEEKEND in article_data.columns:
            # Séparer weekends et semaine
            weekend_mask = article_data[ColumnNames.IS_WEEKEND]

            ax1.scatter(dates[~weekend_mask], quantities[~weekend_mask],
                       alpha=0.6, color='steelblue', label='Semaine', s=30)
            ax1.scatter(dates[weekend_mask], quantities[weekend_mask],
                       alpha=0.8, color='orange', label='Weekend', s=30)
        else:
            ax1.scatter(dates, quantities, alpha=0.6, color='steelblue', s=30)

        # Ligne connectant les points
        ax1.plot(dates, quantities, alpha=0.3, color='gray', linewidth=0.5)

        # Tendance (moyenne mobile si demandée)
        if show_trend and len(article_data) > 7:
            rolling_col = f"{ColumnNames.QUANTITY}_rolling_mean_7d"
            if rolling_col in article_data.columns:
                ax1.plot(dates, article_data[rolling_col],
                        color='red', linewidth=2, label='Tendance (7 jours)', alpha=0.8)

        ax1.set_title(f'Évolution des Quantités Commandées')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Quantité')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Format des dates sur l'axe X
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax1.xaxis.set_minor_locator(mdates.WeekdayLocator())

        # === GRAPHIQUE DES JOURS DE SEMAINE ===
        if ColumnNames.WEEKDAY_NAME in article_data.columns:
            weekday_means = article_data.groupby(ColumnNames.WEEKDAY_NAME)[ColumnNames.QUANTITY].mean()
            weekday_order = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
            weekday_means = weekday_means.reindex([day for day in weekday_order if day in weekday_means.index])

            colors = ['lightcoral' if day in ['Samedi', 'Dimanche'] else 'lightblue' for day in weekday_means.index]
            bars = ax2.bar(weekday_means.index, weekday_means.values, color=colors, alpha=0.8)

            # Ajouter les valeurs sur les barres
            for bar, value in zip(bars, weekday_means.values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.1f}', ha='center', va='bottom', fontsize=9)

            ax2.set_title('Moyenne des Ventes par Jour de Semaine')
            ax2.set_ylabel('Quantité Moyenne')
            ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # Statistiques pour le log
        total_commandes = quantities.sum()
        jours_avec_commandes = (quantities > 0).sum()
        moyenne_quotidienne = quantities.mean()

        logger.info(f"   📈 Total commandes : {total_commandes}")
        logger.info(f"   📅 Jours avec commandes : {jours_avec_commandes}/{len(quantities)}")
        logger.info(f"   📊 Moyenne quotidienne : {moyenne_quotidienne:.2f}")

        # Sauvegarde si demandée
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"   💾 Graphique sauvegardé : {save_path}")

        return fig

    def plot_weekday_analysis(
        self,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[Path] = None
    ) -> Tuple[plt.Figure, pd.DataFrame]:
        """
        Analyse complète des ventes par jour de semaine pour tous les articles.

        OBJECTIF : Identifier quel jour est le plus fort globalement

        Args:
            figsize: Taille de la figure
            save_path: Chemin de sauvegarde (optionnel)

        Returns:
            Tuple: (Figure matplotlib, DataFrame des statistiques)
        """
        if self.enriched_data is None:
            self.load_enriched_data()

        logger.info("📊 Analyse des ventes par jour de semaine")

        # Calcul des statistiques par jour de semaine
        weekday_stats = self.enriched_data.groupby([ColumnNames.WEEKDAY_NAME, ColumnNames.WEEKDAY])[ColumnNames.QUANTITY].agg([
            'mean', 'median', 'sum', 'std', 'count'
        ]).round(2).reset_index()

        # Réorganisation par ordre des jours
        weekday_stats = weekday_stats.sort_values(ColumnNames.WEEKDAY)

        # Création de la figure avec sous-graphiques
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Analyse Complète des Ventes par Jour de Semaine', fontsize=14, fontweight='bold')

        days = weekday_stats[ColumnNames.WEEKDAY_NAME]
        colors = ['lightcoral' if day in ['Samedi', 'Dimanche'] else 'lightblue' for day in days]

        # 1. Moyenne des ventes
        bars1 = ax1.bar(days, weekday_stats['mean'], color=colors, alpha=0.8)
        ax1.set_title('Quantité Moyenne par Jour')
        ax1.set_ylabel('Quantité Moyenne')
        ax1.tick_params(axis='x', rotation=45)

        # Annotation du jour le plus fort
        max_day_idx = weekday_stats['mean'].idxmax()
        max_day = weekday_stats.iloc[max_day_idx][ColumnNames.WEEKDAY_NAME]
        max_value = weekday_stats.iloc[max_day_idx]['mean']

        ax1.annotate(f'MAX\n{max_value:.1f}',
                    xy=(max_day_idx, max_value),
                    xytext=(max_day_idx, max_value + max_value * 0.1),
                    ha='center', fontweight='bold', color='red',
                    arrowprops=dict(arrowstyle='->', color='red'))

        # 2. Total des ventes
        ax2.bar(days, weekday_stats['sum'], color=colors, alpha=0.8)
        ax2.set_title('Volume Total par Jour')
        ax2.set_ylabel('Volume Total')
        ax2.tick_params(axis='x', rotation=45)

        # 3. Nombre d'observations
        ax3.bar(days, weekday_stats['count'], color=colors, alpha=0.8)
        ax3.set_title('Nombre d\'Observations par Jour')
        ax3.set_ylabel('Nombre de Lignes')
        ax3.tick_params(axis='x', rotation=45)

        # 4. Écart-type (variabilité)
        ax4.bar(days, weekday_stats['std'], color=colors, alpha=0.8)
        ax4.set_title('Variabilité des Ventes (Écart-type)')
        ax4.set_ylabel('Écart-type')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # Résultats dans le log
        logger.info(f"   🏆 Jour le plus fort : {max_day} (moyenne: {max_value:.2f})")
        logger.info(f"   📊 Classement par moyenne :")

        for idx, row in weekday_stats.sort_values('mean', ascending=False).iterrows():
            logger.info(f"      {row[ColumnNames.WEEKDAY_NAME]}: {row['mean']:.2f}")

        # Sauvegarde si demandée
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"   💾 Graphique sauvegardé : {save_path}")

        return fig, weekday_stats

    def plot_weekend_vs_weekday_comparison(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Compare les ventes weekend vs semaine.

        Args:
            figsize: Taille de la figure

        Returns:
            Figure: Graphique de comparaison
        """
        if self.enriched_data is None:
            self.load_enriched_data()

        if ColumnNames.IS_WEEKEND not in self.enriched_data.columns:
            raise ValueError("Colonne 'is_weekend' manquante dans les données")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Comparaison Weekend vs Semaine', fontsize=14, fontweight='bold')

        # Données pour la comparaison
        weekend_data = self.enriched_data[self.enriched_data[ColumnNames.IS_WEEKEND]][ColumnNames.QUANTITY]
        weekday_data = self.enriched_data[~self.enriched_data[ColumnNames.IS_WEEKEND]][ColumnNames.QUANTITY]

        # 1. Moyennes
        categories = ['Semaine', 'Weekend']
        means = [weekday_data.mean(), weekend_data.mean()]
        colors = ['lightblue', 'lightcoral']

        bars = ax1.bar(categories, means, color=colors, alpha=0.8)
        ax1.set_title('Quantité Moyenne')
        ax1.set_ylabel('Quantité Moyenne')

        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means) * 0.02,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

        # 2. Histogrammes de distribution
        ax2.hist(weekday_data, bins=30, alpha=0.6, label='Semaine', color='lightblue', density=True)
        ax2.hist(weekend_data, bins=30, alpha=0.6, label='Weekend', color='lightcoral', density=True)
        ax2.set_title('Distribution des Quantités')
        ax2.set_xlabel('Quantité')
        ax2.set_ylabel('Densité')
        ax2.legend()

        plt.tight_layout()

        # Calcul du ratio
        ratio = weekend_data.mean() / weekday_data.mean()
        logger.info(f"   📊 Ratio Weekend/Semaine : {ratio:.2f}")
        logger.info(f"   📈 {'Les weekends sont plus forts' if ratio > 1 else 'La semaine est plus forte'}")

        return fig

    def show_lag_verification(self, n_rows: int = 10, article_ids: Optional[List[int]] = None) -> None:
        """
        Vérifie et affiche les colonnes de retard (quantité jour précédent).

        OBJECTIF : Validation du calcul de 'quantité_jour_prec'

        Args:
            n_rows: Nombre de lignes à afficher
            article_ids: Liste d'articles spécifiques (optionnel)
        """
        if self.enriched_data is None:
            self.load_enriched_data()

        if ColumnNames.QUANTITY_PREV_DAY not in self.enriched_data.columns:
            logger.error("❌ Colonne 'quantité_jour_prec' non trouvée dans les données")
            return

        logger.info(f"🔍 Vérification des variables de retard (premières {n_rows} lignes)")

        # Sélection des données à afficher
        display_data = self.enriched_data.copy()

        if article_ids:
            display_data = display_data[display_data[ColumnNames.ARTICLE_ID].isin(article_ids)]

        # Tri par article puis par date
        display_data = display_data.sort_values([ColumnNames.ARTICLE_ID, ColumnNames.DATE])

        # Colonnes d'intérêt
        columns_to_show = [
            ColumnNames.DATE,
            ColumnNames.ARTICLE_ID,
            ColumnNames.QUANTITY,
            ColumnNames.QUANTITY_PREV_DAY,
            ColumnNames.WEEKDAY_NAME
        ]

        # Ajout des autres colonnes lag si elles existent
        lag_columns = [col for col in display_data.columns if 'lag_' in col and col != ColumnNames.QUANTITY_PREV_DAY]
        columns_to_show.extend(lag_columns[:2])  # Limite à 2 colonnes lag supplémentaires

        preview_data = display_data[columns_to_show].head(n_rows)

        print("\n" + "="*100)
        print(f"📋 VÉRIFICATION DES VARIABLES DE RETARD (lag features)")
        print("="*100)
        print(preview_data.to_string(index=False))
        print("="*100)

        # Vérification manuelle pour quelques lignes
        print("\n🔍 Vérification manuelle (Article + jour suivant) :")

        for article_id in display_data[ColumnNames.ARTICLE_ID].unique()[:2]:  # 2 premiers articles
            article_subset = display_data[display_data[ColumnNames.ARTICLE_ID] == article_id].head(3)
            print(f"\n   📦 Article {article_id} :")

            for i, (_, row) in enumerate(article_subset.iterrows()):
                if i > 0:  # Pas la première ligne
                    prev_qty = article_subset.iloc[i-1][ColumnNames.QUANTITY]
                    current_lag = row[ColumnNames.QUANTITY_PREV_DAY]
                    status = "✅" if prev_qty == current_lag else "❌"
                    print(f"     {status} {row[ColumnNames.DATE].date()}: qty={row[ColumnNames.QUANTITY]}, qty_prev={current_lag} (attendu: {prev_qty})")

        logger.info("✅ Vérification des variables de retard terminée")


# ===== FONCTIONS UTILITAIRES =====

def create_article_dashboard(
    enriched_data: pd.DataFrame,
    article_id: int,
    output_dir: Optional[Path] = None
) -> Dict[str, plt.Figure]:
    """
    Crée un dashboard complet pour un article spécifique.

    Args:
        enriched_data: Données enrichies
        article_id: ID de l'article
        output_dir: Dossier de sortie pour les graphiques

    Returns:
        dict: Dictionnaire des figures générées
    """
    viz = DataVisualization(enriched_data)
    figures = {}

    try:
        # Graphique principal des ventes quotidiennes
        figures['daily_sales'] = viz.plot_daily_sales_by_article(
            article_id=article_id,
            save_path=output_dir / f"article_{article_id}_daily_sales.png" if output_dir else None
        )

        logger.info(f"📊 Dashboard créé pour l'article {article_id}")

    except Exception as e:
        logger.error(f"❌ Erreur lors de la création du dashboard pour l'article {article_id}: {e}")

    return figures


def create_global_analysis(
    enriched_data: pd.DataFrame,
    output_dir: Optional[Path] = None
) -> Dict[str, plt.Figure]:
    """
    Crée une analyse globale de tous les articles.

    Args:
        enriched_data: Données enrichies
        output_dir: Dossier de sortie pour les graphiques

    Returns:
        dict: Dictionnaire des figures et statistiques
    """
    viz = DataVisualization(enriched_data)
    results = {'figures': {}, 'stats': {}}

    try:
        # Analyse par jour de semaine
        fig_weekday, stats_weekday = viz.plot_weekday_analysis(
            save_path=output_dir / "weekday_analysis.png" if output_dir else None
        )
        results['figures']['weekday_analysis'] = fig_weekday
        results['stats']['weekday_stats'] = stats_weekday

        # Comparaison weekend vs semaine
        results['figures']['weekend_comparison'] = viz.plot_weekend_vs_weekday_comparison()

        logger.info("📊 Analyse globale terminée avec succès")

    except Exception as e:
        logger.error(f"❌ Erreur lors de l'analyse globale: {e}")

    return results


if __name__ == "__main__":
    # Test des visualisations si exécuté directement
    try:
        viz = DataVisualization()
        viz.load_enriched_data()

        # Test de vérification des lags
        viz.show_lag_verification(n_rows=15)

        # Test d'analyse globale
        fig_weekday, stats = viz.plot_weekday_analysis()
        plt.show()

    except Exception as e:
        print(f"Erreur lors du test : {e}")
        print("Assurez-vous que les données enrichies existent dans data/processed/")