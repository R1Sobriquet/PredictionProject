"""
Modèle CatBoost pour la prévision de montants de commandes ATM.

Target : amount (DC_Montant_Cmd)
Entité : atm_id (DC_Automate_Id)

Features :
- Horizon de prédiction
- Statistiques historiques par ATM
- Variables temporelles de la date cible
- Features ATM (soldes, cassettes, volatilité, etc.)
- Historique de rechargement (dernières commandes)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import warnings
import pickle

try:
    from catboost import CatBoostRegressor, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost non installé. Installez-le avec: pip install catboost")

from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from ..utils import (
        ColumnNames,
        Messages,
        WEEKDAY_NAMES,
        DMQ_BY_COUPURE,
        get_file_path,
    )
    from ..utils.holidays import is_french_holiday, is_eve_of_holiday, is_payday
    from .baseline import BaselineModel
except ImportError:
    from src.utils import (
        ColumnNames,
        Messages,
        WEEKDAY_NAMES,
        DMQ_BY_COUPURE,
        get_file_path,
    )
    from src.utils.holidays import is_french_holiday, is_eve_of_holiday, is_payday
    from src.models.baseline import BaselineModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CatBoostForecaster(BaselineModel):
    """
    Modèle CatBoost pour la prévision de montants de commandes ATM.

    Caractéristiques :
    - Horizon de prédiction paramétrable (1-90 jours)
    - Feature engineering adaptatif court terme / long terme
    - atm_id traité comme feature catégorielle native
    - Early stopping
    """

    SHORT_TERM_THRESHOLD = 14

    # Presets prêts à l'emploi. Passer ``preset=`` au constructeur surcharge
    # iterations/learning_rate/depth ; les autres kwargs restent actifs.
    PRESETS: Dict[str, Dict[str, Any]] = {
        'fast':    {'iterations': 1500, 'learning_rate': 0.05, 'depth': 6},
        'default': {'iterations': 4000, 'learning_rate': 0.03, 'depth': 6},
        'deep':    {'iterations': 6000, 'learning_rate': 0.02, 'depth': 8},
    }

    def __init__(
        self,
        max_horizon: int = 90,
        iterations: int = 4000,
        learning_rate: float = 0.03,
        depth: int = 6,
        early_stopping_rounds: int = 100,
        random_state: int = 42,
        verbose: bool = False,
        l2_leaf_reg: float = 3.0,
        subsample: float = 0.85,
        rsm: float = 0.85,
        loss_function: str = 'MAE',
        bootstrap_type: str = 'Bernoulli',
        preset: Optional[str] = None,
    ):
        super().__init__(f"CatBoost_H{max_horizon}")

        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost n'est pas installé.")

        # Application du preset si fourni
        if preset is not None:
            if preset not in self.PRESETS:
                raise ValueError(
                    f"Preset inconnu '{preset}'. Choix : {list(self.PRESETS)}"
                )
            cfg = self.PRESETS[preset]
            iterations = cfg['iterations']
            learning_rate = cfg['learning_rate']
            depth = cfg['depth']
            logger.info(f"CatBoost preset '{preset}' appliqué : {cfg}")

        self.max_horizon = max_horizon
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.verbose = verbose

        self.model = None
        self.feature_columns = []
        self.cat_features = []
        self.atm_historical_stats = {}
        self.global_stats = {}

        # Colonne cible : ``amount`` pour le modèle de base ; surchargée à
        # ``dmq_{coupure}`` par CatBoostDmqForecaster. Utilisée par le chemin
        # vectorisé (_prepare_features_for_training / predict_batch).
        self.target_column = ColumnNames.AMOUNT

        # Hyperparamètres tunés :
        # - learning_rate 0.03 (vs 0.85) : convergence beaucoup plus stable
        # - loss_function MAE : robuste aux outliers (queues épaisses des montants)
        # - l2_leaf_reg + subsample/rsm : régularisation + bagging Bernoulli
        self.model_params = {
            'iterations': iterations,
            'learning_rate': learning_rate,
            'depth': depth,
            'loss_function': loss_function,
            'l2_leaf_reg': l2_leaf_reg,
            'subsample': subsample,
            'rsm': rsm,
            'bootstrap_type': bootstrap_type,
            'random_seed': random_state,
            'verbose': verbose,
            'early_stopping_rounds': early_stopping_rounds,
            'use_best_model': True,
        }

    def _compute_historical_stats(self, data: pd.DataFrame) -> None:
        """Calcule les statistiques historiques par ATM."""
        logger.info("Calcul des statistiques historiques par ATM...")

        self.global_stats = {
            'mean': data[ColumnNames.AMOUNT].mean(),
            'std': data[ColumnNames.AMOUNT].std(),
            'median': data[ColumnNames.AMOUNT].median(),
        }

        for atm_id in data[ColumnNames.ATM_ID].unique():
            atm_data = data[data[ColumnNames.ATM_ID] == atm_id].sort_values(ColumnNames.ORDER_DATE)

            weekday_means = {}
            if ColumnNames.WEEKDAY in atm_data.columns:
                weekday_means = atm_data.groupby(ColumnNames.WEEKDAY)[ColumnNames.AMOUNT].mean().to_dict()

            month_means = {}
            if ColumnNames.MONTH in atm_data.columns:
                month_means = atm_data.groupby(ColumnNames.MONTH)[ColumnNames.AMOUNT].mean().to_dict()

            # Fréquence de rechargement
            dates = atm_data[ColumnNames.ORDER_DATE]
            if len(dates) > 1:
                diffs = dates.diff().dt.days.dropna()
                avg_frequency = diffs.mean()
            else:
                avg_frequency = 0

            self.atm_historical_stats[atm_id] = {
                'mean': atm_data[ColumnNames.AMOUNT].mean(),
                'std': atm_data[ColumnNames.AMOUNT].std(),
                'median': atm_data[ColumnNames.AMOUNT].median(),
                'max': atm_data[ColumnNames.AMOUNT].max(),
                'min': atm_data[ColumnNames.AMOUNT].min(),
                'weekday_means': weekday_means,
                'month_means': month_means,
                'total_orders': len(atm_data),
                'avg_frequency': avg_frequency,
            }

        logger.info(f"  Stats calculées pour {len(self.atm_historical_stats)} ATMs")

    def _prepare_features_for_training_loop(self, data: pd.DataFrame) -> pd.DataFrame:
        """Version boucle (référence d'équivalence). Conservée pour valider la
        version vectorisée ``_prepare_features_for_training``."""
        logger.info(f"Préparation des features pour horizons 1 à {self.max_horizon}...")

        all_examples = []
        data = data.sort_values([ColumnNames.ATM_ID, ColumnNames.ORDER_DATE]).reset_index(drop=True)

        for atm_id in data[ColumnNames.ATM_ID].unique():
            atm_data = data[data[ColumnNames.ATM_ID] == atm_id].reset_index(drop=True)
            atm_stats = self.atm_historical_stats.get(atm_id, {})

            for idx in range(len(atm_data) - 1):
                base_row = atm_data.iloc[idx]

                # Horizons à échantillonner
                max_forward = min(len(atm_data) - idx - 1, self.max_horizon)
                horizons = list(range(1, min(15, max_forward + 1)))
                horizons += [h for h in [21, 30, 45, 60, 75, 90] if h <= max_forward]

                for horizon in horizons:
                    target_row = atm_data.iloc[idx + horizon]

                    features = self._build_features(
                        base_row=base_row,
                        target_date=target_row[ColumnNames.ORDER_DATE],
                        horizon=horizon,
                        atm_stats=atm_stats,
                        historical_data=atm_data.iloc[:idx + 1],
                    )
                    features['target'] = target_row[self.target_column]
                    all_examples.append(features)

        training_df = pd.DataFrame(all_examples)
        logger.info(f"  {len(training_df)} exemples d'entraînement générés")
        return training_df

    # ------------------------------------------------------------------
    # Chemin VECTORISÉ (équivalent à la boucle ci-dessus, ~100× plus rapide)
    # ------------------------------------------------------------------

    # Noms des colonnes de lag. Surchargé par CatBoostDmqForecaster.
    LAG_NAMES = {
        'last': 'last_amount', 'prev': 'prev_amount',
        'rm5': 'rolling_mean_5cmd', 'rm10': 'rolling_mean_10cmd',
        'rstd5': 'rolling_std_5cmd', 'trend': 'recent_trend',
        'days': 'days_since_last',
    }

    # Colonnes enrichies reprises telles quelles de la ligne de base.
    _BASE_ENRICHED_COLS = [
        'volatilite_dmq', 'evenement_en_cours', 'risque_atm_vide',
        'total_soldes', 'total_ajustement', 'total_k7hs', 'cassettes_actives',
    ]
    _DMQ_SIGNAL_COLS = ['dmq_trend_7j', 'dmq_trend_28j', 'dmq_debut_mois_ratio']

    def _compute_row_base_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcule, de façon vectorisée, toutes les features dépendant de la
        *ligne de base* (lags court-terme + colonnes enrichies + stats ATM).

        ``data`` doit être trié par ``(atm_id, order_date)`` et ré-indexé.
        Retourne un DataFrame aligné sur l'index de ``data``.
        """
        tc = self.target_column
        atm = data[ColumnNames.ATM_ID]

        # Stats ATM (mappées par atm_id)
        gmean = self.global_stats.get('mean', 0.0)
        gstd = self.global_stats.get('std', 0.0)
        gmed = self.global_stats.get('median', 0.0)
        stats = self.atm_historical_stats
        atm_mean = atm.map({a: s.get('mean', gmean) for a, s in stats.items()}).fillna(gmean)
        atm_std = atm.map({a: s.get('std', gstd) for a, s in stats.items()}).fillna(gstd)
        atm_median = atm.map({a: s.get('median', gmed) for a, s in stats.items()}).fillna(gmed)
        atm_freq = atm.map({a: s.get('avg_frequency', 0) for a, s in stats.items()}).fillna(0)

        # Lags court-terme (fenêtres glissantes sur tc, par ATM)
        g = data.groupby(ColumnNames.ATM_ID, sort=False)[tc]
        st_last = data[tc].astype(float)
        st_prev = g.shift(1)
        st_prev = st_prev.where(st_prev.notna(), atm_mean)
        st_rm5 = (data.groupby(ColumnNames.ATM_ID, sort=False)[tc]
                  .rolling(5, min_periods=1).mean().reset_index(level=0, drop=True))
        st_rm10 = (data.groupby(ColumnNames.ATM_ID, sort=False)[tc]
                   .rolling(10, min_periods=1).mean().reset_index(level=0, drop=True))
        st_rstd5 = (data.groupby(ColumnNames.ATM_ID, sort=False)[tc]
                    .rolling(5, min_periods=2).std().reset_index(level=0, drop=True)).fillna(0.0)
        st_trend = st_rm5 - st_rm10

        if ColumnNames.DAYS_SINCE_LAST_ORDER in data.columns:
            st_days = data[ColumnNames.DAYS_SINCE_LAST_ORDER].astype(float)
        else:
            st_days = pd.Series(0.0, index=data.index)

        bf = pd.DataFrame(index=data.index)
        bf['atm_id'] = atm.values
        bf['atm_mean'] = atm_mean.values
        bf['atm_std'] = atm_std.values
        bf['atm_median'] = atm_median.values
        bf['atm_avg_frequency'] = atm_freq.values

        # Colonnes enrichies + DMQ par coupure + signaux (telles quelles, sinon 0)
        for col in self._BASE_ENRICHED_COLS:
            bf[col] = data[col].values if col in data.columns else 0
        for dmq_col in DMQ_BY_COUPURE.values():
            bf[dmq_col] = data[dmq_col].values if dmq_col in data.columns else 0.0
        for col in self._DMQ_SIGNAL_COLS:
            bf[col] = data[col].values if col in data.columns else 0.0

        # Valeurs lag court-terme (sélection court/long terme faite plus tard)
        bf['_st_last'] = st_last.values
        bf['_st_prev'] = st_prev.values
        bf['_st_rm5'] = st_rm5.values
        bf['_st_rm10'] = st_rm10.values
        bf['_st_rstd5'] = st_rstd5.values
        bf['_st_trend'] = st_trend.values
        bf['_st_days'] = st_days.values
        return bf

    def _weekday_month_mean_lookup(self):
        """Construit deux DataFrames longs (atm_id, weekday/month, moyenne) à
        partir de ``atm_historical_stats`` pour merge vectorisé."""
        wk_rows, mo_rows = [], []
        for atm, s in self.atm_historical_stats.items():
            for wd, v in s.get('weekday_means', {}).items():
                wk_rows.append((atm, int(wd), float(v)))
            for mo, v in s.get('month_means', {}).items():
                mo_rows.append((atm, int(mo), float(v)))
        wk_df = pd.DataFrame(wk_rows, columns=['atm_id', '_wd', 'atm_weekday_mean'])
        mo_df = pd.DataFrame(mo_rows, columns=['atm_id', '_mo', 'atm_month_mean'])
        return wk_df, mo_df

    def _build_matrix(self, bf_rows: pd.DataFrame, target_dates, horizons) -> pd.DataFrame:
        """Assemble la matrice de features finale (sans ``target``) à partir des
        features de base déjà gatherées (``bf_rows``), des dates cibles et des
        horizons. Reproduit exactement l'ordre/les noms de ``_build_features``."""
        n = len(bf_rows)
        horizons = np.asarray(horizons, dtype=float)
        td = pd.DatetimeIndex(pd.to_datetime(target_dates))
        short = horizons <= self.SHORT_TERM_THRESHOLD

        out = {}
        # --- HORIZON ---
        out['horizon'] = horizons
        out['horizon_log'] = np.log1p(horizons)
        out['is_short_term'] = short.astype(int)

        # --- ATM ---
        atm_mean = bf_rows['atm_mean'].to_numpy()
        atm_std = bf_rows['atm_std'].to_numpy()
        atm_freq = bf_rows['atm_avg_frequency'].to_numpy()
        out['atm_id'] = bf_rows['atm_id'].to_numpy()
        out['atm_mean'] = atm_mean
        out['atm_std'] = atm_std
        out['atm_median'] = bf_rows['atm_median'].to_numpy()
        out['atm_avg_frequency'] = atm_freq

        # --- TEMPOREL (date cible) ---
        wd = td.weekday.to_numpy()
        month = td.month.to_numpy()
        doy = td.dayofyear.to_numpy()
        out['target_weekday'] = wd
        out['target_month'] = month
        out['target_day'] = td.day.to_numpy()
        out['target_quarter'] = td.quarter.to_numpy()
        out['target_is_weekend'] = (wd >= 5).astype(int)
        out['target_is_month_start'] = (td.day.to_numpy() <= 5).astype(int)
        out['target_is_month_end'] = (td.day.to_numpy() > 25).astype(int)
        out['target_day_of_year'] = doy
        out['target_weekday_sin'] = np.sin(2 * np.pi * wd / 7)
        out['target_weekday_cos'] = np.cos(2 * np.pi * wd / 7)
        out['target_month_sin'] = np.sin(2 * np.pi * month / 12)
        out['target_month_cos'] = np.cos(2 * np.pi * month / 12)
        out['target_day_of_year_sin'] = np.sin(2 * np.pi * doy / 365.25)
        out['target_day_of_year_cos'] = np.cos(2 * np.pi * doy / 365.25)

        # --- moyennes par jour/mois (lookup par (atm, weekday/month)) ---
        wk_df, mo_df = self._weekday_month_mean_lookup()
        key = pd.DataFrame({'atm_id': out['atm_id'], '_wd': wd, '_mo': month})
        if not wk_df.empty:
            key = key.merge(wk_df, on=['atm_id', '_wd'], how='left')
        else:
            key['atm_weekday_mean'] = np.nan
        if not mo_df.empty:
            key = key.merge(mo_df, on=['atm_id', '_mo'], how='left')
        else:
            key['atm_month_mean'] = np.nan
        out['atm_weekday_mean'] = key['atm_weekday_mean'].fillna(
            pd.Series(atm_mean)).to_numpy()
        out['atm_month_mean'] = key['atm_month_mean'].fillna(
            pd.Series(atm_mean)).to_numpy()

        # --- FEATURES ATM (ligne de base) ---
        for col in self._BASE_ENRICHED_COLS:
            out[col] = bf_rows[col].to_numpy()
        # --- DMQ par coupure ---
        for dmq_col in DMQ_BY_COUPURE.values():
            out[dmq_col] = bf_rows[dmq_col].to_numpy()
        # --- signaux DMQ enrichis ---
        for col in self._DMQ_SIGNAL_COLS:
            out[col] = bf_rows[col].to_numpy()

        # --- CALENDRIER FR (calculé sur dates uniques puis mappé) ---
        uniq = pd.unique(td)
        hol = {d: int(is_french_holiday(pd.Timestamp(d).date())) for d in uniq}
        eve = {d: int(is_eve_of_holiday(pd.Timestamp(d).date())) for d in uniq}
        pay = {d: int(is_payday(pd.Timestamp(d).date())) for d in uniq}
        out['target_is_holiday'] = td.map(hol).to_numpy()
        out['target_is_eve_holiday'] = td.map(eve).to_numpy()
        out['target_is_payday'] = td.map(pay).to_numpy()

        # --- LAGS (sélection court terme / fallback selon horizon) ---
        L = self.LAG_NAMES
        out[L['last']] = np.where(short, bf_rows['_st_last'].to_numpy(), atm_mean)
        out[L['prev']] = np.where(short, bf_rows['_st_prev'].to_numpy(), atm_mean)
        out[L['rm5']] = np.where(short, bf_rows['_st_rm5'].to_numpy(), atm_mean)
        out[L['rm10']] = np.where(short, bf_rows['_st_rm10'].to_numpy(), atm_mean)
        out[L['rstd5']] = np.where(short, bf_rows['_st_rstd5'].to_numpy(), atm_std)
        out[L['trend']] = np.where(short, bf_rows['_st_trend'].to_numpy(), 0.0)
        out[L['days']] = np.where(short, bf_rows['_st_days'].to_numpy(), atm_freq)

        return pd.DataFrame(out, index=np.arange(n))

    def _expand_pairs(self, data: pd.DataFrame):
        """Génère les paires (base_idx, target_idx, horizon) pour ``data`` trié,
        dans le même ordre que la boucle de référence."""
        grp = data.groupby(ColumnNames.ATM_ID, sort=False)
        gpos = grp.cumcount().to_numpy()
        gsize = grp[ColumnNames.ATM_ID].transform('size').to_numpy()
        max_forward = np.minimum(gsize - 1 - gpos, self.max_horizon)
        gidx = np.arange(len(data))

        H_template = [h for h in (list(range(1, 15)) + [21, 30, 45, 60, 75, 90])
                      if h <= self.max_horizon]
        base_list, tgt_list, hor_list = [], [], []
        for h in H_template:
            mask = max_forward >= h
            bi = gidx[mask]
            base_list.append(bi)
            tgt_list.append(bi + h)
            hor_list.append(np.full(len(bi), h))
        if not base_list:
            return (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int))
        base_idx = np.concatenate(base_list)
        target_idx = np.concatenate(tgt_list)
        horizons = np.concatenate(hor_list)
        # Ordre identique à la boucle : par base_idx (atm puis date), puis horizon
        order = np.lexsort((horizons, base_idx))
        return base_idx[order], target_idx[order], horizons[order]

    def _prepare_features_for_training(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prépare les features pour l'entraînement multi-horizon (vectorisé)."""
        logger.info(f"Préparation des features pour horizons 1 à {self.max_horizon}...")
        tc = self.target_column
        if tc not in data.columns:
            raise ValueError(f"Colonne cible '{tc}' absente des données.")

        data = data.sort_values(
            [ColumnNames.ATM_ID, ColumnNames.ORDER_DATE]
        ).reset_index(drop=True)

        bf = self._compute_row_base_frame(data)
        base_idx, target_idx, horizons = self._expand_pairs(data)

        bf_rows = bf.iloc[base_idx].reset_index(drop=True)
        target_dates = data[ColumnNames.ORDER_DATE].to_numpy()[target_idx]
        target_vals = data[tc].to_numpy()[target_idx].astype(float)

        training_df = self._build_matrix(bf_rows, target_dates, horizons)
        training_df['target'] = target_vals
        logger.info(f"  {len(training_df)} exemples d'entraînement générés")
        return training_df

    def predict_batch(
        self,
        atm_ids,
        target_dates,
        context_data: pd.DataFrame,
        horizon: int = 1,
    ) -> np.ndarray:
        """Prédiction vectorisée pour des paires (atm_id, date) en un seul appel.

        La ligne de base de chaque ATM est sa dernière ligne d'historique dans
        ``context_data`` (comme la version scalaire de ``predict`` avec
        ``horizon`` fixe). Tous les ATM/dates sont prédits en un seul
        ``model.predict``.
        """
        if not self.is_fitted:
            raise ValueError(f"Modèle {self.name} non entraîné.")

        ctx = context_data.sort_values(
            [ColumnNames.ATM_ID, ColumnNames.ORDER_DATE]
        ).reset_index(drop=True)
        bf = self._compute_row_base_frame(ctx)
        bf[ColumnNames.ATM_ID] = ctx[ColumnNames.ATM_ID].values
        # Dernière ligne d'historique par ATM
        last_pos = bf.groupby(ColumnNames.ATM_ID, sort=False).tail(1)
        last_by_atm = {a: i for a, i in zip(last_pos[ColumnNames.ATM_ID].values,
                                            last_pos.index.values)}

        atm_ids = np.asarray(atm_ids)
        target_dates = np.asarray(target_dates)
        gmean = self.global_stats.get('mean', 0.0)

        # Index de la ligne de base pour chaque paire (−1 si ATM inconnu)
        rows_idx = np.array([last_by_atm.get(a, -1) for a in atm_ids])
        known = rows_idx >= 0

        preds = np.full(len(atm_ids), max(0.0, gmean), dtype=float)
        if known.any():
            bf_rows = bf.iloc[rows_idx[known]].reset_index(drop=True)
            hor = np.full(known.sum(), horizon)
            X = self._build_matrix(bf_rows, target_dates[known], hor)
            X = X[self.feature_columns]
            p = self.model.predict(X)
            preds[known] = np.clip(p, 0.0, None)
        return preds

    def _build_features(
        self,
        base_row: pd.Series,
        target_date: datetime,
        horizon: int,
        atm_stats: Dict,
        historical_data: pd.DataFrame,
    ) -> Dict:
        """Construit les features pour une prédiction."""
        features = {}

        # === HORIZON ===
        features['horizon'] = horizon
        features['horizon_log'] = np.log1p(horizon)
        features['is_short_term'] = 1 if horizon <= self.SHORT_TERM_THRESHOLD else 0

        # === ATM ===
        features['atm_id'] = base_row[ColumnNames.ATM_ID]
        features['atm_mean'] = atm_stats.get('mean', self.global_stats.get('mean', 0))
        features['atm_std'] = atm_stats.get('std', self.global_stats.get('std', 0))
        features['atm_median'] = atm_stats.get('median', self.global_stats.get('median', 0))
        features['atm_avg_frequency'] = atm_stats.get('avg_frequency', 0)

        # === TEMPOREL (date cible) ===
        target_dt = pd.Timestamp(target_date)
        features['target_weekday'] = target_dt.weekday()
        features['target_month'] = target_dt.month
        features['target_day'] = target_dt.day
        features['target_quarter'] = target_dt.quarter
        features['target_is_weekend'] = 1 if target_dt.weekday() >= 5 else 0
        features['target_is_month_start'] = 1 if target_dt.day <= 5 else 0
        features['target_is_month_end'] = 1 if target_dt.day > 25 else 0
        features['target_day_of_year'] = target_dt.dayofyear

        # Cycliques
        features['target_weekday_sin'] = np.sin(2 * np.pi * target_dt.weekday() / 7)
        features['target_weekday_cos'] = np.cos(2 * np.pi * target_dt.weekday() / 7)
        features['target_month_sin'] = np.sin(2 * np.pi * target_dt.month / 12)
        features['target_month_cos'] = np.cos(2 * np.pi * target_dt.month / 12)
        features['target_day_of_year_sin'] = np.sin(2 * np.pi * target_dt.dayofyear / 365.25)
        features['target_day_of_year_cos'] = np.cos(2 * np.pi * target_dt.dayofyear / 365.25)

        # Moyennes historiques par jour/mois
        weekday_means = atm_stats.get('weekday_means', {})
        features['atm_weekday_mean'] = weekday_means.get(target_dt.weekday(), atm_stats.get('mean', 0))
        month_means = atm_stats.get('month_means', {})
        features['atm_month_mean'] = month_means.get(target_dt.month, atm_stats.get('mean', 0))

        # === FEATURES ATM (de la ligne de base) ===
        for col in ['volatilite_dmq', 'evenement_en_cours', 'risque_atm_vide',
                     'total_soldes', 'total_ajustement', 'total_k7hs', 'cassettes_actives']:
            if col in base_row.index:
                features[col] = base_row[col]
            else:
                features[col] = 0

        # === DMQ PAR COUPURE (colonnes enrichies dmq_5..100) ===
        for coupure, dmq_col in DMQ_BY_COUPURE.items():
            if dmq_col in base_row.index:
                features[dmq_col] = base_row[dmq_col]
            else:
                features[dmq_col] = 0.0

        # === SIGNAUX DMQ ENRICHIS ===
        for col in ['dmq_trend_7j', 'dmq_trend_28j', 'dmq_debut_mois_ratio']:
            if col in base_row.index:
                features[col] = base_row[col]
            else:
                features[col] = 0.0

        # === CALENDRIER FR (date cible) ===
        target_date_py = target_dt.date()
        features['target_is_holiday'] = int(is_french_holiday(target_date_py))
        features['target_is_eve_holiday'] = int(is_eve_of_holiday(target_date_py))
        features['target_is_payday'] = int(is_payday(target_date_py))

        # === FEATURES DE LAG (historique commandes) ===
        if horizon <= self.SHORT_TERM_THRESHOLD and len(historical_data) > 0:
            features['last_amount'] = base_row[ColumnNames.AMOUNT]

            if len(historical_data) >= 2:
                features['prev_amount'] = historical_data.iloc[-2][ColumnNames.AMOUNT]
            else:
                features['prev_amount'] = atm_stats.get('mean', 0)

            # Moyenne des 5 dernières commandes
            tail5 = historical_data.tail(5)[ColumnNames.AMOUNT]
            features['rolling_mean_5cmd'] = tail5.mean()

            # Moyenne des 10 dernières commandes
            tail10 = historical_data.tail(10)[ColumnNames.AMOUNT]
            features['rolling_mean_10cmd'] = tail10.mean()

            # Écart-type récent
            features['rolling_std_5cmd'] = tail5.std() if len(tail5) > 1 else 0

            # Tendance
            features['recent_trend'] = features['rolling_mean_5cmd'] - features['rolling_mean_10cmd']

            # Jours depuis dernière commande
            if ColumnNames.DAYS_SINCE_LAST_ORDER in base_row.index:
                features['days_since_last'] = base_row[ColumnNames.DAYS_SINCE_LAST_ORDER]
            else:
                features['days_since_last'] = 0
        else:
            mean_val = atm_stats.get('mean', 0)
            features['last_amount'] = mean_val
            features['prev_amount'] = mean_val
            features['rolling_mean_5cmd'] = mean_val
            features['rolling_mean_10cmd'] = mean_val
            features['rolling_std_5cmd'] = atm_stats.get('std', 0)
            features['recent_trend'] = 0
            features['days_since_last'] = atm_stats.get('avg_frequency', 0)

        return features

    def fit(self, data: pd.DataFrame, eval_data: Optional[pd.DataFrame] = None) -> 'CatBoostForecaster':
        """Entraîne le modèle CatBoost."""
        logger.info(f"Entraînement du modèle {self.name}")
        logger.info(f"  Données d'entrée : {len(data)} lignes")

        self.training_data = data.copy()

        # 1. Stats historiques
        self._compute_historical_stats(data)

        # 2. Features
        training_df = self._prepare_features_for_training(data)

        # 3. Séparer features / target
        self.feature_columns = [col for col in training_df.columns if col != 'target']
        X_train = training_df[self.feature_columns]
        y_train = training_df['target']

        # 4. Features catégorielles
        self.cat_features = ['atm_id']
        cat_feature_indices = [self.feature_columns.index(f) for f in self.cat_features if f in self.feature_columns]

        logger.info(f"  Features : {len(self.feature_columns)} colonnes")

        # 5. Validation split
        if eval_data is not None:
            eval_df = self._prepare_features_for_training(eval_data)
            X_eval = eval_df[self.feature_columns]
            y_eval = eval_df['target']
        else:
            split_idx = int(len(X_train) * 0.8)
            X_eval = X_train.iloc[split_idx:]
            y_eval = y_train.iloc[split_idx:]
            X_train = X_train.iloc[:split_idx]
            y_train = y_train.iloc[:split_idx]

        eval_set = Pool(X_eval, y_eval, cat_features=cat_feature_indices)
        train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)

        # 6. Entraînement
        self.model = CatBoostRegressor(**self.model_params)
        logger.info("  Début de l'entraînement...")
        self.model.fit(train_pool, eval_set=eval_set, verbose=self.verbose)
        self.is_fitted = True

        best_iter = self.model.get_best_iteration()
        logger.info(f"  Entraînement terminé (meilleure itération : {best_iter})")

        # Feature importance
        importance = self.model.get_feature_importance()
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance,
        }).sort_values('importance', ascending=False)

        logger.info("  Top 5 features :")
        for _, row in importance_df.head(5).iterrows():
            logger.info(f"    - {row['feature']}: {row['importance']:.2f}")

        return self

    def predict(
        self,
        atm_id: int,
        prediction_dates: List[datetime],
        context_data: Optional[pd.DataFrame] = None,
        horizon: Optional[int] = None,
    ) -> np.ndarray:
        """Génère des prédictions pour un ATM sur des dates données."""
        if not self.is_fitted:
            raise ValueError(f"Modèle {self.name} non entraîné.")

        atm_stats = self.atm_historical_stats.get(atm_id, self.global_stats)

        # Récupérer l'historique
        if context_data is not None:
            atm_history = context_data[context_data[ColumnNames.ATM_ID] == atm_id].copy()
            atm_history = atm_history.sort_values(ColumnNames.ORDER_DATE)
        else:
            atm_history = self.training_data[self.training_data[ColumnNames.ATM_ID] == atm_id].copy()
            atm_history = atm_history.sort_values(ColumnNames.ORDER_DATE)

        base_row = atm_history.iloc[-1] if len(atm_history) > 0 else None
        base_date = base_row[ColumnNames.ORDER_DATE] if base_row is not None else None

        predictions = []
        for pred_date in prediction_dates:
            if horizon is not None:
                h = horizon
            elif base_date is not None:
                h = max(1, (pd.Timestamp(pred_date) - pd.Timestamp(base_date)).days)
            else:
                h = 1
            h = min(h, self.max_horizon)

            if base_row is not None:
                features = self._build_features(
                    base_row=base_row,
                    target_date=pred_date,
                    horizon=h,
                    atm_stats=atm_stats,
                    historical_data=atm_history,
                )
            else:
                features = self._build_default_features(atm_id, pred_date, h, atm_stats)

            feature_values = [features.get(col, 0) for col in self.feature_columns]
            X_pred = pd.DataFrame([feature_values], columns=self.feature_columns)
            pred = max(0, self.model.predict(X_pred)[0])
            predictions.append(pred)

        return np.array(predictions)

    def _build_default_features(self, atm_id, target_date, horizon, atm_stats):
        """Features par défaut quand il n'y a pas d'historique."""
        features = {}

        features['horizon'] = horizon
        features['horizon_log'] = np.log1p(horizon)
        features['is_short_term'] = 1 if horizon <= self.SHORT_TERM_THRESHOLD else 0

        features['atm_id'] = atm_id
        features['atm_mean'] = atm_stats.get('mean', self.global_stats.get('mean', 0))
        features['atm_std'] = atm_stats.get('std', self.global_stats.get('std', 0))
        features['atm_median'] = atm_stats.get('median', self.global_stats.get('median', 0))
        features['atm_avg_frequency'] = atm_stats.get('avg_frequency', 0)

        target_dt = pd.Timestamp(target_date)
        features['target_weekday'] = target_dt.weekday()
        features['target_month'] = target_dt.month
        features['target_day'] = target_dt.day
        features['target_quarter'] = target_dt.quarter
        features['target_is_weekend'] = 1 if target_dt.weekday() >= 5 else 0
        features['target_is_month_start'] = 1 if target_dt.day <= 5 else 0
        features['target_is_month_end'] = 1 if target_dt.day > 25 else 0
        features['target_day_of_year'] = target_dt.dayofyear

        features['target_weekday_sin'] = np.sin(2 * np.pi * target_dt.weekday() / 7)
        features['target_weekday_cos'] = np.cos(2 * np.pi * target_dt.weekday() / 7)
        features['target_month_sin'] = np.sin(2 * np.pi * target_dt.month / 12)
        features['target_month_cos'] = np.cos(2 * np.pi * target_dt.month / 12)
        features['target_day_of_year_sin'] = np.sin(2 * np.pi * target_dt.dayofyear / 365.25)
        features['target_day_of_year_cos'] = np.cos(2 * np.pi * target_dt.dayofyear / 365.25)

        weekday_means = atm_stats.get('weekday_means', {})
        features['atm_weekday_mean'] = weekday_means.get(target_dt.weekday(), atm_stats.get('mean', 0))
        month_means = atm_stats.get('month_means', {})
        features['atm_month_mean'] = month_means.get(target_dt.month, atm_stats.get('mean', 0))

        for col in ['volatilite_dmq', 'evenement_en_cours', 'risque_atm_vide',
                     'total_soldes', 'total_ajustement', 'total_k7hs', 'cassettes_actives']:
            features[col] = 0

        mean_val = atm_stats.get('mean', 0)
        features['last_amount'] = mean_val
        features['prev_amount'] = mean_val
        features['rolling_mean_5cmd'] = mean_val
        features['rolling_mean_10cmd'] = mean_val
        features['rolling_std_5cmd'] = atm_stats.get('std', 0)
        features['recent_trend'] = 0
        features['days_since_last'] = atm_stats.get('avg_frequency', 0)

        return features

    def predict_horizon(
        self,
        atm_id: int,
        start_date: datetime,
        horizon_days: int,
        context_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Prédit pour un ATM sur un horizon donné."""
        if horizon_days > self.max_horizon:
            logger.warning(f"Horizon demandé ({horizon_days}j) > max ({self.max_horizon}j)")
            horizon_days = self.max_horizon

        prediction_dates = pd.date_range(start=start_date, periods=horizon_days, freq='D')

        predictions = []
        for i, pred_date in enumerate(prediction_dates):
            pred = self.predict(atm_id, [pred_date], context_data, horizon=i + 1)[0]
            predictions.append({
                ColumnNames.ORDER_DATE: pred_date,
                ColumnNames.ATM_ID: atm_id,
                'prediction': pred,
                'horizon': i + 1,
            })

        return pd.DataFrame(predictions)

    def evaluate_by_horizon(
        self,
        test_data: pd.DataFrame,
        horizons: List[int] = [1, 7, 14, 30, 60, 90],
    ) -> pd.DataFrame:
        """Évalue le modèle par horizon de prédiction."""
        logger.info("Évaluation par horizon...")
        results = []

        test_data_sorted = test_data.sort_values([ColumnNames.ATM_ID, ColumnNames.ORDER_DATE])

        for horizon in horizons:
            if horizon > self.max_horizon:
                continue

            predictions = []
            actuals = []

            for atm_id in test_data[ColumnNames.ATM_ID].unique():
                atm_test = test_data_sorted[test_data_sorted[ColumnNames.ATM_ID] == atm_id]
                if len(atm_test) <= horizon:
                    continue

                for i in range(len(atm_test) - horizon):
                    base_date = atm_test.iloc[i][ColumnNames.ORDER_DATE]
                    target_date = atm_test.iloc[i + horizon][ColumnNames.ORDER_DATE]
                    actual = atm_test.iloc[i + horizon][ColumnNames.AMOUNT]

                    context = test_data[
                        (test_data[ColumnNames.ATM_ID] == atm_id) &
                        (test_data[ColumnNames.ORDER_DATE] <= base_date)
                    ]

                    pred = self.predict(atm_id, [target_date], context, horizon=horizon)[0]
                    predictions.append(pred)
                    actuals.append(actual)

            if predictions:
                predictions = np.array(predictions)
                actuals = np.array(actuals)

                mae = mean_absolute_error(actuals, predictions)
                rmse = np.sqrt(mean_squared_error(actuals, predictions))

                non_zero_mask = actuals != 0
                mape = (
                    np.mean(np.abs((actuals[non_zero_mask] - predictions[non_zero_mask]) / actuals[non_zero_mask])) * 100
                    if non_zero_mask.any() else np.inf
                )

                results.append({
                    'horizon': horizon, 'mae': mae, 'rmse': rmse,
                    'mape': mape, 'n_samples': len(predictions),
                })
                logger.info(f"  H+{horizon:2d}: MAE={mae:.2f}, RMSE={rmse:.2f}, "
                            f"MAPE={mape:.1f}%, n={len(predictions)}")

        return pd.DataFrame(results)

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Modèle non entraîné")
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.get_feature_importance(),
        }).sort_values('importance', ascending=False)

    def save_model(self, path: Union[str, Path]) -> None:
        if not self.is_fitted:
            raise ValueError("Modèle non entraîné")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save_model(str(path / 'catboost_model.cbm'))

        metadata = {
            'name': self.name,
            'max_horizon': self.max_horizon,
            'feature_columns': self.feature_columns,
            'cat_features': self.cat_features,
            'atm_historical_stats': self.atm_historical_stats,
            'global_stats': self.global_stats,
            'model_params': self.model_params,
        }
        with open(path / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Modèle sauvegardé : {path}")

    @classmethod
    def load_model(cls, path: Union[str, Path]) -> 'CatBoostForecaster':
        path = Path(path)

        with open(path / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        instance = cls(max_horizon=metadata['max_horizon'])
        instance.name = metadata['name']
        instance.feature_columns = metadata['feature_columns']
        instance.cat_features = metadata['cat_features']
        instance.atm_historical_stats = metadata['atm_historical_stats']
        instance.global_stats = metadata['global_stats']
        instance.model_params = metadata['model_params']

        instance.model = CatBoostRegressor()
        instance.model.load_model(str(path / 'catboost_model.cbm'))
        instance.is_fitted = True

        logger.info(f"Modèle chargé : {path}")
        return instance


# =============================================================================
#  CatBoost DMQ par coupure
# =============================================================================


class CatBoostDmqForecaster(CatBoostForecaster):
    """Variante de ``CatBoostForecaster`` qui prédit le **DMQ d'une coupure**
    donnée au lieu du montant total ``amount``.

    Surcharge les stats historiques et les features de lag pour utiliser la
    colonne cible (``dmq_{coupure}``) au lieu du montant global.

    On instancie typiquement 5 modèles (un par coupure) via
    :class:`MultiCoupureForecaster`.
    """

    # Noms des colonnes de lag spécifiques au DMQ (cf. _build_features).
    LAG_NAMES = {
        'last': 'last_dmq', 'prev': 'prev_dmq',
        'rm5': 'rolling_mean_5', 'rm10': 'rolling_mean_10',
        'rstd5': 'rolling_std_5', 'trend': 'recent_trend',
        'days': 'days_since_last',
    }

    def __init__(self, coupure: int, target_column: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.coupure = int(coupure)
        self.target_column = target_column or f"dmq_{self.coupure}"
        self.name = f"CatBoostDMQ_{self.coupure}_H{self.max_horizon}"

    def _compute_historical_stats(self, data: pd.DataFrame) -> None:
        """Stats historiques basées sur le DMQ de la coupure, pas sur amount."""
        logger.info("Calcul des statistiques historiques par ATM...")

        tc = self.target_column
        if tc not in data.columns:
            super()._compute_historical_stats(data)
            return

        self.global_stats = {
            'mean': float(data[tc].mean()),
            'std': float(data[tc].std()),
            'median': float(data[tc].median()),
        }

        for atm_id in data[ColumnNames.ATM_ID].unique():
            atm_data = data[data[ColumnNames.ATM_ID] == atm_id].sort_values(
                ColumnNames.ORDER_DATE
            )
            vals = atm_data[tc]

            weekday_means = {}
            if ColumnNames.WEEKDAY in atm_data.columns:
                weekday_means = (
                    atm_data.groupby(ColumnNames.WEEKDAY)[tc].mean().to_dict()
                )

            month_means = {}
            if ColumnNames.MONTH in atm_data.columns:
                month_means = (
                    atm_data.groupby(ColumnNames.MONTH)[tc].mean().to_dict()
                )

            dates = atm_data[ColumnNames.ORDER_DATE]
            if len(dates) > 1:
                avg_frequency = dates.diff().dt.days.dropna().mean()
            else:
                avg_frequency = 0

            self.atm_historical_stats[atm_id] = {
                'mean': float(vals.mean()),
                'std': float(vals.std()) if len(vals) > 1 else 0.0,
                'median': float(vals.median()),
                'max': float(vals.max()),
                'min': float(vals.min()),
                'weekday_means': weekday_means,
                'month_means': month_means,
                'total_orders': len(atm_data),
                'avg_frequency': avg_frequency,
            }

        logger.info(f"  Stats calculées pour {len(self.atm_historical_stats)} ATMs")

    def _build_features(
        self,
        base_row: pd.Series,
        target_date: datetime,
        horizon: int,
        atm_stats: Dict,
        historical_data: pd.DataFrame,
    ) -> Dict:
        """Features alignées sur le DMQ de la coupure cible."""
        features = {}
        tc = self.target_column

        # === HORIZON ===
        features['horizon'] = horizon
        features['horizon_log'] = np.log1p(horizon)
        features['is_short_term'] = 1 if horizon <= self.SHORT_TERM_THRESHOLD else 0

        # === ATM ===
        features['atm_id'] = base_row[ColumnNames.ATM_ID]
        features['atm_mean'] = atm_stats.get('mean', self.global_stats.get('mean', 0))
        features['atm_std'] = atm_stats.get('std', self.global_stats.get('std', 0))
        features['atm_median'] = atm_stats.get('median', self.global_stats.get('median', 0))
        features['atm_avg_frequency'] = atm_stats.get('avg_frequency', 0)

        # === TEMPOREL (date cible) ===
        target_dt = pd.Timestamp(target_date)
        features['target_weekday'] = target_dt.weekday()
        features['target_month'] = target_dt.month
        features['target_day'] = target_dt.day
        features['target_quarter'] = target_dt.quarter
        features['target_is_weekend'] = 1 if target_dt.weekday() >= 5 else 0
        features['target_is_month_start'] = 1 if target_dt.day <= 5 else 0
        features['target_is_month_end'] = 1 if target_dt.day > 25 else 0
        features['target_day_of_year'] = target_dt.dayofyear

        features['target_weekday_sin'] = np.sin(2 * np.pi * target_dt.weekday() / 7)
        features['target_weekday_cos'] = np.cos(2 * np.pi * target_dt.weekday() / 7)
        features['target_month_sin'] = np.sin(2 * np.pi * target_dt.month / 12)
        features['target_month_cos'] = np.cos(2 * np.pi * target_dt.month / 12)
        features['target_day_of_year_sin'] = np.sin(2 * np.pi * target_dt.dayofyear / 365.25)
        features['target_day_of_year_cos'] = np.cos(2 * np.pi * target_dt.dayofyear / 365.25)

        # Moyennes historiques par jour/mois (basées sur le DMQ de la coupure)
        weekday_means = atm_stats.get('weekday_means', {})
        features['atm_weekday_mean'] = weekday_means.get(
            target_dt.weekday(), atm_stats.get('mean', 0)
        )
        month_means = atm_stats.get('month_means', {})
        features['atm_month_mean'] = month_means.get(
            target_dt.month, atm_stats.get('mean', 0)
        )

        # === FEATURES ATM (de la ligne de base) ===
        for col in ['volatilite_dmq', 'evenement_en_cours', 'risque_atm_vide',
                     'total_soldes', 'total_ajustement', 'total_k7hs', 'cassettes_actives']:
            if col in base_row.index:
                features[col] = base_row[col]
            else:
                features[col] = 0

        # === DMQ PAR COUPURE (colonnes enrichies dmq_5..100) ===
        for coupure, dmq_col in DMQ_BY_COUPURE.items():
            if dmq_col in base_row.index:
                features[dmq_col] = base_row[dmq_col]
            else:
                features[dmq_col] = 0.0

        # === SIGNAUX DMQ ENRICHIS ===
        for col in ['dmq_trend_7j', 'dmq_trend_28j', 'dmq_debut_mois_ratio']:
            if col in base_row.index:
                features[col] = base_row[col]
            else:
                features[col] = 0.0

        # === CALENDRIER FR (date cible) ===
        target_date_py = target_dt.date()
        features['target_is_holiday'] = int(is_french_holiday(target_date_py))
        features['target_is_eve_holiday'] = int(is_eve_of_holiday(target_date_py))
        features['target_is_payday'] = int(is_payday(target_date_py))

        # === FEATURES DE LAG (basées sur le DMQ de la coupure, pas amount) ===
        tc_available = tc in base_row.index
        if horizon <= self.SHORT_TERM_THRESHOLD and len(historical_data) > 0 and tc_available:
            features['last_dmq'] = float(base_row[tc])

            if len(historical_data) >= 2 and tc in historical_data.columns:
                features['prev_dmq'] = float(historical_data.iloc[-2][tc])
            else:
                features['prev_dmq'] = atm_stats.get('mean', 0)

            if tc in historical_data.columns:
                tail5 = historical_data.tail(5)[tc]
                tail10 = historical_data.tail(10)[tc]
            else:
                tail5 = pd.Series([atm_stats.get('mean', 0)])
                tail10 = tail5

            features['rolling_mean_5'] = float(tail5.mean())
            features['rolling_mean_10'] = float(tail10.mean())
            features['rolling_std_5'] = float(tail5.std()) if len(tail5) > 1 else 0.0
            features['recent_trend'] = features['rolling_mean_5'] - features['rolling_mean_10']

            if ColumnNames.DAYS_SINCE_LAST_ORDER in base_row.index:
                features['days_since_last'] = base_row[ColumnNames.DAYS_SINCE_LAST_ORDER]
            else:
                features['days_since_last'] = 0
        else:
            mean_val = atm_stats.get('mean', 0)
            features['last_dmq'] = mean_val
            features['prev_dmq'] = mean_val
            features['rolling_mean_5'] = mean_val
            features['rolling_mean_10'] = mean_val
            features['rolling_std_5'] = atm_stats.get('std', 0)
            features['recent_trend'] = 0
            features['days_since_last'] = atm_stats.get('avg_frequency', 0)

        return features

    def _build_default_features(self, atm_id, target_date, horizon, atm_stats):
        """Features par défaut quand il n'y a pas d'historique (DMQ version)."""
        features = {}
        features['horizon'] = horizon
        features['horizon_log'] = np.log1p(horizon)
        features['is_short_term'] = 1 if horizon <= self.SHORT_TERM_THRESHOLD else 0

        features['atm_id'] = atm_id
        features['atm_mean'] = atm_stats.get('mean', self.global_stats.get('mean', 0))
        features['atm_std'] = atm_stats.get('std', self.global_stats.get('std', 0))
        features['atm_median'] = atm_stats.get('median', self.global_stats.get('median', 0))
        features['atm_avg_frequency'] = atm_stats.get('avg_frequency', 0)

        target_dt = pd.Timestamp(target_date)
        features['target_weekday'] = target_dt.weekday()
        features['target_month'] = target_dt.month
        features['target_day'] = target_dt.day
        features['target_quarter'] = target_dt.quarter
        features['target_is_weekend'] = 1 if target_dt.weekday() >= 5 else 0
        features['target_is_month_start'] = 1 if target_dt.day <= 5 else 0
        features['target_is_month_end'] = 1 if target_dt.day > 25 else 0
        features['target_day_of_year'] = target_dt.dayofyear

        features['target_weekday_sin'] = np.sin(2 * np.pi * target_dt.weekday() / 7)
        features['target_weekday_cos'] = np.cos(2 * np.pi * target_dt.weekday() / 7)
        features['target_month_sin'] = np.sin(2 * np.pi * target_dt.month / 12)
        features['target_month_cos'] = np.cos(2 * np.pi * target_dt.month / 12)
        features['target_day_of_year_sin'] = np.sin(2 * np.pi * target_dt.dayofyear / 365.25)
        features['target_day_of_year_cos'] = np.cos(2 * np.pi * target_dt.dayofyear / 365.25)

        weekday_means = atm_stats.get('weekday_means', {})
        features['atm_weekday_mean'] = weekday_means.get(
            target_dt.weekday(), atm_stats.get('mean', 0)
        )
        month_means = atm_stats.get('month_means', {})
        features['atm_month_mean'] = month_means.get(
            target_dt.month, atm_stats.get('mean', 0)
        )

        for col in ['volatilite_dmq', 'evenement_en_cours', 'risque_atm_vide',
                     'total_soldes', 'total_ajustement', 'total_k7hs', 'cassettes_actives']:
            features[col] = 0

        for dmq_col in DMQ_BY_COUPURE.values():
            features[dmq_col] = 0.0
        for col in ['dmq_trend_7j', 'dmq_trend_28j', 'dmq_debut_mois_ratio']:
            features[col] = 0.0

        features['target_is_holiday'] = 0
        features['target_is_eve_holiday'] = 0
        features['target_is_payday'] = 0

        mean_val = atm_stats.get('mean', 0)
        features['last_dmq'] = mean_val
        features['prev_dmq'] = mean_val
        features['rolling_mean_5'] = mean_val
        features['rolling_mean_10'] = mean_val
        features['rolling_std_5'] = atm_stats.get('std', 0)
        features['recent_trend'] = 0
        features['days_since_last'] = atm_stats.get('avg_frequency', 0)

        return features

    def _prepare_features_for_training_loop(self, data: pd.DataFrame) -> pd.DataFrame:
        """Version boucle DMQ (référence d'équivalence). Le chemin de production
        utilise la version vectorisée héritée de ``CatBoostForecaster``."""
        logger.info(
            f"[{self.name}] Préparation features pour la coupure {self.coupure}€"
        )

        if self.target_column not in data.columns:
            raise ValueError(
                f"Colonne cible '{self.target_column}' absente des données. "
                f"Colonnes disponibles : {list(data.columns)[:10]}..."
            )

        all_examples = []
        data = data.sort_values([ColumnNames.ATM_ID, ColumnNames.ORDER_DATE]).reset_index(drop=True)

        for atm_id in data[ColumnNames.ATM_ID].unique():
            atm_data = data[data[ColumnNames.ATM_ID] == atm_id].reset_index(drop=True)
            atm_stats = self.atm_historical_stats.get(atm_id, {})

            for idx in range(len(atm_data) - 1):
                base_row = atm_data.iloc[idx]

                max_forward = min(len(atm_data) - idx - 1, self.max_horizon)
                horizons = list(range(1, min(15, max_forward + 1)))
                horizons += [h for h in [21, 30, 45, 60, 75, 90] if h <= max_forward]

                for horizon in horizons:
                    target_row = atm_data.iloc[idx + horizon]

                    features = self._build_features(
                        base_row=base_row,
                        target_date=target_row[ColumnNames.ORDER_DATE],
                        horizon=horizon,
                        atm_stats=atm_stats,
                        historical_data=atm_data.iloc[:idx + 1],
                    )
                    features['target'] = float(target_row[self.target_column])
                    all_examples.append(features)

        training_df = pd.DataFrame(all_examples)
        logger.info(
            f"[{self.name}] {len(training_df)} exemples générés (target={self.target_column})"
        )
        return training_df


class MultiCoupureForecaster:
    """Wrapper qui entraîne un :class:`CatBoostDmqForecaster` par coupure et
    expose une API de prédiction unifiée.

    Produit un ``Dict[coupure, float]`` exploitable directement par
    ``CommandPipeline`` (via un ``DmqProvider``).
    """

    def __init__(self, coupures: Optional[List[int]] = None, **kwargs):
        self.coupures = coupures or [5, 10, 20, 50, 100]
        self.models: Dict[int, CatBoostDmqForecaster] = {
            c: CatBoostDmqForecaster(coupure=c, **kwargs) for c in self.coupures
        }
        self.is_fitted = False
        # Coupures pour lesquelles le target n'a pas de variance (tout à 0 ou
        # constante) — on stocke la valeur constante et on skip CatBoost.
        self.constant_predictions: Dict[int, float] = {}

    def fit(self, data: pd.DataFrame, eval_data: Optional[pd.DataFrame] = None) -> 'MultiCoupureForecaster':
        for coupure, model in self.models.items():
            logger.info(f"=== Entraînement modèle coupure {coupure}€ ===")

            # Vérifie la variance de la target avant d'appeler CatBoost
            target_col = model.target_column
            if target_col not in data.columns:
                logger.warning(
                    f"  Colonne {target_col} absente : prédicteur constant à 0"
                )
                self.constant_predictions[coupure] = 0.0
                continue

            target_values = data[target_col].dropna()
            if len(target_values) == 0 or target_values.nunique() <= 1:
                const_val = float(target_values.iloc[0]) if len(target_values) else 0.0
                logger.warning(
                    f"  Target {target_col} sans variance (valeur unique = {const_val}) : "
                    f"prédicteur constant"
                )
                self.constant_predictions[coupure] = const_val
                continue

            model.fit(data, eval_data=eval_data)
        self.is_fitted = True
        return self

    def predict_dmq_par_coupure(
        self,
        atm_id: int,
        prediction_date: datetime,
        context_data: Optional[pd.DataFrame] = None,
        horizon: Optional[int] = None,
    ) -> Dict[int, float]:
        """Retourne un dict ``{coupure: dmq_prédit}`` pour un ATM et une date."""
        if not self.is_fitted:
            raise ValueError("MultiCoupureForecaster non entraîné.")

        out: Dict[int, float] = {}
        for coupure, model in self.models.items():
            if coupure in self.constant_predictions:
                out[coupure] = float(max(0.0, self.constant_predictions[coupure]))
                continue
            pred = model.predict(
                atm_id=atm_id,
                prediction_dates=[prediction_date],
                context_data=context_data,
                horizon=horizon,
            )[0]
            out[coupure] = float(max(0.0, pred))
        return out

    def as_dmq_provider(
        self,
        prediction_date: datetime,
        context_data: Optional[pd.DataFrame] = None,
        horizon: Optional[int] = None,
    ):
        """Adaptateur vers l'interface ``DmqProvider`` de ``CommandPipeline``."""

        def _provider(atm_id: int) -> Dict[int, float]:
            return self.predict_dmq_par_coupure(
                atm_id=atm_id,
                prediction_date=prediction_date,
                context_data=context_data,
                horizon=horizon,
            )

        return _provider


# ===== FONCTIONS UTILITAIRES =====

def train_and_evaluate_catboost(
    enriched_data: pd.DataFrame,
    max_horizon: int = 90,
    test_ratio: float = 0.2,
    preset: Optional[str] = None,
) -> Tuple[CatBoostForecaster, pd.DataFrame]:
    """Entraîne et évalue un modèle CatBoost.

    Args:
        preset: Nom d'un preset d'hyperparamètres
            (``CatBoostForecaster.PRESETS``) ou ``None`` pour les défauts.
    """
    logger.info("ENTRAÎNEMENT ET ÉVALUATION CATBOOST")
    logger.info("=" * 50)

    dates = sorted(enriched_data[ColumnNames.ORDER_DATE].unique())
    split_idx = int(len(dates) * (1 - test_ratio))
    split_date = dates[split_idx]

    train_data = enriched_data[enriched_data[ColumnNames.ORDER_DATE] < split_date].copy()
    test_data = enriched_data[enriched_data[ColumnNames.ORDER_DATE] >= split_date].copy()

    logger.info(f"  Train : {len(train_data)} lignes jusqu'au {split_date.date()}")
    logger.info(f"  Test  : {len(test_data)} lignes à partir du {split_date.date()}")

    model = CatBoostForecaster(max_horizon=max_horizon, preset=preset)
    model.fit(train_data)

    results = model.evaluate_by_horizon(test_data)

    global_metrics = model.evaluate(test_data)
    logger.info(f"\nMÉTRIQUES GLOBALES :")
    logger.info(f"  MAE  : {global_metrics['mae']:.2f}")
    logger.info(f"  RMSE : {global_metrics['rmse']:.2f}")
    logger.info(f"  MAPE : {global_metrics['mape']:.1f}%")

    return model, results


if __name__ == "__main__":
    print("=" * 60)
    print("TEST DU MODÈLE CATBOOST ATM")
    print("=" * 60)

    try:
        enriched_file = get_file_path('enriched')

        if not enriched_file.exists():
            print("Fichier de données enrichies non trouvé.")
            print("Exécutez d'abord : python main.py --step enrichment")
        else:
            enriched_data = pd.read_csv(
                enriched_file,
                parse_dates=[ColumnNames.ORDER_DATE],
                date_format='%Y-%m-%d',
            )

            print(f"\nDonnées chargées : {len(enriched_data)} lignes")

            model, results = train_and_evaluate_catboost(enriched_data, max_horizon=90)

            print("\nRÉSULTATS PAR HORIZON :")
            print(results.to_string(index=False))

            model.save_model("data/output/catboost_model")
            print("\nTest terminé avec succès !")

    except Exception as e:
        print(f"\nErreur : {e}")
        import traceback
        traceback.print_exc()
