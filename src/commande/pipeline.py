"""Orchestration des 6 étapes du moteur de commande déterministe.

Entrée : DataFrame enrichi (un automate par ligne au jour de commande) +
DMQ par coupure (dict ou provider) + configurations ATM.

Sortie : DataFrame avec 5 colonnes `predictif_*` + drapeaux métier :
    is_command = any(predictif_* > 0)
    is_command_exceptionnelle, alerte_risque_vide, alerte_commande_supprimee,
    k7hs_5/10/20/50/100, etc.

Règle clé demandée par l'utilisateur :
    « Si une des 5 valeurs est supérieure à 0, alors c'est une commande. »
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, List, Optional

import logging

import pandas as pd

from .command_calculator import (
    CommandeParCoupure,
    compute_command_clic_clac,
    compute_command_exceptionnelle,
    compute_command_per_coupure,
)
from .constants import (
    CommandConfig,
    ColumnNames,
    COUPURES,
    DMQ_BY_COUPURE,
    K7HS_BY_COUPURE,
    NB_CASSETTES_BY_COUPURE,
    PREDICTIF_BY_COUPURE,
    SOLDES_BY_COUPURE,
)
from .exceptional import (
    DEFAULT_TOURNEE_AVAILABLE,
    TourneeAvailableFn,
    evaluate_exceptional,
)
from .insurance import apply_insurance_caps
from .k7hs_detector import detect_k7hs
from .solde_simulator import (
    DEFAULT_IS_HOLIDAY,
    IsHolidayFn,
    PendingCommand,
    simulate_solde_at_loading,
    simulate_solde_evening_after_loading,
)
from .verifications import check_min_command, reduce_for_axytrans


logger = logging.getLogger(__name__)


# Provider DMQ : `atm_id → Dict[coupure, float]`
DmqProvider = Callable[[int], Dict[int, float]]


@dataclass
class AtmConfig:
    """Configuration d'un automate (lue depuis la ligne du DataFrame)."""

    atm_id: int
    nb_cassettes_par_coupure: Dict[int, int] = field(default_factory=dict)
    nb_conteneurs: int = 1
    insurance_amount: float = 0.0
    mode_livraison: str = ""
    mode_chargement: str = ""
    agency_key: Optional[str] = None  # pour regrouper les commandes agence


@dataclass
class CommandDecision:
    """Résultat complet pour un automate (une ligne de sortie)."""

    atm_id: int
    predictif: Dict[int, int] = field(default_factory=dict)
    k7hs: Dict[int, bool] = field(default_factory=dict)
    is_command_exceptionnelle: bool = False
    alerte_risque_vide: bool = False
    alerte_commande_supprimee: bool = False
    alerte_commande_precedente_non_chargee: bool = False

    def to_row(self) -> Dict:
        row: Dict = {ColumnNames.ATM_ID: self.atm_id}
        for c in COUPURES:
            row[PREDICTIF_BY_COUPURE[c]] = int(self.predictif.get(c, 0))
            row[K7HS_BY_COUPURE[c]] = bool(self.k7hs.get(c, False))
        montants = [row[PREDICTIF_BY_COUPURE[c]] * c for c in COUPURES]
        row[ColumnNames.IS_COMMAND] = any(v > 0 for v in montants)
        row[ColumnNames.IS_COMMAND_EXCEPTIONNELLE] = self.is_command_exceptionnelle
        row[ColumnNames.ALERTE_RISQUE_VIDE] = self.alerte_risque_vide
        row[ColumnNames.ALERTE_COMMANDE_SUPPRIMEE] = self.alerte_commande_supprimee
        row[ColumnNames.ALERTE_COMMANDE_PRECEDENTE_NON_CHARGEE] = self.alerte_commande_precedente_non_chargee
        row['montant_total'] = int(sum(montants))
        return row


def _infer_nb_cassettes(
    row: pd.Series,
    history: Optional[pd.DataFrame] = None,
) -> Dict[int, int]:
    """Détermine le nombre de cassettes par coupure pour un ATM.

    Stratégie de fallback (les exports HFSQL ne contiennent pas toujours les
    colonnes ``nb_cassettes_<c>``) :

    1. Si la colonne ``nb_cassettes_<c>`` existe et est > 0 → on utilise sa
       valeur directe.
    2. Sinon, si la coupure a déjà eu un solde > 0 dans l'historique de
       l'ATM (ou en l'absence d'historique, dans la ligne courante) →
       ``DEFAULT_NB_CASSETTES_PAR_COUPURE[c]`` (1 par défaut, surchargeable
       via ``CMD_DEFAULT_NB_CASSETTES_<c>``).
    3. Sinon → 0 (la coupure n'est pas servie par cet ATM, marquée K7 HS).
    """
    nb_cassettes: Dict[int, int] = {}
    # ``row.index`` permet de distinguer :
    #   - colonne ABSENTE  → on infère (cas HFSQL réel sans nb_cassettes_*)
    #   - colonne PRÉSENTE → on respecte la valeur (même si 0 = ATM sans
    #     cette coupure, intention explicite des données ou des tests)
    row_columns = set(row.index)
    for c in COUPURES:
        cassette_col = NB_CASSETTES_BY_COUPURE[c]
        if cassette_col in row_columns:
            raw_value = row.get(cassette_col, 0)
            try:
                nb_cassettes[c] = max(0, int(raw_value or 0))
            except (TypeError, ValueError):
                nb_cassettes[c] = 0
            continue

        # Fallback : colonne absente → inférence à partir du solde
        # (historique + ligne courante).
        sol_col = SOLDES_BY_COUPURE[c]
        served = False
        if history is not None and not history.empty and sol_col in history.columns:
            try:
                served = bool((history[sol_col].fillna(0) > 0).any())
            except (TypeError, ValueError):
                served = False
        if not served:
            try:
                served = float(row.get(sol_col, 0.0) or 0.0) > 0
            except (TypeError, ValueError):
                served = False

        if served:
            nb_cassettes[c] = int(
                CommandConfig.DEFAULT_NB_CASSETTES_PAR_COUPURE.get(c, 1)
            )
        else:
            nb_cassettes[c] = 0
    return nb_cassettes


def _extract_atm_config(
    row: pd.Series,
    history: Optional[pd.DataFrame] = None,
) -> AtmConfig:
    """Lit la configuration d'un ATM depuis une ligne DataFrame.

    ``history`` (optionnel) permet d'inférer le nombre de cassettes quand la
    colonne dédiée est absente des données HFSQL — voir
    :func:`_infer_nb_cassettes`.
    """
    nb_cassettes = _infer_nb_cassettes(row, history)
    return AtmConfig(
        atm_id=int(row[ColumnNames.ATM_ID]),
        nb_cassettes_par_coupure=nb_cassettes,
        nb_conteneurs=int(row.get(ColumnNames.NB_CONTENEURS, 1) or 1),
        insurance_amount=float(row.get(ColumnNames.INSURANCE_AMOUNT, 0.0) or 0.0),
        mode_livraison=str(row.get(ColumnNames.MODE_LIVRAISON, "") or ""),
        mode_chargement=str(row.get(ColumnNames.MODE_CHARGEMENT, "") or ""),
    )


def _extract_solde_jour(row: pd.Series) -> Dict[int, float]:
    return {c: float(row.get(SOLDES_BY_COUPURE[c], 0.0) or 0.0) for c in COUPURES}


def _default_dmq_provider(history_by_atm: Dict[int, pd.DataFrame]) -> DmqProvider:
    """Construit un provider DMQ basé sur les colonnes DMQ_* si présentes,
    sinon sur la moyenne historique des soldes (fallback simple).
    """

    def _provide(atm_id: int) -> Dict[int, float]:
        hist = history_by_atm.get(atm_id)
        if hist is None or hist.empty:
            return {c: 0.0 for c in COUPURES}

        dmq: Dict[int, float] = {}
        for c in COUPURES:
            dmq_col = DMQ_BY_COUPURE[c]
            if dmq_col in hist.columns:
                dmq[c] = float(hist[dmq_col].tail(28).mean() or 0.0)
            else:
                # Fallback : différence moyenne jour-à-jour du solde (~ consommation)
                sol_col = SOLDES_BY_COUPURE[c]
                if sol_col in hist.columns and len(hist) >= 2:
                    diffs = hist[sol_col].diff().dropna()
                    # DMQ ≈ moyenne des baisses observées (valeurs positives)
                    baisses = -diffs[diffs < 0]
                    dmq[c] = float(baisses.mean()) if not baisses.empty else 0.0
                else:
                    dmq[c] = 0.0
        return dmq

    return _provide


class CommandPipeline:
    """Exécute les 6 étapes du moteur de commande pour un ensemble d'ATMs.

    Example:
        >>> pipeline = CommandPipeline()
        >>> output_df = pipeline.run(
        ...     current_data=snapshot_df,
        ...     historical_data=history_df,
        ...     day_commande=date(2026, 3, 30),
        ...     day_livraison=date(2026, 4, 2),
        ... )
        >>> # 5 colonnes predictif_* + is_command
    """

    def __init__(
        self,
        config: Optional[CommandConfig] = None,
        dmq_provider: Optional[DmqProvider] = None,
        is_holiday: IsHolidayFn = DEFAULT_IS_HOLIDAY,
        tournee_available: TourneeAvailableFn = DEFAULT_TOURNEE_AVAILABLE,
    ):
        # CommandConfig est une classe de constantes ; on l'accepte par
        # cohérence d'interface mais on l'utilise via les attributs de classe.
        self.config = config or CommandConfig
        self.dmq_provider = dmq_provider
        self.is_holiday = is_holiday
        self.tournee_available = tournee_available

    # --- API principale ----------------------------------------------------

    def run(
        self,
        current_data: pd.DataFrame,
        historical_data: Optional[pd.DataFrame] = None,
        day_commande: Optional[date] = None,
        day_livraison: Optional[date] = None,
        pending_commands_by_atm: Optional[Dict[int, List[PendingCommand]]] = None,
    ) -> pd.DataFrame:
        """Exécute les 6 étapes pour chaque ligne de ``current_data``.

        Args:
            current_data: DataFrame (une ligne par ATM) avec colonnes
                `atm_id`, `soldes_5..100`, et idéalement les colonnes de
                config (`nb_cassettes_*`, `mode_livraison`, ...).
            historical_data: DataFrame complet pour la détection K7 HS et le
                calcul du DMQ (colonne `atm_id` + historique soldes).
            day_commande: Jour de commande (défaut: `ORDER_DATE` de la 1re ligne).
            day_livraison: Jour de livraison/chargement.
            pending_commands_by_atm: Commandes précédentes non chargées.

        Returns:
            DataFrame à N lignes (une par ATM d'entrée) et colonnes :
            atm_id, predictif_5..100, k7hs_5..100, is_command,
            is_command_exceptionnelle, alerte_risque_vide,
            alerte_commande_supprimee, alerte_commande_precedente_non_chargee,
            montant_total.
        """
        if current_data is None or current_data.empty:
            return pd.DataFrame()

        # Date de commande : soit fournie, soit extraite de la 1re ligne.
        if day_commande is None:
            day_commande = pd.Timestamp(current_data[ColumnNames.ORDER_DATE].iloc[0]).date()
        if day_livraison is None:
            # Défaut : +3 jours (hors férié) conformément à la doc.
            from datetime import timedelta as _td
            day_livraison = day_commande + _td(days=3)

        # Préparer historique groupé par ATM
        history_by_atm: Dict[int, pd.DataFrame] = {}
        if historical_data is not None and not historical_data.empty:
            sorted_hist = historical_data.sort_values(ColumnNames.ORDER_DATE)
            for atm_id, grp in sorted_hist.groupby(ColumnNames.ATM_ID):
                history_by_atm[int(atm_id)] = grp.reset_index(drop=True)

        # Provider DMQ par défaut si non fourni
        dmq_provider = self.dmq_provider or _default_dmq_provider(history_by_atm)

        pending_map = pending_commands_by_atm or {}

        # Regrouper les ATMs par agence pour le cap assurance agence
        # (les commandes en cours se cumulent à l'échelle de l'agence).
        commandes_par_agence: Dict[str, float] = {}

        rows: List[Dict] = []
        decisions: List[CommandDecision] = []

        for _, row in current_data.iterrows():
            atm_id_for_history = int(row[ColumnNames.ATM_ID])
            atm_history = history_by_atm.get(atm_id_for_history, pd.DataFrame())
            atm_cfg = _extract_atm_config(row, history=atm_history)
            decision = self._process_single(
                row=row,
                atm_cfg=atm_cfg,
                history=atm_history,
                dmq_provider=dmq_provider,
                day_commande=day_commande,
                day_livraison=day_livraison,
                pending_commands=pending_map.get(atm_cfg.atm_id, []),
                commandes_en_cours_agence=commandes_par_agence.get(
                    atm_cfg.agency_key or f"atm_{atm_cfg.atm_id}", 0.0
                ),
            )
            # Cumuler la commande pour l'agence (cap assurance étape 6)
            agency = atm_cfg.agency_key or f"atm_{atm_cfg.atm_id}"
            montant = sum(
                decision.predictif.get(c, 0) * c for c in COUPURES
            )
            commandes_par_agence[agency] = commandes_par_agence.get(agency, 0.0) + montant

            decisions.append(decision)
            rows.append(decision.to_row())

        result = pd.DataFrame(rows)

        logger.info(
            "CommandPipeline: %d automates traités, %d commandes générées",
            len(result),
            int(result[ColumnNames.IS_COMMAND].sum()) if not result.empty else 0,
        )
        return result

    # --- Implémentation étape par étape -----------------------------------

    def _process_single(
        self,
        row: pd.Series,
        atm_cfg: AtmConfig,
        history: pd.DataFrame,
        dmq_provider: DmqProvider,
        day_commande: date,
        day_livraison: date,
        pending_commands: List[PendingCommand],
        commandes_en_cours_agence: float,
    ) -> CommandDecision:
        decision = CommandDecision(atm_id=atm_cfg.atm_id)

        # === Étape 0 : K7 HS ==============================================
        k7hs = detect_k7hs(history)
        # Les cassettes absentes (nb_cassettes == 0) sont également HS.
        for c in COUPURES:
            if atm_cfg.nb_cassettes_par_coupure.get(c, 0) <= 0:
                k7hs[c] = True
        decision.k7hs = k7hs

        # === DMQ par coupure ==============================================
        dmq = dmq_provider(atm_cfg.atm_id)

        # === Étape 1 : Solde au chargement ================================
        solde_jour = _extract_solde_jour(row)
        if pending_commands:
            decision.alerte_commande_precedente_non_chargee = True

        solde_chargement = simulate_solde_at_loading(
            solde_jour=solde_jour,
            dmq_par_coupure=dmq,
            day_commande=day_commande,
            day_chargement=day_livraison,
            pending_commands=pending_commands,
            is_holiday=self.is_holiday,
        )

        # === Étape 2 : Calcul par coupure =================================
        commande = compute_command_per_coupure(
            solde_chargement=solde_chargement,
            nb_cassettes_par_coupure=atm_cfg.nb_cassettes_par_coupure,
            k7hs=k7hs,
        )

        # === Étape 3 : Vérifications ======================================
        commande, verif_min = check_min_command(commande)
        decision.alerte_commande_supprimee = verif_min.commande_supprimee

        commande, _ = reduce_for_axytrans(
            commande=commande,
            mode_livraison=atm_cfg.mode_livraison,
            nb_conteneurs=atm_cfg.nb_conteneurs,
        )

        # === Étape 4 : Projection au soir + commande exceptionnelle ======
        solde_soir = simulate_solde_evening_after_loading(
            solde_jour=solde_jour,
            dmq_par_coupure=dmq,
            day_commande=day_commande,
            day_chargement=day_livraison,
            commanded_values={
                c: commande.nb_billets.get(c, 0) for c in COUPURES
            },
            pending_commands=pending_commands,
            is_holiday=self.is_holiday,
        )
        exceptional = evaluate_exceptional(
            solde_soir=solde_soir,
            dmq_par_coupure=dmq,
            day_livraison=day_livraison,
            atm_id=atm_cfg.atm_id,
            k7hs=k7hs,
            tournee_available=self.tournee_available,
        )
        decision.alerte_risque_vide = exceptional.risque_vide

        # === Étape 5 : Sélection finale ===================================
        if exceptional.risque_vide and exceptional.autorisee:
            commande = compute_command_exceptionnelle(
                nb_cassettes_par_coupure=atm_cfg.nb_cassettes_par_coupure,
                k7hs=k7hs,
            )
            decision.is_command_exceptionnelle = True
        elif (atm_cfg.mode_chargement or "").strip().lower() == CommandConfig.CLIC_CLAC_MODE:
            commande = compute_command_clic_clac(
                nb_cassettes_par_coupure=atm_cfg.nb_cassettes_par_coupure,
                k7hs=k7hs,
            )

        # === Étape 6 : Assurance agence + cap global ======================
        commande, ins_flags = apply_insurance_caps(
            commande=commande,
            commandes_en_cours_agence_eur=commandes_en_cours_agence,
            insurance_amount_eur=atm_cfg.insurance_amount,
        )
        if ins_flags.supprimee_apres_reduction:
            decision.alerte_commande_supprimee = True

        decision.predictif = {c: commande.nb_billets.get(c, 0) for c in COUPURES}
        return decision
