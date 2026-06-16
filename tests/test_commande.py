"""Tests unitaires du moteur de commande déterministe (``src/commande``).

Couvre les 6 étapes du pipeline avec des fixtures synthétiques :
0. Détection K7 HS
1. Simulation du solde au jour du chargement
2. Calcul par coupure
3. Vérifications (min command + Axytrans)
4. Commande exceptionnelle
5. Sélection finale + règle "is_command = any > 0"
6. Cap assurance agence + cap global

Exécution :
    pytest tests/test_commande.py -v
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# Permet d'exécuter les tests depuis la racine du projet
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.commande import (
    AtmConfig,
    CommandDecision,
    CommandPipeline,
    CommandeParCoupure,
    PendingCommand,
    apply_insurance_caps,
    check_min_command,
    compute_command_clic_clac,
    compute_command_exceptionnelle,
    compute_command_per_coupure,
    detect_k7hs,
    evaluate_exceptional,
    reduce_for_axytrans,
    simulate_solde_at_loading,
    simulate_solde_evening_after_loading,
)
from src.utils import (
    COUPURES,
    ColumnNames,
    CommandConfig,
    DMQ_BY_COUPURE,
    K7HS_BY_COUPURE,
    NB_CASSETTES_BY_COUPURE,
    PREDICTIF_BY_COUPURE,
    SOLDES_BY_COUPURE,
)


# ==========================================================================
# Fixtures
# ==========================================================================


@pytest.fixture
def dmq_standard():
    """DMQ typique : 100 billets de 5€, 80 de 10€, etc."""
    return {5: 100.0, 10: 80.0, 20: 60.0, 50: 40.0, 100: 20.0}


@pytest.fixture
def cassettes_standard():
    """1 cassette par coupure (config de base)."""
    return {c: 1 for c in COUPURES}


@pytest.fixture
def k7hs_all_ok():
    return {c: False for c in COUPURES}


@pytest.fixture
def soldes_bas():
    """Soldes volontairement bas → doit générer une commande."""
    return {5: 200.0, 10: 150.0, 20: 100.0, 50: 60.0, 100: 30.0}


def make_historical_soldes(
    atm_id: int = 1,
    n_days: int = 20,
    stale_coupure: int = None,
    end_date: date = date(2026, 3, 30),
) -> pd.DataFrame:
    """Construit un historique synthétique de soldes pour un automate.

    Args:
        stale_coupure: si non None, la coupure indiquée aura un solde
            identique sur les 5 derniers jours (→ K7 HS attendu).
    """
    rows = []
    start = end_date - timedelta(days=n_days - 1)
    for i in range(n_days):
        d = start + timedelta(days=i)
        row = {
            ColumnNames.ATM_ID: atm_id,
            ColumnNames.ORDER_DATE: pd.Timestamp(d),
        }
        # Soldes décroissants (consommation normale).
        for c in COUPURES:
            # Valeur décroissante pour simuler de la consommation.
            row[SOLDES_BY_COUPURE[c]] = 500.0 - i * 5.0
        # Applique un gel sur la coupure HS.
        if stale_coupure is not None and stale_coupure in COUPURES:
            if i >= n_days - 5:
                row[SOLDES_BY_COUPURE[stale_coupure]] = 42.0
        rows.append(row)
    return pd.DataFrame(rows)


# ==========================================================================
# Étape 0 — Détection K7 HS
# ==========================================================================


class TestK7HSDetector:
    def test_detect_hs_on_stale_coupure(self):
        hist = make_historical_soldes(stale_coupure=20)
        k7hs = detect_k7hs(hist)
        assert k7hs[20] is True
        # Les autres coupures doivent être OK (variations normales).
        assert all(k7hs[c] is False for c in [5, 10, 50, 100])

    def test_no_hs_when_all_soldes_move(self):
        hist = make_historical_soldes()
        k7hs = detect_k7hs(hist)
        assert all(k7hs[c] is False for c in COUPURES)

    def test_empty_history_marks_all_hs(self):
        empty = pd.DataFrame()
        k7hs = detect_k7hs(empty)
        assert all(k7hs[c] is True for c in COUPURES)

    def test_absent_coupure_marked_hs(self):
        # On retire volontairement la colonne soldes_5.
        hist = make_historical_soldes()
        hist = hist.drop(columns=[SOLDES_BY_COUPURE[5]])
        k7hs = detect_k7hs(hist)
        assert k7hs[5] is True


# ==========================================================================
# Étape 1 — Simulation du solde au chargement
# ==========================================================================


class TestSoldeSimulation:
    def test_linear_decrease(self, dmq_standard):
        """Sans commande pendante, le solde baisse de 2,5 × DMQ."""
        solde_jour = {5: 500.0, 10: 400.0, 20: 300.0, 50: 200.0, 100: 100.0}
        result = simulate_solde_at_loading(
            solde_jour=solde_jour,
            dmq_par_coupure=dmq_standard,
            day_commande=date(2026, 3, 30),
            day_chargement=date(2026, 4, 2),
        )
        for c in COUPURES:
            expected = max(
                0.0,
                solde_jour[c] - dmq_standard[c] * CommandConfig.DMQ_CONSO_JOURS_CHARGEMENT,
            )
            assert result[c] == pytest.approx(expected)

    def test_pending_command_added(self, dmq_standard):
        """Une commande pendante est ajoutée au solde au bon jour."""
        solde_jour = {c: 500.0 for c in COUPURES}
        pending = [
            PendingCommand(
                loading_date=date(2026, 4, 1),
                amounts_par_coupure={5: 1000, 10: 800, 20: 500, 50: 200, 100: 100},
            )
        ]
        result = simulate_solde_at_loading(
            solde_jour=solde_jour,
            dmq_par_coupure=dmq_standard,
            day_commande=date(2026, 3, 30),
            day_chargement=date(2026, 4, 2),
            pending_commands=pending,
        )
        # Solde doit inclure la commande ajoutée (500 + 1000 - 2.5*100 = 1250).
        assert result[5] == pytest.approx(500.0 + 1000.0 - 2.5 * 100.0)

    def test_same_day_returns_solde(self, dmq_standard):
        """Si day_commande == day_chargement → pas d'évolution."""
        solde = {c: 300.0 for c in COUPURES}
        result = simulate_solde_at_loading(
            solde_jour=solde,
            dmq_par_coupure=dmq_standard,
            day_commande=date(2026, 3, 30),
            day_chargement=date(2026, 3, 30),
        )
        assert result == {c: 300.0 for c in COUPURES}

    def test_evening_adds_commanded_values(self, dmq_standard):
        """Au soir, le solde = projection + billets commandés."""
        solde_jour = {c: 500.0 for c in COUPURES}
        commanded = {5: 100, 10: 80, 20: 0, 50: 0, 100: 0}
        result = simulate_solde_evening_after_loading(
            solde_jour=solde_jour,
            dmq_par_coupure=dmq_standard,
            day_commande=date(2026, 3, 30),
            day_chargement=date(2026, 4, 2),
            commanded_values=commanded,
        )
        # Pour coupure 5 : 500 - 3.0*100 + 100 = 300
        assert result[5] == pytest.approx(500.0 - 3.0 * 100.0 + 100.0)


# ==========================================================================
# Étape 2 — Calcul par coupure
# ==========================================================================


class TestCommandCalculator:
    def test_standard_formula(self, k7hs_all_ok):
        """SEUIL_MAX[5]=2500, 1 cassette, solde=200 → commande=2300."""
        solde_chargement = {5: 200, 10: 150, 20: 100, 50: 60, 100: 30}
        cassettes = {c: 1 for c in COUPURES}
        cmd = compute_command_per_coupure(solde_chargement, cassettes, k7hs_all_ok)

        for c in COUPURES:
            capacite = CommandConfig.seuil_max(c) * 1
            expected = max(0, capacite - int(round(solde_chargement[c])))
            assert cmd.nb_billets[c] == expected

    def test_k7hs_forces_zero(self):
        solde_chargement = {c: 0.0 for c in COUPURES}
        cassettes = {c: 1 for c in COUPURES}
        k7hs = {5: True, 10: False, 20: False, 50: False, 100: False}
        cmd = compute_command_per_coupure(solde_chargement, cassettes, k7hs)
        assert cmd.nb_billets[5] == 0
        assert cmd.nb_billets[10] > 0

    def test_no_cassette_returns_zero(self):
        solde_chargement = {c: 0.0 for c in COUPURES}
        cassettes = {c: 0 for c in COUPURES}
        k7hs = {c: False for c in COUPURES}
        cmd = compute_command_per_coupure(solde_chargement, cassettes, k7hs)
        assert cmd.total_billets == 0

    def test_clic_clac_fills_to_max(self, k7hs_all_ok):
        cassettes = {c: 2 for c in COUPURES}
        cmd = compute_command_clic_clac(cassettes, k7hs_all_ok)
        for c in COUPURES:
            assert cmd.nb_billets[c] == CommandConfig.seuil_max(c) * 2

    def test_exceptionnelle_is_half(self, k7hs_all_ok):
        cassettes = {c: 1 for c in COUPURES}
        cmd = compute_command_exceptionnelle(cassettes, k7hs_all_ok)
        for c in COUPURES:
            assert cmd.nb_billets[c] == CommandConfig.seuil_max(c) // 2

    def test_is_command_rule_any_positive(self, k7hs_all_ok):
        """Règle métier clé : any > 0 → commande existe."""
        cmd = CommandeParCoupure(nb_billets={5: 1, 10: 0, 20: 0, 50: 0, 100: 0})
        montants = [cmd.nb_billets[c] * c for c in COUPURES]
        assert any(v > 0 for v in montants)

        cmd_vide = CommandeParCoupure(nb_billets={c: 0 for c in COUPURES})
        assert not any(cmd_vide.nb_billets[c] > 0 for c in COUPURES)


# ==========================================================================
# Étape 3 — Vérifications (min command + Axytrans)
# ==========================================================================


class TestVerifications:
    def test_min_command_zeroes_when_below(self):
        # Seulement 3 billets de 5 = 15 €, bien en dessous de MIN_COMMAND_AMOUNT.
        cmd = CommandeParCoupure(nb_billets={5: 3, 10: 0, 20: 0, 50: 0, 100: 0})
        reduced, flags = check_min_command(cmd)
        assert flags.commande_supprimee is True
        assert reduced.total_billets == 0

    def test_min_command_keeps_when_above(self):
        # 1000 billets de 10 = 10 000 € > 2 000.
        cmd = CommandeParCoupure(nb_billets={5: 0, 10: 1000, 20: 0, 50: 0, 100: 0})
        reduced, flags = check_min_command(cmd)
        assert flags.commande_supprimee is False
        assert reduced.total_billets == 1000

    def test_axytrans_reduces_to_max_amount(self):
        # 1000 billets de 100 = 100 000 € > 75 000.
        cmd = CommandeParCoupure(nb_billets={5: 0, 10: 0, 20: 0, 50: 0, 100: 1000})
        reduced, flags = reduce_for_axytrans(
            cmd, mode_livraison="axytrans", nb_conteneurs=1
        )
        assert flags.reduite_axytrans_montant is True
        assert reduced.montant_total <= CommandConfig.AXYTRANS_MAX_EUR

    def test_axytrans_reduces_to_max_billets(self):
        # 5000 billets de 5 = 25 000 € (sous cap €) mais > 2600 billets.
        cmd = CommandeParCoupure(nb_billets={5: 5000, 10: 0, 20: 0, 50: 0, 100: 0})
        reduced, flags = reduce_for_axytrans(
            cmd, mode_livraison="axytrans", nb_conteneurs=1
        )
        assert flags.reduite_axytrans_billets is True
        assert reduced.total_billets <= CommandConfig.AXYTRANS_MAX_BILLETS_PER_CONTAINER

    def test_non_axytrans_unchanged(self):
        cmd = CommandeParCoupure(nb_billets={5: 0, 10: 0, 20: 0, 50: 0, 100: 1000})
        reduced, flags = reduce_for_axytrans(
            cmd, mode_livraison="fourgon", nb_conteneurs=1
        )
        assert reduced.nb_billets == cmd.nb_billets
        assert flags.reduite_axytrans_montant is False


# ==========================================================================
# Étape 4 — Commande exceptionnelle
# ==========================================================================


class TestExceptional:
    def test_risque_detected_when_solde_under_dmq(self):
        # Solde soir à 50 pour coupure 5 avec DMQ=100 → risque vide.
        solde_soir = {5: 50.0, 10: 500.0, 20: 500.0, 50: 500.0, 100: 500.0}
        dmq = {5: 100.0, 10: 50.0, 20: 50.0, 50: 20.0, 100: 10.0}
        decision = evaluate_exceptional(
            solde_soir=solde_soir,
            dmq_par_coupure=dmq,
            day_livraison=date(2026, 4, 2),
            atm_id=1,
        )
        assert decision.risque_vide is True
        # Stub par défaut → tournée dispo.
        assert decision.autorisee is True

    def test_no_risk_when_solde_sufficient(self):
        solde_soir = {c: 1000.0 for c in COUPURES}
        dmq = {c: 100.0 for c in COUPURES}
        decision = evaluate_exceptional(
            solde_soir=solde_soir,
            dmq_par_coupure=dmq,
            day_livraison=date(2026, 4, 2),
            atm_id=1,
        )
        assert decision.risque_vide is False

    def test_not_authorized_without_tournee(self):
        solde_soir = {5: 10.0, 10: 1000.0, 20: 1000.0, 50: 1000.0, 100: 1000.0}
        dmq = {5: 100.0, 10: 100.0, 20: 100.0, 50: 100.0, 100: 100.0}
        no_tournee = lambda _d, _atm: False  # noqa: E731
        decision = evaluate_exceptional(
            solde_soir=solde_soir,
            dmq_par_coupure=dmq,
            day_livraison=date(2026, 4, 2),
            atm_id=1,
            tournee_available=no_tournee,
        )
        assert decision.risque_vide is True
        assert decision.autorisee is False


# ==========================================================================
# Étape 6 — Assurance agence + cap global
# ==========================================================================


class TestInsurance:
    def test_reduction_on_agency_cap(self):
        # 1000 × 100 € = 100 000 €, assurance agence = 50 000 → réduction.
        cmd = CommandeParCoupure(nb_billets={5: 0, 10: 0, 20: 0, 50: 0, 100: 1000})
        reduced, flags = apply_insurance_caps(
            commande=cmd,
            commandes_en_cours_agence_eur=0.0,
            insurance_amount_eur=50_000.0,
        )
        assert flags.reduite_assurance_agence is True
        assert reduced.montant_total <= 50_000.0

    def test_global_cap_applies(self):
        # Commande 400k€, cap global 300k → réduction.
        cmd = CommandeParCoupure(nb_billets={5: 0, 10: 0, 20: 0, 50: 0, 100: 4000})
        reduced, flags = apply_insurance_caps(
            commande=cmd,
            commandes_en_cours_agence_eur=0.0,
            insurance_amount_eur=1_000_000.0,  # assurance haute, pas limitante
        )
        assert flags.reduite_cap_global is True
        assert reduced.montant_total <= CommandConfig.INSURANCE_GLOBAL_CAP

    def test_suppression_if_below_min_after_reduction(self):
        # Assurance très basse → réduction fait passer sous MIN → suppression.
        cmd = CommandeParCoupure(nb_billets={5: 0, 10: 0, 20: 0, 50: 0, 100: 100})
        reduced, flags = apply_insurance_caps(
            commande=cmd,
            commandes_en_cours_agence_eur=0.0,
            insurance_amount_eur=500.0,  # < MIN_COMMAND_AMOUNT
        )
        assert flags.supprimee_apres_reduction is True
        assert reduced.total_billets == 0


# ==========================================================================
# Pipeline bout-en-bout
# ==========================================================================


def _make_snapshot_row(
    atm_id: int,
    soldes: dict,
    cassettes: dict,
    mode_livraison: str = "fourgon",
    mode_chargement: str = "complement",
    nb_conteneurs: int = 1,
    insurance: float = 500_000.0,
    dmq: dict = None,
) -> dict:
    row = {
        ColumnNames.ATM_ID: atm_id,
        ColumnNames.ORDER_DATE: pd.Timestamp(date(2026, 3, 30)),
        ColumnNames.NB_CONTENEURS: nb_conteneurs,
        ColumnNames.INSURANCE_AMOUNT: insurance,
        ColumnNames.MODE_LIVRAISON: mode_livraison,
        ColumnNames.MODE_CHARGEMENT: mode_chargement,
    }
    for c in COUPURES:
        row[SOLDES_BY_COUPURE[c]] = soldes.get(c, 0.0)
        row[NB_CASSETTES_BY_COUPURE[c]] = cassettes.get(c, 0)
        if dmq is not None:
            row[DMQ_BY_COUPURE[c]] = dmq.get(c, 0.0)
    return row


class TestPipelineEndToEnd:
    def test_pipeline_end_to_end_three_atms(self, dmq_standard, soldes_bas):
        """Scénario à 3 ATMs : un normal, un K7HS, un solde trop haut (pas de cmd)."""
        # ATM 1 — commande attendue (soldes bas, tout OK).
        r1 = _make_snapshot_row(
            atm_id=1, soldes=soldes_bas, cassettes={c: 1 for c in COUPURES},
            dmq=dmq_standard,
        )
        # ATM 2 — soldes très élevés (proche capacité) → pas de commande.
        r2 = _make_snapshot_row(
            atm_id=2,
            soldes={c: CommandConfig.seuil_max(c) - 10 for c in COUPURES},
            cassettes={c: 1 for c in COUPURES},
            dmq={c: 0.5 for c in COUPURES},  # DMQ très faible
        )
        # ATM 3 — pas de cassettes (tout absent / HS) → 0 partout.
        r3 = _make_snapshot_row(
            atm_id=3,
            soldes={c: 0.0 for c in COUPURES},
            cassettes={c: 0 for c in COUPURES},
            dmq=dmq_standard,
        )

        current = pd.DataFrame([r1, r2, r3])

        # Historique synthétique 20 jours par ATM.
        history_frames = [
            make_historical_soldes(atm_id=1),
            make_historical_soldes(atm_id=2),
            make_historical_soldes(atm_id=3),
        ]
        for i, f in enumerate(history_frames, start=1):
            f[ColumnNames.ATM_ID] = i
            # Injecter le DMQ dans l'historique pour que le provider par
            # défaut le retrouve.
            for c in COUPURES:
                f[DMQ_BY_COUPURE[c]] = dmq_standard[c]
        history = pd.concat(history_frames, ignore_index=True)

        pipeline = CommandPipeline()
        result = pipeline.run(
            current_data=current,
            historical_data=history,
            day_commande=date(2026, 3, 30),
            day_livraison=date(2026, 4, 2),
        )

        # Structure
        assert len(result) == 3
        assert set(PREDICTIF_BY_COUPURE.values()).issubset(result.columns)
        assert ColumnNames.IS_COMMAND in result.columns

        # ATM 1 : commande attendue.
        r1_out = result[result[ColumnNames.ATM_ID] == 1].iloc[0]
        assert r1_out[ColumnNames.IS_COMMAND] is True or r1_out[ColumnNames.IS_COMMAND] == True  # noqa: E712

        # ATM 3 : pas de cassettes → pas de commande.
        r3_out = result[result[ColumnNames.ATM_ID] == 3].iloc[0]
        assert bool(r3_out[ColumnNames.IS_COMMAND]) is False

    def test_is_command_rule_matches_predictif_any_positive(self):
        """Règle : `is_command = any(predictif_* > 0)`."""
        r = _make_snapshot_row(
            atm_id=1,
            soldes={c: 10.0 for c in COUPURES},
            cassettes={c: 1 for c in COUPURES},
            dmq={c: 1.0 for c in COUPURES},
        )
        current = pd.DataFrame([r])
        hist = make_historical_soldes(atm_id=1)
        hist[ColumnNames.ATM_ID] = 1
        for c in COUPURES:
            hist[DMQ_BY_COUPURE[c]] = 1.0

        pipeline = CommandPipeline()
        result = pipeline.run(
            current_data=current,
            historical_data=hist,
            day_commande=date(2026, 3, 30),
            day_livraison=date(2026, 4, 2),
        )

        row = result.iloc[0]
        values = [row[PREDICTIF_BY_COUPURE[c]] for c in COUPURES]
        assert bool(row[ColumnNames.IS_COMMAND]) == any(v > 0 for v in values)

    def test_empty_input_returns_empty(self):
        pipeline = CommandPipeline()
        result = pipeline.run(
            current_data=pd.DataFrame(),
            historical_data=None,
            day_commande=date(2026, 3, 30),
            day_livraison=date(2026, 4, 2),
        )
        assert result.empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
