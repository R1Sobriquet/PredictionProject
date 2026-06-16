"""Moteur de commande déterministe PredikATM (module 4.1 de la documentation).

Le pipeline exécute les 6 étapes documentées :
    0. Détection K7 HS (cassettes hors service)
    1. Simulation du solde au jour du chargement (DMQ × 2,5)
    2. Calcul du nombre de billets à commander par coupure (formule doc)
    3. Vérifications spécifiques (seuil min + caps Axytrans)
    4. Projection au soir & commande exceptionnelle
    5. Sélection finale (exceptionnelle / clic-clac / complément)
    6. Cap assurance agence et cap global 300 000 €

Règle clé : une commande existe dès qu'une des 5 valeurs par coupure > 0.
"""

from .command_calculator import (
    CommandeParCoupure,
    compute_command_clic_clac,
    compute_command_exceptionnelle,
    compute_command_per_coupure,
)
from .exceptional import (
    ExceptionalDecision,
    detect_risque_vide,
    evaluate_exceptional,
)
from .insurance import InsuranceFlags, apply_insurance_caps
from .k7hs_detector import detect_k7hs, detect_k7hs_from_row
from .pipeline import AtmConfig, CommandDecision, CommandPipeline
from .solde_simulator import (
    PendingCommand,
    simulate_solde_at_loading,
    simulate_solde_evening_after_loading,
)
from .verifications import (
    VerificationFlags,
    check_min_command,
    reduce_for_axytrans,
)

__all__ = [
    # Pipeline
    'CommandPipeline',
    'AtmConfig',
    'CommandDecision',
    # Étape 0
    'detect_k7hs',
    'detect_k7hs_from_row',
    # Étape 1 & 4 (simulation)
    'simulate_solde_at_loading',
    'simulate_solde_evening_after_loading',
    'PendingCommand',
    # Étape 2
    'CommandeParCoupure',
    'compute_command_per_coupure',
    'compute_command_clic_clac',
    'compute_command_exceptionnelle',
    # Étape 3
    'VerificationFlags',
    'check_min_command',
    'reduce_for_axytrans',
    # Étape 4 (exceptionnelle)
    'ExceptionalDecision',
    'detect_risque_vide',
    'evaluate_exceptional',
    # Étape 6
    'InsuranceFlags',
    'apply_insurance_caps',
]
