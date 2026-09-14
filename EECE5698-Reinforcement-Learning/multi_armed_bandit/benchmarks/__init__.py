# Benchmark integrations for standardized bandit evaluation
# Implements OBP integration, OBD evaluation, and supervised-to-bandit conversion

# OBP Integration
from .obp_integration import (
    OBPEvaluator,
    BanditFeedback,
    create_bandit_feedback_from_run,
    validate_with_obp,
)

# Supervised-to-Bandit
from .supervised_to_bandit import (
    SupervisedToBandit,
    MushroomBandit,
    CovertypeBandit,
    SyntheticDriftBandit,
)

# Replay Evaluation
from .replay_evaluation import (
    ReplayResult,
    ReplayEvaluator,
    SyntheticReplayEvaluator,
    run_replay_benchmark,
)

__all__ = [
    # OBP Integration
    'OBPEvaluator',
    'BanditFeedback',
    'create_bandit_feedback_from_run',
    'validate_with_obp',
    # Supervised-to-Bandit
    'SupervisedToBandit',
    'MushroomBandit',
    'CovertypeBandit',
    'SyntheticDriftBandit',
    # Replay Evaluation
    'ReplayResult',
    'ReplayEvaluator',
    'SyntheticReplayEvaluator',
    'run_replay_benchmark',
]

# Registry for convenience
BENCHMARKS = {
    'mushroom': MushroomBandit,
    'covertype': CovertypeBandit,
    'synthetic_drift': SyntheticDriftBandit,
}
