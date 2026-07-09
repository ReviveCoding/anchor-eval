from pathlib import Path

import pandas as pd
import pytest

from evalforge.dpo import build_preference_training_pairs


def test_dpo_builder_blocks_non_dpo_items(tmp_path: Path) -> None:
    full = pd.DataFrame(
        [
            {
                "item_id": "final_x",
                "source_group": "final_group",
                "split": "final",
                "response_a": "a",
                "response_b": "b",
                "prompt": "p",
                "domain": "x",
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "item_id": "final_x",
                "candidate_id": "C5_routed_production",
                "route": "AUTO_ACCEPT",
                "prediction": "A",
                "p_correct": 0.99,
            }
        ]
    )
    with pytest.raises(ValueError, match="dpo"):
        build_preference_training_pairs(
            predictions, full, tmp_path / "pairs.jsonl", "C5_routed_production", True
        )
