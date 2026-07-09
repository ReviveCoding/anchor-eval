from pathlib import Path

import pandas as pd
import pytest

from evalforge.dpo import validate_dpo_config


def test_dpo_config_rejects_duplicate_source_group(tmp_path: Path) -> None:
    pairs = tmp_path / "pairs.jsonl"
    pd.DataFrame(
        [
            {
                "item_id": "a",
                "source_group": "g",
                "domain": "x",
                "prompt": "p",
                "chosen": "a",
                "rejected": "b",
            },
            {
                "item_id": "b",
                "source_group": "g",
                "domain": "x",
                "prompt": "p",
                "chosen": "a",
                "rejected": "b",
            },
        ]
    ).to_json(pairs, orient="records", lines=True)
    config = {
        "experiment_id": "x",
        "label_source": "raw_judge",
        "model_id": "model",
        "pairs_path": str(pairs),
        "output_dir": str(tmp_path / "out"),
        "seed": 1,
        "max_length": 512,
        "max_prompt_length": 256,
    }
    with pytest.raises(ValueError, match="unique item_id and source_group"):
        validate_dpo_config(config)
