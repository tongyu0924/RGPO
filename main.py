#!/usr/bin/env python3
"""
Entry point for RGPO training (Ranking-Guided Preference Optimization).

See config.py for hyperparameter defaults and losses.py for the
objective terms (Eq. 5-13 of the paper).
"""
import logging
import os

from config import (
    HF_TOKEN,
    DEFAULT_MODEL_NAME,
    DEFAULT_DATA_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_BETA,
    DEFAULT_TAU_BT,
    DEFAULT_TEMPERATURE,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_USE_PAIRWISE,
    CUDA_ALLOC_CONF,
)
from trainer import JsonlGRPOTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = CUDA_ALLOC_CONF

    if not HF_TOKEN:
        logger.warning("HF_TOKEN is not set; only public models can be loaded. "
                        "Set the HF_TOKEN environment variable if a private/gated model is needed.")

    trainer = JsonlGRPOTrainer(
        model_name=DEFAULT_MODEL_NAME,
        beta=DEFAULT_BETA,
        temperature=DEFAULT_TEMPERATURE,
        tau_bt=DEFAULT_TAU_BT,
        hf_token=HF_TOKEN,
    )

    trainer.train(
        data_file=DEFAULT_DATA_FILE,
        output_dir=DEFAULT_OUTPUT_DIR,
        epochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        lr=DEFAULT_LR,
        use_pairwise=DEFAULT_USE_PAIRWISE,
    )

    trainer.evaluate_preferences(DEFAULT_DATA_FILE)


if __name__ == '__main__':
    main()
