# RPRO: Ranked Preference Reinforcement Optimization

A novel framework that enhances medical question answering by combining reinforcement learning with preference-driven reasoning refinement. RPRO automatically identifies and corrects low-quality reasoning chains to improve clinical chain-of-thought performance.

## Key Features

- **Reinforcement Learning with Preference Optimization**: Combines RL with preference-driven reasoning enhancement
- **Automatic Quality Assessment**: Probabilistic evaluation of reasoning chains across multiple dimensions
- **Groupwise Ranking Optimization**: Beyond traditional pairwise methods using Bradley-Terry model
- **Medical Domain Optimization**: Task-adaptive reasoning templates for biomedical and clinical contexts
- **Efficient Training**: Achieves superior performance with smaller models (1.1B vs 7B-13B parameters)

## Results

- Outperforms larger 7B-13B models on PubMedQA and MedQA-USMLE benchmarks
- Demonstrates that quality-driven refinement beats simple parameter scaling
- Achieves 61.95% accuracy on PubMedQA and 27.92% accuracy on MedQA-USMLE

## Installation

```bash
pip install torch transformers openai tqdm matplotlib numpy
```

## Usage

### 1. Data Generation

Generate ranked reasoning pairs for RPRO training using the probabilistic refinement pipeline:

```bash
# Using OpenAI API (recommended)
export OPENAI_API_KEY="your-api-key"
export INPUT_PATH="/path/to/medical_qa_data.json"
export OUTPUT_PATH="/path/to/rpro_cot_pairs.jsonl"
export ACCEPTANCE_THRESHOLD="0.6"

python data_generation.py
```

```bash
# Using HuggingFace models (fallback)
export HF_MODEL="HuggingFaceH4/zephyr-7b-beta"
export INPUT_PATH="/path/to/medical_qa_data.json"
export OUTPUT_PATH="/path/to/rpro_cot_pairs.jsonl"

python data_generation.py
```

## Detailed Usage

### Data Generation Pipeline

The `cot_rpro_pairs_5_to_4.py` script follows this 5→4 selection process:

1. **Generate 5 CoT variants** for each question using task-adaptive templates
2. **Score variants** on Coverage, Factual Accuracy, and Redundancy (0-5 scale)
3. **Rank and select top 4** variants based on aggregate scores
4. **Apply probabilistic refinement** using acceptance threshold
5. **Output ranked data** in RPRO format

#### Environment Variables

```bash
# Data paths
export INPUT_PATH="/path/to/medical_data.json"
export OUTPUT_PATH="/path/to/rpro_cot_pairs.jsonl"

# Model configuration (choose one)
export OPENAI_API_KEY="your-api-key"                    # Use OpenAI
export HF_MODEL="HuggingFaceH4/zephyr-7b-beta"         # Use HuggingFace

# Refinement parameters
export ACCEPTANCE_THRESHOLD="0.6"                       # Probabilistic threshold
export OPENAI_MODEL_GENERATE="gpt-4o-mini"             # Generation model
export OPENAI_MODEL_SCORE="gpt-4o-mini"                # Scoring model
export OPENAI_MODEL_REVISE="gpt-4o-mini"               # Revision model
```

#### Data Format

Input: Medical QA data with `QUESTION`, `CONTEXTS`, and `final_decision` fields
Output: JSONL with ranked reasoning chains for RPRO training

### RPRO Training

The training script uses `JsonlRPROTrainer` class with the following configuration:

```python
# Default configuration from rpro_trainer.py
trainer = JsonlRPROTrainer(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    beta=0.01,        # KL divergence coefficient
    temperature=1.0   # Sampling temperature
)

trainer.train(
    data_file="/content/rpro_cot_pairs.jsonl",
    output_dir="./rpro_output",
    epochs=30,        # Training epochs
    batch_size=1,     # Batch size (memory constrained)
    lr=1e-6,          # Learning rate
    use_pairwise=True,      # Enable Bradley-Terry loss
    pairwise_weight=0.1     # Pairwise loss weight
)
```

#### Training Components

- **Linear Reward Shaping**: `_linear_rewards_from_rank()` converts rankings to advantages
- **KL Divergence Regularization**: Prevents policy drift from reference model
- **Bradley-Terry Ranking Loss**: `compute_pairwise_loss()` for full ranking information
- **Gradient Checkpointing**: Memory-efficient training for longer sequences
- **Dynamic Padding**: Efficient batching with `collate_fn()`

#### Training Outputs

```
rpro_output/
├── checkpoint-epoch-1/          # Model checkpoints
├── checkpoint-epoch-2/
├── ...
├── final_model/                 # Final trained model
├── loss_history.csv            # Training metrics
└── loss_curve.png              # Loss visualization
```

#### Memory Optimization

```python
# Set environment variable for GPU memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

### Evaluation

Evaluate ranking accuracy on test data:

```python
trainer.evaluate_preferences("/content/rpro_cot_pairs.jsonl")
```

The evaluation computes ranking accuracy by checking if policy log-probabilities follow the expected order.

## Algorithm Details

### RPRO Loss Function

The total loss combines three components:

```
L_total = L_ranking + L_KL + L_linear
```

- **L_ranking**: Bradley-Terry pairwise ranking loss
- **L_KL**: KL divergence regularization term  
- **L_linear**: Linear advantage-weighted policy gradient

### Linear Reward Shaping

For K ranked candidates, advantages are calculated as:

```
a_j = (K - r_j) - (K-1)/2
```

Where r_j is the rank of candidate j (1 = best, K = worst).

### Probabilistic Refinement

The acceptance probability is calculated as:

```
P(accept) = P(coverage) × P(accuracy) × (1 - P(redundancy))
```

Refinement is triggered when P(accept) < threshold.

## File Structure

```
├── rpro_trainer.py           # Main RPRO training implementation
├── cot_rpro_pairs_5_to_4.py  # Data generation pipeline
├── loss_history.csv          # Training metrics
├── loss_curve.png           # Loss visualization
└── rpro_output/
    ├── checkpoint-epoch-*/   # Model checkpoints
    └── final_model/         # Final trained model
```

## Citation

```bibtex
@article{hsu2025rpro,
  title={RPRO: Ranked Preference Reinforcement Optimization for Enhancing Medical QA and Diagnostic Reasoning},
  author={Hsu, Chia-Hsuan and Ding, Jun-En and Hsu, Hsin-Ling and Liu, Feng and Hung, Fang-Ming},
  journal={arXiv preprint arXiv:2509.00974},
  year={2025}
}
```

## License

This project is licensed under the MIT License.
