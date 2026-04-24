# RGPO: Ranking-Guided Preference Optimization for Reliable Clinical Reasoning
 
A novel framework that enhances medical question answering by combining reinforcement learning with preference-driven reasoning refinement. RGPO automatically identifies and corrects low-quality reasoning chains to improve clinical chain-of-thought performance.
 
## Key Features
 
- **Reinforcement Learning with Preference Optimization**: Combines RL with preference-driven reasoning enhancement
- **Automatic Quality Assessment**: Probabilistic evaluation of reasoning chains across multiple dimensions
- **Groupwise Ranking Optimization**: Beyond traditional pairwise methods using Bradley-Terry model
- **Medical Domain Optimization**: Task-adaptive reasoning templates for biomedical and clinical contexts
- **Efficient Training**: Achieves superior performance with a smaller 2B model versus 7B–20B baselines
## Results
 
- Outperforms larger 7B–20B models (including medical-specialized variants) on PubMedQA, MedQA-USMLE, and the real-world FEMH clinical dataset
- Demonstrates that quality-driven refinement beats simple parameter scaling
- Achieves **62.02% accuracy on PubMedQA** and **51.67% accuracy on MedQA-USMLE** (5-shot setting)
- On the FEMH real-world clinical dataset, achieves BERTScore-F1 of **0.891** and Cosine Similarity of **0.528** (5-shot setting)
## Installation
 
```bash
pip install torch transformers openai tqdm matplotlib numpy
```
 
## Usage
 
### 1. Data Generation
 
Generate ranked reasoning pairs for RGPO training using the probabilistic refinement pipeline:
 
```bash
# Using OpenAI API (recommended)
export OPENAI_API_KEY="your-api-key"
export INPUT_PATH="/path/to/medical_qa_data.json"
export OUTPUT_PATH="/path/to/rgpo_cot_pairs.jsonl"
export ACCEPTANCE_THRESHOLD="0.6"
 
python cot_rgpo_pairs_5_to_4.py
```
 
```bash
# Using HuggingFace models (fallback)
export HF_MODEL="HuggingFaceH4/zephyr-7b-beta"
export INPUT_PATH="/path/to/medical_qa_data.json"
export OUTPUT_PATH="/path/to/rgpo_cot_pairs.jsonl"
 
python cot_rgpo_pairs_5_to_4.py
```
 
## Detailed Usage
 
### Data Generation Pipeline
 
The `cot_rgpo_pairs_5_to_4.py` script follows this 5→4 selection process:
 
1. **Generate 5 CoT variants** for each question using task-adaptive templates
2. **Score variants** on Coverage (0–5), Factual Accuracy (0–5), and Redundancy (0–1)
3. **Rank and select top 4** variants based on aggregate scores
4. **Apply probabilistic refinement** using acceptance threshold
5. **Output ranked data** in RGPO format
#### Environment Variables
 
```bash
# Data paths
export INPUT_PATH="/path/to/medical_data.json"
export OUTPUT_PATH="/path/to/rgpo_cot_pairs.jsonl"
 
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
Output: JSONL with ranked reasoning chains for RGPO training
 
### RGPO Training
 
The training script uses `JsonlRGPOTrainer` class with the following configuration:
 
```python
# Default configuration from rgpo_trainer.py
trainer = JsonlRGPOTrainer(
    model_name="google/gemma-2b",
    beta=0.1,         # KL divergence coefficient
    temperature=1.0   # Sampling temperature
)
 
trainer.train(
    data_file="/content/rgpo_cot_pairs.jsonl",
    output_dir="./rgpo_output",
    epochs=3,               # Training epochs
    batch_size=16,          # Batch size
    lr=5e-5,                # Learning rate
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
rgpo_output/
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
trainer.evaluate_preferences("/content/rgpo_cot_pairs.jsonl")
```
 
The evaluation computes ranking accuracy by checking if policy log-probabilities follow the expected order.
 
## Algorithm Details
 
### RGPO Loss Function
 
The total loss combines three components:
 
```
L_total = L_CoT-rank + L_KL + L_linear
```
 
- **L_CoT-rank**: Bradley-Terry pairwise ranking loss
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
P_accept(c) = p_cov(c) × p_fact(c) × (1 - p_red(c))
```
 
Refinement is triggered when P_accept(c) < threshold θ (default θ = 0.6).
 
## File Structure
 
```
├── cot_rgpo_pairs_5_to_4.py      # Entry point: RGPO dataset generation (5→4)
├── config.py                     # Runtime configuration and hyperparameters
├── prompts.py                    # Task-adaptive CoT prompt templates (τ_QA / τ_Diag)
├── scoring.py                    # Probabilistic quality assessment (P_accept, Eq. 3–4)
├── generation.py                 # LLM backend, CoT generation and revision
├── rgpo_trainer.py               # RGPO training implementation
├── loss_history.csv              # Training metrics
├── loss_curve.png                # Loss visualization
└── rgpo_output/
    ├── checkpoint-epoch-*/       # Model checkpoints
    └── final_model/             # Final trained model
```
 
## Citation
 
If you find this work useful, please consider citing our paper.
<!--
```bibtex
@article{hsu2025rgpo,
  title={RGPO: Ranking-Guided Preference Optimization for Reliable Clinical Reasoning},
  author={Hsu, Chia-Hsuan and Ding, Jun-En and Hsu, Hsin-Ling and Hsu, Chih-Ho and Yang, Shihao and Yao, Li-Hung and Liao, Chun-Chieh and Liu, Feng and Hung, Fang-Ming},
  journal={IEEE Transactions on Artificial Intelligence},
  year={2025}
}
```
-->
## License
 
This project is licensed under the MIT License.
