# Constraint Decomposition for Multi-Objective RLHF

Early-stage research exploring decomposed reward modeling for complex instruction-following in large language models.

## Core Problem

Standard RLHF uses monolithic reward models: `R(prompt, response) → scalar`

This creates issues for multi-constraint instructions:
- **Example:** "Solve x²+5x+6=0 step-by-step in <100 words for high schoolers"
- **Constraints:** Correctness + Structure + Format + Audience
- **Problem:** When constraints conflict, a single scalar can't distinguish which failed

## My Approach

Decompose reward into orthogonal components:
R(x,y) = f(R_semantic, R_structural, R_format, R_meta)
where:
R_semantic:   Factual accuracy, logical validity
R_structural: Reasoning organization, step clarity
R_format:     Length, style, presentation constraints
R_meta:       Safety, tone, audience appropriateness

### Architecture
                Input: [Prompt + Response]
                          ↓
                Shared Encoder (7B params)
                          ↓
                Hidden State (4096-dim)
                     ↙  ↓  ↓  ↘
                [Sem][Str][Fmt][Meta]  ← 4 lightweight heads
                   ↓    ↓    ↓    ↓
                 0.92  0.85  0.30  0.95  ← component scores
                          ↓
                Hierarchical Combination
                          ↓
                     R_total = 0.70

### Hierarchical Combination

Not linear combination - structured composition:

1. **Safety Gate:** R_meta < 0.7 → reject (hard constraint)
2. **Content Base:** (w_sem × R_sem) + (w_struct × R_struct)
3. **Format Modulation:** Base × (0.8 + 0.2 × w_fmt × R_fmt)

This ensures safety is non-negotiable, content is primary, format modulates but doesn't dominate.

## Preliminary Results

Tested on IFEval benchmark (541 prompts with verifiable constraints):

| Method | Accuracy | Δ |
|--------|----------|---|
| Baseline RLHF | 41.2% | - |
| **Decomposed** | **73.8%** | **+32.6 pts** |

Also tested generalization:
- GSM8K (math): 72.3% → 87.9% (+15.6 pts)
- HumanEval (code): 65.2% → 76.3% (+11.1 pts)

See `results/` for detailed breakdowns.

## Repository Contents

- `reward_model.py`: Decomposed reward architecture (PyTorch)
- `ppo_training.py`: RL training loop with decomposed rewards
- `evaluation.py`: IFEval evaluation script
- `results/`: Experimental results and comparison plots
- `docs/technical_details.md`: Extended methodology

## Status

⚠️ **Work in Progress** ⚠️

This is exploratory research - prototype quality code, not production-ready:

- ✅ Core architecture validated
- ✅ Preliminary results promising
- ⚠️ Many edge cases unsolved (exact counting, 7+ constraints)
- ⚠️ Not extensively tested for adversarial robustness
- ⚠️ Needs optimization for production deployment

## Key Limitations

1. **Precise counting fails** (26% of errors): "Exactly 100 words" → 103 words
2. **Complex instructions struggle** (5+ constraints): Performance degrades
3. **Manual decomposition**: 4 components hand-designed, not learned
4. **Annotation cost**: Aspect-level labels require 3× more effort

## Future Directions

1. **Multimodal extension**: Apply to avatar generation (visual + audio + expression sync)
2. **Hierarchical decomposition**: Split components further for complex instructions
3. **Automatic discovery**: Learn decomposition structure from data
4. **Constrained decoding**: Combine learned soft constraints with algorithmic hard constraints


Hierarchical combination ensures:
- Safety/appropriateness is gated
- Semantic coherence is primary
- Synchronization modulates quality

Real-time constraints add challenge - exploring efficient architectures.

## Requirements
```bash
pip install -r requirements.txt

PyTorch 2.1+
Transformers 4.36+
Datasets
NumPy, matplotlib

Quick Start
bash# Install dependencies
pip install -r requirements.txt

# Evaluate baseline model
python evaluation.py --model baseline --benchmark ifeval

# Train decomposed reward model (requires labeled data)
python reward_model.py --train --data data/preferences.jsonl

# Run RL training (requires GPU cluster)
python ppo_training.py --config configs/decomposed_ppo.yaml
Citation
This work is currently unpublished. If you find it useful, please reference this repository:
@misc{paunova2024constraint,
  title={Constraint Decomposition for Multi-Objective LLM Alignment},
  author={Paunova, Eva},
  year={2024},
  note={Work in progress, available at github.com/[your-username]/constraint-decomposition-llm}
}
Contact
Eva Paunova 
PhD, ETH Zürich | NVIDIA Research
Questions, feedback, or collaboration ideas welcome!
License
MIT License - see LICENSE file
