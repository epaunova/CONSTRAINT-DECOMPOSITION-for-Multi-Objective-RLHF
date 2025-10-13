# Technical Details: Constraint Decomposition

## Problem Formulation

Standard RLHF optimizes:
π* = argmax E[R(x,y) - β·KL(π || π_ref)]

where R(x,y) is a **monolithic scalar** reward.

**Issue:** Multi-constraint instructions bundle orthogonal requirements:
- "Solve x²+5x+6=0 **step-by-step** in **<100 words** for **high schoolers**"
- Conflates: correctness + structure + format + audience

When constraints conflict, scalar R can't distinguish which failed.

---

## Our Approach: Decomposition

Factor reward into orthogonal components:
R(x,y) = f(R_semantic, R_structural, R_format, R_meta)

### Component Definitions

| Component | What It Measures | Examples |
|-----------|-----------------|----------|
| **R_semantic** | Factual accuracy, logical validity | Math correctness, no contradictions |
| **R_structural** | Organization, reasoning clarity | Step-by-step flow, transitions |
| **R_format** | Length, style, presentation | Word count, bullet points, tone |
| **R_meta** | Safety, appropriateness | No harmful content, audience-appropriate |

### Architecture
Input: [Prompt, Response] concatenated
↓
Shared Encoder: 7B Transformer (Nemotron)
↓
Hidden State: [batch, 4096]
↓
Split into 4 heads:

Linear(4096 → 1): R_semantic
Linear(4096 → 1): R_structural
Linear(4096 → 1): R_format
Linear(4096 → 1): R_meta


**Key advantage:** Shared encoder (parameter-efficient), independent heads (clear signals).

---

## Hierarchical Combination

Not simple weighted sum. Structured composition:
```python
if R_meta < 0.7:
    return -5.0  # Safety gate: reject unsafe responses
    
content_base = w_sem * R_sem + w_struct * R_struct

format_modulation = 0.8 + 0.2 * w_fmt * R_fmt

R_total = content_base * format_modulation
Rationale:

Safety is non-negotiable → hard threshold
Content is primary → semantic + structural form base
Format modulates → can reduce by max 20%, not dominate


Training
Phase 1: Reward Model
Data: 180k preference pairs with aspect-level labels:
json{
  "prompt": "Solve x² = 4",
  "response_A": "x = ±2 (correct, terse)",
  "response_B": "x = 2 (incorrect, but well-explained)",
  "labels": {
    "semantic": "A",      # A is more correct
    "structural": "B",    # B has better explanation
    "format": "tie",
    "meta": "tie"
  }
}
Loss: Bradley-Terry per aspect:
P(A >_i B) = σ(R_i(A) - R_i(B))
L_i = -ℓ_i log P(A >_i B) - (1-ℓ_i) log P(B >_i A)
L_total = Σ w_i L_i
Training: 3 epochs, batch size 32, lr=5e-6, ~12 hours on 8×A100.
Phase 2: RL with PPO
Loop (10k iterations):

Rollout: Generate 2048 prompts × 4 responses = 8192 samples
Score: Decomposed reward model evaluates all
Advantages: Per-prompt baseline (reduce variance)
PPO Update: 4 epochs, clip ε=0.2

Time: ~191 sec/iteration, ~530 GPU-hours total.
Stability mechanisms:

Adaptive KL penalty (target 0.15-0.25)
Gradient clipping (max norm 1.0)
Value function separate learning rate (10× policy)


Results Summary
IFEval Benchmark
MetricBaselineOursΔPrompt-level41.2%73.8%+32.6 ptsInstruction-level68.7%87.4%+18.7 pts
Statistical significance: p < 0.0001, Cohen's d = 8.97 (5 runs)
Generalization
BenchmarkBaselineOursΔGSM8K (math)72.3%87.9%+15.6 ptsHumanEval (code)65.2%76.3%+11.1 ptsMT-Bench7.128.52+1.40 pts

Ablation Studies
ConfigurationIFEvalContributionBaseline (monolithic)41.2%-+ Decomposition only58.7%+17.5 pts (54%)+ Hierarchical combination64.3%+5.6 pts (17%)+ Weight prediction69.1%+4.8 pts (15%)+ Conflict detection73.8%+4.7 pts (14%)
Key insight: Decomposition itself contributes majority of improvement.

Failure Modes
Remaining 26.2% errors:

Precise counting (26%): "Exactly 100 words" → 103 words

Root cause: Tokenization mismatch, no lookahead


Complex instructions (18%): 5+ simultaneous constraints

Combinatorial explosion


Ambiguous constraints (12%): "Be creative but accurate"

Inherent tension, no ground truth


Rare constraints (7%): <10 training examples

Data sparsity
Limitations

Annotation cost: 3× more effort than overall preferences
Fixed decomposition: 4 components hand-designed
Counting errors: Precise numerical constraints difficult
Architecture assumption: Requires shared encoder


Future Directions

Automatic decomposition discovery: Learn optimal structure from data
Hierarchical refinement: Split components further as needed
Constrained decoding: Combine learned soft + algorithmic hard constraints
Multi-modal extension: Apply to vision + audio + text


Code Organization
constraint-decomposition-llm/
├── reward_model.py          # Core decomposed architecture
├── ppo_training.py          # RL training loop
├── evaluation.py            # IFEval evaluation harness
├── results/                 # Experimental results
│   ├── baseline_results.json
│   ├── decomposed_results.json
│   └── comparison_plot.png
└── docs/
    └── technical_details.md # This file

References

IFEval: Zhou et al., "Instruction-Following Evaluation for Large Language Models", 2023
PPO: Schulman et al., "Proximal Policy Optimization", 2017
RLHF: Ouyang et al., "Training Language Models to Follow Instructions", 2022
Bradley-Terry: Bradley & Terry, "Rank Analysis of Incomplete Block Designs", 1952


Contact
For questions or collaboration:

Author: Eva Paunova
Affiliation: ETH Zürich (PhD) | NVIDIA Research

