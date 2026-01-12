# Cold Start SFT Training Guide

## Overview

This document explains the Supervised Fine-Tuning (SFT) stage used for "Cold Start" training in the DeepSeek-R1 style reasoning model pipeline. Cold Start SFT is essential for teaching the model the proper reasoning format before reinforcement learning (GRPO) training.

---

## 1. What is Cold Start SFT?

### 1.1 The Full DeepSeek-R1 Pipeline

According to the DeepSeek-R1 technical report [1], the complete training pipeline is:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Cold-Start SFT │ -> │   RL Stage 1    │ -> │ Rejection       │ -> │   RL Stage 2    │
│  (thousands of  │    │   (GRPO with    │    │ Sampling + SFT  │    │ (Helpfulness &  │
│   examples)     │    │  rule rewards)  │    │ (reasoning +    │    │  Harmlessness)  │
│                 │    │                 │    │  non-reasoning) │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

> "In the initial stage, we collect thousands of cold-start data that exhibits a conversational, human-aligned thinking process." — DeepSeek-R1 [1]

### 1.2 Purpose

Cold Start SFT serves as the **initialization phase** before GRPO training. Its primary goals are:

1. **Format Learning**: Teach the model to output in a structured format:
   ```
   <reasoning>step-by-step thinking process</reasoning>
   <answer>final answer</answer>
   ```

2. **Reasoning Template Injection**: Show the model examples of high-quality Chain-of-Thought (CoT) reasoning patterns

3. **Distribution Alignment**: Prevent the model from producing chaotic or unreadable outputs during RL training

### 1.2 Why is Cold Start Necessary?

Research from DeepSeek-R1 [1] shows that pure RL training (without cold start) causes:

- **Language Mixing**: Random switching between languages
- **Format Collapse**: Unstructured, hard-to-parse outputs
- **Inefficient Exploration**: Model struggles to find good reasoning strategies

> "We find that with a small amount of cold-start data, DeepSeek-R1 can achieve reasoning performance comparable to OpenAI-o1-1217 while maintaining coherent and readable outputs." — DeepSeek-R1 Technical Report [1]

---

## 2. LoRA (Low-Rank Adaptation) Training

### 2.1 What is LoRA?

LoRA [2] is a parameter-efficient fine-tuning technique that:

1. **Freezes** the pre-trained model weights
2. **Injects trainable low-rank matrices** into specific layers
3. **Trains only these small adapters** (typically <1% of total parameters)

### 2.2 Mathematical Formulation

For a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA represents the update as:

$$W = W_0 + \Delta W = W_0 + BA$$

Where:
- $B \in \mathbb{R}^{d \times r}$ (down-projection)
- $A \in \mathbb{R}^{r \times k}$ (up-projection)
- $r \ll \min(d, k)$ is the **rank** (typically 8-64)

### 2.3 LoRA Configuration in Our Training

```python
# LoRA parameters used in this notebook
RANK = 64       # Rank of LoRA matrices
ALPHA = 64.0    # Scaling factor (typically equal to rank)

# Target modules for Gemma3
target_modules = [
    "q_einsum",      # Query projection
    "kv_einsum",     # Key-Value projection
    "gate_proj",     # MLP gate
    "down_proj",     # MLP down projection
    "up_proj",       # MLP up projection
    "attn_vec_einsum" # Attention output
]
```

### 2.4 Benefits of LoRA

| Aspect | Full Fine-tuning | LoRA |
|--------|-----------------|------|
| **Memory** | ~4x model size | ~1.1x model size |
| **Training Speed** | Baseline | 2-3x faster |
| **Storage** | Full model per task | Small adapter per task |
| **Risk of Forgetting** | High | Low |

---

## 3. Loss Function for SFT

### 3.1 Cross-Entropy Loss (Standard Language Modeling)

The standard loss function for SFT is **Cross-Entropy Loss** for next-token prediction:

$$\mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \log P(x_t^{(i)} | x_{<t}^{(i)}; \theta)$$

Where:
- $N$ = batch size
- $T$ = sequence length
- $x_t^{(i)}$ = target token at position $t$ for example $i$
- $x_{<t}^{(i)}$ = all previous tokens (context)
- $\theta$ = model parameters

### 3.2 Implementation in JAX/Flax

```python
def compute_sft_loss(logits, targets, loss_mask):
    """
    Compute cross-entropy loss for language modeling.
    
    Args:
        logits: Model output, shape (batch, seq_len, vocab_size)
        targets: Target token IDs, shape (batch, seq_len)
        loss_mask: Binary mask for valid tokens, shape (batch, seq_len)
    
    Returns:
        Scalar loss value
    """
    # Convert logits to log probabilities
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    
    # Gather log probabilities for target tokens
    # Shape: (batch, seq_len)
    target_log_probs = jnp.take_along_axis(
        log_probs, 
        targets[:, :, None], 
        axis=-1
    ).squeeze(-1)
    
    # Apply mask and compute mean loss
    masked_loss = -target_log_probs * loss_mask
    loss = jnp.sum(masked_loss) / (jnp.sum(loss_mask) + 1e-8)
    
    return loss
```

### 3.3 Loss Masking

We use **loss masking** to:
1. **Ignore padding tokens**: Don't penalize predictions on `[PAD]` tokens
2. **Focus on completion**: Optionally mask the prompt portion

```python
# Example: Create loss mask
attention_mask = (tokens != PAD_TOKEN_ID).astype(jnp.float32)
loss_mask = attention_mask[:, 1:]  # Shift for next-token prediction
```

---

## 4. Chain-of-Thought (CoT) Training

### 4.1 What Makes CoT Training Different?

Chain-of-Thought training differs from standard instruction tuning in several ways:

| Aspect | Standard Instruction Tuning | CoT Training |
|--------|---------------------------|--------------|
| **Output Length** | Short, direct answers | Long reasoning traces |
| **Structure** | Free-form | Structured (tags, steps) |
| **Data Quality** | Quantity over quality | Quality over quantity |
| **Loss Focus** | Entire response | Reasoning + Answer |

### 4.2 Loss Functions for CoT Training

#### Option A: Standard Cross-Entropy (Most Common)

The most common approach is to use standard cross-entropy loss on the **entire** response (reasoning + answer):

$$\mathcal{L} = -\sum_{t \in \text{response}} \log P(x_t | x_{<t})$$

This is what we use in our implementation.

#### Option B: Weighted Loss (Research Approach)

Some research [3] proposes weighting different parts of the response:

$$\mathcal{L} = -\alpha \sum_{t \in \text{reasoning}} \log P(x_t | x_{<t}) - \beta \sum_{t \in \text{answer}} \log P(x_t | x_{<t})$$

Where:
- $\alpha$ = weight for reasoning tokens (typically 1.0)
- $\beta$ = weight for answer tokens (can be higher, e.g., 2.0)

#### Option C: Process Reward Model (Advanced)

More advanced approaches use Process Reward Models (PRM) [4] to provide step-level feedback:

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{PRM}$$

This is typically used in RL phases, not SFT.

### 4.3 Best Practices for CoT SFT

1. **Use High-Quality Data**: Datasets like Bespoke-Stratos-17k [5] contain carefully curated reasoning traces

2. **Preserve Long Sequences**: CoT requires longer context (4K-8K tokens)

3. **Include Diverse Reasoning Patterns**:
   - Self-verification: "Let me check..."
   - Backtracking: "Wait, that's wrong..."
   - Step decomposition: "First... Then... Finally..."

4. **Maintain Format Consistency**: Always use the same tags (`<reasoning>`, `<answer>`)

---

## 5. Training Hyperparameters

### 5.1 Recommended Settings for Cold Start SFT

```python
# Learning rate (higher than GRPO)
SFT_LEARNING_RATE = 2e-4

# Batch size (limited by memory)
SFT_BATCH_SIZE = 2
SFT_GRADIENT_ACCUMULATION = 4  # Effective batch = 8

# Training duration
SFT_MAX_STEPS = 500  # ~1 epoch on 17k examples

# Learning rate schedule
SFT_WARMUP_STEPS = 50  # 10% warmup
# Then cosine decay to 0

# Regularization
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 0.1  # Gradient clipping
```

### 5.2 Why These Settings?

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Learning Rate** | 2e-4 | Higher than RL (3e-6) because SFT is more stable |
| **Warmup** | 10% | Prevents early training instability |
| **Cosine Decay** | Yes | Smooth convergence |
| **Gradient Clipping** | 0.1 | Prevents exploding gradients on long sequences |
| **Weight Decay** | 0.1 | Regularization for LoRA parameters |

---

## 6. Monitoring Training

### 6.1 Key Metrics

1. **Training Loss**: Should decrease smoothly
   - Initial: ~2-4 (depending on model)
   - Final: ~0.5-1.5

2. **Gradient Norm**: Monitor for stability
   - Should stay below `MAX_GRAD_NORM` after clipping

3. **Learning Rate**: Verify schedule is working
   - Should follow warmup → peak → decay

### 6.2 Signs of Problems

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| Loss spikes | Learning rate too high | Reduce LR or increase warmup |
| Loss plateaus | Learning rate too low | Increase LR |
| NaN loss | Numerical instability | Add gradient clipping, check data |
| Very low loss (<0.1) | Overfitting | Reduce steps, add regularization |

---

## 7. Dataset: Bespoke-Stratos-17k

### 7.1 Dataset Overview

**Bespoke-Stratos-17k** [5] is a high-quality Chain-of-Thought dataset:

- **Size**: ~17,000 examples
- **Source**: Distilled from DeepSeek-R1
- **Format**: Conversations with `<|begin_of_thought|>` tags
- **Quality**: Long, detailed reasoning traces with self-reflection

### 7.2 Data Format

Original format:
```json
{
  "conversations": [
    {"from": "human", "value": "What is 15% of 80?"},
    {"from": "gpt", "value": "<|begin_of_thought|>\nLet me calculate 15% of 80...\n<|end_of_thought|>\n<|begin_of_solution|>\n12\n<|end_of_solution|>"}
  ]
}
```

Converted format for our training:
```
<start_of_turn>user
You are given a problem. First, think about the problem and provide your reasoning...

What is 15% of 80?<end_of_turn>
<start_of_turn>model
<reasoning>
Let me calculate 15% of 80...
</reasoning>
<answer>12</answer>
```

---

## 8. DeepSeek-R1 GRPO Hyperparameters (Reference)

According to the DeepSeek-R1 paper [1], the official GRPO hyperparameters are:

### 8.1 First RL Stage

```python
# DeepSeek-R1 Official Settings
LEARNING_RATE = 3e-6
KL_COEFFICIENT = 0.001      # Much lower than typical PPO
EPSILON = 10                # Clip ratio (much higher than PPO's 0.2!)
TEMPERATURE = 1.0           # High temperature for diverse exploration
NUM_GENERATIONS = 16        # 16 outputs per question
MAX_LENGTH = 32768          # Very long sequences
BATCH_SIZE = 32             # Questions per step (effective batch = 32 × 16 = 512)
REFERENCE_UPDATE_STEPS = 400  # Update reference model every 400 steps
```

### 8.2 Language Consistency Reward

DeepSeek-R1 introduces a **language consistency reward** to prevent language mixing:

$$Reward_{language} = \frac{Num(Words_{target})}{Num(Words)}$$

This reward is added to both reasoning and non-reasoning data.

### 8.3 Second RL Stage

```python
TEMPERATURE = 0.7           # Lower for coherent generation
TOTAL_STEPS = 1700
# General instruction data added in final 400 steps only
```

### 8.4 Combined Reward Function

$$Reward = Reward_{reasoning} + Reward_{general} + Reward_{language}$$

Where:
- $Reward_{reasoning} = Reward_{rule}$ (format + accuracy)
- $Reward_{general} = Reward_{reward\_model} + Reward_{format}$

---

## References

[1] DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948, 2025. https://arxiv.org/abs/2501.12948
    - Key source for training pipeline and GRPO hyperparameters (EPSILON=10, BETA=0.001, etc.)

[2] Hu, E. J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685, 2021.

[3] Lightman, H., et al. "Let's Verify Step by Step." arXiv:2305.20050, 2023.

[4] Wang, P., et al. "Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations." arXiv:2312.08935, 2023.

[5] Bespoke Labs. "Bespoke-Stratos-17k." HuggingFace Datasets, 2024. https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k

[6] Google. "Tunix: Training Gemma Models on TPU." GitHub, 2024. https://github.com/google/tunix

[7] Google. "Qwix: LoRA and Quantization for JAX." GitHub, 2024. https://github.com/google/qwix

---

## Appendix: Full SFT Training Code

See `gemma_coldstart_grpo_training.ipynb` Cell 12 for the complete implementation.

Key components:
1. `load_bespoke_stratos_dataset()` - Data loading and formatting
2. `sft_train_step()` - JIT-compiled training step
3. Checkpoint saving with Orbax
4. TensorBoard logging for monitoring

