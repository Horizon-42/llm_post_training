# Reasoning Revolution: Two-Stage Training Pipeline for Chain-of-Thought Models

**Cold Start SFT + GRPO: Teaching Gemma 3 to Think Step-by-Step**

---

## Overview

This project implements a comprehensive two-stage training pipeline to enhance the reasoning capabilities of the Gemma 3 1B-IT model. By combining supervised fine-tuning (SFT) with Group Relative Policy Optimization (GRPO), we teach the model to produce structured reasoning outputs and improve its problem-solving accuracy.

## Key Features

### Two-Stage Training Approach

1. **Stage 1: Cold Start SFT (Supervised Fine-Tuning)**
   - Uses `PeftTrainer` from Tunix (following qlora_gemma approach)
   - Teaches the model the structured output format: `<reasoning>...</reasoning><answer>...</answer>`
   - Uses Bespoke-Stratos-17k dataset with `<think>` marker handling
   - Prevents chaotic outputs during reinforcement learning

2. **Stage 2: GRPO (Group Relative Policy Optimization)**
   - Memory-efficient RL algorithm (no separate value model needed)
   - Multiple reward functions for format and accuracy
   - Strengthens reasoning through reward-based learning

### Technical Highlights

- **Efficient Training**: LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- **Modern Framework**: Built with Tunix, JAX, and Flax NNX
- **Checkpoint Management**: Automatic checkpoint saving and loading
- **Comprehensive Evaluation**: Pre-SFT, post-SFT, and post-GRPO evaluation with detailed metrics
- **Data Handling**: Robust processing of `<think>` and `<think>` markers

## Architecture

The training pipeline follows a sequential two-stage architecture:

**Base Model**: Starts with Gemma 3 1B-IT (instruction-tuned) as the foundation model.

**Stage 1 - SFT (Supervised Fine-Tuning)**: 
- Applies LoRA adapters for parameter-efficient fine-tuning
- Uses PeftTrainer from Tunix framework
- Trains on Bespoke-Stratos-17k dataset to teach structured output format
- Output: Format-aware model with reasoning structure capabilities

**Stage 2 - GRPO (Group Relative Policy Optimization)**:
- Continues from SFT-trained model
- Applies reinforcement learning with group-based policy updates
- Trains on GSM8K dataset with reward signals
- Output: Enhanced reasoning model with improved accuracy

The architecture ensures smooth transfer of learned format patterns from SFT to GRPO, enabling the model to leverage structured reasoning during reinforcement learning.

## Datasets

### SFT Stage: Bespoke-Stratos-17k
- **Source**: HuggingFace `bespokelabs/Bespoke-Stratos-17k`
- **Purpose**: Teach reasoning format and structure
- **Features**: 
  - High-quality Chain-of-Thought examples
  - Handles `<think>`, `<think>`, and other thinking tags
  - Converts to standard `<reasoning>...</reasoning><answer>...</answer>` format

### GRPO Stage: GSM8K
- **Source**: Kaggle `thedevastator/grade-school-math-8k-q-a`
- **Purpose**: Strengthen reasoning through reward signals
- **Features**:
  - Verifiable answers for reward computation
  - Math word problems with clear correct/incorrect signals

## Training Pipeline

1. **Environment Setup**: Install dependencies, authenticate services
2. **Model Loading**: Download Gemma 3 1B-IT, apply LoRA adapters
3. **Data Preparation**: Load and format datasets with marker conversion
4. **Cold Start SFT**: Train with PeftTrainer using Bespoke-Stratos-17k
5. **Evaluation**: Compare pre-SFT and post-SFT performance
6. **GRPO Training**: Reinforcement learning with multiple reward functions
7. **Final Evaluation**: Assess complete training pipeline results
8. **Model Export**: Merge LoRA weights and save final model

## Output Format

The trained model produces structured reasoning outputs:

```
<reasoning>
Step-by-step thinking process...
</reasoning>
<answer>
Final answer
</answer>
```

## Requirements

- **Hardware**: Kaggle TPU v6e-1 (recommended) or similar TPU configuration
- **Libraries**: Tunix, Qwix, JAX, Flax NNX, Grain, Transformers
- **Training Time**: ~9 hours for full pipeline
- **Storage**: Sufficient space for model checkpoints and datasets

## Key Innovations

1. **PeftTrainer Integration**: Uses Tunix's built-in trainer instead of custom training loops
2. **Marker Handling**: Robust conversion of various thinking tags (`<think>`, `<think>`, etc.)
3. **Checkpoint Management**: Automatic loading of trained parameters for evaluation
4. **Comprehensive Metrics**: Format accuracy, answer accuracy, and partial accuracy tracking

## Results

The pipeline tracks improvements across three stages:
- **Before SFT**: Baseline performance
- **After SFT**: Format learning and initial reasoning improvements
- **After GRPO**: Enhanced reasoning accuracy through reinforcement learning

## Usage

1. Configure hyperparameters in the configuration cell
2. Run environment setup and authentication
3. Execute SFT training (Stage 1)
4. Evaluate SFT results
5. Execute GRPO training (Stage 2)
6. Evaluate final results
7. Export trained model

## References

- [Tunix GitHub](https://github.com/google/tunix)
- [GRPO Paper](https://arxiv.org/pdf/2402.03300)
- [DeepSeek-R1 Technical Report](https://arxiv.org/abs/2401.02954)
- [Bespoke-Stratos Dataset](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k)

---

**Status**: Production-ready training pipeline for reasoning model development

