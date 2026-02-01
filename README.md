# Annotated S4 (Part 1 & Part 2)

This repository implements **Part 1 and Part 2** of  
**“The Annotated S4: Efficiently Modeling Long Sequences with Structured State Spaces”**
(Rush & Karamcheti, 2022).

---

## Implemented

### Part 1 — State Space Models
- Continuous-time linear SSMs
- Bilinear (Tustin) discretization
- RNN formulation using `jax.lax.scan`
- CNN formulation via convolution
- FFT-based non-circular convolution
- HiPPO matrix construction

### Part 2 — S4 Acceleration
- SSM convolution kernels
- Truncated z-transform
- FFT-based kernel recovery
- Diagonal SSMs (Cauchy kernel)
- Diagonal Plus Low-Rank (DPLR) systems
- Woodbury identity for fast computation
- NPLR form of HiPPO matrices


## Key Ideas

### State Space Models (SSMs)

An SSM maps an input sequence \( u(t) \) to an output \( y(t) \) via a latent state \( x(t) \):

\[
\begin{aligned}
x'(t) &= A x(t) + B u(t) \\
y(t) &= C x(t) + D u(t)
\end{aligned}
\]

After discretization, this can be evaluated either as:
- an **RNN** (sequential scan), or
- a **CNN** (convolution with a learned kernel)

---

### CNN–RNN Duality

The same SSM admits two equivalent forms:

- **RNN form** — sequential, intuitive, and memory-efficient  
- **CNN form** — parallelizable and FFT-accelerated  

Part 1 demonstrates this equivalence explicitly.

---

### HiPPO Matrices

HiPPO (High-Order Polynomial Projection Operators) define structured transition matrices
that preserve long-term history by projecting inputs onto orthogonal polynomial bases.
These matrices are critical for stable long-sequence modeling.

---

### S4 Acceleration (Part 2)

Part 2 exploits structure in the transition matrix \( A \):

- Diagonal systems allow closed-form kernel computation
- DPLR systems enable fast inversion using Woodbury identities
- HiPPO matrices can be transformed into NPLR / DPLR form

This reduces kernel computation from **O(N²L)** to **O(NL)**.
Here is the **clean, professional, fully copyable `README.md`** with **no emojis**.

---

# Fish Speech — Mamba + S4 Autoregressive Core
**VISIT THIS REPO https://github.com/Kushagra-o7/fish-speech FOR THE FILES, THE FILE HYPERLINK IN THE REPO GIVES 404 FOR SOME REASON**

This project replaces the Transformer-based autoregressive (AR) decoder in **Fish Speech** with a **State Space Model (SSM)** stack composed of **Mamba** and **S4**, enabling faster inference, lower memory usage, and better long-context stability while preserving the original VITS-based text-to-speech pipeline.

The work focuses on architectural substitution rather than retraining the entire system, making it a clean testbed for evaluating modern SSMs in neural speech generation.

---

## Motivation

Transformer autoregressive decoders dominate modern TTS systems but suffer from:

* Quadratic memory and latency scaling
* KV-cache complexity at inference
* Poor extrapolation to long sequences

Recent SSM architectures — **Mamba** and **S4** — provide:

* Linear-time inference
* Implicit causality without attention masks
* Strong long-context modeling

This project evaluates whether such models can directly replace the Transformer decoder inside a production-scale TTS system without architectural redesign.

---

## System Overview

### Original Fish Speech (Decoder Path)

```
Text → Embedding → Transformer Decoder × N → Acoustic Tokens → Vocoder → Audio
```

### Modified Architecture

```
Text → Embedding → (Mamba → S4) × N → Acoustic Tokens → Vocoder → Audio
```

Key properties:

* No attention layers
* No rotary embeddings
* No causal masks
* No KV cache
* Fully autoregressive via SSM recurrence

---

## Implementation Summary

The modification targets the **autoregressive decoder stack** in:

```
fish_speech/models/text2semantic/llama.py
```

The Transformer block stack is replaced with a **Mamba + S4 hybrid stack**, implemented in:

```
fish_speech/mamba_s4.py
```

Each layer applies:

1. Pre-norm Mamba block
2. Residual connection
3. Pre-norm S4 block
4. Residual connection

This preserves architectural depth and embedding dimensionality while changing only the sequence modeling core.

---

Changes:

* Replaced `TransformerBlock` stack with `MambaS4Stack`
* Removed rotary embeddings
* Removed attention masks
* Removed KV cache logic
* Simplified forward pass to pure SSM recurrence

## Installation

### Environment

```bash
conda create -n mamba-fish python=3.10 -y
conda activate mamba-fish
conda install pytorch==2.3.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install numpy==1.26.4
```

### Dependencies

```bash
pip install mamba-ssm==2.1.1 causal-conv1d==1.3.0
pip install git+https://github.com/TariqAHassan/S4Torch.git
pip install -e .
```

---

## Usage

### Load Model

```python
from fish_speech.models.text2semantic.llama import BaseTransformer

model = BaseTransformer.from_pretrained("your_checkpoint")
```

### Inference

```python
audio = model.generate("Hello world.")
```

No inference-time caching or masking is required — causality is handled internally by Mamba and S4.

---

## Training

LoRA fine-tuning is supported without modification.

```python
from peft import LoraConfig, get_peft_model

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["mamba", "s4"],
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_cfg)
```

Training scripts remain unchanged:

```bash
python -m fish_speech.train --config configs/train.yaml
```

---

## Observed Properties

| Property               | Transformer | Mamba + S4   |
| ---------------------- | ----------- | ------------ |
| Time Complexity        | O(T²)       | O(T)         |
| Memory                 | Quadratic   | Linear       |
| KV Cache               | Required    | Not required |
| Long-context stability | Degrades    | Stable       |
| Latency                | Higher      | Lower        |

---

## Design Rationale

| Component            | Reason                            |
| -------------------- | --------------------------------- |
| Mamba                | Efficient token mixing and gating |
| S4                   | Long-range memory retention       |
| Pre-norm             | Stable deep stacking              |
| Residual connections | Preserve optimization dynamics    |

This hybrid design stabilizes training while preserving Fish Speech’s original interface.

---

## Limitations

* Pretrained Transformer checkpoints cannot be directly reused
* No attention interpretability
* Requires PyTorch ≥ 2.1
* Streaming generation currently uses full-sequence recurrence (no KV caching)


## Resources used:

https://iclr-blog-track.github.io/2022/03/25/annotated-s4/#training-ssms-the-convolutional-representation

- [A Visual Guide to Mamba and State — Maarten Grootendorst (newsletter)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state)
- [Understanding Mamba and Selective State-Space Models (SSMs) — Towards AI](https://towardsai.net/p/l/understanding-mamba-and-selective-state-space-models-ssms)
- [Mamba Explained — The Gradient](https://thegradient.pub/mamba-explained/)
- [Mamba Appendix](https://ageron.github.io/homlp/HOMLP_Appendix_E.pdf)
- [Mamba Model Blog - HackMD](https://hackmd.io/Btjp7ZMRQGCLh93n1vMAVw#3-Architecture-of-RWKV)
- [State Space Duality (Mamba-2) Part III - The Algorithm | Goomba Lab](https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/)

Paper Explainer Videos
- [Intuition behind Mamba and State Space Models | Enhancing LLMs!](https://youtu.be/BDTVVlUU1Ck?si=Hrumdz6pbEs80fPj)  
(Video that ties in with the first blog)
- [Mamba and S4 explained: architecture, parallel scan, kernel fusion, recurrent, convolution, math](https://youtu.be/8Q_tqwpTpVU?si=Xmnb_5G6UWXFHmMI) 
(By Umar Jamil)
- [MedAI #41: Efficiently Modeling Long Sequences with Structured State Spaces | Albert Gu](https://www.youtube.com/watch?v=luCBXCErkCs&list=PLcyHLdeyXTP0k1XjNlWCMil6BfaRz0LA3&index=2) 
(By the author himself)




