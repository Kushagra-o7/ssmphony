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




