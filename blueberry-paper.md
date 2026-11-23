# Blueberry: A Shared-Weight Recurrent Transformer Architecture

**Research paper draft · Version 0.1 · November 23, 2026**

**Author:** Dustin Loring  
**Email:** Dustinwloring1988@gmail.com  
**Project:** Blueberry  
**Hugging Face:** [CrowdMind/Blueberry](https://huggingface.co/CrowdMind/Blueberry)  
**Source code:** [blueberry-llm/blueberry](https://github.com/blueberry-llm/blueberry)

---

## Abstract

Transformer architectures typically increase computational depth by adding layers, each with its own learned parameters. While this approach provides additional representational capacity, it also increases model size and memory requirements.

We introduce **Blueberry**, a shared-weight recurrent Transformer architecture that increases computational depth by repeatedly executing a subset of Transformer blocks. Instead of assigning independent parameters to every execution step, Blueberry reuses the weights of its middle blocks across multiple recurrent passes. The architecture retains conventional Transformer blocks at the beginning and end of the network, while an optional learned loop-state gate controls the integration of information between recurrent passes.

We implement Blueberry as an experimental research fork of nanochat and evaluate initial gated and ungated configurations. Both configurations use eight unique Transformer blocks, with the middle four blocks executed three times, producing sixteen total block executions per forward pass.

Initial experiments show a small advantage for the gated configuration on several reported evaluation metrics, including pretraining validation bits per byte and CORE. These results provide an initial basis for investigating recurrent depth and shared-weight execution in language models. However, the current experiments are limited to a single model configuration and seed. Further evaluation is required to determine the consistency, scalability, and practical advantages of the architecture.

This paper introduces the Blueberry architecture, describes its implementation, reports preliminary experimental results, and outlines experiments needed to evaluate its effectiveness against conventional Transformer designs.

**Keywords:** Transformer, recurrent neural networks, weight sharing, recurrent depth, language models, parameter efficiency, loop-state gating.

## 1. Introduction

The Transformer architecture has become the dominant foundation for modern language models. Its ability to process sequences through attention and feed-forward transformations has enabled substantial improvements in language understanding, generation, and reasoning.

A common approach to increasing a Transformer's computational depth is to add more layers. Each layer introduces another set of learned parameters, increasing the model's capacity but also increasing its memory requirements. This creates an opportunity to investigate alternative ways of increasing computation without requiring a proportional increase in unique parameters.

Blueberry explores one such approach: **recurrent execution of shared Transformer blocks**.

Rather than constructing a network in which every layer has independent weights, Blueberry divides the Transformer stack into three sections:

1. An initial section that processes the input once.
2. A recurrent middle section that executes multiple times using the same weights.
3. A final section that processes the recurrent representation once.

This arrangement allows the model to perform additional transformations while reusing existing parameters.

An optional learned loop-state gate provides another degree of control. It regulates how information from the original recurrent input is integrated into the state passed through subsequent executions. This creates two configurations for experimentation: gated recurrence and ungated recurrence.

The central objective of this research is to investigate whether shared-weight recurrent execution provides a useful architectural alternative to conventional Transformer depth, and to understand how recurrent passes and loop-state integration affect model performance.

### 1.1 Contributions

This work makes the following initial contributions:

- **Blueberry architecture:** A Transformer architecture that repeatedly executes a subset of its blocks using shared weights.
- **Recurrent depth:** A configurable mechanism for increasing block execution depth without introducing a separate parameter set for every execution.
- **Loop-state gating:** An optional learned gate for controlling information integration between recurrent passes.
- **Nanochat implementation:** An experimental implementation using nanochat's training and evaluation pipeline.
- **Initial evaluation:** A preliminary comparison of gated and ungated recurrent configurations.

These contributions are intended as an initial architecture study. We do not yet claim that Blueberry outperforms conventional Transformers at matched compute or parameter budgets.

## 2. Related Work

Blueberry builds on a line of research exploring parameter sharing and recurrent computation in Transformer networks. Its design is related to several existing approaches, but differs in how recurrence is applied within the network.

### 2.1 Universal Transformers

The [Universal Transformer](https://arxiv.org/abs/1807.03819) introduced recurrent computation into Transformer models. Rather than applying a fixed sequence of independently parameterized layers, it repeatedly applies a shared transformation to the evolving hidden representation. The architecture also explored adaptive computation through a per-position halting mechanism.

Blueberry shares the general idea of recurrent Transformer computation. However, it does not replace the entire network with a single recurrent transformation. It preserves distinct initial and final sections around a recurrent middle section.

### 2.2 Cross-layer Parameter Sharing

[ALBERT](https://arxiv.org/abs/1909.11942) explored cross-layer parameter sharing as a way to reduce the number of learned parameters in BERT-style models. Instead of maintaining independent parameters for every Transformer layer, ALBERT reuses parameters across layers.

This establishes an important precedent for reducing Transformer parameter counts through weight sharing. Blueberry applies sharing within an explicitly recurrent execution structure, allowing the same middle blocks to be revisited multiple times during a forward pass.

### 2.3 Recent Looped Transformer Research

Recent work has examined how repeated Transformer computation can improve model capability and how loop transitions should be designed.

- **DeepLoop: Depth Scaling for Looped Transformers** investigates residual scaling in recurrent Transformer architectures and argues that repeated parameter visits require considering the effects of tied weights on optimization stability.
- **Sparse Layers are Critical to Scaling Looped Language Models** studies dense and mixture-of-experts looped models, reporting that sparse routing can help recover expressivity across recurrent passes.
- **What Makes Recurrence Effective in Looped Language Models?** examines where recurrence should be applied and how recurrent states should be conditioned.
- **Looped Transformers as Optimizers** studies loop transitions through an optimization perspective, deriving alternative transition mechanisms and evaluating them under matched training FLOPs.

These studies make clear that Blueberry belongs to an active research area. Its contribution should therefore be evaluated through its specific architecture, implementation, and experimental results rather than through a broad claim that recurrent Transformer computation is new.

## 3. Blueberry Architecture

### 3.1 Design Overview

Blueberry uses a conventional Transformer stack with a recurrent middle section. The architecture is controlled by two primary configuration parameters:

- `loop_repeats`: number of executions of the recurrent middle section.
- `loop_gate`: enables or disables learned loop-state integration.

For a network with an even number of layers \(N\), the recurrent section contains \(N/2\) blocks. It begins at block index \(\lfloor N/4 \rfloor\). The remaining blocks form the initial and final sections.

For the eight-block configuration used in the initial experiments, the execution sequence is:

```text
Initial section:
  B0 -> B1

Recurrent section (shared weights):
  Pass 1: B2 -> B3 -> B4 -> B5
  Pass 2: B2 -> B3 -> B4 -> B5
  Pass 3: B2 -> B3 -> B4 -> B5

Final section:
  B6 -> B7
```

The resulting execution sequence is:

\[
[B_0,B_1,
\underbrace{B_2,B_3,B_4,B_5}_{\text{Pass 1}},
\underbrace{B_2,B_3,B_4,B_5}_{\text{Pass 2}},
\underbrace{B_2,B_3,B_4,B_5}_{\text{Pass 3}},
B_6,B_7]
\]

The recurrent blocks are not copies. Each pass uses the same underlying parameter objects. Gradients from each use therefore contribute to updates of the same shared parameters.

### 3.2 Recurrent Execution

Let \(x\) represent the hidden state entering the recurrent section, and let \(F_\theta\) represent the composition of the recurrent Transformer blocks with shared parameters \(\theta\).

For an ungated model, recurrent execution can be expressed as:

\[
h_0=x
\]

\[
h_{t+1}=F_\theta(h_t)
\]

where \(t\) indexes recurrent passes.

For three passes:

\[
h_1=F_\theta(x)
\]

\[
h_2=F_\theta(h_1)
\]

\[
h_3=F_\theta(h_2)
\]

The resulting state is passed to the final, independently parameterized Transformer blocks.

This approach increases the number of sequential transformations while keeping the recurrent section's parameter count fixed.

### 3.3 Loop-State Gating

Blueberry optionally introduces a learned gate at recurrent pass boundaries.

The implementation preserves the hidden state entering the recurrent section as a reference state \(p\). At the beginning of subsequent passes, the current recurrent state \(h_t\) is combined with this reference state.

The gate is computed as:

\[
g_t = \sigma\left(
W_g[\operatorname{RMSNorm}(p);
\operatorname{RMSNorm}(h_t)] + b_g
\right)
\]

The updated recurrent state is:

\[
h'_t = (1-g_t)p + g_t h_t
\]

where:

- \(p\) is the hidden state entering the recurrent section.
- \(h_t\) is the current recurrent state.
- \(W_g\) is the learned gate projection.
- \(b_g\) is a learned bias.
- \(\sigma\) is the sigmoid function.
- \(g_t\) is a token-dependent scalar gate.

The gate is applied independently at each token position. It controls the interpolation between the original recurrent input and the state produced by previous recurrent computation.

An important implementation detail is that the reference state remains fixed across recurrent passes. The gate therefore does not simply blend the immediately preceding pass with the next pass. It blends the original loop input with the current accumulated recurrent state.

The gate parameters are initialized with zero projection weights and a bias of \(-4\), initially favoring the reference state. The gate is applied at the start of each recurrent pass after the first.

This design allows the model to learn how much recurrently accumulated information to retain relative to the original input.

### 3.4 Parameter Sharing and Computational Depth

The primary distinction between unique model depth and execution depth is important when evaluating Blueberry.

Let:

- \(N\) be the number of unique Transformer blocks.
- \(R\) be the number of recurrent passes.
- \(L=N/2\) be the number of blocks in the recurrent section.

For an even-depth model using the current implementation, the total number of block executions is:

\[
D_{\text{exec}} = N-L+RL
\]

Substituting \(L=N/2\):

\[
D_{\text{exec}}=N\left(\frac{R+1}{2}\right)
\]

For eight unique blocks and three recurrent passes:

\[
D_{\text{exec}}=8-4+(3\times4)=16
\]

The architecture therefore performs sixteen block executions while retaining eight unique Transformer blocks.

This distinction does not mean that recurrent execution is computationally free. Every additional pass requires additional forward and backward computation. Blueberry trades additional computation for parameter reuse rather than eliminating the cost of increased depth.

### 3.5 Integration with the Transformer

Blueberry retains the underlying nanochat Transformer implementation, including its existing attention, normalization, and feed-forward components.

The recurrent mechanism changes the order in which blocks execute and determines when shared blocks are revisited. It does not replace the attention or feed-forward operations inside those blocks.

The implementation also preserves other model components, including rotary positional embeddings, grouped-query attention, sliding-window attention, value embeddings, and the existing residual-stream mechanisms.

These components are part of the experimental model configuration, not new contributions of the Blueberry recurrence mechanism.

## 4. Implementation

Blueberry is implemented in PyTorch as a research fork of nanochat.

The primary implementation is located in `nanochat/gpt.py`. The model configuration introduces two options:

```python
loop_repeats: int = 3
loop_gate: bool = False
```

The model constructs an execution index sequence that references the same block objects multiple times. This avoids creating additional copies of the recurrent block parameters.

The implementation also tracks the expanded execution sequence separately from the unique block collection. This allows parameter counting, attention-window configuration, FLOP estimation, and key-value cache allocation to account for repeated block executions.

The architecture currently requires an even number of unique blocks, with at least four blocks when recurrence is enabled. A single recurrent pass disables recurrence and produces the ordinary non-recurrent execution sequence.

### 4.1 Computational Considerations

Although shared weights reduce the number of unique parameters relative to a model with independently parameterized blocks at every execution depth, they do not eliminate the computational cost of repeated execution.

For a recurrent section with \(L\) blocks and \(R\) passes, the section performs \(RL\) block executions.

This has several implications:

- Recurrent execution increases training computation.
- Repeated blocks accumulate gradients from multiple uses.
- Inference latency can increase because recurrent passes are sequential.
- Parameter count and computational cost must be reported separately.

Future comparisons should therefore report both parameter counts and training FLOPs. Comparing Blueberry against a conventional Transformer using parameter count alone would not provide a complete picture.

## 5. Experimental Setup

### 5.1 Hardware and Software

The initial experiments were conducted using:

| Component | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti |
| Compute precision | BF16 |
| Attention implementation | Flash Attention 3 |
| Training framework | PyTorch |
| Base implementation | nanochat |

### 5.2 Model Configuration

Both experiments used the same underlying model configuration.

| Parameter | Value |
|---|---:|
| Unique Transformer blocks | 8 |
| Recurrent passes | 3 |
| Total block executions | 16 |
| Embedding dimension | 512 |
| Attention heads | 4 |
| KV heads | 4 |
| Sequence length | 2,048 |
| Vocabulary size | 32,768 |
| Attention window pattern | SSSL |

### 5.3 Training Configuration

| Parameter | Value |
|---|---|
| Dataset | ClimbMix |
| Training tokens | 335,544,320 |
| Global batch size | 262,144 tokens |
| Random seed | 42 |
| Precision | BF16 |

The gated and ungated experiments used the same tokenizer, dataset, seed, optimizer settings, and token budget. The primary architectural difference was whether the learned loop-state gate was enabled.

### 5.4 Experimental Conditions

We evaluate two configurations:

**Blueberry-Gated**

The recurrent middle section executes three times. A learned gate integrates the original recurrent input with the accumulated recurrent state at subsequent pass boundaries.

**Blueberry-Ungated**

The recurrent middle section executes three times using the same shared weights, without learned loop-state gating.

This comparison is intended to isolate the effect of the loop-state gate within the Blueberry architecture.

## 6. Results

### 6.1 Model Parameters

| Metric | Gated | Ungated |
|---|---:|---:|
| Total parameters | 125,830,379 | 125,829,354 |
| Additional gate parameters | 1,025 | 0 |

The loop-state gate adds only 1,025 parameters to the model. Both configurations therefore have effectively the same parameter count.

### 6.2 Pretraining Results

| Metric | Gated | Ungated |
|---|---:|---:|
| Pretraining time | 106.79 min | 107.16 min |
| Minimum validation BPB | **0.965060** | 0.967800 |
| CORE at step 1,280 | **0.1048** | 0.0974 |

The gated configuration achieved a lower minimum validation BPB and a higher reported CORE score.

The absolute BPB difference was:

\[
0.967800-0.965060=0.002740
\]

Pretraining time was similar between the two configurations.

### 6.3 Supervised Fine-Tuning Results

| Metric | Gated | Ungated |
|---|---:|---:|
| SFT time | 158.68 min | 154.63 min |
| Minimum SFT validation BPB | **0.4138** | 0.4150 |
| ChatCORE after SFT | 0.0258 | 0.0257 |

The gated configuration again achieved a slightly lower validation BPB. ChatCORE results were nearly identical.

### 6.4 Final Evaluation

| Metric | Gated | Ungated |
|---|---:|---:|
| Final ChatCORE | **0.0241** | 0.0231 |
| Final HumanEval accuracy | **7.32%** | 6.71% |
| GSM8K accuracy | 0% | 0% |

The final evaluation shows small differences favoring the gated configuration on ChatCORE and HumanEval. Neither configuration demonstrated measurable GSM8K accuracy in this experiment.

### 6.5 Initial Interpretation

The first experiments suggest that loop-state gating may provide a modest benefit to recurrent computation.

The gated configuration achieved better reported pretraining and SFT validation BPB, a higher pretraining CORE score, and slightly higher final ChatCORE and HumanEval results.

However, these observations should be treated cautiously.

The experiments use one model size, one training seed, and one recurrent configuration. They do not establish statistical significance, nor do they demonstrate that Blueberry is more efficient than a conventional Transformer at matched compute.

The results support further investigation of loop-state integration rather than a definitive performance claim.

## 7. Discussion

The initial results raise several questions about the role of the original recurrent input during repeated computation.

In an ungated recurrent architecture, each pass receives the output of the previous pass. Information from the original loop input must therefore be preserved through the recurrent transformations.

Blueberry's gate provides a direct path for incorporating that original representation at subsequent pass boundaries. This may help prevent useful information from being lost or distorted during repeated transformations.

The current experiments are consistent with this possibility, but they do not establish the mechanism responsible for the observed differences.

Further experiments should investigate:

- Whether the gate learns different values across token positions.
- Whether the gate behaves differently across recurrent passes.
- Whether gating becomes more useful as the number of passes increases.
- Whether the same behavior appears at larger model sizes.
- Whether alternative gate formulations improve training stability.

A further question is whether recurrent computation can provide useful improvements when compared against conventional Transformers under matched training FLOPs. This is necessary to determine whether parameter reuse offers a practical advantage rather than simply moving cost from model storage to computation.

## 8. Limitations

This study has several limitations.

1. **Limited scale:** The initial experiments use a relatively small model with approximately 126 million parameters.
2. **Single seed:** The results have not been repeated across multiple random seeds.
3. **Limited recurrence settings:** Only three recurrent passes have been evaluated.
4. **No matched-compute baseline:** The initial results do not compare Blueberry against a conventional Transformer trained with an equivalent compute budget.
5. **Limited downstream performance:** The reported evaluations do not establish broad improvements in language understanding, reasoning, or coding.
6. **Hardware constraints:** The initial implementation and experiments were conducted on a single RTX 4060 Ti.

These limitations define the next stage of research.

## 9. Future Work

We plan to investigate the following directions.

| Experiment | Configurations | Main objective |
|---|---|---|
| Recurrent depth | 1, 2, 3, 4, 6 passes | Measure the effect of additional recurrence |
| Gate ablation | Gated and ungated | Determine the value of loop-state gating |
| Model depth | 4, 8, 12, 16 unique blocks | Study scaling with physical depth |
| Gate alternatives | Scalar, vector, residual, learned projection | Explore state integration |
| Compute comparison | Matched FLOPs | Compare against conventional Transformers |
| Parameter comparison | Matched parameter counts | Measure parameter efficiency |
| Training stability | Multiple seeds | Measure variance and reproducibility |
| Larger models | Increasing embedding and model dimensions | Investigate scaling behavior |

### 9.1 Recurrent Pass Scaling

The first priority is to evaluate the effect of recurrent pass count.

Increasing recurrent passes provides more computation without introducing another set of recurrent block parameters. However, it also increases sequential computation and may introduce optimization difficulties.

We plan to evaluate whether performance improves consistently as recurrent depth increases, and whether there is a point where additional passes provide diminishing returns.

### 9.2 Loop-State Gate Ablations

The current gate blends the original recurrent input with the accumulated recurrent state.

Future work should compare this design against alternatives, including:

- A gate based only on the current recurrent state.
- A gate based on the original input and current state.
- Per-channel gates instead of scalar gates.
- Residual additive state updates.
- Learned gates at every recurrent block rather than only at pass boundaries.

### 9.3 Matched-Compute Comparisons

A key experiment will compare Blueberry against a conventional Transformer under matched training FLOPs.

This will help determine whether recurrent parameter sharing provides a useful trade-off between model size, computational depth, and predictive performance.

## 10. Conclusion

Blueberry is an experimental shared-weight recurrent Transformer architecture designed to increase computational depth by repeatedly executing a subset of Transformer blocks.

The architecture preserves independently parameterized initial and final sections while reusing the weights of its middle section across recurrent passes. An optional learned loop-state gate provides a mechanism for integrating the original recurrent input with the accumulated hidden representation.

Initial experiments comparing gated and ungated configurations show small differences favoring the gated model on several reported metrics. These findings motivate further investigation but are not sufficient to establish a general performance advantage.

The next stage of research will focus on recurrent depth scaling, loop-state gate alternatives, training stability, and comparisons against conventional Transformers under controlled parameter and compute budgets.

## References

The following works provide the initial research foundation. The related-work section should be expanded and checked against the full papers before submission.

1. Dehghani, M., et al. (2019). *Universal Transformers*. ICLR. [arXiv:1807.03819](https://arxiv.org/abs/1807.03819).
2. Lan, Z., et al. (2020). *ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations*. ICLR. [arXiv:1909.11942](https://arxiv.org/abs/1909.11942).
3. Li, S., et al. (2026). *DeepLoop: Depth Scaling for Looped Transformers*. [arXiv:2607.13491](https://arxiv.org/abs/2607.13491).
4. Lee, R., et al. (2026). *Sparse Layers are Critical to Scaling Looped Language Models*. [arXiv:2605.09165](https://arxiv.org/abs/2605.09165).
5. Zhuang, X., et al. (2026). *What Makes Recurrence Effective in Looped Language Models?* [arXiv:2609.36636](https://arxiv.org/abs/2609.36636).
6. Huang, Y., et al. (2026). *Looped Transformers as Optimizers*. [arXiv:2609.37379](https://arxiv.org/abs/2609.37379).

---

## Author and project information

**Dustin Loring**  
Dustinwloring1988@gmail.com

**Project:** Blueberry  
**Hugging Face:** [CrowdMind/Blueberry](https://huggingface.co/CrowdMind/Blueberry)  
**Source code:** [blueberry-llm/blueberry](https://github.com/blueberry-llm/blueberry)

---

## Submission checklist

Before submitting this paper to arXiv, the following items should be completed:

- [ ] Verify the architecture equations and gate implementation against the current source code.
- [ ] Verify all reported metrics against the original training and evaluation logs.
- [ ] Verify bibliographic details and claims against the full cited papers.
- [ ] Run multiple random seeds.
- [ ] Add a conventional Transformer baseline.
- [ ] Add matched-parameter and matched-compute comparisons.
- [ ] Publish exact training commands, configuration files, dataset revisions, and evaluation scripts.
- [ ] Record the code commit and model artifact revisions used for each experiment.
