# Blueberry Research

Research papers, experiment records, benchmarks, and technical notes for **Blueberry**, a shared-weight recurrent Transformer architecture.

Blueberry explores increasing effective computational depth by repeatedly executing a subset of Transformer blocks with shared parameters. This repository is the central record of the architecture's research, experiments, and published findings.

**Research lead:** Dustin Loring  
**Email:** Dustinwloring1988@gmail.com  
**Model:** [CrowdMind/Blueberry](https://huggingface.co/CrowdMind/Blueberry)  
**Implementation:** [blueberry-llm/blueberry](https://github.com/blueberry-llm/blueberry)

## Research goals

The project investigates:

- Shared-weight recurrent execution in Transformer architectures.
- Increasing computational depth without proportionally increasing unique parameters.
- The effect of recurrent pass count on model quality and training stability.
- Learned loop-state gating and alternative state-transition mechanisms.
- Parameter-matched and compute-matched comparisons with conventional Transformers.
- Scaling recurrent architectures across model sizes and training budgets.

## Repository structure

```text
research/
├── README.md
├── papers/
│   ├── blueberry-architecture/
│   │   ├── paper.md
│   │   └── figures/
│   └── ...
├── experiments/
│   ├── gated-vs-ungated/
│   │   ├── README.md
│   │   ├── configs/
│   │   ├── logs/
│   │   └── results/
│   └── ...
├── benchmarks/
│   ├── README.md
│   └── ...
├── notes/
│   └── ...
└── assets/
    ├── diagrams/
    └── figures/
```

The structure is intended to keep papers, raw experiment evidence, processed results, and supporting material separate. Add folders as the research grows; do not commit large model checkpoints or datasets unless there is a clear reason to version them here. Prefer Hugging Face or another artifact store for large files, and record exact revisions and links.

## Current research

### Blueberry: A Shared-Weight Recurrent Transformer Architecture

**Status:** Working draft  
**Author:** Dustin Loring  
**Focus:** Architecture and preliminary gated versus ungated experiments

The first paper introduces the recurrent middle-section design, optional loop-state gate, implementation in nanochat, and initial comparison of gated and ungated configurations.

Initial configuration:

| Setting | Value |
|---|---:|
| Unique Transformer blocks | 8 |
| Recurrent middle blocks | 4 |
| Recurrent passes | 3 |
| Total block executions | 16 |
| Embedding dimension | 512 |
| Parameters, gated | 125,830,379 |
| Parameters, ungated | 125,829,354 |
| Training tokens | 335,544,320 |
| Dataset | ClimbMix |
| Hardware | NVIDIA GeForce RTX 4060 Ti |

The first gated and ungated runs are preliminary. They use one seed and one model configuration and do not establish a general performance advantage or a matched-compute efficiency gain.

## Experiment record standards

Every experiment should include enough information for another researcher to understand and reproduce it.

Record, where applicable:

- Research question and hypothesis.
- Code repository, branch, and exact commit SHA.
- Model configuration and parameter count.
- Dataset name, version, and revision.
- Tokenizer and tokenizer revision.
- Training seed, token budget, batch size, sequence length, optimizer, and schedule.
- Hardware, software versions, precision, and attention implementation.
- Training and evaluation commands.
- Raw logs and checkpoint/artifact locations.
- Evaluation scripts, benchmark versions, and complete results.
- Deviations from the planned setup.
- Known limitations and follow-up work.

Do not report a result without identifying the run and the source log or evaluation artifact that supports it. Distinguish measured results from estimates and hypotheses.

## Reporting principles

- Report unique parameter count separately from total block executions.
- Report training FLOPs or a clearly described compute estimate when making efficiency comparisons.
- State whether comparisons are matched by parameters, training tokens, compute, or another budget.
- Include multiple seeds where practical and report variation, not only the best run.
- Preserve negative and inconclusive results.
- Avoid claiming statistical significance without an appropriate analysis.
- Keep raw logs unchanged; store cleaned tables and plots separately.
- Record the exact model and dataset revisions used for published results.

## Papers and publications

Research papers belong in `papers/`. Each paper folder should contain:

- `paper.md`: editable manuscript source.
- `figures/`: figures used in the manuscript.
- `README.md`: optional notes about status, versions, and related experiments.

When a paper is submitted or published, add the DOI, arXiv identifier, publication venue, version, and date. Keep the source manuscript alongside the exported PDF.

## Experiments

Each experiment should have its own folder under `experiments/`. Include a short `README.md` explaining its purpose, configuration, run identifiers, and outcome.

Suggested naming convention:

```text
YYYY-MM-DD-short-experiment-name/
```

For example:

```text
experiments/2026-11-23-gated-vs-ungated/
```

Keep raw output separate from derived results:

```text
experiment-name/
├── README.md
├── configs/
├── logs/
├── results/
└── plots/
```

## Data and model artifacts

Large datasets and model checkpoints should generally live in dedicated artifact storage rather than Git. The research repository should retain:

- Dataset identifiers and immutable revisions.
- Model identifiers and immutable revisions.
- Download or reproduction instructions.
- Checksums where practical.
- Evaluation outputs and small supporting files.

Current model location: [Hugging Face: CrowdMind/Blueberry](https://huggingface.co/CrowdMind/Blueberry).

## Contributing research

Before adding a new experiment or paper:

1. State the research question.
2. Define the comparison and controls.
3. Record the configuration and code revision.
4. Preserve the raw output.
5. Summarize the results, including failures and limitations.
6. Link the experiment to any paper that uses its findings.

Keep documentation direct and specific. Do not turn preliminary observations into conclusions.

## License

A repository license has not yet been specified. Add a license before accepting external contributions or redistributing research materials.
