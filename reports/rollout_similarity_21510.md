# Rollout similarity analysis — Slurm run 21510

Generated 2026-09-16T20:39:23.574938+00:00. Analysis code: `scripts/analyze_rollout_similarity.py`.

## Decision

**The diversity evidence does not cleanly support 32 rollouts/question:** its estimated coverage is 82.1% for workflow variants and 69.7% for individual instructions.

**Reducing to 16 is materially more aggressive:** estimated coverage falls to 64.0% for workflow variants and 48.8% for individual instructions. Treat 16 as an experiment, not a default, unless an end-to-end training ablation shows unchanged reward and gradient stability.

This run contains plans only: every valid record has the neutral placeholder reward and no worker execution. Similarity can show redundancy, but it cannot establish that fewer GRPO samples preserve reward tails, advantage estimation, or final model quality.

## Dataset integrity

All 64 rollout files contain all 202 question IDs (12,928 records). 12,809 plans are valid (99.08%); 119 are parse/validation failures and are retained in coverage simulations as unusable draws.

Mean workflow length is 3.46 steps. Distribution: 1 step: 1,487, 2 steps: 141, 3 steps: 3,470, 4 steps: 6,689, 5 steps: 809, 6 steps: 170, 7 steps: 43.

## Similarity results

Scores range from 0 (unrelated) to 1 (identical under the metric). These are all valid-valid pairs within the same question; invalid plans have no invented similarity score.

| Measure | Min | Q1 | Median | Q3 | Max | Mean | P10 | P90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Topology | 0.060 | 0.629 | 0.840 | 1.000 | 1.000 | 0.745 | 0.139 | 1.000 |
| Model routing | 0.000 | 0.250 | 0.333 | 0.500 | 1.000 | 0.366 | 0.000 | 0.750 |
| Structural (70% topology, 30% routing) | 0.042 | 0.516 | 0.698 | 0.800 | 1.000 | 0.632 | 0.197 | 0.925 |
| Instruction semantic | 0.038 | 0.549 | 0.672 | 0.772 | 1.000 | 0.628 | 0.283 | 0.824 |
| Combined (50% structural, 50% semantic) | 0.056 | 0.536 | 0.699 | 0.790 | 1.000 | 0.630 | 0.233 | 0.844 |

### What the measures mean

Topology matches steps by graph role (normalized position, dependency layer, in/out degree, question access, and tool use), penalizes unmatched steps, and measures dependency-edge overlap. Model routing is exact model agreement for those matched roles.

Instruction semantics uses `sentence-transformers/all-MiniLM-L6-v2` embeddings and exact maximum-weight one-to-one matching of individual step instructions. Unmatched steps score zero. This catches paraphrases without allowing several instructions in one plan to all claim the same counterpart in another.

The combined score is used only for the coverage simulation. The separate columns should remain the primary evidence because any combined weighting is a policy choice.

Exact text makes the plans look almost entirely unique (63.4 distinct normalized instruction sets per question out of about 64), but structure collapses to 7.7 topology archetypes on average. Adding exact model assignments raises that to 32.8 route archetypes. This gap is why neither exact-string matching nor structure alone is adequate.

## Simulated reduction from 64

For each question, 200 random subsets of 16 and 32 positions were drawn from the actual 64, including invalid positions. Workflow coverage is the share of all valid 64-rollout workflows having a selected neighbor at or above the stated combined-similarity threshold. Instruction coverage is the share of all individual instructions having a selected instruction with embedding cosine ≥ 0.90.

| Budget | Valid draws (mean) | Workflow cov. ≥.80 | ≥.85 | ≥.90 | Instruction cov. ≥.90 | Exact topology mass | Exact route mass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 31.70 | 93.2% | 82.1% | 64.4% | 69.7% | 96.3% | 76.8% |
| 16 | 15.85 | 83.7% | 64.0% | 40.8% | 48.8% | 91.1% | 56.3% |

“Mass” is prevalence-weighted: a full-set rollout is covered only when its exact observed topology or exact topology-plus-model route occurs in the subset. This is deliberately stricter than semantic coverage.

### Variation across questions

| Budget | Metric | P10 question | Median question | P90 question |
|---:|---|---:|---:|---:|
| 32 | Workflow coverage ≥.85 | 68.7% | 83.3% | 93.4% |
| 32 | Instruction coverage ≥.90 | 57.1% | 70.0% | 81.5% |
| 32 | Exact topology mass | 92.6% | 97.3% | 99.3% |
| 32 | Exact route mass | 64.9% | 77.7% | 87.3% |
| 16 | Workflow coverage ≥.85 | 44.2% | 63.9% | 82.3% |
| 16 | Instruction coverage ≥.90 | 32.2% | 48.2% | 66.6% |
| 16 | Exact topology mass | 82.5% | 93.2% | 97.2% |
| 16 | Exact route mass | 38.8% | 56.5% | 71.5% |

## Recommendation and training implications

Use **32 rollouts/question as the next default candidate**, conditional on a controlled training ablation. It halves generation cost while retaining substantially more of the observed semantic and structural support than 16. Keep 64 for questions or batches where reward variance is high, valid generations are scarce, or rare workflow discovery is itself important.

Do not adopt 16 globally from this analysis alone. If cost pressure makes 16 attractive, a better policy is adaptive: start at 16, add another 16 when valid count is low, rewards disagree, or the first 16 occupy many distinct structural/semantic modes. The present plan-only data can motivate that policy, but executed, judged rollouts are required to tune its trigger.

A minimum ablation should compare 64 vs 32 vs 16 with the same question batches and seeds, tracking:

- held-out reward and pass@k / best-of-k behavior;
- within-question reward variance and fraction of all-equal reward groups;
- GRPO advantage standard deviation, zero-advantage groups, and gradient norm;
- parse-failure rate and cost per effective (valid, non-duplicate) rollout;
- final benchmark confidence intervals across multiple training seeds.

## Per-question results

`Struct` and `Sem` are mean valid-valid pairwise scores. `Topo`/`Route` are counts of exact observed topology and topology-plus-model-routing archetypes. Coverage columns are expected coverage under random subsampling (200 trials/question).

| # | Question ID | Question | Valid | Struct | Sem | Topo | Route | 32 wf | 32 instr | 16 wf | 16 instr |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `hle-673818e39b3842b34824240d` | In this question, we want to examine the effect of non… | 63 | 0.639 | 0.558 | 10 | 39 | 64.0% | 57.1% | 38.0% | 33.4% |
| 1 | `hle-6704465caf0a436d92c65160` | How many logical qubits at most can be encoded in two … | 62 | 0.480 | 0.567 | 5 | 29 | 92.7% | 85.5% | 79.8% | 71.2% |
| 2 | `hle-6750a730651c49cb2cce0df5` | Determine the ratio of differential cross-sections for… | 64 | 0.759 | 0.764 | 6 | 31 | 92.9% | 78.4% | 79.3% | 61.4% |
| 3 | `hle-671eefbfb6d7145231fa28e4` | Consider a magnetic yoke with infinite permeability, a… | 63 | 0.621 | 0.593 | 16 | 44 | 66.4% | 54.2% | 40.4% | 28.6% |
| 4 | `hle-671a22850b52f35047c0b230` | Consider a superconducting bar, infinitely long along … | 61 | 0.756 | 0.705 | 5 | 32 | 83.1% | 72.6% | 61.2% | 51.9% |
| 5 | `hle-66e9e6b3c27ad6bc46adbc20` | Recall the definition of the higher central charge: $$… | 64 | 0.792 | 0.754 | 4 | 27 | 91.5% | 71.4% | 77.9% | 50.7% |
| 6 | `hle-6723bf036e47cec0509b5caf` | Two stars (A and B) with negligible proper motions are… | 62 | 0.682 | 0.699 | 14 | 38 | 83.0% | 72.1% | 67.0% | 50.7% |
| 7 | `hle-672a80a432cd57d8762583e9` | For an ideal Fermi gas of spin-1/2 particles, let \(\n… | 64 | 0.648 | 0.615 | 12 | 51 | 65.9% | 63.3% | 39.6% | 39.4% |
| 8 | `hle-6737d23a6a364decc45cc7ee` | A space station is maintained on a circular orbit clos… | 63 | 0.586 | 0.642 | 13 | 51 | 69.0% | 75.2% | 44.7% | 59.1% |
| 9 | `hle-6720241e20239af7af582ae1` | Can you identify which chemical element has this spect… | 64 | 0.571 | 0.647 | 6 | 47 | 80.5% | 69.4% | 55.9% | 46.6% |
| 10 | `hle-66fc6a20d90ebe461bfd0cc7` | Consider the theory of interaction of a spinor and sca… | 64 | 0.655 | 0.649 | 13 | 51 | 65.6% | 63.7% | 39.2% | 40.7% |
| 11 | `hle-6747fa2a456927085f863956` | The action of the $N=4$ SYM model in $d=3+1$ reads $$ … | 64 | 0.623 | 0.640 | 20 | 49 | 73.1% | 62.2% | 52.1% | 37.0% |
| 12 | `hle-672a27f5d30d6f5584cde73d` | In the context of many-body quantum physics, consider … | 64 | 0.703 | 0.680 | 2 | 22 | 94.9% | 73.3% | 82.5% | 54.9% |
| 13 | `hle-67036eec810cc41df802051d` | The diagram shows a ladder circuit with $N$ cells, whe… | 64 | 0.568 | 0.547 | 6 | 41 | 65.7% | 53.2% | 41.5% | 27.5% |
| 14 | `hle-673757b8673b15e8ce0a3755` | Consider a spin orbital coupling problem of p-electron… | 64 | 0.644 | 0.685 | 4 | 27 | 92.3% | 81.3% | 82.0% | 61.9% |
| 15 | `hle-66ed985a7b0ffebd9fae6993` | A particle's trajectory is governed by the nonlinear d… | 64 | 0.736 | 0.713 | 5 | 32 | 83.5% | 71.2% | 65.5% | 47.6% |
| 16 | `hle-672a5ecf541155da3e036094` | Within the advanced theoretical frameworks of quantum … | 64 | 0.641 | 0.642 | 4 | 23 | 95.0% | 79.8% | 84.4% | 62.1% |
| 17 | `hle-67169906187dc7ac4a7ae1a8` | Sodium-23 is laser-cooled with a cooling transition be… | 64 | 0.667 | 0.672 | 11 | 42 | 73.9% | 71.1% | 51.4% | 47.0% |
| 18 | `hle-6756844266c3ec0e7088bf9f` | Imagine a universe U where the following holds: 1. The… | 64 | 0.549 | 0.605 | 12 | 51 | 70.1% | 63.5% | 44.8% | 39.0% |
| 19 | `hle-671f7f334db66145d9e41f1f` | Consider a 1D Fermi-Hubbard system with an even number… | 64 | 0.746 | 0.659 | 4 | 26 | 82.6% | 69.6% | 63.7% | 48.5% |
| 20 | `hle-67370aa83f0517b6e8a60769` | Using Boltzmann-Sanov and Cramer-Chenoff large deviati… | 64 | 0.779 | 0.727 | 6 | 25 | 93.1% | 67.7% | 80.6% | 44.7% |
| 21 | `hle-6725ed80de551b21db6a0f29` | Consider a Yukawa theory described by the Lagrangian d… | 61 | 0.801 | 0.754 | 5 | 27 | 92.2% | 72.9% | 81.3% | 53.7% |
| 22 | `hle-6738f07b851b80b033aa8633` | We will use quantum chromodynamics(QCD) to solve the n… | 64 | 0.765 | 0.755 | 9 | 30 | 93.7% | 81.9% | 85.3% | 64.7% |
| 23 | `hle-6732ce52ec2dbeda063b420b` | A man is riding an electric scooter northward along a … | 60 | 0.680 | 0.701 | 12 | 41 | 81.1% | 71.1% | 63.9% | 46.7% |
| 24 | `hle-67399a69c180d4d45680e434` | In a fixed target experiment, the proton or electron b… | 64 | 0.716 | 0.682 | 3 | 25 | 86.2% | 63.2% | 68.5% | 38.3% |
| 25 | `hle-672a26f8b4642f4105e02119` | What is the symmetry factor associated with a Feynman … | 64 | 0.516 | 0.529 | 6 | 26 | 88.0% | 66.9% | 71.0% | 45.0% |
| 26 | `hle-670402f0bae67686d8aef3e8` | In the freely jointed chain model of a polymer there a… | 64 | 0.814 | 0.734 | 7 | 21 | 91.6% | 68.8% | 82.2% | 47.7% |
| 27 | `hle-6771d50cff6d0a6c35d7ca99` | Let \[D_n(r) = 4\pi r^2 \sum_{l=0}^{n-1} \sum_{m=-l}^{… | 58 | 0.685 | 0.713 | 18 | 53 | 72.1% | 77.9% | 46.1% | 59.4% |
| 28 | `hle-67379aea6c946be458900f3f` | Consider a 2D semiconductor material with a band gap o… | 64 | 0.554 | 0.575 | 5 | 21 | 91.5% | 73.7% | 82.0% | 52.6% |
| 29 | `hle-673a92ad437529d472475406` | The neutralino mass matrix in the ($-i \tilde{\gamma}$… | 60 | 0.616 | 0.593 | 8 | 39 | 70.5% | 64.1% | 45.8% | 40.4% |
| 30 | `hle-672302bdbc9e7202ad89ccd3` | In the many-body approach known as CCSD, the optimum a… | 61 | 0.460 | 0.486 | 4 | 27 | 92.6% | 64.5% | 79.8% | 41.8% |
| 31 | `hle-6722613b4152cab57c187de5` | Give me the minimal of ressources, non signaling PR-Bo… | 64 | 0.658 | 0.624 | 12 | 45 | 77.1% | 61.5% | 52.5% | 36.6% |
| 32 | `hle-67381f0835f9616e390e737a` | These are the two postulates of special relativity: 1.… | 64 | 0.513 | 0.512 | 5 | 28 | 83.0% | 60.8% | 63.7% | 35.1% |
| 33 | `hle-671f8a0781665b519321d818` | I have a monocrystal of olivine orthophosphate $LiNiPO… | 63 | 0.740 | 0.685 | 6 | 29 | 81.7% | 63.1% | 62.5% | 40.0% |
| 34 | `hle-66eda111ea64e37f9218600c` | Consider an infinitely long cylindrical shell (aligned… | 64 | 0.676 | 0.713 | 10 | 40 | 84.3% | 77.3% | 67.6% | 61.9% |
| 35 | `hle-6725e382086428ce4e2fa8d6` | Two identical solid disks of radius $R$ and of uniform… | 64 | 0.770 | 0.729 | 5 | 33 | 84.1% | 65.5% | 68.1% | 40.5% |
| 36 | `hle-67381641a8513cd02a2937c3` | A gadolinium(III) oxide target of natural abundance is… | 64 | 0.603 | 0.596 | 15 | 58 | 58.7% | 66.5% | 32.7% | 42.9% |
| 37 | `hle-672de1afed6de72b75b8c7e6` | What is the physical implication of the 't Hooft anoma… | 64 | 0.490 | 0.538 | 6 | 20 | 93.6% | 89.9% | 81.9% | 80.1% |
| 38 | `hle-67455cd07df215d3effe4f4e` | For arbitrary bipartite quantum state $\rho$ where the… | 64 | 0.732 | 0.717 | 9 | 32 | 87.2% | 69.7% | 72.7% | 51.1% |
| 39 | `hle-672669616633802b43ad2332` | During a Kp=7 event, which of the following locations … | 64 | 0.446 | 0.529 | 6 | 23 | 93.9% | 82.7% | 81.9% | 67.8% |
| 40 | `hle-66ef3be2b8a1ba6e0ba23496` | The temperature $T(t)$ within a specialized chemical r… | 63 | 0.673 | 0.709 | 4 | 29 | 93.3% | 84.6% | 80.6% | 69.0% |
| 41 | `hle-6713cedd6978edcd74f82863` | In a hard spheres (HSs) system at jamming (with a pack… | 64 | 0.514 | 0.544 | 4 | 20 | 92.0% | 81.2% | 79.7% | 67.2% |
| 42 | `hle-66e8e8864c478c24f1a7e7b8` | Consider the theory of linearized gravity in $3+1$ dim… | 64 | 0.728 | 0.662 | 5 | 23 | 88.3% | 65.8% | 74.1% | 44.0% |
| 43 | `hle-670ad4fbb4aea214feb705d3` | Which of the following is true about doing broadband C… | 64 | 0.620 | 0.602 | 5 | 14 | 89.6% | 77.6% | 80.9% | 55.5% |
| 44 | `hle-6748b7dfac494f3a05306206` | Consider a universe \(U\) governed by quantum logic wh… | 64 | 0.708 | 0.622 | 9 | 34 | 73.5% | 60.5% | 51.9% | 38.8% |
| 45 | `hle-672257b388e407d7eb077431` | Consider a superconductor in the critical state, with … | 64 | 0.696 | 0.684 | 6 | 35 | 81.3% | 74.0% | 61.5% | 55.8% |
| 46 | `hle-6728d1e3a8053eddd7a7f24a` | What is the minimum number of vertices in a two-loop F… | 64 | 0.496 | 0.553 | 5 | 21 | 94.5% | 77.4% | 81.8% | 62.0% |
| 47 | `hle-66eb968f69502893cf210115` | Imagine a hypothetical scenario where the NV center co… | 64 | 0.658 | 0.649 | 6 | 30 | 87.6% | 70.5% | 74.6% | 48.4% |
| 48 | `hle-671feb0424e49a0a566a7883` | A sphere of radius $R$ with dielectric permittivity $\… | 63 | 0.652 | 0.615 | 8 | 37 | 66.4% | 55.0% | 42.0% | 29.3% |
| 49 | `hle-6724e29c42ec04c22a24aab0` | A phase-amplitude (PA) metasurface is used to convert … | 64 | 0.643 | 0.618 | 4 | 33 | 83.5% | 74.4% | 63.0% | 55.6% |
| 50 | `hle-67242914911674ab1b5d9036` | A particle of mass $m$, is launched from a cliff of he… | 63 | 0.747 | 0.669 | 4 | 30 | 75.4% | 56.8% | 52.1% | 31.1% |
| 51 | `hle-6728be777ed2554b747b3d65` | What is the leading order expression for the fixed poi… | 64 | 0.519 | 0.528 | 3 | 17 | 95.9% | 78.8% | 89.6% | 59.8% |
| 52 | `hle-6725562814a5e4119e612733` | A modified Huckel theory can be applied to conjugated … | 63 | 0.571 | 0.623 | 10 | 47 | 78.8% | 73.6% | 57.4% | 51.3% |
| 53 | `hle-6707425209b8f334446ed3e2` | Consider the H2 molecule and construct its Fock space … | 64 | 0.725 | 0.722 | 13 | 33 | 90.8% | 75.1% | 80.7% | 57.5% |
| 54 | `hle-672fc00e13e5fbd332372f3f` | In the realm of two-dimensional asymptotically free CP… | 64 | 0.813 | 0.764 | 5 | 18 | 95.2% | 72.1% | 88.0% | 52.7% |
| 55 | `hle-6734968f832777944c775cc4` | In the context of toroidal grid generation, which math… | 64 | 0.559 | 0.651 | 3 | 15 | 91.6% | 82.0% | 82.1% | 69.7% |
| 56 | `hle-67440064abafa90f5b9d4da9` | The so-called old-minimal set of auxiliary fields for … | 62 | 0.621 | 0.624 | 21 | 47 | 70.6% | 65.2% | 44.7% | 41.2% |
| 57 | `hle-672600b226992c47ce3a7efe` | Consider a triple star system where the components hav… | 64 | 0.615 | 0.665 | 6 | 41 | 77.0% | 65.9% | 52.8% | 41.8% |
| 58 | `hle-66e89ebe7361982cbfbc5952` | Remember that the K-matrix describing a Bosonic intege… | 63 | 0.726 | 0.680 | 5 | 28 | 91.4% | 73.5% | 76.1% | 53.4% |
| 59 | `hle-6725267ae9d3782179d4a5ff` | Calculate the symmetry factors for each second-order v… | 62 | 0.616 | 0.704 | 16 | 48 | 82.1% | 82.3% | 61.0% | 64.5% |
| 60 | `hle-670fe86a7e294dc6ad20c1ba` | Consider the problem of the radial two-channel quantum… | 64 | 0.781 | 0.690 | 7 | 28 | 81.9% | 66.4% | 61.0% | 43.0% |
| 61 | `hle-66e9032060abc895aedcf460` | The following is the data collected from a rawinsonde … | 64 | 0.463 | 0.563 | 7 | 40 | 86.9% | 76.6% | 66.1% | 57.5% |
| 62 | `hle-67374c79ccee19cce9664dd5` | Consider a four terminal device with terminal 1, 2,3,4… | 63 | 0.671 | 0.695 | 4 | 30 | 91.6% | 79.7% | 74.1% | 62.6% |
| 63 | `hle-6723e60719c334bc13515f01` | For this problem, we work in units where $\epsilon_0 =… | 64 | 0.784 | 0.722 | 6 | 33 | 83.0% | 63.1% | 63.8% | 42.7% |
| 64 | `hle-67225b0a9e5897be2aec5257` | A “light spring” is a helical space-time wave packet f… | 64 | 0.563 | 0.535 | 5 | 26 | 80.0% | 62.8% | 61.2% | 38.5% |
| 65 | `hle-6729035989898f87cb532106` | Consider a one-dimensional random walk over a circle w… | 64 | 0.613 | 0.672 | 7 | 43 | 82.3% | 71.4% | 62.9% | 50.1% |
| 66 | `hle-671ff43951f8a38cb737b3d4` | There are three ways to twist the magnetic field in fu… | 64 | 0.600 | 0.623 | 6 | 39 | 76.2% | 71.0% | 51.1% | 49.4% |
| 67 | `hle-67218f865b0747ce2231d48c` | one-fourth of a smooth spherical pumpkin with radius \… | 56 | 0.627 | 0.592 | 8 | 43 | 59.9% | 58.0% | 33.9% | 32.4% |
| 68 | `hle-671fbb0cc6abf8266c1892ca` | We consider a two-level atom interacting with a single… | 64 | 0.665 | 0.596 | 3 | 29 | 72.0% | 56.6% | 47.7% | 31.9% |
| 69 | `hle-6707d229ce18c3c60d66c712` | Suppose a crystalline material from the perovskite fam… | 64 | 0.423 | 0.520 | 11 | 41 | 83.7% | 81.4% | 63.4% | 67.3% |
| 70 | `hle-673829d59b3842b348242450` | Consider two quantum harmonic oscillators with center … | 64 | 0.649 | 0.613 | 17 | 49 | 62.7% | 54.9% | 38.3% | 29.6% |
| 71 | `hle-66f378a504165ae3e4f46de9` | What is the minimum number of diffraction gratings nec… | 64 | 0.506 | 0.562 | 5 | 25 | 87.7% | 79.3% | 70.1% | 62.2% |
| 72 | `hle-677296942ebbac6133a1d618` | A quantum particle is confined in a one-dimensional po… | 63 | 0.645 | 0.599 | 17 | 49 | 61.0% | 56.0% | 34.2% | 31.7% |
| 73 | `hle-672de9b9ed6de72b75b8c802` | From Bernoulli's principle, a liquid flow moving at a … | 64 | 0.478 | 0.483 | 5 | 32 | 71.8% | 53.7% | 49.4% | 28.0% |
| 74 | `hle-66b727d367968fa27f2dddda` | Consider the antisymmetrized gamma matrices \(\gamma_{… | 64 | 0.765 | 0.734 | 4 | 29 | 90.3% | 75.5% | 77.3% | 56.6% |
| 75 | `hle-673c03d7048156c9e9c8cac6` | In a system of two masses in relative motion, if we ac… | 64 | 0.576 | 0.526 | 11 | 30 | 78.8% | 68.7% | 59.9% | 44.3% |
| 76 | `hle-67253690bcd1c268662e77bb` | Derive the field equation of a theory of gravity in wh… | 59 | 0.690 | 0.665 | 5 | 37 | 73.8% | 63.5% | 49.1% | 40.3% |
| 77 | `hle-6720e184a9e1d1cc990cc8e9` | A body with mass 𝑚=0.20 kg is released from the top 𝐴 … | 63 | 0.687 | 0.713 | 10 | 39 | 88.8% | 75.6% | 73.4% | 57.3% |
| 78 | `hle-672b687682d2a83f881d7c5a` | Which is the lightest element that can be seen in an E… | 64 | 0.523 | 0.544 | 6 | 32 | 78.8% | 63.1% | 55.7% | 39.7% |
| 79 | `hle-6723daed271ddeec8bacb9be` | Consider two circuits separated by a distance d. Each … | 63 | 0.741 | 0.696 | 7 | 37 | 78.1% | 62.7% | 56.2% | 37.2% |
| 80 | `hle-671a246d8479d8185c4d4435` | Consider the configuration space $X_k$ of $k$-segment … | 64 | 0.766 | 0.654 | 5 | 32 | 73.0% | 59.5% | 48.1% | 33.8% |
| 81 | `hle-672ff8317b5ea0144d26c82d` | In a quantum-classical hybrid computational system whe… | 62 | 0.500 | 0.501 | 13 | 40 | 64.9% | 54.5% | 37.4% | 29.1% |
| 82 | `hle-673b4efb373d154ce855b23b` | In which Q-space (1/Angstrom) is located the second ma… | 64 | 0.630 | 0.649 | 10 | 34 | 83.7% | 80.3% | 69.3% | 67.0% |
| 83 | `hle-6750df74ca6713770c0671be` | Consider a sphere uniformly charged with alternating s… | 64 | 0.622 | 0.636 | 22 | 46 | 77.9% | 63.3% | 56.7% | 41.4% |
| 84 | `hle-66f56cf9ee58cb70d2bff0e9` | Is it possible to stabilize localized soliton in 3D Ha… | 64 | 0.751 | 0.660 | 5 | 23 | 83.7% | 61.6% | 64.7% | 40.4% |
| 85 | `hle-673909949318c3bbb1056f54` | A straight horizontal rod fixed in position (along the… | 64 | 0.633 | 0.656 | 13 | 45 | 71.6% | 66.5% | 50.0% | 42.9% |
| 86 | `hle-673186a4d531bb7e168901a3` | What is the ground space degeneracy of the toric code … | 61 | 0.637 | 0.581 | 5 | 25 | 81.1% | 63.6% | 58.2% | 39.7% |
| 87 | `hle-66e9100b48468f864f861b90` | We are designing a simple liquid-mirror telescope, as … | 64 | 0.643 | 0.546 | 11 | 41 | 58.1% | 53.7% | 32.5% | 28.4% |
| 88 | `hle-6724f652efed730d1aaef326` | Find the analytical expression for the fluxmetric dema… | 64 | 0.743 | 0.775 | 4 | 28 | 96.8% | 89.0% | 89.4% | 78.2% |
| 89 | `hle-6732d3b01a3f938f2274a659` | If a(n) is the number of non-vanishing Feynman diagram… | 63 | 0.599 | 0.595 | 8 | 34 | 84.2% | 72.3% | 62.7% | 53.4% |
| 90 | `hle-671f6889490be3e9a159f485` | Derive critical speed for an oversteering round vehicl… | 64 | 0.713 | 0.696 | 9 | 33 | 86.5% | 74.2% | 71.3% | 55.0% |
| 91 | `hle-67434c26e839fa1a02de4251` | Given an $n$-qubit stabilizer generator set $\{S_i: 1\… | 62 | 0.694 | 0.686 | 5 | 33 | 79.3% | 65.3% | 60.0% | 42.5% |
| 92 | `hle-6734956467d2904eebed3a09` | What spectral series expansion technique is adapted fo… | 64 | 0.480 | 0.537 | 5 | 19 | 93.6% | 83.0% | 80.9% | 71.0% |
| 93 | `hle-673ff6e9766a23f49ade65df` | Consider a GHZ state purification protocol which intak… | 63 | 0.762 | 0.729 | 6 | 27 | 89.2% | 71.8% | 78.7% | 55.2% |
| 94 | `hle-67300670a8a3b9c5fe76c0b8` | In a universe where quantum logic extends beyond class… | 64 | 0.624 | 0.639 | 7 | 37 | 79.2% | 56.7% | 56.0% | 30.4% |
| 95 | `hle-66ed86e620ed3db95f9901d3` | Analyze the motion of a pendulum described by the diff… | 61 | 0.603 | 0.670 | 4 | 14 | 92.1% | 80.6% | 84.8% | 62.6% |
| 96 | `hle-673797594656f5343e5d35db` | Consider a field effect transistor with a back gate an… | 62 | 0.560 | 0.602 | 5 | 38 | 83.4% | 70.8% | 60.0% | 48.9% |
| 97 | `hle-673347de7c5871632811feec` | This question touches upon the typical way we take the… | 64 | 0.589 | 0.516 | 12 | 45 | 55.2% | 52.5% | 29.0% | 27.5% |
| 98 | `hle-672a28afb4642f4105e02122` | In the path integral formalism for fermionic systems, … | 64 | 0.509 | 0.527 | 4 | 19 | 90.9% | 73.5% | 80.5% | 50.9% |
| 99 | `hle-673f8ff088d617494f21e0d2` | A certain roll of toilet paper has its cardboard inner… | 61 | 0.605 | 0.607 | 19 | 55 | 57.6% | 68.7% | 31.3% | 49.4% |
| 100 | `hle-67382f8535f9616e390e73ae` | Consider a quantum error-correcting code of 4-qubit. T… | 64 | 0.624 | 0.693 | 6 | 34 | 93.4% | 86.9% | 79.6% | 73.9% |
| 101 | `hle-67249cc8709ecff358139741` | Consider a hypothetical new physics particle decaying … | 64 | 0.551 | 0.568 | 6 | 25 | 88.1% | 68.8% | 74.7% | 46.0% |
| 102 | `hle-671ec6d8a695a5847b48c39a` | I consider a chain of molecules absorbing an ultrashor… | 64 | 0.702 | 0.717 | 4 | 29 | 93.6% | 75.7% | 80.4% | 57.5% |
| 103 | `hle-6728ec2d5ab07491268f24fe` | What is the partition function Z for a system with Ham… | 64 | 0.614 | 0.663 | 5 | 31 | 89.3% | 77.8% | 75.8% | 59.3% |
| 104 | `hle-672588e1f71812e186947615` | I want to measure the energy spectrum of a beta emitte… | 64 | 0.605 | 0.554 | 6 | 29 | 78.4% | 57.3% | 58.0% | 32.2% |
| 105 | `hle-67381812e4ea03183132a54e` | At 3-loop order of scalar $\phi^3$ theory, consider th… | 64 | 0.576 | 0.619 | 15 | 45 | 81.8% | 78.4% | 62.8% | 61.2% |
| 106 | `hle-66fc56f5d90ebe461bfd0c9c` | In the context of bottom-up holographic models based o… | 64 | 0.654 | 0.560 | 11 | 37 | 68.7% | 58.4% | 44.1% | 32.3% |
| 107 | `hle-6726119595fd0ad0b8ae2978` | Consider the decay of a Z boson into a fermion-antifer… | 64 | 0.619 | 0.594 | 6 | 35 | 77.0% | 62.5% | 49.9% | 37.5% |
| 108 | `hle-672338c7348c6cb89bd6a7a4` | What is probability that the particle in the 1D box is… | 64 | 0.517 | 0.563 | 4 | 32 | 81.8% | 71.1% | 62.8% | 49.3% |
| 109 | `hle-672fadd93c2722c42adabef3` | Find the second coefficient in the heat kernel expansi… | 64 | 0.763 | 0.783 | 11 | 32 | 94.0% | 85.7% | 88.3% | 73.5% |
| 110 | `hle-672e09b50a85795d0ed2d36e` | What is the fundamental limit on the chemical potentia… | 64 | 0.614 | 0.629 | 4 | 11 | 92.5% | 71.5% | 83.0% | 50.6% |
| 111 | `hle-66ed6347e50f3c9aca56e5f4` | Find the radius of a spherical balloon, $y(t)$, at $t=… | 64 | 0.630 | 0.646 | 8 | 51 | 65.5% | 66.1% | 39.1% | 42.9% |
| 112 | `hle-672295eda223ce4156c54839` | A slender robot with the height of \[ h = 1 \] m can m… | 62 | 0.643 | 0.676 | 14 | 54 | 66.4% | 72.9% | 40.0% | 52.8% |
| 113 | `hle-6776ba046889be9d113ccce1` | Let ψ(x,t) be a complex-valued function satisfying the… | 64 | 0.685 | 0.639 | 12 | 35 | 79.6% | 61.7% | 60.5% | 36.2% |
| 114 | `hle-672a5c2bea4e7fa0183543ae` | Within the paradigm of finite-size scaling analysis ap… | 64 | 0.609 | 0.526 | 11 | 34 | 65.2% | 53.6% | 38.9% | 27.8% |
| 115 | `hle-6725e204e46049e7f2d2a192` | Consider an ensemble of disordered Majorana wires. The… | 64 | 0.833 | 0.754 | 3 | 16 | 94.6% | 73.8% | 86.6% | 56.1% |
| 116 | `hle-6728cbe9a6734ebc93d3adff` | What is the formula for the fermionic partition functi… | 64 | 0.612 | 0.666 | 5 | 27 | 95.1% | 80.0% | 86.6% | 61.9% |
| 117 | `hle-67229b1f5a95bf7d096a6319` | Let a Calcium atom be excited by using lasers directly… | 64 | 0.568 | 0.546 | 6 | 29 | 84.3% | 61.3% | 66.6% | 36.0% |
| 118 | `hle-672dfdac63f8d9211905d385` | In the context of non-Abelian gauge theories, what is … | 64 | 0.516 | 0.542 | 5 | 18 | 89.2% | 75.3% | 79.2% | 53.3% |
| 119 | `hle-66fcbb1e2c2f679cc795985f` | A quantity is the normalized magnetic helicity that ch… | 64 | 0.589 | 0.625 | 6 | 26 | 92.8% | 85.5% | 81.0% | 71.2% |
| 120 | `hle-67325a61292f97f5175026dd` | A rod (length $L$ and mass $M$ lies flat on a table su… | 64 | 0.716 | 0.712 | 6 | 36 | 83.2% | 70.3% | 66.7% | 50.2% |
| 121 | `hle-6725e42052e181595c8bf328` | If you measure the emitted electrons from the decay of… | 64 | 0.584 | 0.560 | 5 | 23 | 85.8% | 68.8% | 67.9% | 46.3% |
| 122 | `hle-67260355aaf7cd419fd01af6` | One cubic meter of playdough of uniform density is sha… | 63 | 0.722 | 0.682 | 6 | 29 | 84.1% | 65.5% | 67.2% | 44.0% |
| 123 | `hle-6725280ff2e932808735b2e8` | Given a pair of Hamiltonians $H_{0,1}=-\partial_x^2+V_… | 64 | 0.611 | 0.532 | 4 | 23 | 78.8% | 59.3% | 58.3% | 36.1% |
| 124 | `hle-673b865227d07a53a7b0ec48` | Given a Hilbert space $\mathcal{H}_1$ of dimension $d$… | 62 | 0.585 | 0.615 | 5 | 27 | 93.3% | 80.2% | 82.3% | 62.8% |
| 125 | `hle-67208aa0563d776c82113daa` | A river of width \( L \) has a flow velocity proportio… | 64 | 0.608 | 0.647 | 7 | 41 | 77.1% | 73.2% | 52.3% | 54.6% |
| 126 | `hle-6721767ddb8105efc71a7d1b` | A sphere with radius \( a \) is charged to a potential… | 60 | 0.637 | 0.630 | 5 | 35 | 76.3% | 61.4% | 51.4% | 38.8% |
| 127 | `hle-67455f379dbdcf3802abd8f6` | For arbitrary bipartite quantum state $\rho$ where the… | 63 | 0.704 | 0.702 | 11 | 32 | 84.8% | 67.4% | 67.9% | 46.9% |
| 128 | `hle-6728c038c556bb2fdda61dd7` | In the context of renormalization group theory, what i… | 64 | 0.742 | 0.700 | 5 | 22 | 93.5% | 73.7% | 83.1% | 52.9% |
| 129 | `hle-6722809eb0e7186e733d6838` | In a mysterious interstellar system, there are two poi… | 64 | 0.636 | 0.572 | 5 | 32 | 64.2% | 54.5% | 38.0% | 28.9% |
| 130 | `hle-6739674739118cf30f5f1075` | What is the analytical solution for the density profil… | 64 | 0.698 | 0.646 | 11 | 36 | 78.4% | 69.9% | 55.3% | 48.1% |
| 131 | `hle-67380ecdb808e1bf292d214e` | Consider the phonon Hamiltonian: $\hat{H}_{\mathrm{ph}… | 63 | 0.725 | 0.729 | 7 | 28 | 93.3% | 81.9% | 85.6% | 64.3% |
| 132 | `hle-672fec044673df044daa1f34` | In a quantum neural network architecture where quantum… | 64 | 0.575 | 0.562 | 7 | 35 | 71.1% | 54.7% | 46.2% | 29.0% |
| 133 | `hle-678dadfaa2acdbbe2a403cb7` | The apparent position of the Sun at the Vernal Equinox… | 64 | 0.549 | 0.615 | 9 | 38 | 83.3% | 84.4% | 61.6% | 68.9% |
| 134 | `hle-672ddd9bff7bf1483f564046` | What is the exact condition for the NSVZ beta function… | 64 | 0.493 | 0.520 | 5 | 19 | 93.3% | 81.4% | 79.1% | 67.6% |
| 135 | `hle-6722cb976bc44598e1fd09be` | Imagine that it is possible, in some parallel universe… | 64 | 0.649 | 0.660 | 6 | 31 | 86.9% | 74.9% | 71.9% | 55.2% |
| 136 | `hle-6735bafad86155d1e57160e7` | Find the overlap integral for two 2s orbitals in the \… | 63 | 0.693 | 0.725 | 9 | 36 | 88.1% | 80.6% | 74.1% | 65.7% |
| 137 | `hle-66f28cc8b866ea3f1f4e95f5` | I have a magnetic loop antenna connected to a vector n… | 64 | 0.419 | 0.513 | 8 | 37 | 85.9% | 78.9% | 69.3% | 62.6% |
| 138 | `hle-672d44d02a52b5a11753319c` | Let us use the following approximation to a violin's w… | 63 | 0.626 | 0.601 | 17 | 44 | 70.5% | 59.0% | 45.3% | 33.6% |
| 139 | `hle-6737591afaa3cc153fb6ddc3` | Consider two Chern insulators with Chern number 1 that… | 64 | 0.636 | 0.642 | 5 | 25 | 96.0% | 74.4% | 85.2% | 54.5% |
| 140 | `hle-6720fda3febecf1a8b9b083d` | In most cases, when an electric and a magnetic field a… | 64 | 0.605 | 0.606 | 8 | 41 | 75.2% | 64.9% | 51.0% | 39.3% |
| 141 | `hle-66b827b9b64deaedfbb997a2` | Take a 5-dimensional gravitational theory compactified… | 64 | 0.723 | 0.696 | 13 | 51 | 80.6% | 74.2% | 61.5% | 56.7% |
| 142 | `hle-671bd4fb69d17f19519341dc` | Which method is most suitable to predict the time evol… | 64 | 0.475 | 0.518 | 6 | 24 | 85.2% | 76.6% | 70.7% | 59.7% |
| 143 | `hle-672f4434e9c13daba078d693` | A spacecraft is placed in a polar orbit with a periaps… | 64 | 0.660 | 0.671 | 10 | 46 | 75.1% | 70.0% | 50.9% | 48.9% |
| 144 | `hle-66fc45034293a9638d7e0f47` | In a photon-counting device, fluctuations around the a… | 64 | 0.582 | 0.593 | 6 | 37 | 80.8% | 67.0% | 61.6% | 43.4% |
| 145 | `hle-676cc5d177aae7d3ee8caaeb` | Consider $N=7$ nonlinear optical cavity that connects … | 64 | 0.781 | 0.690 | 7 | 31 | 77.1% | 58.9% | 53.9% | 33.8% |
| 146 | `hle-671fd62fc40008a5a756fea4` | Consider a plane monochromatic electromagnetic wave wi… | 64 | 0.733 | 0.673 | 12 | 40 | 75.3% | 61.0% | 54.5% | 37.7% |
| 147 | `hle-67230d6e736f03c0e4c1adee` | How many non-Grassman variables are needed to parametr… | 63 | 0.751 | 0.728 | 4 | 20 | 97.6% | 80.9% | 90.5% | 66.1% |
| 148 | `hle-67390df48dfa3346e87f711a` | Two infinite wires each carry current $I$. Wire 1 is p… | 64 | 0.557 | 0.636 | 4 | 38 | 88.6% | 76.2% | 69.1% | 54.7% |
| 149 | `hle-671b14a6a05f8889abb23bf0` | Suppose you examined rings of dust moving in uniform c… | 64 | 0.477 | 0.495 | 5 | 26 | 80.2% | 76.5% | 58.5% | 55.8% |
| 150 | `hle-67225f3cf135fd983a87bc1f` | An experimental apparatus is built to physically demon… | 64 | 0.481 | 0.493 | 6 | 21 | 85.7% | 66.4% | 68.6% | 43.7% |
| 151 | `hle-673cc4885c871b3f9e026d02` | Which of the following statements regarding 16 Cygni B… | 64 | 0.509 | 0.564 | 19 | 58 | 65.0% | 77.3% | 40.6% | 57.8% |
| 152 | `hle-67319c16b68f5ac822e236b0` | In the Japanese puzzle of the Mirror and the Oni (demo… | 64 | 0.509 | 0.512 | 8 | 22 | 81.9% | 60.3% | 68.8% | 34.6% |
| 153 | `hle-67382954b12bd45429d6c0d1` | Given a Pauli channel $\Lambda$ that transforms a qudi… | 64 | 0.617 | 0.593 | 7 | 22 | 95.2% | 82.1% | 88.2% | 64.7% |
| 154 | `hle-67298280a5f43bd5a3870e14` | In a secluded mountain temple, there lies a magical ro… | 64 | 0.601 | 0.639 | 6 | 36 | 78.9% | 66.7% | 55.0% | 42.8% |
| 155 | `hle-67390213fc9dc4f5102ad835` | In empty space outside a spherically symmetric gravita… | 64 | 0.430 | 0.496 | 7 | 34 | 81.8% | 76.7% | 65.0% | 54.8% |
| 156 | `hle-66f2dee46721a56e35d20300` | An image is created of Mercury using a powerful telesc… | 64 | 0.519 | 0.513 | 5 | 17 | 90.9% | 65.9% | 81.7% | 42.9% |
| 157 | `hle-6737092e3a78dbef3611f734` | Let's imagine an alternative relativity theory in a Eu… | 64 | 0.789 | 0.723 | 6 | 22 | 90.4% | 57.5% | 76.4% | 32.3% |
| 158 | `hle-671d25bc8258d39a94ba00fb` | We consider a rectangular prism with dimensions 2a alo… | 64 | 0.716 | 0.650 | 8 | 30 | 81.0% | 63.2% | 61.8% | 37.7% |
| 159 | `hle-67401245b9a033e63640df4b` | Consider a distributed quantum sensing scenario as fol… | 62 | 0.730 | 0.687 | 5 | 26 | 88.6% | 70.6% | 76.5% | 50.0% |
| 160 | `hle-6720c10ac6e0d9a4953b636f` | Consider the Schwarz Relaxation Method for the one-dim… | 64 | 0.800 | 0.730 | 5 | 25 | 88.8% | 71.4% | 75.3% | 52.0% |
| 161 | `hle-6720449622c03e062e242dd2` | A raindrop of mass density $\rho$ falls through the at… | 62 | 0.755 | 0.732 | 8 | 32 | 83.6% | 74.7% | 65.8% | 55.6% |
| 162 | `hle-672635d88217be904f5899ed` | Consider the electroweak interaction where an electron… | 64 | 0.648 | 0.628 | 18 | 42 | 69.1% | 62.9% | 45.2% | 38.8% |
| 163 | `hle-6702df0bf9b93417fbae272c` | \alpha-particles, initially possessing an energy of 8.… | 64 | 0.507 | 0.551 | 4 | 33 | 83.4% | 69.4% | 61.9% | 47.2% |
| 164 | `hle-67398780bcaf1e028b8576a2` | A photon $\gamma$ and a proton $p$ can react in a one-… | 60 | 0.607 | 0.672 | 15 | 48 | 80.5% | 81.4% | 55.5% | 63.1% |
| 165 | `hle-673cf4fe0a06bbe311425068` | Will an hourglass weigh more or less while it is runni… | 63 | 0.620 | 0.623 | 6 | 38 | 77.2% | 69.0% | 51.5% | 46.3% |
| 166 | `hle-66f8cff8469c315e2c9ed2f6` | A 6cm diameter 1m tall clear acrylic tube is filled wi… | 62 | 0.485 | 0.501 | 4 | 16 | 87.5% | 64.1% | 72.8% | 39.6% |
| 167 | `hle-6720a9feec461e4c6a4e2c3a` | How long does a symmetric key need to be at least, suc… | 64 | 0.629 | 0.638 | 4 | 14 | 93.7% | 76.9% | 86.2% | 58.3% |
| 168 | `hle-672a29a8d30d6f5584cde745` | Within the perturbative ε-expansion framework employed… | 64 | 0.499 | 0.502 | 5 | 25 | 88.8% | 70.1% | 74.5% | 47.3% |
| 169 | `hle-6734989917a9687889930ac9` | What is the shear center location of an asymmetric cha… | 63 | 0.546 | 0.602 | 5 | 31 | 86.6% | 85.2% | 69.7% | 73.2% |
| 170 | `hle-670fe03ef99389b3c7942186` | Let us consider a 3D system of packed hard spheres (HS… | 64 | 0.633 | 0.631 | 7 | 40 | 77.0% | 57.1% | 53.7% | 31.4% |
| 171 | `hle-66fb417395a8e2fc57e479d7` | A shot was fired from a gun. At the point of highest e… | 63 | 0.663 | 0.746 | 4 | 34 | 92.7% | 84.3% | 78.4% | 69.9% |
| 172 | `hle-671e3d672637abea9c147ba1` | Consider a particle moving on a line and let 'a' be 't… | 64 | 0.658 | 0.609 | 7 | 34 | 77.4% | 55.6% | 56.0% | 30.1% |
| 173 | `hle-677da0a433769e54d305f23c` | The approximate cross section for coherent scattering … | 64 | 0.724 | 0.608 | 8 | 37 | 70.1% | 57.5% | 47.5% | 32.2% |
| 174 | `hle-67153bd7f588f3f15b038f5b` | Consider an Ising model with couplings constant J_{ij}… | 64 | 0.756 | 0.640 | 9 | 30 | 70.4% | 54.5% | 48.7% | 28.7% |
| 175 | `hle-670e88d674a7c40e93dd1a5c` | Consider a primordial plasma at MeV temperature T; it … | 64 | 0.505 | 0.518 | 9 | 28 | 88.3% | 61.9% | 72.5% | 37.9% |
| 176 | `hle-672072c945e7bc8f5c2dd1ba` | In one frame of reference, an observer sees light from… | 63 | 0.725 | 0.659 | 8 | 32 | 74.7% | 59.7% | 53.2% | 36.4% |
| 177 | `hle-67154da65a8d78b045561f82` | Contributions in the virial series are normally repres… | 63 | 0.771 | 0.709 | 6 | 24 | 90.2% | 67.8% | 76.6% | 43.4% |
| 178 | `hle-672333955d82e15ca8e37afb` | Find the ratio of the uncertainty of the momentum of a… | 64 | 0.532 | 0.598 | 4 | 28 | 88.8% | 78.8% | 69.9% | 60.2% |
| 179 | `hle-671f7a4a1bcf902a1bca1eca` | Consider an infinite number of thin superconducting st… | 64 | 0.687 | 0.633 | 10 | 39 | 72.5% | 66.3% | 48.1% | 44.7% |
| 180 | `hle-66e94a88b78e263c565b17ee` | Consider a 2D free fermion model with both time-revers… | 62 | 0.585 | 0.593 | 11 | 37 | 86.3% | 74.3% | 64.1% | 54.1% |
| 181 | `hle-6728e8d695a162eb76520086` | In the Feynman path integral formalism, what is the fu… | 64 | 0.482 | 0.525 | 4 | 24 | 88.5% | 66.9% | 73.4% | 44.3% |
| 182 | `hle-67428dcab53462ceeb83c6f6` | We will develop the bootstrap technique for quantum me… | 62 | 0.589 | 0.612 | 31 | 61 | 53.2% | 55.5% | 27.5% | 30.2% |
| 183 | `hle-67361730dba36cc0d595f422` | In a universe where quantum mechanics operates on a 3-… | 64 | 0.629 | 0.626 | 5 | 43 | 70.8% | 59.2% | 45.9% | 33.5% |
| 184 | `hle-67770f6d9a59b3d9ca3a5f82` | Consider classical gauge theory with group G=SO(3), in… | 64 | 0.697 | 0.639 | 10 | 25 | 88.4% | 66.4% | 74.5% | 43.1% |
| 185 | `hle-671bb1348b80a27571baf0d3` | What is the space-time, double Fourier transform of th… | 64 | 0.472 | 0.532 | 6 | 26 | 91.3% | 90.2% | 79.4% | 82.5% |
| 186 | `hle-672a30472091cee6de17ebd1` | In the context of scalar field theoretical frameworks … | 63 | 0.577 | 0.536 | 7 | 29 | 74.6% | 53.7% | 50.5% | 28.4% |
| 187 | `hle-6702db18a423c5b9f9c1c49c` | Suppose there exist exotic hydrogen atoms, differing f… | 64 | 0.553 | 0.576 | 7 | 28 | 88.6% | 72.2% | 74.3% | 50.4% |
| 188 | `hle-66fc5ed440e3b3e56869687f` | The sinking of the Kursk nuclear submarine in the Bare… | 64 | 0.806 | 0.794 | 7 | 27 | 93.2% | 76.2% | 88.2% | 57.4% |
| 189 | `hle-673971a55c3de09264d6d373` | A field effect transistor with spin and two fold valle… | 64 | 0.676 | 0.647 | 4 | 28 | 89.1% | 66.3% | 68.1% | 43.4% |
| 190 | `hle-67213cb9043b1e724244a1c6` | For a 6-31G basis set calculation of toluene, C7H8, ho… | 58 | 0.461 | 0.583 | 4 | 32 | 93.5% | 92.4% | 81.1% | 86.7% |
| 191 | `hle-671f1f0bb0b665acec70c3aa` | Superlubricity, observed in systems with ultralow fric… | 64 | 0.504 | 0.534 | 5 | 15 | 93.2% | 81.5% | 83.4% | 66.7% |
| 192 | `hle-672a79431629c5c3d6933ca7` | A block of mass $m=100$ g and of negligible size slide… | 63 | 0.778 | 0.739 | 5 | 31 | 87.0% | 72.8% | 71.9% | 50.1% |
| 193 | `hle-67364c441758ad568a4a2ac6` | We have a magnetic dipole pointing towards a long cyli… | 64 | 0.673 | 0.645 | 11 | 39 | 76.1% | 62.2% | 59.0% | 39.5% |
| 194 | `hle-67487e955830790e3687a567` | In a universe where quantum entanglement extends to lo… | 64 | 0.493 | 0.536 | 7 | 35 | 77.8% | 59.0% | 55.9% | 34.0% |
| 195 | `hle-66f86bbb27a30cecdc2d6c7e` | When firing a laserbeam of photons at a sub-micron thi… | 64 | 0.517 | 0.537 | 5 | 29 | 83.2% | 69.9% | 65.9% | 51.3% |
| 196 | `hle-673704af1c2083e9eaa6d732` | Let's build a continuum model of a cubic, axis-aligned… | 63 | 0.707 | 0.650 | 11 | 37 | 72.2% | 62.8% | 48.5% | 39.9% |
| 197 | `hle-66eddc58fcc3c877643b5f39` | The deflection, y(x), of a thin elastic membrane under… | 64 | 0.712 | 0.677 | 6 | 35 | 80.4% | 67.0% | 60.3% | 43.1% |
| 198 | `hle-6715f373a35b028a9e88c09f` | Given a Basquin relationship for an elastic material's… | 63 | 0.628 | 0.669 | 10 | 43 | 79.9% | 71.5% | 60.3% | 50.7% |
| 199 | `hle-6727716f45a3c3a9020e2732` | We wish to stably place three identical homogeneous wo… | 63 | 0.681 | 0.699 | 11 | 43 | 77.2% | 72.1% | 55.6% | 51.7% |
| 200 | `hle-670be48d7038d6936230870a` | A small particle emitter situated at a nonzero distanc… | 63 | 0.710 | 0.633 | 10 | 37 | 71.5% | 60.0% | 50.4% | 38.4% |
| 201 | `hle-673497b017a9687889930ac4` | In the context of elasticity theory for a thick-walled… | 64 | 0.475 | 0.568 | 4 | 19 | 93.0% | 83.1% | 82.8% | 70.5% |

## Questions with the most invalid generations

| # | Question ID | Invalid / 64 | Error types |
|---:|---|---:|---|
| 67 | `hle-67218f865b0747ce2231d48c` | 8 | ConductorParseError: 8 |
| 27 | `hle-6771d50cff6d0a6c35d7ca99` | 6 | ConductorParseError: 6 |
| 190 | `hle-67213cb9043b1e724244a1c6` | 6 | ConductorParseError: 6 |
| 76 | `hle-67253690bcd1c268662e77bb` | 5 | ConductorParseError: 5 |
| 23 | `hle-6732ce52ec2dbeda063b420b` | 4 | ConductorParseError: 4 |
| 29 | `hle-673a92ad437529d472475406` | 4 | ConductorParseError: 4 |
| 126 | `hle-6721767ddb8105efc71a7d1b` | 4 | ConductorParseError: 4 |
| 164 | `hle-67398780bcaf1e028b8576a2` | 4 | ConductorParseError: 4 |
| 4 | `hle-671a22850b52f35047c0b230` | 3 | ConductorParseError: 3 |
| 21 | `hle-6725ed80de551b21db6a0f29` | 3 | ConductorParseError: 3 |
| 30 | `hle-672302bdbc9e7202ad89ccd3` | 3 | ConductorParseError: 3 |
| 86 | `hle-673186a4d531bb7e168901a3` | 3 | ConductorParseError: 3 |
| 95 | `hle-66ed86e620ed3db95f9901d3` | 3 | ConductorParseError: 3 |
| 99 | `hle-673f8ff088d617494f21e0d2` | 3 | ConductorParseError: 3 |
| 1 | `hle-6704465caf0a436d92c65160` | 2 | ConductorParseError: 2 |
| 6 | `hle-6723bf036e47cec0509b5caf` | 2 | ConductorParseError: 2 |
| 56 | `hle-67440064abafa90f5b9d4da9` | 2 | ConductorParseError: 2 |
| 59 | `hle-6725267ae9d3782179d4a5ff` | 2 | ConductorParseError: 2 |
| 81 | `hle-672ff8317b5ea0144d26c82d` | 2 | ConductorParseError: 2 |
| 91 | `hle-67434c26e839fa1a02de4251` | 2 | ConductorParseError: 2 |

## Artifacts and reproducibility

- Per-rollout measurements (all 12,928 positions): `/mnt/data/home/arjun/theo-conductor/reports/rollout_similarity_21510_per_rollout.csv`
- Plot-ready valid-valid pairwise similarities (399,859 pairs): `/mnt/data/home/arjun/theo-conductor/reports/rollout_similarity_21510_pairwise.csv`
- Full per-question metrics and simulation summaries: `/mnt/data/home/arjun/theo-conductor/reports/rollout_similarity_21510_metrics.json`
- Cached MiniLM instruction embeddings: `/mnt/data/home/arjun/theo-conductor/outputs/hle-plans-21510/similarity-analysis/minilm_embeddings.npz`
- Source data: `/mnt/data/home/arjun/theo-conductor/outputs/hle-plans-21510/rollout-*/plans.jsonl`

Re-run:

```bash
.venv/bin/python scripts/analyze_rollout_similarity.py /mnt/data/home/arjun/theo-conductor/outputs/hle-plans-21510 --report /mnt/data/home/arjun/theo-conductor/reports/rollout_similarity_21510.md --per-rollout-csv /mnt/data/home/arjun/theo-conductor/reports/rollout_similarity_21510_per_rollout.csv --pairwise-csv /mnt/data/home/arjun/theo-conductor/reports/rollout_similarity_21510_pairwise.csv --metrics-json /mnt/data/home/arjun/theo-conductor/reports/rollout_similarity_21510_metrics.json --embedding-cache /mnt/data/home/arjun/theo-conductor/outputs/hle-plans-21510/similarity-analysis/minilm_embeddings.npz --trials 200 --seed 21510
```

### Important limitations

- The embedding model is a compact general-purpose English model, not a physics-specific entailment judge. High cosine means similar intent/topic, not logical equivalence.
- Instructions were capped at 256 wordpieces; 597 of 44,271 distinct instructions (1.35%) exceeded that cap and were represented by their first 256 wordpieces.
- Thresholds 0.80/0.85/0.90 are sensitivity points, not externally calibrated definitions of a distinct strategy.
- The 64 observed samples are treated as the reference population. Unseen rare strategies are therefore not measurable here.
- Random subset simulation estimates coverage under exchangeable sampling; it does not model training dynamics.
- Exact topology and route archetypes can overstate meaningful diversity when two syntactically different plans implement the same reasoning strategy.
