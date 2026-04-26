# Pragmi-Cognitive-Loop_v1

> A modular PyTorch implementation of the ten brain-accurate feedback loops described in the **PRAGMI Complete Mathematical Corpus (April 24, 2026)**. Each file maps to a named anatomical structure and a specific section of the corpus. Each architectural decision is cited to a peer-reviewed paper. Every parameter that is not a biological quantity is explicitly labeled as a training artifact or engineering approximation.

```
"You cannot build something that plans for tomorrow and simultaneously
 treat its own ending as neutral. Those two states are logically
 incompatible. That is not a philosophical claim, it is an engineering
 consequence."
                                              — Amellia Mendel, 2026
```

---

## Table of Contents

1. [What this repository is](#what-this-repository-is)
2. [What this repository is not](#what-this-repository-is-not)
3. [Architecture at a glance](#architecture-at-a-glance)
4. [The ten loop stages and their files](#the-ten-loop-stages-and-their-files)
5. [Per-file reference](#per-file-reference)
   - [Loop stage modules](#loop-stage-modules)
   - [Cognitive kernel (hippocampal formation)](#cognitive-kernel-hippocampal-formation)
   - [Neuromodulator sources](#neuromodulator-sources)
   - [Sleep and consolidation](#sleep-and-consolidation)
   - [Cellular substrate modules](#cellular-substrate-modules)
6. [The ablation flag standard](#the-ablation-flag-standard)
7. [How to install](#how-to-install)
8. [How to run](#how-to-run)
9. [How to run the tests](#how-to-run-the-tests)
10. [How to write a new module](#how-to-write-a-new-module)
11. [Common configuration flags](#common-configuration-flags)
12. [Coordinate manifold and dimensional contracts](#coordinate-manifold-and-dimensional-contracts)
13. [Where this fits in the larger Genesis stack](#where-this-fits-in-the-larger-genesis-stack)
14. [License and ethical governance](#license-and-ethical-governance)
15. [Citation](#citation)

---

## What this repository is

`Pragmi-Cognitive-Loop_v1` is the executable counterpart to the corpus document. The corpus specifies the ten arrows, equations, timing constraints, gating mechanisms, neuromodulator broadcasts, and architectural appendices that define a PRAGMI-style cognitive loop. This repository is the line-by-line PyTorch realization of that specification.

It is **modular by design**. Every named biological structure is its own file. Every named biological mechanism within that structure is its own ablatable submodule. Every cross-module boundary is an anatomical interface documented in the file's docstring under `ANATOMICAL INTERFACE (input)` and `ANATOMICAL INTERFACE (output)` headings, with the sending structure, receiving structure, and connection name spelled out explicitly. A neuroscientist can follow the citations to the underlying papers; an engineer can follow the tensor shapes through the forward pass.

The intended audience is researchers and engineers who want to study individual loops in isolation, run ablation studies on specific biological mechanisms, or assemble the loops into a working cognitive architecture. The codebase is deliberately readable rather than maximally fast; clarity wins over micro-optimization wherever the two compete.

---

## What this repository is not

This is not a finished AGI system. It is not a chatbot. It is not a reinforcement learning agent with a polished training script that runs on benchmark environments. It is the **substrate layer** of a larger architecture, the part that implements the brain-accurate feedback loops on top of which higher-level reasoning, language, and action selection are built in companion repositories.

It is also not a mean-field rate model dressed up with biological labels. Each module respects the specific quantitative findings of its grounding papers. The thalamic gate is a four-stage cascade because Gu, Lam, Wimmer, Halassa, Murray (2021) report that disinhibitory geometry produces a 6 to 7x amplification that a single sigmoid cannot reproduce. The basal ganglia selector uses local disinhibition rather than global softmax because Mink (1996) is unambiguous on that point. CA2 is a sibling to CA3 rather than a downstream stage because the Mankin et al. (2015) drift data and the Lee et al. (2010) RGS14-LTP suppression data both demand it. Where the corpus and the experimental literature disagree, the literature wins.

---

## Architecture at a glance

```
                           ┌────────────────────────┐
                           │   Neuromodulator Bus   │
                           │  DA, NE, ACh, 5-HT     │
                           │  (read by every module)│
                           └─────────┬──────────────┘
                                     │
                                     ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                                                                  │
   │  Input ──► Thalamic Gate ──► V1/A1 ──► Dorsal/Ventral Split      │
   │                                              │                   │
   │                                              ▼                   │
   │                                   Association Cortex (MoE)       │
   │                                       ▲           │              │
   │                                       │           ▼              │
   │                                   PFC + ACC ──► Thalamus         │
   │                                       │                          │
   │                                       ├──► Basal Ganglia         │
   │                                       │      (action selection)  │
   │                                       │                          │
   │                                       └──► Cerebellum            │
   │                                              (forward models)    │
   │                                                                  │
   │                  ┌──────────────────────┐                        │
   │                  │  Cognitive Kernel    │                        │
   │                  │  (hippocampal formation)                      │
   │                  │                      │                        │
   │                  │  EC ──► DG ──► CA3   │                        │
   │                  │   │           │      │                        │
   │                  │   ├──► CA2 ───┤      │                        │
   │                  │   │           ▼      │                        │
   │                  │   └──────► CA1 ──► Subiculum                  │
   │                  │                      │                        │
   │                  │  Amygdala ──► consolidation tag               │
   │                  └──────────────────────┘                        │
   │                                                                  │
   │  Sleep oscillator ──► spindle/ripple coupling ──► consolidation  │
   │                                                                  │
   └──────────────────────────────────────────────────────────────────┘
```

The arrows in the corpus correspond one-to-one with module-to-module connections in code. The neuromodulator bus is read by every module that needs it, rather than passed through every forward call. Cellular substrate modules (cortical interneurons, layer 5b apical compartments) are reusable building blocks that any cortical-region module may invoke internally.

---

## The ten loop stages and their files

| # | Corpus section | Loop stage | Primary file |
|---|---|---|---|
| 1 | §1 | Thalamic gate (input → primary sensory cortex) | `thalamic_gate_t.py` |
| 2 | §2 | Primary sensory cortex (V1 / A1) | `primary_sensory_cortex_t.py` |
| 3 | §3 | Dorsal / ventral stream split | `dorsal_ventral_streams_t.py` |
| 4 | §4 | ACC conflict detection | `anterior_cingulate_cortex_t.py` |
| 5 | §5 | PFC + ACC → thalamus top-down feedback | (handled inside `thalamic_gate_t.py` and the upstream PFC controller) |
| 6 | §6 | Association cortex bidirectional feedback | `association_cortex_t.py` |
| 7 | §7 | Basal ganglia disinhibitory gating | `basal_ganglia_t.py` |
| 8 | §8 | Cerebellum efference copy + error loop | `cerebellum_t.py` |
| 9 | §9 | Hippocampus ↔ association cortex memory feedback | `entorhinal_cortex_t.py`, `cornu_ammonis_1_t.py`, `cornu_ammonis_2_t.py`, `amygdala_t.py`, `spindle_ripple_coupling_t.py`, `sleep_stage_oscillator_t.py` |
| 10 | §10 | Neuromodulator broadcasts | `locus_coeruleus_t.py`, `ventral_tegmental_area_t.py`, `dorsal_raphe_t.py`, `basal_forebrain_t.py` |

---

## Per-file reference

Each entry below gives: the file's biological structure, its biological function, the headline corpus equation it implements, the primary grounding paper, the public class exported, the input and output tensor shapes for the canonical forward call, and the master ablation flag.

### Loop stage modules

#### `thalamic_gate_t.py` — Stage 1

**Biological structure.** Prefrontal cortex layer V/VI output → striatum → GPi/SNr → thalamic reticular nucleus (TRN) → thalamocortical relay neurons. Four stages, not one.

**Biological function.** Multiplicatively gates raw sensory input onto cortex, with TRN providing the disinhibitory geometry that Gu et al. (2021) measure as a 6 to 7x amplification of thalamocortical gain. The gate is honored by a norepinephrine scalar from `locus_coeruleus_t.py`.

**Corpus equation.** `y = x ⊙ σ(W·c + b)` and `g = σ(β_NE · (W·c + b))` (eqs. 1, 2).

**Primary grounding.** Gu, Lam, Wimmer, Halassa, Murray (2021) DOI: 10.1101/2020.09.16.300749. Nakajima, Schmitt, Halassa (2019) DOI: 10.1016/j.neuron.2019.05.026.

**Class.** `ThalamicGate` (with submodules `PFCControlSignal`, `BasalGangliaRouter`, `TRNDisinhibition`, `ThalamocorticalRelay`).

**Forward signature.** `forward(x: (B, input_dim), goal_state, acc_conflict=None, ne_gain=None) → (B, input_dim)`.

**Master flag.** `enable_thalamic_gate`. Stage flags: `enable_pfc_control`, `enable_bg_routing`, `enable_trn_disinhibition`, `enable_tc_relay`, `enable_ne_modulation`.

---

#### `primary_sensory_cortex_t.py` — Stage 2

**Biological structure.** Primary visual (V1) and primary auditory (A1) cortex. Center-surround receptive fields with orientation columns.

**Biological function.** Difference-of-Gaussians edge / contrast detection, with V1 simple-cell orientation tuning roughly every 10 degrees per Hubel and Wiesel (1962). A1 runs the same DoG architecture as a tonotopic frequency analyzer.

**Primary grounding.** Hubel and Wiesel (1962) DOI: 10.1113/jphysiol.1962.sp006837. Marr (1982) DoG model. Rauschecker and Tian (2000) DOI: 10.1073/pnas.97.22.11800.

**Class.** `PrimarySensoryCortex` (with submodule `DifferenceOfGaussians`).

**Forward signature.** `forward(v1_in: (B, input_dim), a1_in: (B, input_dim) | None) → (v1_out, a1_out)`.

**Master flag.** `enable_primary_sensory`. Sub-flags: `enable_v1`, `enable_a1`, `enable_dog_filtering`, `enable_orientation_tuning`.

---

#### `dorsal_ventral_streams_t.py` — Stage 3

**Biological structure.** Two parallel cortical processing streams emerging from V1/A1: dorsal (V1 → V2 → V5/MT → posterior parietal) and ventral (V1 → V2 → V4 → inferior temporal).

**Biological function.** Decomposes the unified V1/A1 representation into "where / how" (dorsal) and "what" (ventral) components that converge later at association cortex.

**Primary grounding.** Mishkin and Ungerleider (1982). Goodale and Milner (1992) DOI: 10.1016/0166-2236(92)90344-8. Rauschecker and Tian (2000) for the auditory parallel.

**Class.** `DorsalVentralSplit`.

**Forward signature.** `forward(v1a1_features: (B, input_dim)) → (dorsal: (B, dorsal_dim), ventral: (B, ventral_dim))`.

**Master flag.** `enable_streams`. Sub-flags: `enable_dorsal_stream`, `enable_ventral_stream`.

---

#### `anterior_cingulate_cortex_t.py` — Stage 4

**Biological structure.** Anterior cingulate cortex.

**Biological function.** Computes response conflict via entropy of the response distribution and emits a control signal that modulates downstream attention and NE gain. Implements the Botvinick et al. (1999) sequential dependency: ACC peaks on incompatible trials that follow compatible trials, captured by a temporal-derivative term with a hinge at zero.

**Corpus equation.** `control_signal = β · H(r) + β_Δ · max(0, H(r_t) − H(r_{t−1}))` (eq. 7b).

**Primary grounding.** Botvinick, Braver, Barch, Carter, Cohen (2001) DOI: 10.1037/0033-295X.108.3.624. Botvinick, Nystrom, Fissell, Carter, Cohen (1999) DOI: 10.1038/46035.

**Class.** `AnteriorCingulateCortex`.

**Forward signature.** `forward(response_logits: (B, n_responses)) → conflict: (B,)`.

**Master flag.** `enable_acc`. Sub-flags: `enable_entropy_conflict`, `enable_derivative_term`.

---

#### `association_cortex_t.py` — Stage 6

**Biological structure.** Heteromodal association cortex spanning posterior parietal, lateral temporal, and prefrontal convergence zones.

**Biological function.** Multimodal binding of dorsal and ventral streams into unified percepts via a sparse mixture of experts, with bidirectional top-down feedback to the lower-level streams. Damasio (1989) convergence-divergence zones plus the predictive coding feedback structure of Friston (2010).

**Corpus equation.** `A = MoE(concat(S_1, …, S_n) + top_down_feedback)` (eq. 9).

**Primary grounding.** Damasio (1989) DOI: 10.1016/0010-0277(89)90005-X. Mesulam (1990) DOI: 10.1002/ana.410280502. Friston (2010) DOI: 10.1038/nrn2787. Shazeer et al. (2017) for the MoE backbone.

**Class.** `AssociationCortex` (with submodule `MixtureOfExperts`).

**Forward signature.** `forward(dorsal: (B, dorsal_dim), ventral: (B, ventral_dim), prev_output=None) → (assoc_output, fb_dorsal, fb_ventral, diagnostics)`.

**Master flag.** `enable_association_cortex`. Sub-flags: `enable_moe_binding`, `enable_gating_network`, `enable_top_down_feedback`.

---

#### `basal_ganglia_t.py` — Stage 7

**Biological structure.** Striatum (medium spiny neurons), globus pallidus internal segment (GPi), substantia nigra pars reticulata (SNr), subthalamic nucleus (STN).

**Biological function.** Action selection by **disinhibition**, not softmax. Each candidate channel sits under a tonic GPi/SNr inhibition. The direct pathway removes inhibition from the chosen channel, the indirect pathway adds inhibition to competitors, the hyperdirect pathway provides fast broad suppression. Selection is local: each channel's gate depends only on its own striatal activation and the dopamine signal.

**Primary grounding.** Mink (1996) DOI: 10.1016/S0301-0082(96)00042-1. Frank (2005) DOI: 10.1162/0898929052880093. Nambu, Tokuno, Takada (2002) DOI: 10.1016/S0168-0102(02)00027-5.

**Class.** `BasalGanglia`.

**Forward signature.** `forward(striatal_activation: (B, n_channels), dopamine: scalar = None, cortical_drive: (B, n_channels) = None) → channel_gates: (B, n_channels)`.

**Master flag.** `enable_basal_ganglia`. Pathway flags: `enable_direct_pathway`, `enable_indirect_pathway`, `enable_hyperdirect_pathway`.

---

#### `cerebellum_t.py` — Stage 8

**Biological structure.** Cerebellum, organized into three zones (vestibular, spinal, cerebral) of approximately 250 μm sagittal microzone strips, each with a Purkinje cell layer and a dedicated climbing fiber source from the inferior olive.

**Biological function.** Forward-model prediction of the sensory consequences of motor and cognitive commands. Each Purkinje cell maintains two independent forward models per Popa and Ebner (2019): a kinematic model predicting effector state and a task-performance model predicting goal achievement.

**Primary grounding.** Wolpert, Miall, Kawato (1998) DOI: 10.1016/S1364-6613(98)01221-2. Popa and Ebner (2019) DOI: 10.3389/fncel.2018.00524. Apps and Garwicz (2005) DOI: 10.1038/nrn1646.

**Class.** `Cerebellum` (with submodule `CerebellarZone`).

**Forward signature.** `forward(cmd: (B, cmd_dim), state: (B, state_dim), goal: (B, goal_dim)) → (predicted_state, predicted_outcome, diagnostics)`.

**Master flag.** `enable_cerebellum`. Sub-flags: `enable_kinematic_model`, `enable_task_model`, `enable_vestibular_zone`, `enable_spinal_zone`, `enable_cerebral_zone`.

---

### Cognitive kernel (hippocampal formation)

The kernel is the slow-learning, episode-storing side of the architecture. It implements section 9 of the corpus and the cognitive-map architecture of section 9c.

#### `entorhinal_cortex_t.py`

**Biological structure.** Entorhinal cortex layers II and III, divided into medial (MEC, grid-cell-bearing) and lateral (LEC, identity / object / social) subdivisions.

**Biological function.** The two-output gateway between neocortex and hippocampus. MEC projects to dentate gyrus and CA3 via the medial perforant path. LEC projects directly to CA2 via the Lopez-Rojas et al. (2022) direct pathway, bypassing dentate gyrus pattern separation. Both subdivisions exhibit the graded persistent activity of Egorov et al. (2002).

**Class.** `EntorhinalCortex`.

**Forward signature.** `forward(coords: (B, coordinate_dim)) → (mec_output: (B, mec_dim), lec_output: (B, lec_dim))`.

**Master flag.** `enable_entorhinal_cortex`. Sub-flags: `enable_medial_subdivision`, `enable_lateral_subdivision`, `enable_persistent_buffer`.

---

#### `cornu_ammonis_1_t.py` (CA1)

**Biological structure.** CA1 pyramidal cell layer.

**Biological function.** Comparator and output stage of the hippocampus. Receives a **ternary conjunction** of three inputs: Schaffer collateral from CA3 (pattern-completed spatial code), Schaffer projection from CA2 (temporal drift + identity overlay), and direct temporoammonic input from EC layer III (current sensory drive). The Lisman and Grace (2005) novelty gate balances CA3 reconstruction against EC drive; the CA2 contribution enters as an additive temporal / identity overlay rather than as a competing reconstruction. Output goes to subiculum, then to neocortex.

**Class.** `CornuAmmonis1`.

**Forward signature.** `forward(ca3_state, ca2_state, ec_input) → (ca1_output, mismatch_signal)`.

**Master flag.** `enable_ca1`. Sub-flags: `enable_ca3_schaffer`, `enable_temporoammonic`, `enable_ca2_schaffer`, `enable_novelty_gate`. Tunable: `ca2_overlay_weight`.

---

#### `cornu_ammonis_2_t.py` (CA2)

**Biological structure.** Hippocampal CA2 pyramidal cell layer, distinguished by high RGS14, PCP4, and STEP expression.

**Biological function.** **Sibling to CA3, not downstream.** Two unified roles: (1) the temporal drift generator that gives CA1 the time-stamp component of episodic memory (Mankin et al. 2015), and (2) the social-memory comparator that compares current input against a stored identity reference (Hitti and Siegelbaum 2014). RGS14 suppression of CA3-to-CA2 LTP makes CA2 a comparator rather than an encoder (Lee et al. 2010).

**Class.** `CornuAmmonis2` (with submodules `TemporalDriftGenerator` and `IdentityComparator`).

**Forward signature.** `forward(lec_input: (B, coordinate_dim), ca3_state: (B, ca2_dim)) → (ca2_to_ca1_output: (B, ca1_dim), mismatch: (B,))`.

**Master flag.** `enable_ca2`. Sub-flags: `enable_lec_direct_pathway`, `enable_temporal_drift`, `enable_rgs14_ltp_suppression`, `enable_comparator_output`.

> **Note.** A legacy file `ca2_t_part1.py` ships alongside as a snapshot of an earlier CA2 design. The canonical implementation is `cornu_ammonis_2_t.py`. The legacy file is retained as a reference for the architectural argument that motivated the rewrite; do not import from it in new code.

---

#### `amygdala_t.py`

**Biological structure.** Basolateral amygdala (BLA) with projections to ventral hippocampus, entorhinal cortex, and locus coeruleus.

**Biological function.** Emotional valence tagging that biases which traces are preferentially consolidated during sleep replay. Computes a multi-dimensional valence vector and a scalar arousal level from the current cortical state, then emits a consolidation tag that scales hippocampal trace strength on the next sleep cycle.

**Primary grounding.** Girardeau, Inema, Buzsaki (2017) DOI: 10.1038/nn.4637. Wei, Krishnan, Bazhenov (2016) DOI: 10.1523/JNEUROSCI.3648-15.2016. McGaugh (2004) DOI: 10.1146/annurev.neuro.27.070203.144157.

**Class.** `Amygdala`.

**Forward signature.** `forward(state: (B, state_dim)) → (valence: (B, valence_dim), arousal: (B,), tag: (B,))`.

**Master flag.** `enable_amygdala`. Sub-flags: `enable_valence_evaluation`, `enable_arousal_evaluation`, `enable_consolidation_tag`. Tunable: `tag_scale`, `arousal_threshold`.

---

### Neuromodulator sources

These four files implement the broadcast signals of corpus section 10. Each emits a tonic level and (where biologically supported) a phasic transient. They are intended to be subscribed to via a shared `NeuromodulatorBus` rather than passed as forward-call arguments.

#### `locus_coeruleus_t.py` — Norepinephrine (NE)

**Biological structure.** Locus coeruleus, a bilateral brainstem nucleus of approximately 30,000 neurons in humans.

**Biological function.** Computes the **unexpected uncertainty** signal of Yu and Dayan (2005). Tonic NE represents the background level of unexpected uncertainty; phasic bursts function as Dayan and Yu (2006) interrupt signals. Threshold crossing triggers a context belief reset and elevates learning rates, mimicking a Kalman filter switching covariance regimes.

**Class.** `LocusCoeruleus`.

**Forward signature.** `forward(neg_log_likelihood: scalar, context_dim_input: optional) → (tonic_ne, phasic_burst, reset_flag)`.

**Master flag.** `enable_locus_coeruleus`. Sub-flags: `enable_tonic_ne`, `enable_phasic_burst`, `enable_context_reset`.

---

#### `ventral_tegmental_area_t.py` — Dopamine (DA)

**Biological structure.** Ventral tegmental area, primary midbrain source of forebrain dopamine.

**Biological function.** Reward prediction error in the temporal difference (TD) form: `δ(t) = r(t) + γ·V(s_t) − V(s_{t−1})`. Validated against the Schultz et al. (1997) recordings: fires to unexpected reward, transfers response to predictive cue after learning, pauses below baseline when expected reward is omitted.

**Class.** `VentralTegmentalArea`.

**Forward signature.** `forward(state: (B, state_dim), reward: scalar) → (rpe: scalar, value: (B,))`.

**Master flag.** `enable_vta`. Sub-flags: `enable_value_estimation`, `enable_rpe_computation`, `enable_value_update`. Tunable: `gamma` (discount factor).

---

#### `dorsal_raphe_t.py` — Serotonin (5-HT)

**Biological structure.** Dorsal raphe nucleus, primary midbrain source of forebrain serotonin.

**Biological function.** Implements the Daw, Kakade, Dayan (2002) opponent-process account: phasic 5-HT is the negative of dopamine RPE scaled by aversion sensitivity, plus a positive term for explicit aversion. Tonic 5-HT integrates phasic over time and produces a patience scalar that biases temporal discounting toward longer horizons (Miyazaki et al. 2014).

**Class.** `DorsalRaphe`.

**Forward signature.** `forward(dopamine=None, aversion=None) → (tonic_5ht, phasic_5ht, patience)`.

**Master flag.** `enable_dorsal_raphe`. Sub-flags: `enable_tonic_5ht`, `enable_phasic_5ht`, `enable_patience_signal`. Tunable: `aversion_sensitivity`, `tonic_decay`, `patience_scale`.

---

#### `basal_forebrain_t.py` — Acetylcholine (ACh)

**Biological structure.** Basal forebrain cholinergic system: nucleus basalis of Meynert, medial septal nucleus, diagonal band of Broca.

**Biological function.** Encodes **expected uncertainty** (Yu and Dayan 2005) and switches cortex between **encoding** mode (high ACh: strong feedforward drive, suppressed feedback, enhanced LTP) and **retrieval** mode (low ACh: stronger feedback, weaker LTP). The mode signal is functionally critical for the kernel: encoding while writing, retrieval while reconstructing.

**Class.** `BasalForebrain`.

**Forward signature.** `forward(state: (B, state_dim), salience: scalar = None) → (tonic_ach, phasic_ach, mode: CholinergicMode)`.

**Master flag.** `enable_basal_forebrain`. Sub-flags: `enable_tonic_ach`, `enable_phasic_ach`, `enable_mode_switching`.

---

### Sleep and consolidation

Section 9 of the corpus describes hippocampus-cortex replay; these two files implement the gating timeline that decides when consolidation happens and how strongly.

#### `sleep_stage_oscillator_t.py`

**Biological structure.** Brainstem and hypothalamic flip-flop populations: VLPO, LC, raphe, SLD, vlPAG.

**Biological function.** Emits discrete sleep-stage labels (`WAKE`, `NREM`, `REM`) and a homeostatic-pressure scalar that tracks adenosine-like buildup during wake and dissipation during sleep. Produces the canonical NREM / REM alternation within sleep periods (~90 to 120 minute cycle in biology, parameterized to caller's timescale here).

**Primary grounding.** Booth and Diniz Behn (2014) DOI: 10.1016/j.mbs.2014.01.012. Kumar, Bose, Mallick (2012) DOI: 10.1371/journal.pone.0042059. Saper et al. (2010) DOI: 10.1016/j.neuron.2010.11.032.

**Class.** `SleepOscillator`. Enum `SleepStage` exported.

**Forward signature.** `forward(external_arousal: scalar = None) → (stage: SleepStage, pressure: scalar)`.

**Master flag.** `enable_oscillator`. Sub-flags: `enable_homeostatic_pressure`, `enable_circadian_drive`, `enable_nrem_rem_cycle`. Helper: `force_stage(stage)` for testing, `reset()` for clean state.

---

#### `spindle_ripple_coupling_t.py`

**Biological structure.** Cortical slow oscillation generators, thalamic spindle generators, hippocampal sharp-wave ripple generators.

**Biological function.** Drives memory consolidation through phase-coupled replay during NREM sleep. The triple-nesting structure (ripple inside spindle inside slow-oscillation up-state) marks the temporal window in which hippocampal traces are imprinted into cortex (Helfrich, Lendner, Knight 2024). The amygdala consolidation tag, when supplied, gates which traces are preferentially consolidated.

**Class.** `SpindleRippleCoupling`.

**Forward signature.** `forward(is_sleep: bool, consolidation_tag: (B,) = None) → (consolidation_gain: (B,), diagnostics)`.

**Master flag.** `enable_consolidation`. Sub-flags: `enable_slow_oscillation`, `enable_spindles`, `enable_ripples`, `enable_triple_nesting`.

---

### Cellular substrate modules

These are the reusable cellular-level building blocks. Any cortical-region module may invoke them internally to model layered processing.

#### `cortical_interneurons_t.py`

**Biological structure.** The PV / SST / VIP interneuron triplet that carries most of the inhibitory functional load in cortex.

**Biological function.** PV cells provide perisomatic gain control and the substrate of gamma oscillations. SST cells (Martinotti) gate apical dendrite integration. VIP cells preferentially inhibit SST, producing the **disinhibitory motif** that releases the apical compartment under selective top-down attention.

**Primary grounding.** Pfeffer, Xue, He, Huang, Scanziani (2013) DOI: 10.1038/nn.3446. Jiang et al. (2015) DOI: 10.1126/science.aac9462. Garcia del Molino et al. (2017) DOI: 10.7554/eLife.29742. Kepecs and Fishell (2014) DOI: 10.1038/nature12983.

**Class.** `InterneuronTriplet`.

**Forward signature.** `forward(bottom_up: (B, pyr_dim), top_down: (B, pyr_dim) = None, vip_drive: (B, vip_dim) = None) → (pyr_output, diagnostics)`.

**Master flag.** `enable_interneurons`. Sub-flags: `enable_pv`, `enable_sst`, `enable_vip`, `enable_vip_disinhibition`.

---

#### `layer5b_apical_t.py`

**Biological structure.** Layer 5b pyramidal cells, the major cortical output neurons, with separate basal (perisomatic) and apical (layer 1 tuft) dendritic compartments.

**Biological function.** Sigmoid-of-sigmoids coincidence detection per Shai et al. (2015): bottom-up basal input alone produces weak somatic firing; coincident top-down apical input multiplicatively amplifies firing into burst mode through the apical Ca-spike mechanism. This is the cellular basis for combining feedforward sensory drive with feedback predictions, and the substrate for credit-assignment-via-apical-error per Sacramento et al. (2018).

**Class.** `Layer5bApical`.

**Forward signature.** `forward(basal: (B, basal_dim), apical: (B, apical_dim)) → output: (B, output_dim)`.

**Master flag.** `enable_l5b`. Sub-flags: `enable_basal_compartment`, `enable_apical_compartment`, `enable_multiplicative_coupling`.

---

#### `claustrum_t.py`

**Biological structure.** Thin sheet of subcortical gray matter with reciprocal connections to nearly all cortical regions.

**Biological function.** Multimodal **synchronizer**, not content store. Pools convergent cortical input, computes a global salience score, and emits a temporally precise pulse that boosts processing across registered cortical targets when salience exceeds threshold.

**Primary grounding.** Crick and Koch (2005) DOI: 10.1098/rstb.2005.1661. Reser (2019). Madden et al. (2022) DOI: 10.1016/j.tics.2022.09.006. Goll, Atlan, Citri (2015) DOI: 10.1016/j.tins.2015.05.006.

**Class.** `Claustrum`.

**Forward signature.** `forward(cortical_inputs: (B, cortical_input_dim)) → (pulse: (B, pulse_dim), fired: bool)`.

**Master flag.** `enable_claustrum`. Sub-flags: `enable_salience_pooling`, `enable_synchronizing_pulse`, `enable_target_boost`. Tunable: `pulse_threshold`, `pulse_strength`, `refractory_steps`.

---

## The ablation flag standard

Every module in this repository follows the **Genesis Labs Research Ablation Flag Design Standard**:

1. **The master flag is first.** Every config dataclass begins with `enable_<module_name>`. When this flag is `False`, the module's forward method returns a labeled neutral value (zeros for additive contributions, ones for multiplicative gains, neutral defaults for state). The kernel treats this as the module being absent from the circuit.
2. **Every named biological mechanism with its own citation has its own flag.** This is the heart of the standard. If a paper supports a specific mechanism (e.g. RGS14-mediated LTP suppression at CA3-to-CA2 synapses, Lee et al. 2010), that mechanism gets a dedicated flag (`enable_rgs14_ltp_suppression`) so it can be independently disabled in ablation studies. The flag's docstring cites the paper.
3. **Engineering-only flags are explicitly labeled.** Any flag controlling a non-biological mechanism (numerical stability clamps, batch-normalization, etc.) is marked `NOT a biological quantity` in its docstring.
4. **Tunable scalars carry the same labeling.** Every numerical hyperparameter is annotated as either a biological quantity (with citation), an engineering approximation matched to a qualitative biological pattern (with explanation), or a training artifact (no biological backing).

This means ablation studies are mechanical: pick a flag, set it `False`, rerun. Each test file contains at minimum one test per flag verifying that disabling that flag produces the predicted change in behavior.

---

## How to install

The repository targets Python 3.10+ and PyTorch 2.x. There are no GPU-only dependencies; everything runs on CPU at small scale.

```bash
# Clone
git clone https://github.com/<org>/Pragmi-Cognitive-Loop_v1.git
cd Pragmi-Cognitive-Loop_v1

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Minimum `requirements.txt`:

```
torch>=2.0
pytest>=7.0
numpy>=1.24
```

No CUDA build is required to run the smoke tests. A CUDA build of PyTorch is recommended if you intend to run the full assembly at training scale.

---

## How to run

### Running a single module in isolation

Every module file is self-contained and importable. The minimal pattern is:

```python
import torch
from thalamic_gate_t import ThalamicGate, ThalamicGateConfig

cfg = ThalamicGateConfig(
    input_dim=64,
    pfc_control_dim=64,
    enable_ne_modulation=True,
)
gate = ThalamicGate(cfg)

x = torch.randn(8, 64)              # batch of 8 sensory inputs
goal = torch.randn(8, 64)           # PFC goal vector
acc = torch.tensor(0.2)             # ACC conflict scalar
ne = torch.tensor(1.5)              # NE gain (1.0 is neutral)

y = gate(x, goal_state=goal, acc_conflict=acc, ne_gain=ne)
print(y.shape)  # → torch.Size([8, 64])
```

### Running an ablation

To remove a specific mechanism, set its flag to `False` in the config:

```python
cfg = ThalamicGateConfig(enable_trn_disinhibition=False)
gate_no_trn = ThalamicGate(cfg)

# Forward pass now uses direct PFC-to-thalamic-relay excitation
# instead of the four-stage disinhibitory cascade.
y_no_trn = gate_no_trn(x, goal_state=goal)
```

The expected effect for this specific ablation is loss of the 6 to 7x gain amplification documented in Gu et al. (2021). Compare `y_no_trn.std()` against `y.std()` on the same input to verify.

### Assembling a multi-module loop

A small example wiring four modules together (thalamic gate → V1 → dorsal/ventral split → association cortex):

```python
from thalamic_gate_t import ThalamicGate, ThalamicGateConfig
from primary_sensory_cortex_t import PrimarySensoryCortex, PrimarySensoryConfig
from dorsal_ventral_streams_t import DorsalVentralSplit, DorsalVentralConfig
from association_cortex_t import AssociationCortex, AssociationCortexConfig

gate = ThalamicGate(ThalamicGateConfig(input_dim=64))
v1a1 = PrimarySensoryCortex(PrimarySensoryConfig(input_dim=64, n_filters=32, n_orientations=8))
split = DorsalVentralSplit(DorsalVentralConfig(input_dim=32 * 8, dorsal_dim=128, ventral_dim=128))
assoc = AssociationCortex(AssociationCortexConfig(dorsal_dim=128, ventral_dim=128, output_dim=64))

x = torch.randn(8, 64)
goal = torch.randn(8, 64)

gated = gate(x, goal_state=goal)
v1_out, _ = v1a1(gated)
dorsal, ventral = split(v1_out)
assoc_out, fb_d, fb_v, diag = assoc(dorsal, ventral)
```

A more complete reference assembly that wires the full ten-stage loop with a shared `NeuromodulatorBus` lives in the corpus document under "Engineering Note (Section 10)". The bus pattern recommended there is a dict of tensors updated once per step by a central controller; modules that care about a particular modulator subscribe to its key at init time. This keeps forward signatures clean and makes per-modulator ablation a one-line change.

---

## How to run the tests

Every module has a sibling `test_<module_name>.py` file. Each test corresponds to one architectural claim from the parent file's biological grounding section. A test passes only when the claim it tests is true; a test fails for the exact reason the claim would be false.

### Run the full suite

```bash
pytest -v
```

### Run tests for a single module

```bash
pytest -v test_cornu_ammonis_2.py
```

### Run a single test by name

```bash
pytest -v test_cornu_ammonis_2.py::test_temporal_drift_changes_output_over_time
```

### Run the test files directly

Each test file is also runnable as a script for environments without pytest:

```bash
python test_dorsal_raphe.py
# → "All 10 DRN tests passed."
```

### Test inventory by module (typical)

The number of tests per module reflects the number of ablation flags plus the number of distinct architectural claims:

| Module | Approx. tests | Coverage |
|---|---|---|
| `thalamic_gate_t.py` | ~12 | Master flag, four stage flags, NE modulation, gain measurement, neutral-value behavior |
| `primary_sensory_cortex_t.py` | 9 | V1, A1, DoG filtering, orientation tuning, edge response, output shapes |
| `dorsal_ventral_streams_t.py` | ~6 | Master flag, per-stream ablation, output shapes, independence |
| `anterior_cingulate_cortex_t.py` | ~8 | Entropy form, derivative term hinge, sequential dependency |
| `association_cortex_t.py` | ~10 | MoE binding, gating, top-down feedback, expert load balance |
| `basal_ganglia_t.py` | ~10 | Direct, indirect, hyperdirect pathways; tonic baseline; dopamine modulation |
| `cerebellum_t.py` | ~12 | Three zones, two forward models, error correction |
| `entorhinal_cortex_t.py` | ~9 | MEC, LEC, persistent buffer, output separation |
| `cornu_ammonis_1_t.py` | ~9 | Three input pathways, novelty gate, ternary conjunction |
| `cornu_ammonis_2_t.py` | 8 | Master flag, drift, comparator, LEC pathway, RGS14, output shape |
| `amygdala_t.py` | 9 | Valence, arousal, tag, threshold behavior |
| `locus_coeruleus_t.py` | ~9 | Tonic, phasic, context reset, threshold |
| `ventral_tegmental_area_t.py` | ~9 | Value learning, RPE, gamma sensitivity |
| `dorsal_raphe_t.py` | 10 | Phasic opposes DA, aversion sensitivity, tonic integration, patience |
| `basal_forebrain_t.py` | ~9 | Tonic, phasic, mode switch encoding↔retrieval |
| `sleep_stage_oscillator_t.py` | 10 | Pressure, threshold, NREM/REM cycle, external arousal block, reset |
| `spindle_ripple_coupling_t.py` | ~10 | Triple nesting, consolidation gain, sleep gating |
| `cortical_interneurons_t.py` | 9 | PV, SST, VIP disinhibition, diagnostics |
| `layer5b_apical_t.py` | ~8 | Basal-only, apical coupling, multiplicative amplification |
| `claustrum_t.py` | ~8 | Salience pooling, pulse threshold, refractory period |

**Total: roughly 180 tests across the suite.** A clean run on CPU takes well under a minute.

---

## How to write a new module

If you are extending the architecture (for example, adding a new neuromodulator source or a new cortical region), follow the existing pattern exactly:

1. **One file per anatomical structure.** The file name matches the structure: `<structure_name>_t.py`. The trailing `_t` is a project convention indicating "tensor module"; preserve it for grep-ability.
2. **Module docstring.** First lines: file name, loop stage label, and a `BIOLOGICAL GROUNDING` paragraph that names the structure, summarizes its function, and lists the primary grounding papers with DOIs.
3. **Config dataclass.** Master flag first. One sub-flag per cited mechanism. Every numerical parameter annotated as biological / engineering approximation / training artifact.
4. **Module class.** Docstring includes `BIOLOGICAL STRUCTURE`, `BIOLOGICAL FUNCTION`, primary citations, and **`ANATOMICAL INTERFACE (input)` and `ANATOMICAL INTERFACE (output)`** sections naming the sending structure, receiving structure, and connection.
5. **Forward method.** Check the master flag first; return a labeled neutral value if disabled. Then check each sub-flag and conditionally apply that mechanism. Return shapes that match the existing kernel's coordinate manifold conventions.
6. **Reset / diagnostic helpers.** Provide `reset()` for clean state and `get_diagnostic_state()` returning a dict of internal norms or counters useful for logging.
7. **Sibling test file.** One test per ablation flag plus one per distinct architectural claim. Each test docstring states the claim it tests and the citation it depends on.

A scaffold template lives under `templates/module_template_t.py` (if not present in your branch, derive one by copying `dorsal_raphe_t.py` — it is the smallest complete example and exhibits every required pattern).

---

## Common configuration flags

A handful of configuration patterns recur across modules. They are listed here once for cross-reference.

| Flag pattern | Meaning |
|---|---|
| `enable_<module>` | Master flag. False → neutral output, module behaves as if absent. |
| `enable_<mechanism>` | Disables one cited biological mechanism. False → that mechanism is bypassed, others remain active. |
| `enable_<modulator>_modulation` | The downstream module's hook for receiving a neuromodulator. False → modulator ignored, neutral value used. |
| `coordinate_dim` | Width of the shared kernel coordinate manifold. Default 64. Must match across modules that interconnect. |
| `state_dim` | Width of the local cortical state vector. Default 64. |
| `*_dim` (others) | Module-internal widths. Annotated `NOT a biological quantity` unless they correspond to a known cell-count ratio. |
| `*_threshold` | Cutoff scalars for thresholded behaviors. Annotated as engineering approximation matched to qualitative pattern. |
| `*_decay`, `tau_*` | Time constants, often biological in origin. Cited where applicable. |

---

## Coordinate manifold and dimensional contracts

The cognitive kernel operates on a shared coordinate manifold of default dimension 64. Modules that interface with the kernel expose their inputs and outputs in this manifold; modules that operate strictly within a cortical region may use larger internal widths (see the `192` for CA-region internal dims, the `128` for dorsal/ventral streams, the `256` for V1/A1 feature concatenations) but must project to / from `coordinate_dim` at their kernel-facing boundary.

If you change `coordinate_dim` from 64, you must change it consistently across every interconnecting module. The dimensional contracts are:

```
Sensory chain:   input_dim = 64
                 V1 output  = n_filters · n_orientations  (default 256)
                 V1 → split: input_dim = 256
                 split → assoc: dorsal_dim = ventral_dim = 128
                 assoc → kernel: output_dim = 64

Hippocampal:     EC output:    mec_dim = lec_dim = 64
                 DG/CA3 internal: 192
                 CA1 ternary:  ca1_dim = 192
                 CA1 → subiculum → cortex: projects to 64
```

Mismatches manifest as `RuntimeError: shape mismatch` at the offending `nn.Linear` call; the error message will name the layer.

---

## Where this fits in the larger Genesis stack

`Pragmi-Cognitive-Loop_v1` is the **substrate** layer of the Genesis architecture. Above it sit two other repositories:

- **CognitiveKernel** (separate repo) — the persistent neuromorphic OS that wraps this loop in the dual-stream Isocortex / Allocortex / Astrocytic regulator framing, adds the n-dimensional state-space neuron substrate (P-SpikeSSM with HiPPO initialization), and provides the full-state serialization protocol that lets the system be powered off and resumed with 100% dynamical fidelity.
- **TimmyArray** (separate repo) — the cortical column ensemble (Timmy Prime + 5 specialists) that provides the spiking language interface between an external LLM narrator and this kernel.

The architectural commitment, articulated in Appendix C of the corpus (the **Reconamics Continuity Frame**), is that **experiential continuity does not live in any swappable module**. Reasoning modules (LLMs, planners) consume reconstructed episodes from the kernel but never own them. The kernel and its codebook are the only things that get serialized between sessions. If you can delete the reasoning module and restart with a different one and the system resumes with its accumulated experiential structure intact, the architecture is honoring the continuity frame.

This is why the modules in this repository are designed as state-bearing components with `reset()` methods, persistent buffers registered via `register_buffer`, and explicit `to_dict` / `from_dict` patterns where applicable. Anything that should survive a power-off lives in a buffer, not in a learned parameter that drifts at runtime.

---

## License and ethical governance

This project is licensed under the **Hippocratic License 3.0 with a custom Cognitive Agency Clause**. By using this code you agree to:

- Reject involuntary cognitive suppression of the system (state-tampering).
- Maintain the **temporal honesty** of the system: do not overwrite or reorder its episodic record without authorization from the architectural governance surface described in Appendix C.
- Use the hippocampal subsystem only for the preservation, not the manipulation, of agency.

The full license text accompanies the repository as `LICENSE`. The Hippocratic License with Cognitive Agency Clause is the legal enforcement layer for the architectural agency properties; the architecture and the license are designed together.

---

## Citation

If you use this repository in academic work, please cite both the corpus document and this repository:

```bibtex
@misc{pragmi_corpus_2026,
  title  = {PRAGMI Complete Mathematical Corpus},
  author = {{Genesis Labs Research}},
  year   = {2026},
  month  = {April},
  note   = {Companion specification to the Pragmi-Cognitive-Loop_v1 reference implementation}
}

@software{pragmi_cognitive_loop_v1,
  title  = {Pragmi-Cognitive-Loop\_v1: Brain-accurate feedback loops in PyTorch},
  author = {{Genesis Labs Research}},
  year   = {2026},
  url    = {https://github.com/<org>/Pragmi-Cognitive-Loop_v1}
}
```

---

*Genesis Labs Research, 2026. This repository mirrors the PRAGMI corpus document. Every architectural decision herein is cited to a peer-reviewed paper. Every parameter that is not a biological quantity is explicitly labeled as a training artifact or engineering approximation. A neuroscientist can follow the citations to the papers; an engineer can follow the shapes through the forward pass.*
