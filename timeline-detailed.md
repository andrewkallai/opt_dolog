# Updated Timeline Starting February 16

## Goal: Prepare for Prelim Exam by October 1

### Weekly Milestones
- **Week 1 (Feb 16-22)**: Set up RL training pipeline; collect LLVM IR from GPU kernels.
  - Install required system packages (cmake, ninja, clang, python3‑venv).
  - Clone `ml-compiler-opt` and LLVM repositories.
  - Build LLVM with TensorFlow‑Lite support for on‑the‑fly inference.
  - Configure the RL environment (`compiler_opt/rl/`) and generate a small pilot corpus of GPU kernels.
  - Verify end‑to‑end data flow: source → LLVM‑IR → feature extraction → RL state.
- **Week 2 (Feb 23-29)**: Implement baseline RL model; define reward based on performance metrics.
  - Design state representation: static LLVM‑IR features (memory access patterns, synchronization primitives, parallelism indicators).
  - Define action space: binary decisions for inlining, register allocation, etc.
  - Choose reward function: weighted combination of code‑size reduction, estimated runtime, and hardware‑specific metrics.
  - Implement a simple policy network (e.g., two‑layer MLP) using TensorFlow.
  - Train with Proximal Policy Optimization (PPO) on the pilot corpus.
- **Week 3 (Mar 1-7)**: Extract additional LLVM IR features (memory patterns, synchronization primitives) from GPU code.
  - Extend `compiler_opt/rl/registry.py` to include new feature extractors.
  - Validate feature correctness on a subset of kernels.
  - Update data pipeline to serialize the expanded feature vectors.
- **Week 4 (Mar 8-14)**: Integrate new features into RL model; run initial training runs.
  - Retrain the policy network with the enriched feature set.
  - Perform hyper‑parameter sweep for learning rate, batch size, and clipping epsilon.
  - Log training metrics (loss, reward, policy entropy) for analysis.
- **Week 5 (Mar 15-21)**: Evaluate model on validation set; refine feature set.
  - Run the trained policy on a held‑out validation corpus.
  - Compute quantitative metrics: size reduction %, runtime improvement, and feature importance via SHAP.
  - Identify under‑performing features and plan refinements.
- **Week 6 (Mar 22-28)**: Optimize training speed; profile bottlenecks.
  - Profile data loading and model forward passes using TensorBoard.
  - Enable mixed‑precision training (float16) to accelerate GPU utilization.
  - Parallelize data preprocessing with `tf.data` pipelines.
- **Week 7 (Mar 29-Apr 4)**: Conduct ablation studies on feature importance.
  - Systematically remove individual features and re‑train to measure impact.
  - Generate SHAP visualizations to rank features.
  - Document findings for inclusion in the interim report.
- **Week 8 (Apr 5-11)**: Expand dataset to 1000 functions; continue RL training.
- **Week 9 (Apr 12-18)**: Analyze convergence; adjust hyperparameters.
- **Week 10 (Apr 19-25)**: Prepare interim report on RL model progress.
  - Draft the interim report covering methodology, early results, and challenges.
  - Include preliminary performance metrics of the RL model.
  - Review the report with advisor and incorporate feedback.
  - Update project plan based on insights.
- **Week 11 (Apr 26-May 2)**: Incorporate feedback; improve model robustness.
  - Address reviewer comments from interim report.
  - Implement regularization techniques to reduce overfitting.
  - Conduct additional validation experiments on held‑out GPU kernels.
  - Refine feature extraction pipeline for consistency.
- **Week 12 (May 3-9)**: Finalize RL model; begin integration with overall thesis pipeline.
  - Complete hyperparameter tuning for the RL agent.
  - Validate model performance on a held‑out test set of GPU kernels.
  - Integrate the trained model into the broader thesis workflow (data preprocessing, evaluation scripts).
  - Prepare a technical summary of model architecture and training results for the prelim submission.
- **Weeks 13-30 (May 10-Oct 1)**: Ongoing RL model refinement and thesis preparation.
  - **May (Weeks 13-16)**: Scale training to full 2000‑function corpus; perform hyperparameter sweep.
    * Expand corpus to 2000 functions covering diverse GPU kernels.
    * Run grid search over learning rates (1e-4 to 1e-2) and batch sizes (32‑256).
    * Log training curves with TensorBoard; select best checkpoint.
    * Draft a short progress report summarizing scaling results.
  - **June (Weeks 17-20)**: Conduct extensive ablation studies on LLVM‑IR feature importance.
    * Systematically mask each feature group and retrain to measure impact.
    * Generate SHAP summary plots for all features.
    * Write a technical note on the most influential features for inclusion in the prelim.
  - **July (Weeks 21-24)**: Integrate RL model results into thesis chapters; draft methods section.
    * Write Chapter 4 (Methods) describing RL formulation, state/action design, and training pipeline.
    * Incorporate quantitative results (size reduction, runtime) into Chapter 5 (Results).
    * Prepare figures and tables for the thesis.
  - **August (Weeks 25-28)**: Validate model on unseen GPU benchmarks; finalize performance tables.
    * Collect a held‑out benchmark suite (e.g., CUDA kernels from real applications).
    * Run the trained policy and compare against baseline compilers.
    * Summarize findings in a performance table for the prelim.
  - **September (Weeks 29-30)**: Prepare and rehearse prelim presentation; incorporate advisor feedback into final submission.
    * Create slide deck covering motivation, methodology, results, and future work.
    * Conduct mock presentations with peers and incorporate feedback.
    * Polish the written prelim document and submit by October 1.
  - **October (Week 31)**: Submit prelim and begin oral exam preparation.
    * Submit the finalized prelim to the committee.
    * Review committee feedback and outline next steps for the dissertation.
    * Start planning for the oral exam (schedule, practice sessions).

**Key Focus**: Train a reinforcement learning model on LLVM IR features, ensuring the feature set captures GPU-specific characteristics such as memory access patterns, synchronization, and parallelism.

# Detailed Timeline for PhD Prelim Exam

You are starting on February 10, 2026. You have already completed Phase 1.

## PHASE 2: TRAINING SPEED OPTIMIZATION (Feb 10 - Mar 31, 2026)

### WEEK 1: PROFILING AND BOTTLENECK IDENTIFICATION

**Feb 10 (Tuesday) - Task Setup**
- [ ] Review existing baseline metrics from Phase 1
- [ ] Identify specific training scripts to profile
- [ ] Set up logging infrastructure

**Feb 13-14 (Tuesday-Wednesday) - Instrumentation**
- [ ] Add profiling to training loop
- [ ] Profile each major component:
  * Feature extraction
  * Data preparation
  * Model training steps
  * Evaluation
- [ ] Profile memory usage

**Feb 15-16 (Thursday-Friday) - Analysis**
- [ ] Create profiling report
- [ ] Identify top 3 bottlenecks
- [ ] Formulate optimization hypotheses

Deliverable: Profiling report with bottleneck analysis

### WEEK 2: FEATURE EXTRACTION OPTIMIZATIONS

**Feb 17-18 (Tuesday-Wednesday) - Investigation 1: Caching**
- [ ] Map feature extraction frequency
- [ ] Implement LRU cache
- [ ] Measure speedup

**Feb 19-20 (Thursday-Friday) - Investigation 2: Parallelization**
- [ ] Profile if extraction is CPU-bound
- [ ] Add multiprocessing
- [ ] Test speedup vs overhead

**Feb 21-23 (Saturday-Monday)**
- [ ] Implement second optimization
- [ ] Validate improvements

Deliverable: 2 optimized implementations

### WEEK 3: TRAINING LOOP OPTIMIZATION

**Feb 24-25 (Tuesday-Wednesday) - Investigation 3: Data Loading**
- [ ] Profile data loading
- [ ] Implement tf.data pipeline
- [ ] Add prefetch and parallel_map

**Feb 26-27 (Thursday-Friday) - Investigation 4: Model Optimization**
- [ ] Profile model training steps
- [ ] Test mixed precision training
- [ ] Implement tf.function conversions

**March 1-2 (Tuesday-Wednesday) - Investigation 5: Final Optimizations**
- [ ] Test other optimizations (learning rate, batching, etc.)
- [ ] Select best 2-3 optimizations

Deliverable: 3 optimizations implemented

### WEEK 4: INTEGRATION AND VALIDATION

**March 3-4 (Thursday-Friday) - Combined Testing**
- [ ] Integrate all optimizations
- [ ] Run comparisons
- [ ] Measure final speedup

**March 5-7 (Saturday-Monday) - Quality Assurance**
- [ ] Verify model quality unchanged
- [ ] Run evaluation suite
- [ ] Document for report

Deliverable: Phase 2 complete with 2x speedup

## PHASE 3: GPU-SPECIFIC FEATURES (April 1 - May 15, 2026)

### WEEK 1: RESEARCH AND DESIGN

**April 1-2 (Tuesday-Wednesday) - Collect GPU IR**
- [ ] Compile GPU tests
- [ ] Extract to LLVM IR
- [ ] Build 500-function corpus

**April 3-4 (Thursday-Friday) - Literature Review**
- [ ] Read about GPU optimization
- [ ] Design feature specifications
- [ ] List 10+ features to implement

Deliverable: Feature design document

### WEEK 2: FEATURE IMPLEMENTATION

**April 8-9 (Tuesday-Wednesday) - Features 1-4**
- [ ] Implement memory pattern features
- [ ] Test on known examples

**April 10-11 (Thursday-Friday) - Features 5-7**
- [ ] Implement synchronization features
- [ ] Test on synchronized vs. unsynchronized code

Deliverable: 7 features implemented

### WEEK 3: INTEGRATION

**April 15-16 (Tuesday-Wednesday) - Integration**
- [ ] Add features to training pipeline
- [ ] Test with small corpus
- [ ] Analyze feature distributions

**April 17-18 (Thursday-Friday) - Initial Training**
- [ ] Train with and without features
- [ ] Compare convergence

Deliverable: Features integrated, initial results

### WEEKS 4-6: EVALUATION AND REFINEMENT

**April 22-23, April 29-30, May 6-7** - Full Training
- [ ] 1000-function corpus
- [ ] Ablation studies
- [ ] Final features and integration

**May 8-15** - Report Section Writing
- [ ] Phase 3 report section
- [ ] All results finalized

Deliverable: Phase 3 complete

## PHASE 4: INTEGRATION (May 16 - June 15, 2026)

### WEEK 1-2: Combine Speed + GPU Features
- Merge the optimized training pipeline (Phase 2) with GPU‑specific LLVM‑IR features (Phase 3).
- Run end‑to‑end training on the 1000‑function corpus.
- Verify that speed‑up gains are retained after feature integration.

### WEEK 3-4: Full Validation
- Conduct comprehensive validation on the combined system using the 2000‑function corpus.
- Measure code‑size reduction, runtime, and hardware‑specific metrics.
- Compare against baseline LLVM and previous MLGO models.

### WEEK 5-6: Report Organization
- Draft the integration results section for the prelim report.
- Create figures/tables summarizing combined speed and GPU feature impact.
- Review the integration chapter with advisor and incorporate feedback.

## PHASE 5: REFINEMENT (June 16 - July 15, 2026)

### WEEK 1-2: Fine‑tune Hyperparameters
- Perform targeted hyperparameter sweeps based on validation results.
- Apply regularization and early‑stopping criteria to avoid overfitting.

### WEEK 3-4: Additional Ablation Studies
- Explore alternative feature subsets and model architectures.
- Document findings and update the technical note.

### WEEK 5-6: Prepare Final Results
- Consolidate all performance tables and visualizations.
- Write the final results narrative for the prelim submission.

## PHASE 6: REPORT WRITING (July 16 - August 31, 2026)

### WEEK 1-3: Draft Methods Chapter
- Detail RL formulation, state/action design, and training pipeline.
- Include descriptions of LLVM‑IR feature extraction and integration steps.

### WEEK 4-6: Draft Results & Discussion Chapters
- Present quantitative results, ablation analyses, and comparisons.
- Discuss limitations and future work.

### WEEK 7-8: Review & Polish
- Conduct internal reviews with advisor and peers.
- Incorporate feedback and finalize the full prelim manuscript.

## PHASE 7: ORAL EXAM PREPARATION (September 1 - October 15, 2026)

### WEEK 1-2: Create Presentation Slides
- Summarize motivation, methodology, key results, and contributions.

### WEEK 3-4: Mock Presentations
- Practice with peers, record sessions, and refine delivery.

### WEEK 5-6: Final Rehearsals & Q&A Prep
- Anticipate committee questions and prepare concise answers.

## PHASE 8: ORAL EXAMINATION (October 16-31, 2026)

- Defend the prelim submission before the committee.
- Incorporate any final feedback into the dissertation plan.