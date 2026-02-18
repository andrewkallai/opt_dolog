# Timeline Starting February 16

Below is a concise checklist for each week, focusing on extracting and validating GPU‑specific LLVM‑IR features. The detailed timeline that follows covers the broader RL training workflow, so weeks are not duplicated here.

## PHASE 1: GPU‑SPECIFIC FEATURE EXTRACTION (Feb 16 - May 9, 2026)

### February 2026
- [ ] **Week 1 (Feb 16‑22)** – Set up feature‑extraction pipeline:
  - Install required tools and clone repositories.
  - Build LLVM with TensorFlow‑Lite support.
  - Implement initial extractors for memory‑access patterns and synchronization primitives.
  - Verify end‑to‑end flow: source → LLVM‑IR → feature vector.
- [ ] **Week 2 (Feb 23‑29)** – Expand feature set:
  - Add extractors for parallelism indicators and register‑allocation hints.
  - Create unit tests to validate each new feature against known kernels.
  - Serialize feature vectors for downstream RL training.

### March 2026
- [ ] **Week 3 (Mar 1‑7)** – Refine and benchmark features:
  - Profile extraction runtime and identify bottlenecks.
  - Optimize code paths (caching, parallel processing).
  - Generate a benchmark report of feature extraction speed.
- [ ] **Week 4 (Mar 8‑14)** – Integrate features with RL model:
  - Update RL state representation to include new features.
  - Run short training runs to verify compatibility.
  - Perform hyper‑parameter sweep (learning rate, batch size, clipping epsilon).
  - Log training metrics (loss, reward, policy entropy).
- [ ] **Week 5 (Mar 15‑21)** – Validation and refinement:
  - Evaluate model on a held‑out validation set.
  - Compute size‑reduction, runtime improvement, and SHAP feature importance.
  - Identify under‑performing features and plan refinements.
- [ ] **Week 6 (Mar 22‑28)** – Optimize training speed:
  - Profile data loading and model forward passes (TensorBoard).
  - Enable mixed‑precision training (float16).
  - Parallelize data preprocessing with `tf.data` pipelines.
- [ ] **Week 7 (Mar 29‑Apr 4)** – Ablation studies:
  - Systematically disable individual features and re‑train.
  - Record impact on reward and convergence.
  - Generate SHAP visualizations for feature ranking.

### April 2026
- [ ] **Week 8 (Apr 5‑11)** – Expand dataset:
  - Expand the corpus to 1 000 functions using the refined pipeline.
  - Ensure feature extraction remains robust at scale.
- [ ] **Week 9 (Apr 12‑18)** – Convergence analysis:
  - Adjust hyper‑parameters based on observed trends.
  - Document convergence patterns and update the pipeline.
- [ ] **Week 10 (Apr 19‑25)** – Interim report preparation:
  - Draft interim report covering methodology, early results, and challenges.
  - Incorporate advisor feedback.
  - Update project plan based on insights.
- [ ] **Week 11 (Apr 26‑May 2)** – Refine and regularize features:
  - Apply regularization to mitigate over‑fitting.
  - Validate consistency across held‑out kernels.

### May 2026
- [ ] **Week 12 (May 3‑9)** – Finalize feature pipeline:
  - Complete hyper‑parameter tuning for the full model.
  - Integrate the stable feature set into the overall thesis workflow.

## PHASE 2: TRAINING SPEED OPTIMIZATION (May 10 - May 31, 2026)

### May 2026
- [ ] **Week 13 (May 10‑17)** – Profile and identify bottlenecks in the RL training pipeline:
  - Instrument training loops to capture runtime and memory usage.
  - Analyze data loading, model forward pass, and optimizer steps.
  - Document top three performance bottlenecks.
- [ ] **Week 14 (May 18‑31)** – Implement optimizations and validate speed‑up:
  - Apply mixed‑precision training (float16) where applicable.
  - Parallelize data preprocessing with `tf.data` pipelines.
  - Measure end‑to‑end training time improvement (target 2× speed‑up).

## PHASE 3: INTEGRATION (May 16 - June 15, 2026)

### May 2026
- [ ] **Week 15 (May 16‑31)** – Combine Speed + GPU Features:
  - Merge the optimized training pipeline (Phase 1) with GPU‑specific LLVM‑IR features (Phase 3).
  - Run end‑to‑end training on the 1000‑function corpus.
  - Verify that speed‑up gains are retained after feature integration.

### June 2026
- [ ] **Week 16 (June 1‑7)** – Full Validation:
  - Conduct comprehensive validation on the combined system using the 2000‑function corpus.
  - Measure code‑size reduction, runtime, and hardware‑specific metrics.
  - Compare against baseline LLVM and previous MLGO models.
- [ ] **Week 17 (June 8‑14)** – Report Organization:
  - Draft the integration results section for the prelim report.
  - Create figures/tables summarizing combined speed and GPU feature impact.
  - Review the integration chapter with advisor and incorporate feedback.

## PHASE 4: REFINEMENT (June 16 - July 15, 2026)

### June 2026
- [ ] **Week 18 (June 15‑21)** – Fine‑tune Hyperparameters:
  - Perform targeted hyperparameter sweeps based on validation results.
  - Apply regularization and early‑stopping criteria to avoid overfitting.

### July 2026
- [ ] **Week 19 (June 22‑28)** – Additional Ablation Studies:
  - Explore alternative feature subsets and model architectures.
  - Document findings and update the technical note.
- [ ] **Week 20 (July 1‑7)** – Prepare Final Results:
  - Consolidate all performance tables and visualizations.
  - Write the final results narrative for the prelim submission.

## PHASE 5: REPORT WRITING (July 16 - August 31, 2026)

### July 2026
- [ ] **Week 21 (July 8‑14)** – Draft Methods Chapter:
  - Detail RL formulation, state/action design, and training pipeline.
  - Include descriptions of LLVM‑IR feature extraction and integration steps.

### August 2026
- [ ] **Week 22 (July 15‑21)** – Draft Results & Discussion Chapters:
  - Present quantitative results, ablation analyses, and comparisons.
  - Discuss limitations and future work.
- [ ] **Week 23 (July 22‑28)** – Review & Polish:
  - Conduct internal reviews with advisor and peers.
  - Incorporate feedback and finalize the full prelim manuscript.

## PHASE 6: ORAL EXAM PREPARATION (September 1 - October 15, 2026)

### September 2026
- [ ] **Week 24 (Sept 1‑7)** – Create Presentation Slides:
  - Summarize motivation, methodology, key results, and contributions.

### October 2026
- [ ] **Week 25 (Sept 8‑14)** – Mock Presentations:
  - Practice with peers, record sessions, and refine delivery.
- [ ] **Week 26 (Oct 1‑7)** – Final Rehearsals & Q&A Prep:
  - Anticipate committee questions and prepare concise answers.

## PHASE 7: ORAL EXAMINATION (October 16 - October 31, 2026)

### October 2026
- [ ] **Week 27 (Oct 8‑14)** – Defend the prelim submission before the committee.
- [ ] **Week 28 (Oct 15‑21)** – Incorporate any final feedback into the dissertation plan.
