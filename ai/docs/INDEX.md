# AI Docs Index

This directory contains the documentation for the `eye-project/ai` module.
The primary documentation has been consolidated into `AI_HANDOFF.md`.

## Current Documentation Structure

1. **[AI_HANDOFF.md](./AI_HANDOFF.md)**
   - **Primary Source of Truth**.
   - Read this first for the current architecture, configuration, and runtime flow.
   - Includes implementation details for inference and training.

2. **[DEVLOG.md](./DEVLOG.md)**
   - **Historical Reference Only**.
   - Contains the project history, experimental logs, and previous design iterations.
   - Use this to understand *why* certain decisions were made in the past.

3. **[EXPERIMENT_REGISTRY.md](./EXPERIMENT_REGISTRY.md)**
   - Canonical classification index for existing checkpoints, evaluation JSONs, and XAI results.
   - Groups runs by research question and documents the migrated `artifacts/runs/<primary_group>/<run_id>/` layout.

4. **[SPRINT1_Devlog.md](./SPRINT1_Devlog.md)**
   - Sprint 1 retrospective / submission summary for the AI part.

5. **[SPRINT2_Devlog.md](./SPRINT2_Devlog.md)**
   - Sprint 2 retrospective / submission summary for the AI part.

6. **[SPRINT3_Devlog.md](./SPRINT3_Devlog.md)**
   - Sprint 3 retrospective / submission summary for the AI part.

7. **[SPRINT4_Devlog.md](./SPRINT4_Devlog.md)**
   - Sprint 4 retrospective / submission summary for the AI part.
   - Summarizes the v31 active decision, XAI/shortcut diagnostics, Phase 4-E/F results, and Sprint 5 carry-over items.

8. **[SPRINT5_Devlog.md](./SPRINT5_Devlog.md)**
   - Sprint 5 retrospective / running summary for the AI part.
   - Tracks Phase 4-G, TJDR/DDR_SEG integration, MAPLES ROI correction, v8b evidence, v31+v8b late fusion diagnostics, and `v31_v8b_fusion_v2` AI-side deployment packaging.

---

**Notes:**
- When documentation conflicts with the current `eye-project/ai` code or active configs (`configs/base.yaml`), **always trust the code and configs**.
- Historical documents may still contain paths or names from the original `fundus_dr_ai` project.
