# Rewrite Matrix

Maps original logic units (U01-U24 from original_logic_map.md) to final manuscript units. Each row records the transformation type and justification.

| Original Unit | Final Location | Transformation | Justification |
|---|---|---|---|
| U01 (title candidates) | Title (F2) | Structural — select and refine | Writing plan's 4 candidates are descriptive but not argumentative. Selected title must carry the "search, not generation" claim. |
| U02 (core positioning) | Introduction P4-P5 (F7-F8) | Rhetorical — reframe as gap+proposal | Positioning statements from writing plan are correct but need to be split: first state the gap (what's missing), then the proposal (what EG-RSA does). |
| U03 (contribution 1: generation→search) | Introduction P5 (F8), Method 3.1 (F20-F22) | Structural — split across introduction and method | The contribution claim stays in Introduction. The mechanism detail moves to Method. |
| U04 (contribution 2: attribution) | Introduction P5 (F8), Method 3.3 (F24-F26) | Structural — split + add evidence | Attribution claim stays in Introduction. Method adds computation detail and evidence from E3. |
| U05 (contribution 3: memory) | Introduction P5 (F8), Method 3.4 (F27) | Structural — split + add evidence | Memory claim stays in Introduction. Method adds lesson format detail and evidence from E4. |
| U06 (contribution 4: audit) | Introduction P5 (F8), Method 3.6 (F29-F31) | Structural — split + add evidence + add limitation | Audit claim stays in Introduction. Method adds audit rules, outcomes, and the hand-design limitation. |
| U07 (section structure) | Whole-work framework (F1) | Structural — adopted as blueprint | The 7-section structure from the writing plan is sound and aligns with confirmed motivation. Adopted with minor reordering. |
| U08 (Introduction logic, 5 moves) | Introduction P1-P6 (F4-F9) | Rewrite — expand from outline to prose | The 5-move logic is correct but needs: (a) a concrete problem opening, (b) citation anchors, (c) explicit contribution list, (d) a roadmap sentence. |
| U09 (Related Work 2.1, reward shaping) | Related Work 2.1 P1-P2 (F10-F11) | Light rewrite — tighten prose, sharpen contrast | Prose is solid. Contrast paragraph needs mechanism-level specificity ("Unlike IRL... Unlike potential-based shaping..."). |
| U10 (Related Work 2.2, LLM reward) | Related Work 2.2 P1-P3 (F12-F14) | Light rewrite — add paradigm framing, sharpen contrast | Group by paradigm (code generation) rather than chronologically. Add mechanism-level contrast (schema vs. code, audit vs. no audit, memory vs. no memory). |
| U11 (Related Work 2.3, memory agents) | Related Work 2.3 P1-P2 (F14-F15) | Light rewrite — add survey context, sharpen object contrast | Add memory survey citation (C028). Sharpen the contrast: EG-RSA stores reward-edit outcome lessons, not verbal reflections or skill code. |
| U12 (Related Work 2.4, reward hacking) | Related Work 2.4 P1-P3 (F16-F18) | Rewrite — expand from 2 to 3 paragraphs | Current draft is thin (2 citations). Add formal taxonomies (C018), Goodhart theory (C049), recent mitigations (C019, C020, C021), and connect to EG-RSA's audit. |
| U13 (Related Work 2.5, positioning) | Related Work 2.5 (F19) | Light rewrite — sharpen final sentence | Final sentence must use confirmed motivation language: "search," "schema," "memory," "audit." |
| U14 (method positioning) | Method 3.1 P1-P2 (F20-F21) | Rewrite — convert from manual tone to journal method | Design doc S1 is written as a manual. Rewrite as a method overview: search loop, components, contrast with Stable-Eureka. |
| U15 (Stable-Eureka difference) | Method 3.1 P2 (F21) | Move + compress — merge into overview contrast | The 5-item difference list is effective but too long for a journal method. Compress to side-by-side text diagrams + 1 paragraph. |
| U16 (task metric specification) | Method 3.3 P1 (F24) — partial | Rewrite — reframe as attribution motivation | Task metrics are the "what we measure." Rewrite as the motivation for attribution, not as a standalone subsection. |
| U17 (componentized reward schema) | Method 3.2 (F23) | Rewrite — method prose with design justification | The JSON example is good. Add: why schema over code (versioning, attribution, operator constraint). Add semantic role taxonomy. |
| U18 (reward compiler) | Method 3.2 or 3.5 — merged | Merge — too implementation-level for standalone subsection | The compiler is an implementation detail. Merge into schema (3.2) or editing (3.5) section. |
| U19 (policy training) | Method 3.1 or Experiments 4.1 — merged | Merge — "fixed sufficient budget" concept moves to overview | Training details are setup, not method. "Fixed sufficient budget" concept moves to overview (3.1) or setup (4.1). |
| U20 (trajectory recorder) | Method 3.3 P2 (F25) | Rewrite — connect to attribution computation | The JSON format is important for understanding attribution. Describe it in the attribution computation paragraph. |
| U21 (EXP-01 10-iteration run) | Experiments 4.2-4.4 (F33-F40) | Rewrite — structure as three case studies, not one aggregate | The 10-iteration data tells three stories (effective edit, regression, deadlock). Structure as case studies, not one table. |
| U22 (EXP-02 relaxed audit) | Experiments 4.5 (F41-F42) | Rewrite — present as audit policy comparison | The relaxed audit data is the counterfactual that proves the deadlock was audit-induced. Present as comparison, not as "the fix." |
| U23 (ablation configs) | Omit or move to Appendix | Delete from main text — no results exist | Configs exist but have no results. Mention as "planned" in Discussion future work. Do not include in main experiments. |
| U24 (experiment positioning) | Experiments section framing (F33) | Keep — the "mechanism verification + case study" framing is correct | Use this framing explicitly in the experiment section opening. |

## Transformation Summary

| Type | Count |
|---|---|
| Rewrite (substantive new prose) | 10 |
| Light rewrite (tighten + sharpen) | 6 |
| Structural (split/move/reorder) | 5 |
| Merge (compress into parent) | 3 |
| Keep (logic preserved) | 2 |
| Delete (no evidence) | 1 |
