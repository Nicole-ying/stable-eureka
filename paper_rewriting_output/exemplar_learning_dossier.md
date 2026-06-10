# Exemplar Learning Dossier

## 1. Exemplar Inventory

| # | Title | Venue | Year | Why Selected |
|---|---|---|---|---|
| 1 | EUREKA: Human-Level Reward Design via Coding Large Language Models | NeurIPS (spotlight) | 2023 | Closest SOTA work; demonstrates the LLM-as-reward-generator paradigm that EG-RSA extends and contrasts against; teaches how to structure a method paper that introduces a new LLM+RL paradigm |
| 2 | Voyager: An Open-Ended Embodied Agent with Large Language Models | NeurIPS (oral) | 2023 | Teaches how to present a complex multi-component LLM agent with skill memory as a coherent system; demonstrates effective iterative-feedback + memory narrative arc |
| 3 | Reflexion: Language Agents with Verbal Reinforcement Learning | NeurIPS | 2023 | Teaches how to frame memory/reflection as a core mechanism rather than an add-on; minimal but effective experiment design for mechanism verification |

## 2. Structural Patterns

**Pattern 1: The "paradigm contrast" opening.** All three exemplars open by characterizing the existing paradigm (code generation, skill execution, single-pass reasoning) and then introduce a new mechanism that changes the paradigm (evolutionary optimization, skill library + curriculum, verbal reflection). The contrast is structural, not rhetorical: it operates at the level of the search/learning loop, not the claim. EG-RSA can adopt this by positioning its opening around "from reward generation to reward search."

**Pattern 2: Component-by-component method exposition with a closing integration paragraph.** Rather than a monolithic Method section, each exemplar decomposes the system into 3–5 named components, explains each in a self-contained subsection, and then closes with a paragraph showing how they interact in a single iteration. This prevents the Method from reading as a code walkthrough.

**Pattern 3: The "failure is insight" experiment arc.** EUREKA dedicates space to cases where the method fails or produces surprising behavior, treating these as evidence of understanding rather than weakness. Voyager analyzes curriculum gaps. This pattern is directly applicable to EG-RSA's audit-deadlock case study.

## 3. Rhetorical Patterns

**Opening technique:** Each exemplar's introduction begins with a concrete problem statement (reward design is hard; open-ended exploration is unsolved; agents make mistakes), NOT a broad "AI is important" statement. The first paragraph names the specific bottleneck and why prior solutions are structurally limited.

**Closing technique:** Discussions return to the opening motivation and explicitly state what the method does NOT solve. Limitations are not buried — they are given a dedicated paragraph or subsection that connects forward to future work. Voyager's limitation paragraph is particularly effective: it lists 3–4 specific constraints (no visual input, curriculum gaps, environment assumptions) without weakening the contribution.

## 4. Language Patterns

- **Register:** Formal but not pompous. Use "we" (first-person plural) consistently. Avoid passive-voice hedging ("it can be observed that") in favor of direct statements ("we observe," "EG-RSA constrains").
- **Claim calibration:** Exemplars use explicit hedging only where evidence is partial ("suggests," "motivates," "is consistent with") and unhedged language only where the mechanism is demonstrated ("EG-RSA stores," "the audit blocks"). Do not hedge mechanisms that are directly implemented.
- **Contribution count:** Each exemplar lists 3–4 contributions in the introduction, not 6+. Contributions are mechanisms or framings, not "we ran experiments on X environments."
- **Figure rhetoric:** Every figure is referenced before it appears. Captions are self-contained one-sentence summaries, not descriptions ("Overview of the EG-RSA search loop" not "This figure shows...").
