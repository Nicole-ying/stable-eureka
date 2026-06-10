# Research Dossier

## 1. Venue Requirements

Target venue: **NeurIPS / ICML** — top-tier ML conferences with hard page limits, double-blind review, and deadline-driven cycles.

| Requirement | Detail |
|---|---|
| Format | LaTeX (conference style file, e.g., `neurips_2025.sty`); double-blind; no author info |
| Page limit | 8 pages main + unlimited references/appendix (NeurIPS 2025); 9 pages (ICML 2025) |
| Abstract | Single paragraph, ~150-200 words, placed at top of first page |
| Structure | Compact IMRaD; Related Work often compressed to 0.5-1 page; Method must justify design choices efficiently |
| Figures/Tables | Embedded; vector preferred; must be readable in grayscale; tight captions |
| Code/Data | Code release strongly encouraged; artifact evaluation optional |
| AI-use policy | Disclose LLM usage; many conferences now require an AI-use statement |
| Supplementary | Unlimited appendix for proofs, ablations, hyperparameters, extended results |

## 2. Review Criteria

Conference reviewers evaluate under time pressure (~30-45 min per paper). They look for:

1. **Novelty signal in first 2 pages** — If the reviewer can't state the contribution after reading the Introduction, the paper is at risk. The gap must be crisp, and the mechanism contrast must be structural, not rhetorical.
2. **Technical soundness under page constraints** — Every claim in the main paper must be backed by evidence visible in the main paper. Mechanism verification is acceptable but must show clear cause-effect relationships, not just correlation.
3. **Clarity at speed** — Section openings, figure captions, and the first sentence of each paragraph must carry the argument. A reviewer skimming at 200 wpm must still follow the logic.
4. **Honest positioning** — Overclaiming is penalized. A paper that says "we show X is possible, with this tension Y" scores higher than "we solve X."
5. **Reproducibility** — Environment, hyperparameters, seeds, and code availability matter. Conference reviewers increasingly check if the setup is reproducible.

## 3. Accepted Paper Patterns

**Pattern 1: The "first-page pitch."** Top conference papers place their entire argument on page 1: opening sentence names the bottleneck, by paragraph 2 the gap is clear, by paragraph 4 the method is introduced, and the contribution list appears before the page break. There is no throat-clearing.

**Pattern 2: One clean experiment story.** Under page limits, successful papers don't run 4+ disjoint experiments. They run one primary experiment, structure it as a narrative (before → intervention → after → counterfactual), and put ablations in the appendix. EG-RSA's 3 case studies (effective edit, deadlock, relaxed) form exactly this narrative arc.

**Pattern 3: Compressed Related Work as "positioning," not "survey."** Conference Related Work sections are 3-4 paragraphs, not 5 subsections. Each paragraph covers one research line, cites 3-5 key papers, and ends with an explicit contrast to the proposed method. The section closes with a positioning paragraph.

## 4. Constraints for This Paper

1. **8-page target.** The current journal manuscript is ~7000 words / 12 pages. Must compress to ~4500 words / 8 pages. Primary compression targets: merge Related Work subsections, tighten Method, move ablation discussion to appendix.
2. **First-page pitch required.** The Introduction must deliver the full argument arc before the page break. Move the contribution list higher.
3. **No fabricated benchmarks.** Maintain mechanism-verification framing. Case studies are the evidence.
4. **No oracle leakage claims.** Preserve the strict separation of oracle evaluation from search decisions.
5. **Audit deadlock is the headline finding.** For a conference paper, lead with the most interesting result. The deadlock → relaxed-audit story is more memorable than the effective-edit case study.
6. **Appendix for ablations and extended traces.** Move planned ablations, environment details, and extended iteration traces to the supplementary appendix.
