# LaTeX Report

## Compilation Status

**Compilation: COMPILED** — Conference version (scene: NeurIPS/ICML). Full chain: pdflatex → bibtex → pdflatex ×2, zero warnings.

## LaTeX Source

- **Main file:** `final_paper/main.tex` — Complete, compilable LaTeX source (~4500 words)
- **Bibliography:** `final_paper/references.bib` — 23 BibTeX entries, all resolved
- **Class:** `article` (11pt, margin=1in, letter paper)
- **Citation style:** `plainnat` (natbib author-year)
- **Sections:** 6 (Introduction, Related Work, Method, Experiments, Discussion, Conclusion) + Appendix
- **Figures:** 0 (placeholder workflow diagram commented out — figure file exists at `figures/stable-eureka_workflow.png`)
- **Tables:** 2 (strict audit run history, relaxed audit run history)

## Source Checks

| Check | Status | Note |
|---|---|---|
| All `\section{}` have `\label{}` | PASS | All 6 sections + appendix labeled |
| All `\cite{}` keys in references.bib | PASS | All 23 citation keys resolved, zero undefined |
| No unescaped special chars | PASS | Checked `&`, `%`, `$`, `_` in text |
| Tables use booktabs | PASS | Both tables use `\toprule`, `\midrule`, `\bottomrule` |
| Abstract present | PASS | ~150 words |
| No `\input`/`\include` macros | PASS | Single self-contained file |

## Compilation Log

| Metric | Value |
|---|---|
| Output pages | 10 |
| Warnings | 0 |
| Undefined citations | 0 |
| Overfull/underfull boxes | 0 |
| PDF size | ~197 KB |
| Compile time | <5 s |

## Notes

- 10 pages with `article` class / 1-inch margins roughly corresponds to ~8 pages in NeurIPS/ICML two-column format. For actual submission, adopt the target conference's official style file.
- The workflow diagram (`figures/stable-eureka_workflow.png`) exists but is commented out — uncomment the `\includegraphics` block and adjust figure placement when ready.
- Author metadata is anonymized for double-blind review.

## Compile Command

```bash
cd final_paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
