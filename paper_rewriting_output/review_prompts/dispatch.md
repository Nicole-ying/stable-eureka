# Review Dispatch Instructions

Launch **three sub-agents in parallel** using the Agent tool. Each agent reads only its own prompt file and the manuscript — they must NOT see each other's outputs or the other prompts.

### Agent 1: Methods Reviewer
Read `review_prompts/methods_reviewer.md` and produce `review_prompts/methods_review_output.md`

### Agent 2: Contribution Reviewer
Read `review_prompts/contribution_reviewer.md` and produce `review_prompts/contribution_review_output.md`

### Agent 3: Clarity Reviewer
Read `review_prompts/clarity_reviewer.md` and produce `review_prompts/clarity_review_output.md`

### After all three complete:
Run `python scripts/structured_review.py paper_rewriting_output --validate review_prompts` to check independence. Then produce the Editor Synthesis.
