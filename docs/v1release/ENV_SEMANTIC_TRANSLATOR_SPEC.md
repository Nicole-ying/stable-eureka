# Environment Semantic Translator Spec

## Goal
Reduce dependence on hand-written environment-specific files. The translator turns an environment description into an executable semantic interface for EG-RSA v1.

Current v0 depends on manually designed files:
- task_description.txt
- diagnostic_spec.yml
- initial_schema.json

v1 target:
- LLM proposes these files from env metadata and few-shot examples.
- Validators reject non-executable metrics/events.
- Smoke tests verify signal quality before long training.

## Inputs
- env_id or env.py path
- observation space and action space
- short natural-language task description
- optional masked/original reward description for reporting only
- few-shot semantic packages from solved tasks, e.g. LunarLander

## Outputs
- generated_task_description.txt
- generated_diagnostic_spec.yml
- generated_initial_schema.json
