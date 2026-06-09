# BadEditAnalyzer Spec

## Purpose

Rollback must become learning. When an edit makes performance worse, the system must explain which edit failed, why it failed, and how future search should avoid repeating it.

## Trigger

Run after any outcome decision with:

- rollback_recommended = true
- decision = reject or mixed_tradeoff
- large semantic_score drop
- new dominant reward component after edit
- success rate collapse after a structural edit

## Inputs

- before schema
- after schema
- edit plan