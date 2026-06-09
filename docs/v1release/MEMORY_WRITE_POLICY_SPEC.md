# Memory Write Policy Spec

## Purpose

Memory is not a log dump. It must turn experience into reusable search knowledge.

## Three memory levels

1. EpisodicMemory
   - Stores what happened in one iteration.
   - Includes schema hash, edit plan, diagnostics, outcome, rollback flag.

2. LessonMemory
   - Stores distilled good or bad edit patterns.
   - Must explain why an edit helped or failed.

3. PolicyMemory
   - Stores high-level search policies.
   - Example: if success is high