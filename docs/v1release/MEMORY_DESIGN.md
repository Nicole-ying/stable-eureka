# v1release Memory Design

## Purpose

Memory is the core mechanism that turns EG-RSA from a log-producing reward editor into a self-improving agent system.

The v0 memory mostly stores what happened. The v1 memory must store what was learned, why it matters, when it applies again, and what future actions it should encourage or forbid.

## Memory levels

### 1. Episodic Memory

Episodic memory records factual iteration-level