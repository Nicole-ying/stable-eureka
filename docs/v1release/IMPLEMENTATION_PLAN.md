# v1release Implementation Plan

## Goal

Upgrade EG-RSA from a constrained LLM reward-schema editor into a memory-driven, tool-using, agentic reward-search system.

The v0 baseline is useful but limited: it edits a human-designed reward schema through a fixed loop. v1 should make the LLM choose actions, use diagnostic tools, write reusable lessons, and expand the reward search space when evidence supports it.

## Milestone 1: Agent action layer

Add `eg_rsa/agent/action_controller.py`.

Responsibilities:

- read