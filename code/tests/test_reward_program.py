from llm_reward_evolver.reward import RewardProgram


def test_reward_program_runs_and_clips():
    code = """
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    return 999.0
"""
    program = RewardProgram(code, reward_clip=10.0)
    assert program([0], [0], [0], 1.0, {}, 0.0) == 10.0


def test_reward_program_rejects_imports():
    code = """
import os
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    return 0.0
"""
    try:
        RewardProgram(code)
    except ValueError as exc:
        assert "Forbidden" in str(exc)
    else:
        raise AssertionError("expected validation failure")

