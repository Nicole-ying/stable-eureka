import copy
import time
from pathlib import Path

import numpy as np
import yaml
from datetime import datetime
import os
import shutil
from typing import Dict

from stable_eureka.logger import get_logger, EmptyLogger
from stable_eureka.ollama_generator import OllamaGenerator
from stable_eureka.openai_generator import OpenAIGenerator
from stable_eureka.utils import (read_from_file,
                                 get_code_from_response, append_and_save_to_txt,
                                 indent_code, save_to_txt, save_to_json,
                                 make_env, reflection_component_to_str, read_from_json,
                                 summarize_final_eval)
from stable_eureka.compact_prompt import build_compact_reflection_prompt, reward_code_hash
from stable_eureka.rl_trainer import RLTrainer
from stable_eureka.rl_evaluator import RLEvaluator
from gymnasium.envs.registration import register

import multiprocessing
import torch


class StableEureka:
    def __init__(self, config_path: str):
        if not Path(config_path).exists():
            raise ValueError(f'Config file {config_path} not found')

        self._config = yaml.safe_load(open(config_path, 'r'))

        self._root_path = Path(os.getcwd())
        self._experiment_path = self._root_path / self._config['experiment']['parent'] / self._config['experiment'][
            'name']

        self._experiment_datetime = None
        if self._config['experiment']['use_datetime']:
            self._experiment_datetime = datetime.utcnow().strftime('%Y-%m-%d')
            self._experiment_path /= self._experiment_datetime

        self._experiment_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(config_path, self._experiment_path / 'config.yaml')

        self._regex = [r'```python(.*?)```']

        self._prompts = {
            'initial_system': read_from_file(self._root_path / 'stable_eureka' / 'prompts' / 'initial_system_prompt.txt'),
            'coding_instructions': read_from_file(self._root_path / 'stable_eureka' / 'prompts' / 'coding_instructions_prompt.txt'),
            'task_description': read_from_file(self._root_path / 'envs' / self._config['environment']['name'] / 'task_description.txt'),
            'env_code': read_from_file(self._root_path / 'envs' / self._config['environment']['name'] / 'step.py'),
            'reward_reflection_init': read_from_file(self._root_path / 'stable_eureka' / 'prompts' / 'reward_reflection_init_prompt.txt'),
            'reward_reflection_end': read_from_file(self._root_path / 'stable_eureka' / 'prompts' / 'reward_reflection_end_prompt.txt'),
            'reward_reflection': '',
        }

        initial_reward_prompt_path = self._root_path / 'envs' / self._config['environment']['name'] / 'initial_reward_prompt.txt'
        if self._config['eureka']['use_initial_reward_prompt'] and initial_reward_prompt_path.exists():
            self._prompts['initial_reward'] = read_from_file(initial_reward_prompt_path)

        self._best_reward = ('', -float('inf'), None, None)  # (reward code, fitness value, iteration, sample)
        self._record_results: Dict = {}
        self._reward_history = []

        (self._experiment_path / 'code').mkdir(parents=True, exist_ok=True)
        self._reward_history_path = self._experiment_path / 'code' / 'reward_history'
        self._reward_history_path.mkdir(parents=True, exist_ok=True)

        for iteration in range(self._config['eureka']['iterations']):
            for sample in range(self._config['eureka']['samples']):
                sample_path = self._experiment_path / 'code' / f'iteration_{iteration}' / f'sample_{sample}'
                sample_path.mkdir(parents=True, exist_ok=True)
                shutil.copytree(self._root_path / 'envs' / self._config['environment']['name'] / 'env_code',
                                sample_path / 'env_code')

        torch.multiprocessing.set_start_method('spawn')

        if self._config['eureka']['backend'] == 'ollama':
            self._llm_generator = OllamaGenerator(model=self._config['eureka']['model'])
        elif self._config['eureka']['backend'] == 'openai':
            self._llm_generator = OpenAIGenerator(model=self._config['eureka']['model'])
        else:
            raise ValueError(f"Backend {self._config['eureka']['backend']} not available. Choose from ['ollama', 'openai']")

    def _build_generation_prompt(self, iteration: int) -> str:
        prompt = self._prompts['initial_system'] + '\nCoding instructions: ' + self._prompts['coding_instructions']

        if iteration == 0:
            # Bootstrap needs the full task and environment interface. Later iterations
            # use compact reflection only; full env code is intentionally not resent.
            prompt += '\nTask description: ' + self._prompts['task_description']
            prompt += '\nEnvironment code:\n' + self._prompts['env_code']
            if 'initial_reward' in self._prompts:
                prompt += '\nInitial reward proposal:\n' + self._prompts['initial_reward']
                prompt += ('\nYou must provide a variation from the initial reward proposal. '
                           'This is only a suggestion; provide valid reward code using the coding tips.')
        else:
            prompt += '\nReward reflection:\n' + self._prompts['reward_reflection']
            if self._config['eureka']['pretraining_with_best_model']:
                prompt += ('\nThe next training will take the best model weights so it reuses relevant '
                           'information from the previous training.')

        prompt += '\nYour reward code is: '
        return prompt

    def run(self, verbose: bool = True):
        init_run_time = time.time()
        logger = get_logger() if verbose else EmptyLogger()

        logger.info(f"Starting stable-eureka optimization. Iterations: {self._config['eureka']['iterations']}, "
                    f"samples: {self._config['eureka']['samples']}")
        logger.info(f"Using LLM: {self._config['eureka']['model']} with T={self._config['eureka']['temperature']}")

        if self._config['environment']['benchmark'] is not None:
            log_dir = self._experiment_path / 'code' / 'benchmark'
            log_dir.mkdir(parents=True, exist_ok=True)
            benchmark_env = make_env(env_class=self._config['environment']['benchmark'],
                                     env_kwargs=self._config['environment'].get('kwargs', None),
                                     n_envs=self._config['rl']['training'].get('num_envs', 1),
                                     is_atari=self._config['rl']['training'].get('is_atari', False),
                                     state_stack=self._config['rl']['training'].get('state_stack', 1),
                                     multithreaded=self._config['rl']['training'].get('multithreaded', False))

            eval_env = make_env(env_class=self._config['environment']['benchmark'],
                                env_kwargs=self._config['environment'].get('kwargs', None),
                                n_envs=1,
                                is_atari=self._config['rl']['training'].get('is_atari', False),
                                state_stack=self._config['rl']['training'].get('state_stack', 1),
                                multithreaded=self._config['rl']['training'].get('multithreaded', False))

            rl_trainer = RLTrainer(benchmark_env, config=self._config['rl'], log_dir=log_dir, name='benchmark')
            process = multiprocessing.Process(target=rl_trainer.run,
                                              args=(eval_env,
                                                    self._config['rl']['training']['eval']['seed'],
                                                    self._config['rl']['training']['eval']['num_episodes'],
                                                    self._config['rl']['training']['eval']['num_evals'],
                                                    logger, True))
            process.start()

        for iteration in range(self._config['eureka']['iterations']):
            prompt = self._build_generation_prompt(iteration)
            save_to_txt(self._experiment_path / 'code' / f'iteration_{iteration}' / 'prompt.txt', prompt)

            init_t = time.time()
            rewards = self._llm_generator.generate(temperature=self._config['eureka']['temperature'],
                                                   prompt=prompt,
                                                   k=self._config['eureka']['samples'],
                                                   logger=logger)
            elapsed = time.time() - init_t
            logger.info("++++++++++++++++++++++++++++++++++++++++++++++++++")
            logger.info(f"Iteration {iteration}/{self._config['eureka']['iterations'] - 1} - LLM generation time: {elapsed:.2f}s")

            if isinstance(self._llm_generator, OllamaGenerator) and self._config['eureka'].get('sleep_time_per_iteration', False):
                logger.info("Sleeping to avoid CUDA memory issues...")
                time.sleep(self._config['eureka'].get('sleep_time_per_iteration') * 60)

            reward_codes = []
            processes = []
            for idx, reward_response in enumerate(rewards):
                sample_dir = self._experiment_path / 'code' / f'iteration_{iteration}' / f'sample_{idx}'
                save_to_txt(sample_dir / 'llm_response.txt', reward_response)
                code = get_code_from_response(reward_response, self._regex)
                save_to_txt(sample_dir / 'reward_code.txt', code)

                logger.info(f"Sample {idx}/{self._config['eureka']['samples'] - 1}")
                logger.info(f"Reward: \n{code}")
                logger.info("--------------------------------------------------")

                code = indent_code(code, signature='# Generated code by stable-eureka')
                reward_codes.append(code)
                append_and_save_to_txt(sample_dir / 'env_code' / 'env.py', code)

                process = None
                try:
                    module_name = f"{self._config['experiment']['parent']}.{self._config['experiment']['name']}"
                    if self._config['experiment']['use_datetime']:
                        module_name += f".{self._experiment_datetime}"
                    module_name += f".code.iteration_{iteration}.sample_{idx}.env_code.env"

                    register(id=f'iteration_{iteration}_sample_{idx}_env-v0',
                             entry_point=f"{module_name}:{self._config['environment']['class_name']}",
                             max_episode_steps=self._config['environment']['max_episode_steps'])

                    env = make_env(env_class=f'iteration_{iteration}_sample_{idx}_env-v0',
                                   env_kwargs=self._config['environment'].get('kwargs', None),
                                   n_envs=self._config['rl']['training'].get('num_envs', 1),
                                   is_atari=self._config['rl']['training'].get('is_atari', False),
                                   state_stack=self._config['rl']['training'].get('state_stack', 1),
                                   multithreaded=self._config['rl']['training'].get('multithreaded', False))

                    eval_env = make_env(env_class=f'iteration_{iteration}_sample_{idx}_env-v0',
                                        env_kwargs=self._config['environment'].get('kwargs', None),
                                        n_envs=1,
                                        is_atari=self._config['rl']['training'].get('is_atari', False),
                                        state_stack=self._config['rl']['training'].get('state_stack', 1),
                                        multithreaded=self._config['rl']['training'].get('multithreaded', False))

                    pretrained_model = None
                    if self._config['eureka'].get('pretraining_with_best_model', False):
                        best_iteration = self._best_reward[2]
                        best_sample = self._best_reward[3]
                        if best_iteration is not None and best_sample is not None:
                            best_model_path = self._experiment_path / 'code' / f'iteration_{best_iteration}' / f'sample_{best_sample}' / 'model.zip'
                            if best_model_path.exists():
                                pretrained_model = str(best_model_path).split('.zip')[0]

                    rl_trainer = RLTrainer(env, config=self._config['rl'], log_dir=sample_dir,
                                           pretrained_model=pretrained_model,
                                           name=f'iteration_{iteration}_sample_{idx}')
                    process = multiprocessing.Process(target=rl_trainer.run,
                                                      args=(eval_env,
                                                            self._config['rl']['training']['eval']['seed'],
                                                            self._config['rl']['training']['eval']['num_episodes'],
                                                            self._config['rl']['training']['eval']['num_evals'],
                                                            logger,))
                    process.start()

                except Exception as e:
                    logger.error(f"Error in training: {e}, for sample {idx}")

                processes.append(process)

            while active_processes := np.sum([process.is_alive() for process in processes if process is not None]):
                logger.info(f"Active processes: {active_processes}")
                time.sleep(20)

            for process in processes:
                if isinstance(process, multiprocessing.Process) and process.is_alive():
                    process.join()

            logger.info("Training loop finished...")

            best_eval = None
            best_fitness = -float('inf')
            best_idx = -1
            for idx in range(self._config['eureka']['samples']):
                eval_path = self._experiment_path / 'code' / f'iteration_{iteration}' / f'sample_{idx}' / 'evals.json'
                if not eval_path.exists():
                    continue

                evals = read_from_json(eval_path)
                fitness_scores = evals['fitness_score']
                max_value = np.max(fitness_scores) if len(fitness_scores) < 2 else np.max(fitness_scores[2:])
                if max_value > best_fitness:
                    best_fitness = max_value
                    best_idx = idx
                    best_eval = copy.deepcopy(evals)

            if best_eval is None:
                logger.info("No successful train found for this iteration. Moving to the next one with the same compact reflection as before.")
                continue

            best_reward_code = reward_codes[best_idx]
            self._record_results[iteration] = {
                'fitness': float(best_fitness),
                'sample': int(best_idx),
                'code_hash': reward_code_hash(best_reward_code),
            }

            previous_elite_iteration = self._best_reward[2]
            previous_elite_sample = self._best_reward[3]
            previous_elite_fitness = self._best_reward[1]
            is_new_elite = best_fitness > previous_elite_fitness

            current_label = f'iter_{iteration:03d}_sample_{best_idx}'
            previous_parent_label = None
            if previous_elite_iteration is not None and previous_elite_sample is not None:
                previous_parent_label = f'iter_{previous_elite_iteration:03d}_sample_{previous_elite_sample}'

            reward_code_filename = f'{current_label}_reward.py'
            reward_code_path = self._reward_history_path / reward_code_filename
            save_to_txt(reward_code_path, best_reward_code)

            history_record = {
                'iteration': int(iteration),
                'sample': int(best_idx),
                'parent': previous_parent_label,
                'is_elite': bool(is_new_elite),
                'fitness': float(best_fitness),
                'final_eval_summary': summarize_final_eval(best_eval),
                'reward_code_path': str(reward_code_path.relative_to(self._experiment_path)),
                'code_hash': reward_code_hash(best_reward_code),
                'reward_code': best_reward_code,
            }
            self._reward_history.append(history_record)
            save_to_json(self._reward_history_path / 'reward_history.json', {'records': self._reward_history})

            component_feedback = reflection_component_to_str(best_eval)
            if is_new_elite or self._best_reward[0] == '':
                parent_reward_code = best_reward_code
                parent_label = current_label
            else:
                parent_reward_code = self._best_reward[0]
                parent_label = previous_parent_label

            reward_reflection_prompt = build_compact_reflection_prompt(
                reflection_init=self._prompts['reward_reflection_init'],
                reflection_end=self._prompts['reward_reflection_end'],
                task_description=self._prompts['task_description'],
                env_code=self._prompts['env_code'],
                component_feedback=component_feedback,
                parent_reward_code=parent_reward_code,
                parent_label=parent_label,
                reward_history=self._reward_history,
                max_total_chars=15000,
            )

            self._prompts['reward_reflection'] = reward_reflection_prompt
            save_to_txt(self._experiment_path / 'code' / f'iteration_{iteration}' / 'reflection_input_compact.txt',
                        reward_reflection_prompt)

            if is_new_elite:
                logger.info(f"New best reward found with fitness score of: {best_fitness}, previous best: {previous_elite_fitness}")
                logger.info(f"Reward code:\n{best_reward_code}")
                self._best_reward = (best_reward_code, best_fitness, iteration, best_idx)
                save_to_json(self._experiment_path / 'code' / 'best_reward.json',
                             {'reward': best_reward_code, 'fitness': float(best_fitness),
                              'iteration': int(iteration), 'sample': int(best_idx),
                              'code_hash': reward_code_hash(best_reward_code)})

            save_to_json(self._experiment_path / 'code' / 'best_iteration_rewards.json', self._record_results)

        model_path = self._experiment_path / 'code' / f'iteration_{self._best_reward[2]}' / f'sample_{self._best_reward[3]}' / 'model.zip'
        env_name = f'iteration_{self._best_reward[2]}_sample_{self._best_reward[3]}_env-v0'

        env = make_env(env_class=env_name,
                       env_kwargs=self._config['environment'].get('kwargs', None),
                       n_envs=1,
                       is_atari=self._config['rl']['training'].get('is_atari', False),
                       state_stack=self._config['rl']['training'].get('state_stack', 1),
                       multithreaded=self._config['rl']['training'].get('multithreaded', False))

        evaluator = RLEvaluator(model_path, algo=self._config['rl']['algo'])
        evaluator.run(env, seed=self._config['rl']['evaluation']['seed'],
                      n_episodes=self._config['rl']['evaluation']['num_episodes'],
                      logger=logger, save_gif=self._config['rl']['evaluation']['save_gif'])

        end_run_time = time.time()
        delta_time = end_run_time - init_run_time
        logger.info(f"Stable-Eureka optimization finished in {delta_time:.2f}s ("
                    f"{delta_time / 60:.2f}m) ({delta_time / 3600:.2f}h)")
