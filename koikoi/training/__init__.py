"""
Training module for Koi-Koi AI.

Contains experience buffers, simulators, and trainers.
"""

from koikoi.training.buffer import ExperienceBuffer, Experience, RolloutBuffer
from koikoi.training.simulator import TraceSimulator, create_parallel_sampler
from koikoi.training.trainer import Trainer, TrainingConfig, Arena, epsilon_schedule

__all__ = [
    # Buffer
    "Experience",
    "ExperienceBuffer",
    "RolloutBuffer",
    # Simulator
    "TraceSimulator",
    "create_parallel_sampler",
    # Trainer
    "Trainer",
    "TrainingConfig", 
    "Arena",
    "epsilon_schedule",
]
