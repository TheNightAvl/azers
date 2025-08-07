"""
Monte Carlo simulation package for Azers game analysis.
"""

from .game_engine import (
    play_game, random_strategy, create_mode1_board, 
    get_all_mode2_starting_positions, GameResult
)
from .tactical_ai import create_tactical_ai

__all__ = [
    'play_game', 'random_strategy', 'create_mode1_board',
    'get_all_mode2_starting_positions', 'GameResult', 'create_tactical_ai'
]
