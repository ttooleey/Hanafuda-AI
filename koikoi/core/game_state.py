"""
Game state management for multi-round Koi-Koi games.

This module manages the overall game state across multiple rounds,
tracking points, determining game end conditions, and providing
feature tensors for neural network input.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from koikoi.core.constants import (
    TOTAL_CARDS,
    DEFAULT_ROUNDS,
    DEFAULT_INITIAL_POINTS,
    BANKRUPTCY_THRESHOLD,
    PlayerID,
)
from koikoi.core.round_state import KoiKoiRoundState


class KoiKoiGameState:
    """
    Manages the overall state of a multi-round Koi-Koi game.
    
    This class tracks:
    - Current round and round state
    - Player points across rounds
    - Game end conditions (bankruptcy, round limit)
    - Game logging and record saving
    - Feature tensor generation for neural network
    
    Attributes:
        round: Current round number (1-indexed)
        round_total: Total number of rounds in the game
        point: Points for each player {PlayerID: int}
        round_state: Current round's state
        game_over: Whether the game has ended
        winner: Winner of the game (if ended)
    
    Example:
        >>> game = KoiKoiGameState()
        >>> while not game.game_over:
        ...     # Play round
        ...     while not game.round_state.round_over:
        ...         action = agent.get_action(game)
        ...         game.round_state.step(action)
        ...     game.new_round()
    """
    
    def __init__(
        self,
        round_num: int = 1,
        round_total: int = DEFAULT_ROUNDS,
        init_point: Optional[List[int]] = None,
        init_dealer: Optional[int] = None,
        player_name: Optional[List[str]] = None,
        record_path: str = '',
        save_record: bool = False,
    ) -> None:
        """
        Initialize a new game.
        
        Args:
            round_num: Starting round number
            round_total: Total rounds in the game
            init_point: Initial points [player1, player2]
            init_dealer: Initial dealer (1 or 2), random if None
            player_name: Player names [name1, name2]
            record_path: Path to save game records
            save_record: Whether to save game records
        """
        # Default values
        if init_point is None:
            init_point = [DEFAULT_INITIAL_POINTS, DEFAULT_INITIAL_POINTS]
        if player_name is None:
            player_name = ['Player1', 'Player2']
        
        # Store init params for new_game
        self._init_point = init_point
        self._init_dealer = init_dealer
        self._round_total = round_total
        self._player_name = {1: player_name[0], 2: player_name[1]}
        self._record_path = record_path
        self._save_record = save_record
        
        # Initialize game state
        dealer = PlayerID(init_dealer) if init_dealer else None
        self._round_state = KoiKoiRoundState(dealer=dealer)
        self._round = round_num
        self._point = {
            PlayerID.PLAYER_1: init_point[0],
            PlayerID.PLAYER_2: init_point[1],
        }
        self._game_over = False
        self._winner: Optional[PlayerID] = None
        
        # Logging
        self._log: Dict[str, Any] = {}
        self._init_record()
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def round(self) -> int:
        """Get current round number."""
        return self._round
    
    @property
    def round_total(self) -> int:
        """Get total number of rounds."""
        return self._round_total
    
    @property
    def round_state(self) -> KoiKoiRoundState:
        """Get current round state."""
        return self._round_state
    
    @property
    def point(self) -> Dict[int, int]:
        """Get points in legacy format {1: pts, 2: pts}."""
        return {
            1: self._point[PlayerID.PLAYER_1],
            2: self._point[PlayerID.PLAYER_2],
        }
    
    @property
    def game_over(self) -> bool:
        """Check if game has ended."""
        return self._game_over
    
    @property
    def winner(self) -> Optional[int]:
        """Get winner (1, 2, or 0 for draw)."""
        if self._winner is None:
            return None
        return int(self._winner) if self._winner != PlayerID.NONE else 0
    
    @property
    def player_name(self) -> Dict[int, str]:
        """Get player names in legacy format {1: name, 2: name}."""
        return {1: self._player_name[1], 2: self._player_name[2]}
    
    @property
    def log(self) -> Dict[str, Any]:
        """Get game log."""
        return self._log
    
    # =========================================================================
    # Game Flow Methods
    # =========================================================================
    
    def new_game(self) -> None:
        """Start a new game with same settings."""
        self.__init__(
            round_num=1,
            round_total=self._round_total,
            init_point=self._init_point,
            init_dealer=self._init_dealer,
            player_name=[self._player_name[1], self._player_name[2]],
            record_path=self._record_path,
            save_record=self._save_record,
        )
    
    def new_round(self) -> None:
        """
        End current round and start a new one.
        
        Updates points based on round result and checks for game end.
        """
        assert self._round_state.round_over
        
        # Update points
        round_points = self._round_state.round_point
        self._point[PlayerID.PLAYER_1] += round_points[1]
        self._point[PlayerID.PLAYER_2] += round_points[2]
        
        # Record round
        self._round_result_record()
        
        # Check game end conditions
        p1_pts = self._point[PlayerID.PLAYER_1]
        p2_pts = self._point[PlayerID.PLAYER_2]
        
        if (p1_pts <= BANKRUPTCY_THRESHOLD or 
            p2_pts <= BANKRUPTCY_THRESHOLD or 
            self._round == self._round_total):
            
            self._game_over = True
            if p1_pts > p2_pts:
                self._winner = PlayerID.PLAYER_1
            elif p2_pts > p1_pts:
                self._winner = PlayerID.PLAYER_2
            else:
                self._winner = PlayerID.NONE  # Draw
            self._game_result_record()
        else:
            # Start new round
            self._round_state.new_round()
            self._round += 1
    
    # =========================================================================
    # Logging Methods
    # =========================================================================
    
    def _init_record(self) -> None:
        """Initialize game record."""
        self._log = {
            'info': {
                'startTime': time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
                'endTime': None,
                'player1Name': self._player_name[1],
                'player2Name': self._player_name[2],
                'player1InitPts': self._point[PlayerID.PLAYER_1],
                'player2InitPts': self._point[PlayerID.PLAYER_2],
                'numRound': self._round_total,
            },
            'result': {
                'isOver': False,
                'gameWinner': None,
                'player1EndPts': None,
                'player2EndPts': None,
            },
            'save': {},
            'record': {},
        }
    
    def _round_result_record(self) -> None:
        """Record round result."""
        self._log['record'][f'round{self._round}'] = self._round_state.log
    
    def _game_result_record(self) -> None:
        """Record game result."""
        self._log['info']['endTime'] = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self._log['result'] = {
            'isOver': True,
            'gameWinner': self.winner,
            'player1EndPts': self._point[PlayerID.PLAYER_1],
            'player2EndPts': self._point[PlayerID.PLAYER_2],
        }
        if self._save_record:
            self._save_record_to_file()
    
    def _save_record_to_file(self) -> None:
        """Save game record to JSON file."""
        filename = (
            f"{self._record_path}{self._log['info']['startTime']} "
            f"{self._player_name[1]} vs {self._player_name[2]}.json"
        )
        with open(filename, 'w') as f:
            json.dump(self._log, f)
    
    # =========================================================================
    # Feature Tensor Methods (for Neural Network)
    # =========================================================================
    
    @property
    def game_status_array(self) -> np.ndarray:
        """
        Get game status as numpy array for neural network features.
        
        Includes:
        - Point difference
        - Yaku points
        - Round number
        - Turn number
        - Dealer
        - Koi-koi counts
        """
        def feature_tuple(x: float, power: List[float], weight: List[float]) -> np.ndarray:
            """Create polynomial feature."""
            return np.abs(x) ** np.array(power) * np.sign(x) * np.array(weight)
        
        def feature_one_hot(pos: int, length: int) -> np.ndarray:
            """Create one-hot encoding."""
            x = np.zeros(length)
            x[pos] = 1
            return x
        
        turn_player = self._round_state.turn_player
        idle_player = self._round_state.idle_player
        
        # Build feature dictionary
        f_dict = {}
        
        # Point difference
        point_diff = self.point[turn_player] - self.point[idle_player]
        f_dict['GamePoint'] = feature_tuple(point_diff / 2, [0.5, 1, 1.5], [1, 0.5, 0.1])
        
        # Yaku points
        f_dict['MyYakuPoint'] = feature_tuple(
            self._round_state.yaku_point(turn_player), [0.5, 1, 1.5], [1, 0.5, 0.1]
        )
        f_dict['OpYakuPoint'] = feature_tuple(
            self._round_state.yaku_point(idle_player), [0.5, 1, 1.5], [1, 0.5, 0.1]
        )
        
        # Game progress
        f_dict['Round'] = feature_one_hot(self._round - 1, 8)
        f_dict['Turn'] = feature_one_hot(self._round_state.turn_16 - 1, 16)
        f_dict['Dealer'] = feature_one_hot(self._round_state.dealer - 1, 2)
        
        # Koi-koi counts
        f_dict['MyKoiKoiNum'] = feature_tuple(
            self._round_state.koikoi_num[turn_player], [1, 2], [1, 1]
        )
        f_dict['OpKoiKoiNum'] = feature_tuple(
            self._round_state.koikoi_num[idle_player], [1, 2], [1, 1]
        )
        
        # Koi-koi history
        f_dict['MyKoiKoi'] = np.array(self._round_state.koikoi[turn_player])
        f_dict['OpKoiKoi'] = np.array(self._round_state.koikoi[idle_player])
        
        # Concatenate and tile
        f_array = np.concatenate(list(f_dict.values()))
        f_array = np.tile(f_array, (TOTAL_CARDS, 1)).T
        return f_array
    
    @property
    def reserve_array(self) -> np.ndarray:
        """Get reserved feature space (for future use)."""
        return np.zeros([17, TOTAL_CARDS])
    
    @property
    def feature_tensor(self) -> torch.Tensor:
        """
        Get complete feature tensor for neural network input.
        
        The tensor has shape (n_features, n_cards) where n_cards is 48
        (or 50 for koi-koi decisions).
        
        Features include:
        - Reserve array (17 x 48)
        - Game status (varies x 48)
        - Yaku status (varies x 48)
        - Card suit encoding (12 x 48)
        - Card positions (varies x 48)
        - Card pairing state (2 x 48)
        - Card log history (varies x 48)
        
        For koi-koi decisions, two extra columns are prepended
        representing the stop/continue choices.
        """
        f = np.vstack([
            self.reserve_array,
            self.game_status_array,
            self._round_state.yaku_status_array,
            self._round_state.card_suit_array,
            self._round_state.card_init_position_array,
            self._round_state.card_current_position_array,
            self._round_state.card_pairing_state_array,
            self._round_state.card_log_array,
        ])
        
        # For koi-koi state, add two columns for stop/continue choices
        if self._round_state.state == 'koikoi':
            f_token = np.zeros([f.shape[0], 2])
            f_token[0:137, :] = f[0:137, 0:2]
            f_token[0, 0] = 1  # Stop token
            f_token[1, 1] = 1  # Continue token
            f = np.hstack([f_token, f])
        
        return torch.Tensor(f)
    
    # =========================================================================
    # Display Methods
    # =========================================================================
    
    def __call__(self) -> None:
        """Display current game state."""
        print('-----------------------------------------------')
        print(f'Round: {self._round} / {self._round_total}')
        print(f'{self._player_name[1]}: {self.point[1]}, '
              f'{self._player_name[2]}: {self.point[2]}')
        if self._game_over:
            print('Game Over')
        print('-----------------------------------------------')
