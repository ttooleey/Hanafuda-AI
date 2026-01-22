#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Play Koi-Koi against AI using GUI.

This module provides the main entry point for playing Koi-Koi
using the refactored koikoi package with FreeSimpleGUI interface.

Usage:
    python -m koikoi.play [--ai SL|RL-Point|RL-WP] [--name YourName]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# Import from refactored package
from koikoi.core.game_state import KoiKoiGameState
from koikoi.ai.agent import KoiKoiAgent
from koikoi.ai.strategies import ModelBasedStrategy
from koikoi.ui.gui import (
    KoiKoiGUI,
    PATH_CARD,
    PATH_CARD_DARK,
)


# AI model configurations
AI_MODELS = {
    'SL': {
        'discard': 'model_agent/discard_sl.pt',
        'pick': 'model_agent/pick_sl.pt',
        'koikoi': 'model_agent/koikoi_sl.pt',
    },
    'RL-Point': {
        'discard': 'model_agent/discard_rl_point.pt',
        'pick': 'model_agent/pick_rl_point.pt',
        'koikoi': 'model_agent/koikoi_rl_point.pt',
    },
    'RL-WP': {
        'discard': 'model_agent/discard_rl_wp.pt',
        'pick': 'model_agent/pick_rl_wp.pt',
        'koikoi': 'model_agent/koikoi_rl_wp.pt',
    },
}


def load_ai_agent(ai_name: str, model_dir: Path) -> KoiKoiAgent:
    """
    Load AI agent with trained models.
    
    Args:
        ai_name: Name of AI model ('SL', 'RL-Point', 'RL-WP')
        model_dir: Base directory containing model files
        
    Returns:
        Configured KoiKoiAgent
    """
    # Import torch_compat for older model compatibility
    try:
        import torch_compat
    except ImportError:
        pass
    
    model_paths = AI_MODELS[ai_name]
    
    discard = torch.load(model_dir / model_paths['discard'], map_location='cpu')
    pick = torch.load(model_dir / model_paths['pick'], map_location='cpu')
    koikoi = torch.load(model_dir / model_paths['koikoi'], map_location='cpu')
    
    strategy = ModelBasedStrategy(
        discard_model=discard,
        pick_model=pick,
        koikoi_model=koikoi,
        temperature=10.0,
    )
    
    return KoiKoiAgent(
        strategy=strategy,
        models={'discard': discard, 'pick': pick, 'koikoi': koikoi}
    )


class GameController:
    """
    Controller for GUI-based Koi-Koi game.
    
    Coordinates between game state, AI agent, and GUI.
    """
    
    def __init__(
        self,
        player_name: str,
        ai_name: str,
        model_dir: Path,
        save_records: bool = False,
        record_path: Optional[Path] = None,
    ):
        """
        Initialize game controller.
        
        Args:
            player_name: Human player's name
            ai_name: AI model name
            model_dir: Directory containing model files
            save_records: Whether to save game records
            record_path: Path to save records
        """
        self.player_name = player_name
        self.ai_name = ai_name
        
        # Setup record path
        if record_path and save_records:
            self.record_path = record_path / ai_name
            self.record_path.mkdir(parents=True, exist_ok=True)
        else:
            self.record_path = None
        
        # Initialize game state
        self.game_state = KoiKoiGameState(
            player_name=[player_name, ai_name],
            record_path=str(self.record_path) + '/' if self.record_path else '',
            save_record=save_records,
        )
        
        # Load AI agent
        self.ai_agent = load_ai_agent(ai_name, model_dir)
        
        # Initialize GUI
        self.gui = KoiKoiGUI()
    
    def run(self) -> None:
        """Run the main game loop."""
        # Initial GUI update
        self.gui.update_game_status(self.game_state)
        
        while True:
            rs = self.game_state.round_state
            state = rs.state
            turn_player = rs.turn_player
            wait_action = rs.wait_action
            
            action = None
            
            # Game over
            if self.game_state.game_over:
                self.gui.show_game_over(self.game_state)
                self.gui.close()
                break
            
            # Round over
            elif state == 'round-over':
                self.gui.show_round_over(self.game_state)
                self.game_state.new_round()
                self.gui.clear_board()
                self.gui.update_game_status(self.game_state)
                self.gui.update_all_cards(self.game_state)
            
            # Player's turn (Player 1)
            elif turn_player == 1:
                self._handle_player_turn(rs, state, wait_action)
            
            # AI's turn (Player 2)
            elif turn_player == 2:
                self._handle_ai_turn(rs, state, wait_action)
    
    def _handle_player_turn(self, rs: Any, state: str, wait_action: bool) -> None:
        """Handle human player's turn."""
        action = None  # Default when no action needed
        
        if state == 'discard':
            self.gui.update_turn_player(self.game_state)
            self.gui.update_all_cards(self.game_state)
            action = self.gui.wait_discard(self.game_state)
            rs.discard(action)
            
        elif state == 'discard-pick':
            if wait_action:
                action = self.gui.wait_pick(self.game_state)
            rs.discard_pick(action)
            
        elif state == 'draw':
            self.gui.update_all_cards(self.game_state)
            self.gui.wait_any_click()
            rs.draw(None)
            
        elif state == 'draw-pick':
            self._show_pile_card()
            if wait_action:
                action = self.gui.wait_pick(self.game_state)
            else:
                self.gui.wait_any_click()
            rs.draw_pick(action)
            
        elif state == 'koikoi':
            self.gui.update_all_cards(self.game_state)
            # Player can only choose koikoi when wait_action is True
            # (means they have yaku points higher than previous turn points)
            if wait_action:
                action = self.gui.wait_koikoi()
            rs.claim_koikoi(action)
    
    def _handle_ai_turn(self, rs: Any, state: str, wait_action: bool) -> None:
        """Handle AI's turn."""
        mask = np.ones(2 if state == 'koikoi' else 48)
        action = None  # Default when no action needed
        
        if state == 'discard':
            self.gui.update_turn_player(self.game_state)
            self.gui.update_all_cards(self.game_state)
            action = self.ai_agent.select_action(self.game_state, mask)
            rs.discard(action)
            self.gui.wait_any_click()
            self._show_opponent_discard()
            
        elif state == 'discard-pick':
            if wait_action:
                action = self.ai_agent.select_action(self.game_state, mask)
            rs.discard_pick(action)
            self.gui.wait_any_click()
            
        elif state == 'draw':
            self.gui.update_all_cards(self.game_state)
            rs.draw(None)
            self.gui.wait_any_click()
            
        elif state == 'draw-pick':
            self._show_pile_card()
            if wait_action:
                action = self.ai_agent.select_action(self.game_state, mask)
            self.gui.wait_any_click()
            rs.draw_pick(action)
            
        elif state == 'koikoi':
            self.gui.update_all_cards(self.game_state)
            # AI always needs to make koikoi decision when wait_action is True
            # If wait_action is False, action should be None (game will auto-set to False)
            if wait_action:
                action = self.ai_agent.select_action(self.game_state, mask)
                self.gui.show_opponent_koikoi(self.game_state, action)
            rs.claim_koikoi(action)
    
    def _show_pile_card(self) -> None:
        """Show the drawn pile card."""
        rs = self.game_state.round_state
        card = rs.show[0]
        self.gui.window['PileCard'].update(
            image_filename=f'{PATH_CARD}{card[0]}-{card[1]}.png'
        )
    
    def _show_opponent_discard(self) -> None:
        """Show opponent's discarded card."""
        rs = self.game_state.round_state
        card = rs.show[0]
        self.gui.window['OpHand1'].update(
            image_filename=f'{PATH_CARD}{card[0]}-{card[1]}.png'
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Play Koi-Koi against AI with GUI'
    )
    parser.add_argument(
        '--ai',
        type=str,
        default='RL-Point',
        choices=['SL', 'RL-Point', 'RL-WP'],
        help='AI model to play against (default: RL-Point)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='Player',
        help='Your player name (default: Player)'
    )
    parser.add_argument(
        '--save-records',
        action='store_true',
        help='Save game records'
    )
    parser.add_argument(
        '--record-path',
        type=str,
        default='gamerecords_player/',
        help='Path to save game records'
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Find project root (where model_agent/ is located)
    # Try current directory first, then parent directories
    model_dir = Path.cwd()
    while model_dir != model_dir.parent:
        if (model_dir / 'model_agent').exists():
            break
        model_dir = model_dir.parent
    
    if not (model_dir / 'model_agent').exists():
        print("Error: Cannot find model_agent/ directory")
        print("Please run from the KoiKoi-AI project directory")
        sys.exit(1)
    
    # Import torch_compat for model compatibility
    sys.path.insert(0, str(model_dir))
    try:
        import torch_compat
    except ImportError:
        pass
    
    # Create and run game controller
    controller = GameController(
        player_name=args.name,
        ai_name=args.ai,
        model_dir=model_dir,
        save_records=args.save_records,
        record_path=Path(args.record_path) if args.save_records else None,
    )
    
    print(f"Starting Koi-Koi: {args.name} vs {args.ai}")
    controller.run()


if __name__ == '__main__':
    main()
