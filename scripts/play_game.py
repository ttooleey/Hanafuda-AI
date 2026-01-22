#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Play Koi-Koi against a trained AI agent.

This script provides an interactive GUI for playing Koi-Koi
card games against various trained AI models.

Usage:
    python scripts/play_game.py [--ai SL|RL-Point|RL-WP] [--name YourName]

Example:
    python scripts/play_game.py --ai RL-Point --name Player
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch_compat  # Load compatibility layer for older PyTorch models

from koikoi import KoiKoiGameState
from koikoi.ai import KoiKoiAgent
from koikoi.ui.gui import KoiKoiGUI


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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Play Koi-Koi against a trained AI agent.'
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
        default=True,
        help='Save game records (default: True)'
    )
    parser.add_argument(
        '--record-path',
        type=str,
        default='gamerecords_player/',
        help='Path to save game records'
    )
    return parser.parse_args()


def load_legacy_agent(model_paths: dict) -> "AgentForTest":
    """
    Load AI agent using legacy model loading.
    
    For compatibility with existing trained models.
    """
    from koikoilearn import AgentForTest
    
    discard = torch.load(model_paths['discard'], map_location='cpu')
    pick = torch.load(model_paths['pick'], map_location='cpu')
    koikoi = torch.load(model_paths['koikoi'], map_location='cpu')
    
    return AgentForTest(discard, pick, koikoi)


def main():
    """Main game loop."""
    args = parse_args()
    
    # Setup record directory
    record_path = Path(args.record_path) / args.ai
    record_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize game state
    game_state = KoiKoiGameState(
        player_name=[args.name, args.ai],
        record_path=str(record_path) + '/',
        save_record=args.save_records
    )
    
    # Load AI agent (using legacy loader for compatibility)
    ai_agent = load_legacy_agent(AI_MODELS[args.ai])
    
    # Initialize GUI
    gui = KoiKoiGUI()
    gui.update_game_status(game_state)
    
    # Main game loop
    while True:
        state = game_state.round_state.state
        turn_player = game_state.round_state.turn_player
        wait_action = game_state.round_state.wait_action
        action = None
        
        # Check game over
        if game_state.game_over:
            gui.show_game_over(game_state)
            gui.close()
            break
        
        # Round over
        elif state == 'round-over':
            gui.show_round_over(game_state)
            game_state.new_round()
            gui.clear_board()
            gui.update_game_status(game_state)
            gui.update_all_cards(game_state)
        
        # Player's turn
        elif turn_player == 1:
            if state == 'discard':
                gui.update_turn_player(game_state)
                gui.update_all_cards(game_state)
                action = gui.wait_discard(game_state)
                game_state.round_state.discard(action)
                
            elif state == 'discard-pick':
                if wait_action:
                    action = gui.wait_pick(game_state)
                game_state.round_state.discard_pick(action)
                
            elif state == 'draw':
                gui.update_all_cards(game_state)
                gui.wait_any_click()
                game_state.round_state.draw(action)
                
            elif state == 'draw-pick':
                gui._update_pile_card()  # Show drawn card
                if wait_action:
                    action = gui.wait_pick(game_state)
                else:
                    gui.wait_any_click()
                game_state.round_state.draw_pick(action)
                
            elif state == 'koikoi':
                gui.update_all_cards(game_state)
                if wait_action:
                    action = gui.wait_koikoi()
                game_state.round_state.claim_koikoi(action)
        
        # AI's turn
        elif turn_player == 2:
            if state == 'discard':
                gui.update_turn_player(game_state)
                gui.update_all_cards(game_state)
                action = ai_agent.auto_action(game_state)
                game_state.round_state.discard(action)
                gui.wait_any_click()
                # Show discarded card
                
            elif state == 'discard-pick':
                action = ai_agent.auto_action(game_state)
                game_state.round_state.discard_pick(action)
                gui.wait_any_click()
                
            elif state == 'draw':
                gui.update_all_cards(game_state)
                game_state.round_state.draw(action)
                gui.wait_any_click()
                
            elif state == 'draw-pick':
                action = ai_agent.auto_action(game_state)
                gui.wait_any_click()
                game_state.round_state.draw_pick(action)
                
            elif state == 'koikoi':
                gui.update_all_cards(game_state)
                action = ai_agent.auto_action(game_state)
                gui.show_opponent_koikoi(game_state, action)
                game_state.round_state.claim_koikoi(action)


if __name__ == '__main__':
    main()
