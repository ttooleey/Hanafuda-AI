"""
GUI components for Koi-Koi game using FreeSimpleGUI.

This module provides the graphical user interface for playing
Koi-Koi against the AI or other players.

Design Notes:
    - Uses FreeSimpleGUI (a PySimpleGUI-compatible fork)
    - Separates layout creation from update logic
    - Provides both class-based and function-based interfaces
      for backward compatibility
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import FreeSimpleGUI as sg

if TYPE_CHECKING:
    from koikoi.core.game_state import KoiKoiGameState


# Resource paths - resolve to absolute path
def _find_resource_dir() -> Path:
    """Find the resource directory containing card images."""
    # Try relative to this file first
    gui_file = Path(__file__).resolve()
    resource_dir = gui_file.parent.parent.parent / 'resource'
    if resource_dir.exists():
        return resource_dir
    
    # Try current working directory
    cwd_resource = Path.cwd() / 'resource'
    if cwd_resource.exists():
        return cwd_resource
    
    # Search parent directories
    search_dir = Path.cwd()
    for _ in range(5):
        if (search_dir / 'resource').exists():
            return search_dir / 'resource'
        search_dir = search_dir.parent
    
    # Fallback to relative path
    return Path('resource')

RESOURCE_DIR = _find_resource_dir()
PATH_CARD = str(RESOURCE_DIR / 'cardpng') + '/'
PATH_CARD_DARK = str(RESOURCE_DIR / 'cardpngdark') + '/'
PATH_CARD_LIGHT = str(RESOURCE_DIR / 'cardpnglight') + '/'
PATH_CARD_SMALL = str(RESOURCE_DIR / 'cardpngsmall') + '/'
PATH_CARD_SMALL_DARK = str(RESOURCE_DIR / 'cardpngsmalldark') + '/'
PATH_CARD_SMALL_LIGHT = str(RESOURCE_DIR / 'cardpngsmalllight') + '/'


# Card classification for display
BRIGHT_CARDS = [[1, 1], [3, 1], [8, 1], [11, 1], [12, 1]]
SEED_CARDS = [[2, 1], [4, 1], [5, 1], [6, 1], [7, 1], [8, 2], [9, 1], [10, 1], [11, 2]]
RIBBON_CARDS = [[1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [9, 2], [10, 2], [11, 3]]


def classify_cards(card_list: List[List[int]]) -> Tuple[List, List, List, List]:
    """
    Classify cards by type for display purposes.
    
    Args:
        card_list: List of cards as [suit, rank] pairs
        
    Returns:
        Tuple of (bright_cards, seed_cards, ribbon_cards, dross_cards)
        
    Note:
        The sake cup (9,1) is both a seed and dross for display.
    """
    brights, seeds, ribbons, dross = [], [], [], []
    
    for card in card_list:
        if card == [9, 1]:
            # Sake cup is both seed and dross
            seeds.append(card)
            dross.append(card)
        elif card in BRIGHT_CARDS:
            brights.append(card)
        elif card in SEED_CARDS:
            seeds.append(card)
        elif card in RIBBON_CARDS:
            ribbons.append(card)
        else:
            dross.append(card)
    
    return brights, seeds, ribbons, dross


# Legacy alias
CardClassify = classify_cards


class KoiKoiGUI:
    """
    Koi-Koi game GUI manager.
    
    Provides methods for creating and updating the game display,
    handling user input, and showing game events.
    
    Attributes:
        window: FreeSimpleGUI window object
        
    Example:
        >>> gui = KoiKoiGUI()
        >>> gui.update_game_status(game_state)
        >>> card = gui.wait_discard(game_state)
    """
    
    def __init__(self):
        """Create and initialize the GUI window."""
        self.window = self._create_window()
    
    def _create_window(self) -> sg.Window:
        """Create the main game window."""
        sg.theme('Material1')
        
        # Score board layout (left panel)
        layout_score = [
            [sg.Text('Round', font=('Helvetica', 20), pad=((2, 2), (0, 0)))],
            [sg.Text('12 / 12', font=('Helvetica', 25), 
                    pad=((2, 2), (0, 3)), key='RoundCounter')],
            [sg.Text('            ', font=('Helvetica', 12), key='gameNum')],
            [sg.T('')],
            [sg.Text('Player2Name', font=('Helvetica', 20), key='opName')],
            [sg.Text('30 Points', font=('Helvetica', 18), key='opPoints')],
            [sg.Text('            ', font=('Helvetica', 12), key='opDealer')],
            [sg.T('')],
            [sg.T(''), sg.Button(image_filename=PATH_CARD + '0-0.png', key='PileCard')],
            [sg.T('')],
            [sg.Text('Player1Name', font=('Helvetica', 20), key='myName')],
            [sg.Text('30 Points', font=('Helvetica', 18), key='myPoints')],
            [sg.Text('            ', font=('Helvetica', 12), key='myDealer')],
            [sg.T('')],
            [sg.T('', size=(3, 1), key=f'PointsRound{i}') for i in [1, 2, 3]],
            [sg.T('', size=(3, 1), key=f'PointsRound{i}') for i in [4, 5, 6]],
            [sg.T('', size=(3, 1), key=f'PointsRound{i}') for i in [7, 8, 9]],
            [sg.T('', size=(3, 1), key=f'PointsRound{i}') for i in [10, 11, 12]],
            [sg.T('')],
            [sg.T('')],
            [sg.Button('Quit', size=(10, 1))]
        ]
        
        # Opponent collected cards
        op_brights = [
            [sg.Image(PATH_CARD_SMALL + 'null.png', pad=((0, 0), (0, 0)))],
            [sg.Image(PATH_CARD_SMALL + 'null.png', pad=((0, 0), (0, 8)), 
                     key=f'OpBrights{i}') for i in range(1, 6)]
        ]
        op_seeds = [
            [sg.Image(PATH_CARD_SMALL + 'null.png', pad=((0, 0), (0, 0)), 
                     key=f'OpSeeds{i}') for i in range(6, 11)],
            [sg.Image(PATH_CARD_SMALL + 'null.png', pad=((0, 0), (0, 8)), 
                     key=f'OpSeeds{i}') for i in range(1, 6)]
        ]
        op_ribbons = [
            [sg.Image(PATH_CARD_SMALL + 'null.png', pad=((0, 0), (0, 0)), 
                     key=f'OpRibbons{i}') for i in range(6, 11)],
            [sg.Image(PATH_CARD_SMALL + 'null.png', pad=((0, 0), (0, 8)), 
                     key=f'OpRibbons{i}') for i in range(1, 6)]
        ]
        op_dross = [
            [sg.Image(PATH_CARD_SMALL + 'null.png', pad=((0, 0), (0, 0)), 
                     key=f'OpDross{i}') for i in [6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 24, 25, 26]],
            [sg.Image(PATH_CARD_SMALL + 'null.png', pad=((0, 0), (0, 8)), 
                     key=f'OpDross{i}') for i in [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 21, 22, 23]]
        ]
        
        layout_op_collected = [
            [sg.Column(op_brights), sg.Column(op_seeds), 
             sg.Column(op_ribbons), sg.Column(op_dross)]
        ]
        
        # Hand and board cards
        layout_op_hand = [
            [sg.Button(image_filename=PATH_CARD + '0-0.png', 
                      key=f'OpHand{i}') for i in range(1, 9)]
        ]
        
        layout_board = [
            [sg.T('')],
            [sg.Button(image_filename=PATH_CARD + 'null.png', 
                      key=f'Board{i}') for i in [1, 3, 5, 7, 9, 11, 13, 15]],
            [sg.Button(image_filename=PATH_CARD + 'null.png', 
                      key=f'Board{i}') for i in [2, 4, 6, 8, 10, 12, 14, 16]],
            [sg.T('')]
        ]
        
        layout_my_hand = [
            [sg.Button(image_filename=PATH_CARD + '0-0.png', 
                      key=f'MyHand{i}') for i in range(1, 9)]
        ]
        
        # My collected cards
        my_brights = [
            [sg.Image(PATH_CARD_SMALL + 'null.png', pad=((0, 0), (8, 0)), 
                     key=f'MyBrights{i}') for i in range(1, 6)],
            [sg.Image(PATH_CARD_SMALL + 'null.png', pad=((0, 0), (0, 0)))]
        ]
        my_seeds = [
            [sg.Image(PATH_CARD_SMALL + 'null.png', pad=((0, 0), (8, 0)), 
                     key=f'MySeeds{i}') for i in range(1, 6)],
            [sg.Image(PATH_CARD_SMALL + 'null.png', pad=((0, 0), (0, 0)), 
                     key=f'MySeeds{i}') for i in range(6, 11)]
        ]
        my_ribbons = [
            [sg.Image(PATH_CARD_SMALL + 'null.png', pad=((0, 0), (8, 0)), 
                     key=f'MyRibbons{i}') for i in range(1, 6)],
            [sg.Image(PATH_CARD_SMALL + 'null.png', pad=((0, 0), (0, 0)), 
                     key=f'MyRibbons{i}') for i in range(6, 11)]
        ]
        my_dross = [
            [sg.Image(PATH_CARD_SMALL + 'null.png', pad=((0, 0), (8, 0)), 
                     key=f'MyDross{i}') for i in [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 21, 22, 23]],
            [sg.Image(PATH_CARD_SMALL + 'null.png', pad=((0, 0), (0, 0)), 
                     key=f'MyDross{i}') for i in [6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 24, 25, 26]]
        ]
        
        layout_my_collected = [
            [sg.Column(my_brights), sg.Column(my_seeds), 
             sg.Column(my_ribbons), sg.Column(my_dross)]
        ]
        
        # Yaku display
        layout_op_yakus = [
            [sg.Text('', size=(16, 1), key=f'OpYaku{i}'), 
             sg.Text('', size=(2, 1), key=f'OpYakuPt{i}')] 
            for i in range(1, 11)
        ]
        layout_hint = [[sg.Text('', size=(17, 1), key='Hint', text_color='blue')]]
        layout_my_yakus = [
            [sg.Text('', size=(16, 1), key=f'MyYaku{i}'), 
             sg.Text('', size=(2, 1), key=f'MyYakuPt{i}')] 
            for i in range(1, 11)
        ]
        
        # Combine board layout
        layout_board_area = [
            [sg.Column(layout_op_hand + layout_board + layout_my_hand), 
             sg.Column(layout_op_yakus + layout_hint + layout_my_yakus)]
        ]
        
        # Full layout
        layout = [
            [sg.Column(layout_score), 
             sg.Column(layout_op_collected + layout_board_area + layout_my_collected)]
        ]
        
        window = sg.Window('Koi-Koi', layout, finalize=True)
        
        # Bind events
        window.bind("<Button-1>", 'Any Click')
        for i in range(1, 9):
            window[f'MyHand{i}'].bind("<Enter>", '-Enter')
            window[f'MyHand{i}'].bind("<Leave>", '-Leave')
        
        return window
    
    def update_game_status(self, game_state: "KoiKoiGameState") -> None:
        """Update the game status display."""
        round_state = game_state.round_state
        
        self.window['RoundCounter'].update(
            f'{game_state.round} / {game_state.round_total}'
        )
        self.window['gameNum'].update('  ')
        self.window['myName'].update(game_state.player_name[1])
        self.window['opName'].update(game_state.player_name[2])
        self.window['myPoints'].update(f'{game_state.point[1]} Points')
        self.window['opPoints'].update(f'{game_state.point[2]} Points')
        
        if round_state.dealer == 1:
            self.window['myDealer'].update('Dealer')
            self.window['opDealer'].update('      ')
        else:
            self.window['myDealer'].update('      ')
            self.window['opDealer'].update('Dealer')
        
        # Update round point history
        for i in range(1, game_state.round):
            pts = game_state.log['record'][f'round{i}']['basic']['player1RoundPts']
            self.window[f'PointsRound{i}'].update(pts)
    
    def update_turn_player(self, game_state: "KoiKoiGameState") -> None:
        """Highlight the current turn player."""
        if game_state.round_state.turn_player == 1:
            self.window['myName'].update(
                game_state.player_name[1], text_color='blue'
            )
            self.window['opName'].update(
                game_state.player_name[2], text_color='black'
            )
        else:
            self.window['myName'].update(
                game_state.player_name[1], text_color='black'
            )
            self.window['opName'].update(
                game_state.player_name[2], text_color='blue'
            )
    
    def clear_board(self) -> None:
        """Clear all cards from the display."""
        for prefix in ['My', 'Op']:
            # Clear collected cards
            for i in range(1, 6):
                self.window[f'{prefix}Brights{i}'].update(PATH_CARD_SMALL + 'null.png')
            for i in range(1, 11):
                self.window[f'{prefix}Seeds{i}'].update(PATH_CARD_SMALL + 'null.png')
                self.window[f'{prefix}Ribbons{i}'].update(PATH_CARD_SMALL + 'null.png')
            for i in range(1, 27):
                self.window[f'{prefix}Dross{i}'].update(PATH_CARD_SMALL + 'null.png')
            
            # Clear hand cards
            for i in range(1, 9):
                self.window[f'{prefix}Hand{i}'].update(
                    image_filename=PATH_CARD + 'null.png', visible=True
                )
            
            # Clear yakus
            for i in range(1, 11):
                self.window[f'{prefix}Yaku{i}'].update('')
                self.window[f'{prefix}YakuPt{i}'].update('')
    
    def update_all_cards(self, game_state: "KoiKoiGameState") -> None:
        """Update all card displays."""
        self._update_hand_cards(game_state)
        self._update_board_cards(game_state)
        self._update_collected_cards(game_state)
        self._update_pile_card()
        self._update_yakus(game_state)
    
    def _update_collected_cards(self, game_state: "KoiKoiGameState") -> None:
        """Update the collected cards display for both players."""
        round_state = game_state.round_state
        
        for prefix, pile in [('My', round_state.pile[1]), ('Op', round_state.pile[2])]:
            brights, seeds, ribbons, dross = classify_cards(pile)
            
            for i, cards, key in [
                (5, brights, 'Brights'),
                (10, seeds, 'Seeds'),
                (10, ribbons, 'Ribbons'),
                (26, dross, 'Dross'),
            ]:
                for j in range(1, len(cards) + 1):
                    card = cards[j - 1]
                    self.window[f'{prefix}{key}{j}'].update(
                        f'{PATH_CARD_SMALL}{card[0]}-{card[1]}.png'
                    )
    
    def _update_hand_cards(self, game_state: "KoiKoiGameState") -> None:
        """Update hand card displays."""
        round_state = game_state.round_state
        my_cards = round_state.hand[1]
        op_cards = round_state.hand[2]
        board_suits = [round_state.field_slot[i][0] for i in range(16)]
        
        # My hand - show actual cards, darken non-matching
        for i in range(1, len(my_cards) + 1):
            card = my_cards[i - 1]
            if card[0] in board_suits:
                path = f'{PATH_CARD}{card[0]}-{card[1]}.png'
            else:
                path = f'{PATH_CARD_DARK}{card[0]}-{card[1]}.png'
            self.window[f'MyHand{i}'].update(image_filename=path, visible=True)
        
        for i in range(len(my_cards) + 1, 9):
            self.window[f'MyHand{i}'].update(
                image_filename=PATH_CARD + 'null.png', visible=True
            )
        
        # Opponent hand - show card backs
        for i in range(1, len(op_cards) + 1):
            self.window[f'OpHand{i}'].update(
                image_filename=PATH_CARD + '0-0.png', visible=True
            )
        
        for i in range(len(op_cards) + 1, 9):
            self.window[f'OpHand{i}'].update(
                image_filename=PATH_CARD + 'null.png', visible=True
            )
    
    def _update_board_cards(self, game_state: "KoiKoiGameState") -> None:
        """Update the field cards display."""
        round_state = game_state.round_state
        
        for i in range(1, 17):
            card = round_state.field_slot[i - 1]
            if card == [0, 0]:
                path = PATH_CARD + 'null.png'
            else:
                path = f'{PATH_CARD}{card[0]}-{card[1]}.png'
            self.window[f'Board{i}'].update(image_filename=path)
    
    def _update_pile_card(self) -> None:
        """Update the draw pile display."""
        self.window['PileCard'].update(image_filename=PATH_CARD + '0-0.png')
    
    def _update_yakus(self, game_state: "KoiKoiGameState") -> None:
        """Update yaku display for both players."""
        round_state = game_state.round_state
        
        for prefix, player in [('My', 1), ('Op', 2)]:
            yakus = round_state.yaku(player)
            total_pts = round_state.yaku_point(player)
            
            if len(yakus) >= 10:
                self.window[f'{prefix}Yaku1'].update('Too Many Yakus')
                self.window[f'{prefix}Yaku2'].update('--------TOTAL--------')
                self.window[f'{prefix}YakuPt2'].update(str(total_pts))
                continue
            
            for i in range(1, len(yakus) + 1):
                yaku = yakus[i - 1]
                self.window[f'{prefix}Yaku{i}'].update(yaku[1])
                if yaku[0] == 16 and yaku[2] >= 4:
                    self.window[f'{prefix}YakuPt{i}'].update(f'x{yaku[2] - 2}')
                else:
                    self.window[f'{prefix}YakuPt{i}'].update(str(yaku[2]))
            
            if yakus:
                self.window[f'{prefix}Yaku{len(yakus) + 1}'].update('--------TOTAL--------')
                self.window[f'{prefix}YakuPt{len(yakus) + 1}'].update(str(total_pts))
    
    def wait_discard(self, game_state: "KoiKoiGameState") -> List[int]:
        """Wait for player to select a card to discard."""
        round_state = game_state.round_state
        my_hand = round_state.hand[1]
        
        self.window['Hint'].update('-> Select a Hand Card')
        
        while True:
            event, values = self.window.read()
            
            # Handle hover events
            if event in [f'MyHand{i}-Enter' for i in range(1, len(my_hand) + 1)]:
                idx = int(event[6]) - 1
                self._highlight_matching_cards(game_state, my_hand[idx])
            elif event in [f'MyHand{i}-Leave' for i in range(1, len(my_hand) + 1)]:
                self._update_board_cards(game_state)
                self._update_collected_cards(game_state)
            elif event in [f'MyHand{i}' for i in range(1, len(my_hand) + 1)]:
                idx = int(event[6]) - 1
                self._update_collected_cards(game_state)
                return my_hand[idx]
            elif event in ['Quit', None]:
                self.close()
                sys.exit(0)
    
    def wait_pick(self, game_state: "KoiKoiGameState") -> List[int]:
        """Wait for player to select a card to pick from the field."""
        round_state = game_state.round_state
        discard = round_state.show[0]
        pairing = round_state.pairing_card
        valid_indices = [
            i + 1 for i in range(16) 
            if round_state.field_slot[i] in pairing
        ]
        
        self.window['Hint'].update('-> Select a Field Card')
        self._highlight_matching_cards(game_state, discard)
        
        while True:
            event, values = self.window.read()
            
            if event in [f'Board{i}' for i in valid_indices]:
                idx = int(event[5:]) - 1
                return round_state.field_slot[idx]
            elif event in ['Quit', None]:
                self.close()
                sys.exit(0)
    
    def wait_any_click(self) -> None:
        """Wait for any mouse click to continue."""
        self.window['Hint'].update('-> Click to Continue')
        
        while True:
            event, values = self.window.read()
            
            if event in ['Quit', None]:
                self.close()
                sys.exit(0)
            elif event == 'Any Click':
                return
    
    def wait_koikoi(self) -> bool:
        """Show koi-koi decision dialog."""
        self.window['Hint'].update('-> Koi-Koi?')
        
        while True:
            event = sg.popup_yes_no('Koi-Koi?')
            if event == 'Yes':
                return True
            elif event == 'No':
                return False
            elif event is None:
                self.close()
                sys.exit(0)
    
    def show_opponent_koikoi(
        self, 
        game_state: "KoiKoiGameState", 
        action: bool
    ) -> None:
        """Show opponent's koi-koi decision."""
        player_name = game_state.player_name[game_state.round_state.turn_player]
        msg = f'{player_name}: Koi-Koi' if action else f'{player_name}: Stop'
        sg.popup(msg, title='Koi-Koi')
    
    def show_round_over(self, game_state: "KoiKoiGameState") -> None:
        """Show round over summary."""
        round_state = game_state.round_state
        p1_pts = round_state.round_point[1]
        p2_pts = round_state.round_point[2]
        
        self.window['Hint'].update('-> Round Over')
        sg.popup(
            f'{game_state.player_name[1]}: {p1_pts}     '
            f'{game_state.player_name[2]}: {p2_pts}',
            title='Round Over'
        )
    
    def show_game_over(self, game_state: "KoiKoiGameState") -> None:
        """Show game over summary."""
        self.window['Hint'].update('-> Game Over')
        sg.popup(
            f'{game_state.player_name[1]}: {game_state.point[1]}     '
            f'{game_state.player_name[2]}: {game_state.point[2]}',
            title='Game Over'
        )
    
    def _highlight_matching_cards(
        self, 
        game_state: "KoiKoiGameState", 
        card: List[int]
    ) -> None:
        """Highlight cards matching the selected card's suit."""
        round_state = game_state.round_state
        
        # Highlight board cards
        for i in range(1, 17):
            board_card = round_state.field_slot[i - 1]
            if board_card[0] == card[0]:
                path = f'{PATH_CARD}{board_card[0]}-{board_card[1]}.png'
            elif board_card != [0, 0]:
                path = f'{PATH_CARD_DARK}{board_card[0]}-{board_card[1]}.png'
            else:
                path = PATH_CARD + 'null.png'
            self.window[f'Board{i}'].update(image_filename=path)
        
        # Highlight collected cards
        for prefix, pile in [('My', round_state.pile[1]), ('Op', round_state.pile[2])]:
            brights, seeds, ribbons, dross = classify_cards(pile)
            
            for i, cards, key in [
                (5, brights, 'Brights'),
                (10, seeds, 'Seeds'),
                (10, ribbons, 'Ribbons'),
                (26, dross, 'Dross'),
            ]:
                for j in range(1, len(cards) + 1):
                    c = cards[j - 1]
                    if c[0] == card[0]:
                        path = f'{PATH_CARD_SMALL_DARK}{c[0]}-{c[1]}.png'
                    else:
                        path = f'{PATH_CARD_SMALL}{c[0]}-{c[1]}.png'
                    self.window[f'{prefix}{key}{j}'].update(path)
    
    def close(self) -> None:
        """Close the window."""
        self.window.close()


# =============================================================================
# Legacy function-based interface for backward compatibility
# =============================================================================

def InitGUI() -> sg.Window:
    """Legacy function to create GUI window."""
    gui = KoiKoiGUI()
    return gui.window


def UpdateGameStatusGUI(window: sg.Window, game_state: Any) -> sg.Window:
    """Legacy function to update game status."""
    round_state = game_state.round_state
    
    window['RoundCounter'].update(f'{game_state.round} / {game_state.round_total}')
    window['gameNum'].update('  ')
    window['myName'].update(game_state.player_name[1])
    window['opName'].update(game_state.player_name[2])
    window['myPoints'].update(f'{game_state.point[1]} Points')
    window['opPoints'].update(f'{game_state.point[2]} Points')
    
    if round_state.dealer == 1:
        window['myDealer'].update('Dealer')
        window['opDealer'].update('      ')
    else:
        window['myDealer'].update('      ')
        window['opDealer'].update('Dealer')
    
    for i in range(1, game_state.round):
        pts = game_state.log['record'][f'round{i}']['basic']['player1RoundPts']
        window[f'PointsRound{i}'].update(pts)
    
    return window
