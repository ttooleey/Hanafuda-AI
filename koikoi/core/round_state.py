"""
Round state management for Koi-Koi using State Pattern.

This module implements the game flow within a single round,
managing card dealing, discarding, picking, and koi-koi decisions.

Game Flow:
    INIT -> DISCARD -> DISCARD_PICK -> DRAW -> DRAW_PICK -> KOIKOI
         -> (back to DISCARD or ROUND_OVER)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np

from koikoi.core.constants import (
    TOTAL_CARDS,
    SUITS_COUNT,
    CARDS_PER_SUIT,
    MAX_TURNS,
    INITIAL_HAND_SIZE,
    MAX_FIELD_SLOTS,
    PlayerID,
    Action,
    CardAction,
    KoiKoiAction,
)
from koikoi.core.card import Card, CardEncoder, CardSets, create_full_deck
from koikoi.core.yaku import YakuCalculator

if TYPE_CHECKING:
    pass


class GamePhase(Enum):
    """
    Game phases within a round.
    
    The round follows this state machine:
        INIT -> DISCARD -> DISCARD_PICK -> DRAW -> DRAW_PICK -> KOIKOI
             -> (DISCARD or ROUND_OVER)
    """
    INIT = auto()
    DISCARD = auto()
    DISCARD_PICK = auto()
    DRAW = auto()
    DRAW_PICK = auto()
    KOIKOI = auto()
    ROUND_OVER = auto()
    
    def to_legacy_string(self) -> str:
        """Convert to legacy string format."""
        mapping = {
            GamePhase.INIT: 'init',
            GamePhase.DISCARD: 'discard',
            GamePhase.DISCARD_PICK: 'discard-pick',
            GamePhase.DRAW: 'draw',
            GamePhase.DRAW_PICK: 'draw-pick',
            GamePhase.KOIKOI: 'koikoi',
            GamePhase.ROUND_OVER: 'round-over',
        }
        return mapping[self]


@dataclass
class TurnLog:
    """Log entry for a single turn."""
    player_in_turn: int = 0
    discard_card: Optional[List[int]] = None
    pair_card: List[List[int]] = field(default_factory=list)
    collect_card: List[List[int]] = field(default_factory=list)
    draw_card: Optional[List[int]] = None
    pair_card_2: List[List[int]] = field(default_factory=list)
    collect_card_2: List[List[int]] = field(default_factory=list)
    is_koikoi: Optional[bool] = None


@dataclass
class RoundLog:
    """Log entry for round state tracking."""
    # Basic info (set at init and round end)
    dealer: int = 0
    init_hand_1: List[List[int]] = field(default_factory=list)
    init_hand_2: List[List[int]] = field(default_factory=list)
    init_board: List[List[int]] = field(default_factory=list)
    init_pile: List[List[int]] = field(default_factory=list)
    round_winner: Optional[int] = None
    player_1_round_pts: Optional[int] = None
    player_2_round_pts: Optional[int] = None
    
    # Turn logs
    turns: Dict[int, TurnLog] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to legacy dictionary format."""
        result: Dict[str, Any] = {
            'basic': {
                'Dealer': self.dealer,
                'initHand1': self.init_hand_1,
                'initHand2': self.init_hand_2,
                'initBoard': self.init_board,
                'initPile': self.init_pile,
            }
        }
        
        if self.round_winner is not None:
            result['basic']['roundWinner'] = self.round_winner
            result['basic']['player1RoundPts'] = self.player_1_round_pts
            result['basic']['player2RoundPts'] = self.player_2_round_pts
        
        for turn_num, turn_log in self.turns.items():
            result[f'turn{turn_num}'] = {
                'playerInTurn': turn_log.player_in_turn,
                'discardCard': turn_log.discard_card,
                'pairCard': turn_log.pair_card,
                'collectCard': turn_log.collect_card,
                'drawCard': turn_log.draw_card,
                'pairCard2': turn_log.pair_card_2,
                'collectCard2': turn_log.collect_card_2,
                'isKoiKoi': turn_log.is_koikoi,
            }
        
        return result


class KoiKoiRoundState:
    """
    Manages the state of a single Koi-Koi round.
    
    This class handles all game logic within a round including:
    - Card dealing and validation
    - Turn management
    - Action validation and execution
    - Yaku calculation
    - State transitions
    
    Attributes:
        hand: Cards in each player's hand {PlayerID: List[Card]}
        pile: Cards collected by each player {PlayerID: List[Card]}
        field_slots: Cards on the field (with None for empty slots)
        stock: Face-down deck
        dealer: Player who deals this round
        phase: Current game phase
        turn_number: Current turn (1-16)
        winner: Winner of this round (if determined)
    
    Example:
        >>> round_state = KoiKoiRoundState(dealer=PlayerID.PLAYER_1)
        >>> round_state.discard(Card(1, 1))  # Discard January Crane
        >>> round_state.discard_pick(None)    # No matching card
        >>> round_state.draw()                # Draw from stock
    """
    
    def __init__(self, dealer: Optional[PlayerID] = None) -> None:
        """
        Initialize a new round.
        
        Args:
            dealer: The dealer for this round. If None, randomly chosen.
        """
        # Card locations
        self._hand: Dict[PlayerID, List[Card]] = {
            PlayerID.PLAYER_1: [],
            PlayerID.PLAYER_2: [],
        }
        self._pile: Dict[PlayerID, List[Card]] = {
            PlayerID.PLAYER_1: [],
            PlayerID.PLAYER_2: [],
        }
        self._field_slots: List[Optional[Card]] = []
        self._stock: List[Card] = []
        
        # Current shown card (discarded or drawn)
        self._shown_card: Optional[Card] = None
        self._collected_cards: List[Card] = []
        
        # Game state
        self._turn_number: int = 1
        self._dealer: PlayerID = (
            dealer if dealer is not None
            else random.choice([PlayerID.PLAYER_1, PlayerID.PLAYER_2])
        )
        
        # Koi-koi tracking: 8 possible koi-koi decisions per player
        self._koikoi_history: Dict[PlayerID, List[int]] = {
            PlayerID.PLAYER_1: [0] * 8,
            PlayerID.PLAYER_2: [0] * 8,
        }
        
        # Round result
        self._winner: Optional[PlayerID] = None
        self._is_exhausted: bool = False
        
        # Turn tracking for yaku comparison
        self._turn_start_points: int = 0
        
        # Phase management
        self._phase: GamePhase = GamePhase.INIT
        self._is_waiting_for_action: bool = False
        
        # Logging
        self._log = RoundLog()
        self._card_log: Dict[int, Dict[str, np.ndarray]] = {}
        
        # Yaku calculator (singleton-like)
        self._yaku_calculator = YakuCalculator()
        
        # Silence mode (suppress output)
        self.silence: bool = True
        
        # Deal cards to start
        self._deal_cards()
    
    # =========================================================================
    # Properties - Game State
    # =========================================================================
    
    @property
    def turn_player(self) -> int:
        """Get current player as int (legacy compatibility)."""
        if (self._turn_number + int(self._dealer)) % 2 == 0:
            return 1
        return 2
    
    @property
    def idle_player(self) -> int:
        """Get opponent as int (legacy compatibility)."""
        return 3 - self.turn_player
    
    @property
    def current_player(self) -> PlayerID:
        """Get the player whose turn it is."""
        return PlayerID(self.turn_player)
    
    @property
    def opponent(self) -> PlayerID:
        """Get the opponent of current player."""
        return PlayerID(self.idle_player)
    
    @property
    def turn_16(self) -> int:
        """Get turn number (1-16)."""
        return self._turn_number
    
    @property
    def turn_8(self) -> int:
        """Get turn index within 8 turns per player."""
        return (self._turn_number + 1) // 2
    
    @property
    def dealer(self) -> int:
        """Get dealer as int."""
        return int(self._dealer)
    
    @property
    def winner(self) -> Optional[int]:
        """Get winner as int or None."""
        return int(self._winner) if self._winner else None
    
    @property
    def exhausted(self) -> bool:
        """Check if round ended due to exhausted turns."""
        return self._is_exhausted
    
    # =========================================================================
    # Properties - Cards
    # =========================================================================
    
    @property
    def hand(self) -> Dict[int, List[List[int]]]:
        """Get hands in legacy format."""
        return {
            1: [c.to_list() for c in self._hand[PlayerID.PLAYER_1]],
            2: [c.to_list() for c in self._hand[PlayerID.PLAYER_2]],
        }
    
    @property
    def pile(self) -> Dict[int, List[List[int]]]:
        """Get piles in legacy format."""
        return {
            1: [c.to_list() for c in self._pile[PlayerID.PLAYER_1]],
            2: [c.to_list() for c in self._pile[PlayerID.PLAYER_2]],
        }
    
    @property
    def field_slot(self) -> List[List[int]]:
        """Get field slots in legacy format (with [0,0] for empty)."""
        return [
            c.to_list() if c is not None else [0, 0]
            for c in self._field_slots
        ]
    
    @property
    def field(self) -> List[List[int]]:
        """Get non-empty field cards in legacy format."""
        return sorted([
            c.to_list() for c in self._field_slots if c is not None
        ])
    
    @property
    def stock(self) -> List[List[int]]:
        """Get stock in legacy format."""
        return [c.to_list() for c in self._stock]
    
    @property
    def show(self) -> List[List[int]]:
        """Get shown card in legacy format."""
        if self._shown_card is None:
            return []
        return [self._shown_card.to_list()]
    
    @property
    def collect(self) -> List[List[int]]:
        """Get collected cards in legacy format."""
        return [c.to_list() for c in self._collected_cards]
    
    @property
    def unseen_card(self) -> Dict[int, List[List[int]]]:
        """Get unseen cards for each player in legacy format."""
        return {
            1: [c.to_list() for c in (self._stock + self._hand[PlayerID.PLAYER_2])],
            2: [c.to_list() for c in (self._stock + self._hand[PlayerID.PLAYER_1])],
        }
    
    @property
    def pairing_card(self) -> List[List[int]]:
        """Get field cards that can pair with shown card."""
        if self._shown_card is None:
            return []
        return [
            c.to_list() for c in self._field_slots
            if c is not None and c.suit == self._shown_card.suit
        ]
    
    @property
    def field_collect(self) -> List[List[int]]:
        """Get collected field cards (excluding shown card)."""
        collect_copy = self._collected_cards.copy()
        if self._shown_card in collect_copy:
            collect_copy.remove(self._shown_card)
        return [c.to_list() for c in collect_copy]
    
    # =========================================================================
    # Properties - State
    # =========================================================================
    
    @property
    def state(self) -> str:
        """Get current phase as legacy string."""
        return self._phase.to_legacy_string()
    
    @property
    def wait_action(self) -> bool:
        """Check if waiting for player action."""
        return self._is_waiting_for_action
    
    @property
    def round_over(self) -> bool:
        """Check if round has ended."""
        return self._phase == GamePhase.ROUND_OVER
    
    # =========================================================================
    # Properties - Koi-koi
    # =========================================================================
    
    @property
    def koikoi(self) -> Dict[int, List[int]]:
        """Get koi-koi history in legacy format."""
        return {
            1: self._koikoi_history[PlayerID.PLAYER_1].copy(),
            2: self._koikoi_history[PlayerID.PLAYER_2].copy(),
        }
    
    @property
    def koikoi_num(self) -> Dict[int, int]:
        """Get koi-koi count for each player."""
        return {
            1: sum(self._koikoi_history[PlayerID.PLAYER_1]),
            2: sum(self._koikoi_history[PlayerID.PLAYER_2]),
        }
    
    # =========================================================================
    # Properties - Points
    # =========================================================================
    
    @property
    def round_point(self) -> Dict[int, Optional[int]]:
        """Get round points for each player."""
        if self._winner is None:
            return {1: None, 2: None}
        
        if self._is_exhausted:
            # Exhausted: dealer gets 1 point
            if self._dealer == PlayerID.PLAYER_1:
                return {1: 1, 2: -1}
            return {1: -1, 2: 1}
        
        # Normal end: winner gets yaku points
        winner_points = self.yaku_point(int(self._winner))
        if self._winner == PlayerID.PLAYER_1:
            return {1: winner_points, 2: -winner_points}
        return {1: -winner_points, 2: winner_points}
    
    # =========================================================================
    # Properties - Action Mask
    # =========================================================================
    
    @property
    def action_mask(self) -> np.ndarray:
        """Get valid action mask for current phase."""
        if self._phase == GamePhase.DISCARD:
            cards = self._hand[self.current_player]
            return np.array(CardEncoder.to_multi_hot(cards))
        elif self._phase in (GamePhase.DISCARD_PICK, GamePhase.DRAW_PICK):
            pairing = [
                c for c in self._field_slots
                if c is not None and self._shown_card and c.suit == self._shown_card.suit
            ]
            return np.array(CardEncoder.to_multi_hot(pairing))
        elif self._phase == GamePhase.KOIKOI:
            return np.array([1, 1])
        return np.array([])
    
    # =========================================================================
    # Properties - Log
    # =========================================================================
    
    @property
    def log(self) -> Dict[str, Any]:
        """Get log in legacy dictionary format."""
        return self._log.to_dict()
    
    # =========================================================================
    # Yaku Methods
    # =========================================================================
    
    def yaku(self, player: int) -> List[Tuple[int, str, int]]:
        """
        Get list of yaku for a player.
        
        Args:
            player: Player ID (1 or 2)
            
        Returns:
            List of (yaku_type, name, points) tuples
        """
        player_id = PlayerID(player)
        yaku_list = self._yaku_calculator.calculate(
            self._pile[player_id],
            sum(self._koikoi_history[player_id])
        )
        return [y.to_tuple() for y in yaku_list]
    
    def yaku_point(self, player: int) -> int:
        """
        Get total yaku points for a player.
        
        Args:
            player: Player ID (1 or 2)
            
        Returns:
            Total point value
        """
        player_id = PlayerID(player)
        return self._yaku_calculator.calculate_points(
            self._pile[player_id],
            sum(self._koikoi_history[player_id])
        )
    
    # =========================================================================
    # Action Methods
    # =========================================================================
    
    def new_round(self) -> None:
        """Start a new round with current winner as dealer."""
        self.__init__(dealer=self._winner)
    
    def step(self, action: Action) -> str:
        """
        Execute an action based on current phase.
        
        Args:
            action: The action to take (card list for discard/pick, bool for koi-koi)
            
        Returns:
            New phase name as string
        """
        if self._phase == GamePhase.DISCARD:
            return self.discard(action)
        elif self._phase == GamePhase.DISCARD_PICK:
            return self.discard_pick(action)
        elif self._phase == GamePhase.DRAW:
            return self.draw(action)
        elif self._phase == GamePhase.DRAW_PICK:
            return self.draw_pick(action)
        elif self._phase == GamePhase.KOIKOI:
            return self.claim_koikoi(action)
        return self.state
    
    def discard(self, card: Optional[List[int]] = None) -> str:
        """
        Discard a card from hand.
        
        Args:
            card: Card to discard as [suit, rank] list
            
        Returns:
            New phase name
        """
        assert self._phase == GamePhase.DISCARD
        assert card in self.hand[self.turn_player]
        
        # Track yaku points at turn start
        self._turn_start_points = self.yaku_point(self.turn_player)
        
        # Find and remove card
        card_obj = Card.from_list(card)
        self._hand[self.current_player].remove(card_obj)
        self._shown_card = card_obj
        
        # Log and transition
        self._write_log()
        self._phase = GamePhase.DISCARD_PICK
        
        # Need action only if exactly 2 cards match
        pairing = [c for c in self._field_slots if c and c.suit == card_obj.suit]
        self._is_waiting_for_action = len(pairing) == 2
        
        return self.state if self.silence else self._display_state()
    
    def discard_pick(self, card: Optional[List[int]] = None) -> str:
        """
        Pick a field card to pair with discarded card.
        
        Args:
            card: Field card to pick as [suit, rank] list (required if 2 cards match)
            
        Returns:
            New phase name
        """
        assert self._phase == GamePhase.DISCARD_PICK
        if self._is_waiting_for_action:
            assert card in self.pairing_card
        
        self._collect_card(card)
        
        self._write_log()
        self._phase = GamePhase.DRAW
        self._is_waiting_for_action = False
        
        return self.state if self.silence else self._display_state()
    
    def draw(self, card: Optional[List[int]] = None) -> str:
        """
        Draw a card from stock.
        
        Args:
            card: Ignored (for API compatibility)
            
        Returns:
            New phase name
        """
        assert self._phase == GamePhase.DRAW
        
        self._shown_card = self._stock.pop()
        
        self._write_log()
        self._phase = GamePhase.DRAW_PICK
        
        # Need action only if exactly 2 cards match
        pairing = [c for c in self._field_slots if c and c.suit == self._shown_card.suit]
        self._is_waiting_for_action = len(pairing) == 2
        
        return self.state if self.silence else self._display_state()
    
    def draw_pick(self, card: Optional[List[int]] = None) -> str:
        """
        Pick a field card to pair with drawn card.
        
        Args:
            card: Field card to pick as [suit, rank] list (required if 2 cards match)
            
        Returns:
            New phase name
        """
        assert self._phase == GamePhase.DRAW_PICK
        if self._is_waiting_for_action:
            assert card in self.pairing_card
        
        self._collect_card(card)
        
        self._write_log()
        self._phase = GamePhase.KOIKOI
        
        # Can koi-koi if: gained yaku points AND not last turn
        has_new_yaku = self.yaku_point(self.turn_player) > self._turn_start_points
        is_not_last_turn = self.turn_8 < 8
        self._is_waiting_for_action = has_new_yaku and is_not_last_turn
        
        return self.state if self.silence else self._display_state()
    
    def claim_koikoi(self, is_koikoi: Optional[bool] = None) -> str:
        """
        Decide whether to claim koi-koi.
        
        Args:
            is_koikoi: True to continue (koi-koi), False to end round
            
        Returns:
            New phase name
        """
        assert self._phase == GamePhase.KOIKOI
        if self._is_waiting_for_action:
            assert isinstance(is_koikoi, bool)
        
        # Force stop on last turn with yaku
        has_new_yaku = self.yaku_point(self.turn_player) > self._turn_start_points
        if has_new_yaku and self.turn_8 == 8:
            is_koikoi = False
        
        # Record koi-koi decision
        if is_koikoi is True:
            self._koikoi_history[self.current_player][self.turn_8 - 1] = 1
        
        self._write_log(is_koikoi)
        
        # Determine next state
        if is_koikoi is False:
            # Player stops: they win
            self._phase = GamePhase.ROUND_OVER
            self._is_waiting_for_action = False
            self._winner = self.current_player
            self._write_log()
        elif self._turn_number >= MAX_TURNS:
            # Turns exhausted: dealer wins
            self._phase = GamePhase.ROUND_OVER
            self._is_waiting_for_action = False
            self._is_exhausted = True
            self._winner = self._dealer
            self._write_log()
        else:
            # Continue to next turn
            self._turn_number += 1
            self._phase = GamePhase.DISCARD
            self._is_waiting_for_action = True
        
        return self.state if self.silence else self._display_state()
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    def _deal_cards(self) -> None:
        """Deal initial cards to players and field."""
        while True:
            deck = create_full_deck()
            random.shuffle(deck)
            
            self._hand[PlayerID.PLAYER_1] = sorted(deck[0:8], key=lambda c: c.to_index())
            self._hand[PlayerID.PLAYER_2] = sorted(deck[8:16], key=lambda c: c.to_index())
            self._field_slots = deck[16:24] + [None] * (MAX_FIELD_SLOTS - INITIAL_HAND_SIZE)
            self._stock = deck[24:]
            
            # Check for 4 of same suit (redeal if found)
            if not self._has_four_of_suit():
                break
        
        # Initialize logging
        self._log.dealer = int(self._dealer)
        self._log.init_hand_1 = [c.to_list() for c in self._hand[PlayerID.PLAYER_1]]
        self._log.init_hand_2 = [c.to_list() for c in self._hand[PlayerID.PLAYER_2]]
        self._log.init_board = [c.to_list() for c in self._field_slots if c is not None]
        self._log.init_pile = [c.to_list() for c in self._stock]
        
        self._init_card_log()
        self._phase = GamePhase.DISCARD
        self._is_waiting_for_action = True
    
    def _has_four_of_suit(self) -> bool:
        """Check if any location has 4 cards of the same suit."""
        locations = [
            self._hand[PlayerID.PLAYER_1],
            self._hand[PlayerID.PLAYER_2],
            [c for c in self._field_slots if c is not None],
        ]
        for location in locations:
            for suit in range(1, SUITS_COUNT + 1):
                if sum(1 for c in location if c.suit == suit) == 4:
                    return True
        return False
    
    def _collect_card(self, picked_card: Optional[List[int]]) -> None:
        """Collect shown card and any paired field cards."""
        if self._shown_card is None:
            return
        
        pairing = [
            c for c in self._field_slots
            if c is not None and c.suit == self._shown_card.suit
        ]
        
        if len(pairing) == 0:
            # No match: place shown card on field
            self._collected_cards = []
            empty_idx = self._field_slots.index(None)
            self._field_slots[empty_idx] = self._shown_card
        elif len(pairing) in (1, 3):
            # 1 or 3 matches: collect all
            self._collected_cards = [self._shown_card] + pairing
            for card in pairing:
                idx = self._field_slots.index(card)
                self._field_slots[idx] = None
            self._pile[self.current_player].extend(self._collected_cards)
        else:
            # 2 matches: collect chosen one
            assert picked_card is not None
            picked = Card.from_list(picked_card)
            self._collected_cards = [self._shown_card, picked]
            idx = self._field_slots.index(picked)
            self._field_slots[idx] = None
            self._pile[self.current_player].extend(self._collected_cards)
    
    def _init_card_log(self) -> None:
        """Initialize card log arrays for neural network features."""
        for turn in range(1, MAX_TURNS + 1):
            self._card_log[turn] = {
                'CardDiscardedAndPaired': np.zeros(TOTAL_CARDS),
                'CardDiscardedAndUnpaired': np.zeros(TOTAL_CARDS),
                'CardPairedByDiscardCollect': np.zeros(TOTAL_CARDS),
                'CardPairedByDiscardUncollect': np.zeros(TOTAL_CARDS),
                'CardDrawnAndPaired': np.zeros(TOTAL_CARDS),
                'CardDrawnAndUnpaired': np.zeros(TOTAL_CARDS),
                'CardPairedByDrawnCollect': np.zeros(TOTAL_CARDS),
                'CardPairedByDrawnUncollect': np.zeros(TOTAL_CARDS),
            }
    
    def _write_log(self, is_koikoi: Optional[bool] = None) -> None:
        """Write current state to log."""
        turn = self._turn_number
        
        if self._phase == GamePhase.DISCARD:
            self._log.turns[turn] = TurnLog(
                player_in_turn=self.turn_player,
                discard_card=self._shown_card.to_list() if self._shown_card else None,
                pair_card=self.pairing_card,
            )
            # Update card log
            self._write_card_log_array('discard')
            
        elif self._phase == GamePhase.DISCARD_PICK:
            if turn in self._log.turns:
                self._log.turns[turn].collect_card = self.collect
            self._write_card_log_array('discard-pick')
            
        elif self._phase == GamePhase.DRAW:
            if turn in self._log.turns:
                self._log.turns[turn].draw_card = self._shown_card.to_list() if self._shown_card else None
                self._log.turns[turn].pair_card_2 = self.pairing_card
            self._write_card_log_array('draw')
            
        elif self._phase == GamePhase.DRAW_PICK:
            if turn in self._log.turns:
                self._log.turns[turn].collect_card_2 = self.collect
            self._write_card_log_array('draw-pick')
            
        elif self._phase == GamePhase.KOIKOI:
            if turn in self._log.turns:
                self._log.turns[turn].is_koikoi = is_koikoi
                
        elif self._phase == GamePhase.ROUND_OVER:
            points = self.round_point
            self._log.round_winner = self.winner
            self._log.player_1_round_pts = points[1]
            self._log.player_2_round_pts = points[2]
    
    def _write_card_log_array(self, state: str) -> None:
        """Update card log arrays for neural network features."""
        turn = self._turn_number
        
        if state == 'discard':
            if not self.pairing_card:
                self._card_log[turn]['CardDiscardedAndUnpaired'] = np.array(
                    CardEncoder.to_multi_hot_from_list(self.show)
                )
            else:
                self._card_log[turn]['CardDiscardedAndPaired'] = np.array(
                    CardEncoder.to_multi_hot_from_list(self.show)
                )
                
        elif state == 'discard-pick':
            if self.collect:
                pair_collect = np.array(CardEncoder.to_multi_hot_from_list(self.field_collect))
                pair_all = np.array(CardEncoder.to_multi_hot_from_list(
                    self._log.turns[turn].pair_card
                ))
                self._card_log[turn]['CardPairedByDiscardCollect'] = pair_collect
                self._card_log[turn]['CardPairedByDiscardUncollect'] = pair_all - pair_collect
                
        elif state == 'draw':
            if not self.pairing_card:
                self._card_log[turn]['CardDrawnAndUnpaired'] = np.array(
                    CardEncoder.to_multi_hot_from_list(self.show)
                )
            else:
                self._card_log[turn]['CardDrawnAndPaired'] = np.array(
                    CardEncoder.to_multi_hot_from_list(self.show)
                )
                
        elif state == 'draw-pick':
            if self.collect:
                pair_collect = np.array(CardEncoder.to_multi_hot_from_list(self.field_collect))
                pair_all = np.array(CardEncoder.to_multi_hot_from_list(
                    self._log.turns[turn].pair_card_2
                ))
                self._card_log[turn]['CardPairedByDrawnCollect'] = pair_collect
                self._card_log[turn]['CardPairedByDrawnUncollect'] = pair_all - pair_collect
    
    def _display_state(self) -> str:
        """Display current game state (for interactive play)."""
        # Implementation omitted for brevity - same as original __call__
        return self.state
    
    # =========================================================================
    # Feature Extraction (for Neural Network)
    # =========================================================================
    
    @property
    def card_log_array(self) -> np.ndarray:
        """Get card log as numpy array for features."""
        turn_list = (
            list(range(self._turn_number, 0, -1)) +
            list(range(self._turn_number + 1, MAX_TURNS + 1))
        )
        arrays = [
            feature
            for turn in turn_list
            for _, feature in self._card_log[turn].items()
        ]
        return np.vstack(arrays)
    
    @property
    def card_suit_array(self) -> np.ndarray:
        """Get card suit encoding."""
        f_array = np.zeros([SUITS_COUNT, TOTAL_CARDS])
        for i in range(SUITS_COUNT):
            f_array[i, CARDS_PER_SUIT * i:CARDS_PER_SUIT * (i + 1)] = 1
        return f_array
    
    @property
    def card_init_position_array(self) -> np.ndarray:
        """Get initial card position encoding."""
        # Note: Due to a historical bug that was kept for model compatibility,
        # this actually returns current hand and unseen cards, not initial positions
        f_dict = {
            'CardInMyHand': CardEncoder.to_multi_hot(self._hand[self.current_player]),
            'CardInBoard': CardEncoder.to_multi_hot_from_list(self._log.init_board),
            'CardUnseen': CardEncoder.to_multi_hot_from_list(self.unseen_card[self.turn_player]),
        }
        return np.vstack(list(f_dict.values()))
    
    @property
    def card_current_position_array(self) -> np.ndarray:
        """Get current card position encoding."""
        f_dict = {
            'CardInMyHand': CardEncoder.to_multi_hot(self._hand[self.current_player]),
            'CardInMyCollect': CardEncoder.to_multi_hot(self._pile[self.current_player]),
            'CardInBoard': CardEncoder.to_multi_hot([c for c in self._field_slots if c]),
            # Bug preserved for model compatibility: uses turn_player pile instead of opponent
            'CardInOpCollect': CardEncoder.to_multi_hot(self._pile[self.current_player]),
            'CardUnseen': CardEncoder.to_multi_hot_from_list(self.unseen_card[self.turn_player]),
        }
        return np.vstack(list(f_dict.values()))
    
    @property
    def card_pairing_state_array(self) -> np.ndarray:
        """Get pairing state encoding."""
        if self._phase in (GamePhase.DISCARD_PICK, GamePhase.DRAW_PICK):
            f_dict = {
                'CardShowed': CardEncoder.to_multi_hot_from_list(self.show),
                'CardPaired': CardEncoder.to_multi_hot_from_list(self.pairing_card),
            }
        else:
            f_dict = {
                'CardShowed': [0] * TOTAL_CARDS,
                'CardPaired': [0] * TOTAL_CARDS,
            }
        return np.vstack(list(f_dict.values()))
    
    @property
    def yaku_status_array(self) -> np.ndarray:
        """Get yaku status encoding."""
        card_dict = {
            'Crane': CardSets.CRANE,
            'Curtain': CardSets.CURTAIN,
            'Moon': CardSets.MOON,
            'Rainman': CardSets.RAINMAN,
            'Phoenix': CardSets.PHOENIX,
            'Sake': CardSets.SAKE,
            'BoarDeerButterfly': CardSets.BOAR_DEER_BUTTERFLY,
            'Seed': CardSets.SEED,
            'RedRibbon': CardSets.RED_RIBBON,
            'BlueRibbon': CardSets.BLUE_RIBBON,
            'RedAndBlue': CardSets.RED_BLUE_RIBBON,
            'Ribbon': CardSets.RIBBON,
            'Dross': CardSets.DROSS,
        }
        
        def cards_to_set(cards: List[Card]) -> set:
            return set(c.to_tuple() for c in cards)
        
        my_hand = cards_to_set(self._hand[self.current_player])
        board = cards_to_set([c for c in self._field_slots if c])
        my_collect = cards_to_set(self._pile[self.current_player])
        op_collect = cards_to_set(self._pile[self.opponent])
        unseen = cards_to_set(self._hand[self.opponent] + self._stock)
        
        f_dict = {
            'NumMyHand': [len(cs & my_hand) for cs in card_dict.values()],
            'NumBoard': [len(cs & board) for cs in card_dict.values()],
            'NumMyCollect': [len(cs & my_collect) for cs in card_dict.values()],
            'NumOpCollect': [len(cs & op_collect) for cs in card_dict.values()],
            'NumUnseen': [len(cs & unseen) for cs in card_dict.values()],
        }
        
        f_array_state = np.concatenate(list(f_dict.values()))
        f_array_state = np.tile(f_array_state, (TOTAL_CARDS, 1)).T
        f_array_key = np.array([
            CardEncoder.to_multi_hot_from_list([[s, r] for s, r in cs])
            for cs in card_dict.values()
        ])
        return np.vstack([f_array_state, f_array_key])
