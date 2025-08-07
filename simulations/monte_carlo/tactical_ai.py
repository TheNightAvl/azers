"""
Tactical AI for Azers using Alpha-Beta pruning with learning capabilities.
"""
import math
import json
import os
import random
from collections import defaultdict
from typing import List, Optional, Tuple, Dict, Any
from game_engine import (
    get_legal_moves, apply_move, get_opponent, get_tokens, 
    has_3_or_4_in_centre, check_opposition_pattern, CENTER_SQUARES
)

class TacticalAI:
    def __init__(self, max_depth: int = 8, learning_enabled: bool = False, opening_book_file: str = "opening_book.json", randomness: float = 0.15):
        self.max_depth = max_depth
        self.nodes_evaluated = 0
        self.learning_enabled = learning_enabled
        self.opening_book_file = opening_book_file
        self.randomness = randomness  # Controls how much randomisation to apply (0.0 = none, 1.0 = maximum)
        
        # Learning components
        if learning_enabled:
            self.opening_book = self.load_opening_book()
            self.game_history = []  # Store (board_hash, move) for current game
            self.position_outcomes = defaultdict(lambda: {'wins': 0, 'draws': 0, 'losses': 0, 'total': 0})
            self.games_learned = 0
        else:
            self.opening_book = {}
            self.game_history = []
            self.position_outcomes = {}
            self.games_learned = 0
        
    def get_dynamic_randomness(self, board: List[List[Optional[str]]], player: str) -> float:
        """Calculate dynamic randomness based on game state"""
        base_randomness = self.randomness
        
        # Reduce randomness in critical situations
        player_centre = sum(1 for (x, y) in CENTER_SQUARES if board[y][x] == player)
        opponent_centre = sum(1 for (x, y) in CENTER_SQUARES if board[y][x] == get_opponent(player))
        
        # If either player is close to winning, be more deterministic
        if player_centre >= 2 or opponent_centre >= 2:
            base_randomness *= 0.3  # Much less random in critical positions
        
        # Check token counts - be more careful when low on tokens
        player_tokens = len(get_tokens(board, player))
        opponent_tokens = len(get_tokens(board, get_opponent(player)))
        
        if player_tokens <= 3 or opponent_tokens <= 3:
            base_randomness *= 0.5  # Less random when massacre is possible
        
        # Early game can be more random
        total_tokens = player_tokens + opponent_tokens
        if total_tokens >= 10:  # Early game (both players have 5+ tokens)
            base_randomness *= 1.5  # More experimental in opening
        
        return min(1.0, max(0.0, base_randomness))

    def get_move(self, board: List[List[Optional[str]]], player: str, moves: List) -> Tuple:
        """Get the best move using alpha-beta search with optional learning"""
        self.nodes_evaluated = 0
        
        # Calculate dynamic randomness based on position
        current_randomness = self.get_dynamic_randomness(board, player)
        
        # Check opening book if learning is enabled and we're in early game
        if self.learning_enabled and len(self.game_history) < 8:
            book_move = self.check_opening_book(board, moves)
            if book_move:
                self.game_history.append((self._board_to_hash(board), book_move))
                return book_move
        
        # Use regular alpha-beta search
        _, best_move = self.alpha_beta(board, player, self.max_depth, -math.inf, math.inf, True, current_randomness)
        best_move = best_move if best_move else moves[0]  # Fallback to first move if search fails
        
        # Enhanced randomisation for more varied play
        if len(moves) > 1 and current_randomness > 0:
            # Get evaluation of all moves
            move_evaluations = []
            for move in moves:
                new_board = apply_move(board, player, move)
                # Use shallow search for quick evaluation
                eval_score, _ = self.alpha_beta(new_board, get_opponent(player), 2, -math.inf, math.inf, False, current_randomness)
                move_evaluations.append((eval_score, move))
            
            if move_evaluations:
                # Sort moves by score (best first)
                move_evaluations.sort(key=lambda x: x[0], reverse=True)
                best_score = move_evaluations[0][0]
                
                # Calculate dynamic tolerance based on position complexity and randomness setting
                base_tolerance = abs(best_score) * 0.1 + 10.0
                randomness_factor = current_randomness * 2.0  # Scale randomness
                tolerance = base_tolerance * randomness_factor
                
                # Find moves within tolerance of best move
                good_moves = [move for score, move in move_evaluations if score >= best_score - tolerance]
                
                if len(good_moves) > 1:
                    # Weight selection towards better moves but allow some randomness
                    weights = []
                    for score, move in move_evaluations:
                        if move in good_moves:
                            # Higher scores get higher weights, but not exclusively
                            weight = max(0.1, score - (best_score - tolerance) + 1.0)
                            weights.append(weight)
                        else:
                            weights.append(0.0)
                    
                    # Randomly select based on weights
                    if weights and sum(weights) > 0:
                        # Add small random noise to prevent identical play in similar positions
                        noise_factor = current_randomness * 0.2
                        for i in range(len(weights)):
                            if weights[i] > 0:
                                weights[i] += random.uniform(-noise_factor, noise_factor)
                        
                        # Normalize weights to ensure they're positive
                        min_weight = min(w for w in weights if w > 0) if any(w > 0 for w in weights) else 0
                        weights = [max(0.01, w - min_weight + 0.01) for w in weights]
                        
                        # Weighted random selection
                        total_weight = sum(weights)
                        if total_weight > 0:
                            rand_val = random.uniform(0, total_weight)
                            cumulative = 0
                            for i, weight in enumerate(weights):
                                cumulative += weight
                                if rand_val <= cumulative:
                                    best_move = move_evaluations[i][1]
                                    break
        
        # Record move for learning
        if self.learning_enabled:
            self.game_history.append((self._board_to_hash(board), best_move))
        
        return best_move
    
    def alpha_beta(self, board: List[List[Optional[str]]], player: str, depth: int, 
                   alpha: float, beta: float, maximising: bool, randomness: float = 0.0) -> Tuple[float, Optional[Tuple]]:
        """Alpha-beta search with pruning"""
        self.nodes_evaluated += 1
        
        # Base case: depth limit or terminal position
        if depth == 0:
            return self.evaluate_position(board, player, randomness), None
        
        # Check for terminal positions
        terminal_score = self.check_terminal_position(board, player)
        if terminal_score is not None:
            return terminal_score, None
        
        moves = get_legal_moves(board, player)
        if not moves:
            # No moves available - opponent wins by defence
            return -1000 if maximising else 1000, None
        
        # Add move ordering with some randomisation to prevent deterministic ordering
        def move_priority(move):
            base_priority = 0
            if move[0] == "SWAP":
                base_priority = 0  # Highest priority (captures)
            elif move[0] == "DROP" and move[2] in CENTER_SQUARES:
                base_priority = 1  # Second priority (centre drops)
            else:
                base_priority = 2  # Lower priority (other moves)
            
            # Add small random component to break ties
            if randomness > 0:
                random_component = random.uniform(0, randomness * 0.1)
                return base_priority + random_component
            return base_priority
        
        moves.sort(key=move_priority)
        
        best_move = None
        
        if maximising:
            max_eval = -math.inf
            for move in moves:
                new_board = apply_move(board, player, move)
                eval_score, _ = self.alpha_beta(new_board, get_opponent(player), 
                                              depth - 1, alpha, beta, False, randomness)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval, best_move
        else:
            min_eval = math.inf
            for move in moves:
                new_board = apply_move(board, player, move)
                eval_score, _ = self.alpha_beta(new_board, get_opponent(player), 
                                              depth - 1, alpha, beta, True, randomness)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval, best_move
    
    def check_terminal_position(self, board: List[List[Optional[str]]], player: str) -> Optional[float]:
        """Check if position is terminal and return evaluation if so"""
        # Check if current player has no tokens (MASSACRE)
        player_tokens = get_tokens(board, player)
        if not player_tokens:
            return -1000  # Current player loses
        
        opponent = get_opponent(player)
        opponent_tokens = get_tokens(board, opponent)
        if not opponent_tokens:
            return 1000  # Current player wins
        
        # Check EXILE conditions
        win, count = has_3_or_4_in_centre(board, player)
        if win:
            return 1000  # Current player wins
        
        opp_win, opp_count = has_3_or_4_in_centre(board, opponent)
        if opp_win:
            return -1000  # Opponent wins
        
        # Check OPPOSITION pattern (draw)
        if check_opposition_pattern(board):
            return 0  # Draw
        
        return None  # Not terminal
    
    def evaluate_position(self, board: List[List[Optional[str]]], player: str, randomness: float = 0.0) -> float:
        """
        Evaluate position from the perspective of the given player.
        Positive values favour the player, negative values favour opponent.
        """
        opponent = get_opponent(player)
        
        # Check for immediate terminal positions
        terminal_score = self.check_terminal_position(board, player)
        if terminal_score is not None:
            return terminal_score
        
        score = 0.0
        
        # 1. Token count advantage
        player_tokens = get_tokens(board, player)
        opponent_tokens = get_tokens(board, opponent)
        token_diff = len(player_tokens) - len(opponent_tokens)
        score += token_diff * 50  # Each token is worth 50 points
        
        # 2. Centre control (most important factor)
        player_center = sum(1 for (x, y) in CENTER_SQUARES if board[y][x] == player)
        opponent_center = sum(1 for (x, y) in CENTER_SQUARES if board[y][x] == opponent)
        center_diff = player_center - opponent_center
        score += center_diff * 100  # Each centre square is worth 100 points
        
        # 3. Mobility (number of legal moves)
        player_moves = len(get_legal_moves(board, player))
        opponent_moves = len(get_legal_moves(board, opponent))
        mobility_diff = player_moves - opponent_moves
        score += mobility_diff * 5  # Each additional move is worth 5 points
        
        # 4. Positional factors
        score += self.evaluate_positional_factors(board, player)
        
        # 5. Threat assessment
        score += self.evaluate_threats(board, player)
        
        # 6. Add small random component to break evaluation ties
        if randomness > 0:
            # Small random noise that doesn't affect major decisions
            noise_magnitude = min(5.0, randomness * 10.0)
            score += random.uniform(-noise_magnitude, noise_magnitude)
        
        return score
    
    def evaluate_positional_factors(self, board: List[List[Optional[str]]], player: str) -> float:
        """Evaluate positional factors like token placement and board control"""
        score = 0.0
        opponent = get_opponent(player)
        
        # Prefer tokens closer to centre
        for x, y in get_tokens(board, player):
            # Distance from centre of board (1.5, 1.5)
            center_distance = abs(x - 1.5) + abs(y - 1.5)
            score += (3 - center_distance) * 2  # Closer to centre is better
            
            # Bonus for being adjacent to centre squares
            for cx, cy in CENTER_SQUARES:
                if abs(x - cx) <= 1 and abs(y - cy) <= 1 and (x, y) not in CENTER_SQUARES:
                    score += 10
        
        # Penalty for opponent tokens near centre
        for x, y in get_tokens(board, opponent):
            for cx, cy in CENTER_SQUARES:
                if abs(x - cx) <= 1 and abs(y - cy) <= 1 and (x, y) not in CENTER_SQUARES:
                    score -= 8
        
        return score
    
    def evaluate_threats(self, board: List[List[Optional[str]]], player: str) -> float:
        """Evaluate immediate threats and opportunities"""
        score = 0.0
        opponent = get_opponent(player)
        
        # Check how close each player is to winning by exile
        player_center = sum(1 for (x, y) in CENTER_SQUARES if board[y][x] == player)
        opponent_center = sum(1 for (x, y) in CENTER_SQUARES if board[y][x] == opponent)
        
        # Big bonus/penalty for being close to exile win
        if player_center == 3:
            score += 500  # Three in centre - almost won!
        elif player_center == 2:
            score += 200  # Two in centre - one away from win
        elif opponent_center == 3:
            score -= 600  # Opponent almost wins - critical!
        elif opponent_center == 2:
            score -= 300  # Opponent close to winning - urgent to stop
        
        # Check for immediate capture threats
        player_moves = get_legal_moves(board, player)
        for move in player_moves:
            if move[0] == "SWAP":  # Swap moves capture opponent tokens
                score += 40  # Increased from 30
        
        opponent_moves = get_legal_moves(board, opponent)
        for move in opponent_moves:
            if move[0] == "SWAP":  # Opponent can capture our tokens
                score -= 35  # Increased from 25
        
        # Encourage aggressive play - bonus for having moves that threaten centre
        for move in player_moves:
            if move[0] == "DROP" and move[2] in CENTER_SQUARES:
                score += 25  # Bonus for moves that can place in centre
        
        # Penalty for having very few tokens (getting close to massacre)
        player_tokens = get_tokens(board, player)
        opponent_tokens = get_tokens(board, opponent)
        
        if len(player_tokens) <= 2:
            score -= 100 * (3 - len(player_tokens))  # Severe penalty for few tokens
        if len(opponent_tokens) <= 2:
            score += 80 * (3 - len(opponent_tokens))  # Bonus when opponent has few tokens
        
        return score

    def load_opening_book(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Load opening book from file"""
        if not os.path.exists(self.opening_book_file):
            return {}
        
        try:
            with open(self.opening_book_file, 'r') as f:
                data = json.load(f)
                return data
        except (json.JSONDecodeError, IOError):
            print(f"Warning: Could not load opening book from {self.opening_book_file}")
            return {}
    
    def save_opening_book(self):
        """Save opening book to file"""
        if not self.learning_enabled:
            return
            
        try:
            with open(self.opening_book_file, 'w') as f:
                json.dump(self.opening_book, f, indent=2)
        except IOError:
            print(f"Warning: Could not save opening book to {self.opening_book_file}")
    
    def check_opening_book(self, board: List[List[Optional[str]]], available_moves: List) -> Optional[Tuple]:
        """Check if we have a good move in our opening book"""
        board_hash = self._board_to_hash(board)
        
        if board_hash not in self.opening_book:
            return None
        
        position_data = self.opening_book[board_hash]
        best_move = None
        best_score = -1.0
        
        for move_str, stats in position_data.items():
            try:
                # Convert string back to move tuple
                move = eval(move_str)
                if move in available_moves and stats['total'] >= 3:  # Need at least 3 games
                    win_rate = (stats['wins'] + 0.5 * stats['draws']) / stats['total']
                    if win_rate > best_score:
                        best_score = win_rate
                        best_move = move
            except (ValueError, SyntaxError):
                continue
        
        # Use book move if it has decent performance (>40% score)
        if best_move and best_score > 0.4:
            return best_move
        
        return None
    
    def learn_from_game(self, result: str, player: str):
        """Update opening book and position evaluations based on game result"""
        if not self.learning_enabled:
            return
        
        self.games_learned += 1
        
        # Only learn from first 8 moves (opening)
        for i, (board_hash, move) in enumerate(self.game_history[:8]):
            if board_hash not in self.opening_book:
                self.opening_book[board_hash] = {}
            
            move_str = str(move)
            if move_str not in self.opening_book[board_hash]:
                self.opening_book[board_hash][move_str] = {
                    'wins': 0, 'draws': 0, 'losses': 0, 'total': 0
                }
            
            stats = self.opening_book[board_hash][move_str]
            stats['total'] += 1
            
            # Determine outcome from this player's perspective
            if result == 'win':  # Win
                stats['wins'] += 1
            elif result == 'draw' or result in ['BOREDOM', 'OPPOSITION']:  # Draw
                stats['draws'] += 1
            else:  # Loss (result == 'loss')
                stats['losses'] += 1
        
        # Clear history for next game
        self.game_history = []
        
        # Save opening book periodically
        if self.games_learned % 50 == 0:
            self.save_opening_book()
            # Show learning progress update
            total_positions = len(self.opening_book)
            total_moves = sum(len(moves) for moves in self.opening_book.values())
            print(f"📚 Learning book updated! Games: {self.games_learned}, Positions: {total_positions}, Moves: {total_moves}")
            
            if self.games_learned % 200 == 0:
                self.print_learning_stats()
    
    def print_learning_stats(self):
        """Print detailed statistics about what the AI has learned"""
        if not self.learning_enabled:
            return
        
        total_positions = len(self.opening_book)
        total_moves = sum(len(moves) for moves in self.opening_book.values())
        
        print(f"\n🧠 === AI Learning Statistics ===")
        print(f"📊 Games analysed: {self.games_learned}")
        print(f"🎯 Opening positions learned: {total_positions}")
        print(f"♟️  Opening moves in database: {total_moves}")
        
        # Find best and worst performing moves
        if total_moves > 0:
            all_moves = []
            for position, moves in self.opening_book.items():
                for move_str, stats in moves.items():
                    if stats['total'] >= 5:  # Only consider moves with enough data
                        win_rate = (stats['wins'] + 0.5 * stats['draws']) / stats['total']
                        all_moves.append((win_rate, move_str, stats['total']))
            
            if all_moves:
                all_moves.sort(reverse=True)
                best_move = all_moves[0]
                worst_move = all_moves[-1]
                
                print(f"🏆 Best performing move: {best_move[1]} (win rate: {best_move[0]:.1%}, games: {best_move[2]})")
                print(f"⚠️  Worst performing move: {worst_move[1]} (win rate: {worst_move[0]:.1%}, games: {worst_move[2]})")
        
        print("=" * 35)
    
    def _board_to_hash(self, board: List[List[Optional[str]]]) -> str:
        """Convert board to a string hash for learning"""
        # Convert None to '.' for cleaner representation
        board_str = ""
        for row in board:
            for cell in row:
                if cell is None:
                    board_str += "."
                else:
                    board_str += cell
        return board_str
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get current learning statistics"""
        total_positions = len(self.opening_book) if self.learning_enabled else 0
        total_moves = sum(len(moves) for moves in self.opening_book.values()) if self.learning_enabled else 0
        
        return {
            "learning_enabled": self.learning_enabled,
            "games_learned": self.games_learned,
            "positions_learned": total_positions,
            "moves_in_book": total_moves,
            "opening_book_file": self.opening_book_file,
            "randomness_setting": self.randomness
        }

def create_tactical_ai(search_depth: int = 8, learning_enabled: bool = False, opening_book_file: str = "opening_book.json", randomness: float = 0.15):
    """Factory function to create tactical AI strategy
    
    Args:
        search_depth: Maximum search depth for alpha-beta
        learning_enabled: Whether to enable learning capabilities
        opening_book_file: File to store/load opening book
        randomness: Amount of randomisation (0.0 = deterministic, 0.5 = moderate, 1.0 = high)
    """
    ai = TacticalAI(search_depth, learning_enabled, opening_book_file, randomness)
    
    def tactical_strategy(board: List[List[Optional[str]]], player: str, moves: List) -> Tuple:
        return ai.get_move(board, player, moves)
    
    # Attach learning methods to the strategy function for external access
    tactical_strategy.ai = ai
    tactical_strategy.learn_from_game = ai.learn_from_game
    tactical_strategy.get_learning_stats = ai.get_learning_stats
    tactical_strategy.save_opening_book = ai.save_opening_book
    
    return tactical_strategy
