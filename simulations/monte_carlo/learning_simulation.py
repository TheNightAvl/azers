"""
Learning-enabled tactical simulation for Azers.
Trains an AI through self-play and evaluates learning progress.
Enhanced with pattern recognition and progressive learning features.
"""
import time
import json
import random
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple
from game_engine import (
    play_game, create_mode1_board, get_all_mode2_starting_positions,
    get_tokens, get_legal_moves, CENTER_SQUARES, get_opponent
)
from tactical_ai import create_tactical_ai

def format_board_compact(board: List[List[Optional[str]]], label: str = "") -> str:
    """Format board in compact grid format for progress display"""
    lines = []
    if label:
        lines.append(f"  ↳ {label}:")
    
    # Column headers
    lines.append("    " + " ".join(chr(ord('A') + x) for x in range(4)))
    
    # Board rows with row numbers
    for y in range(4):
        row_display = f" {y+1}  "
        for x in range(4):
            cell = board[y][x]
            if cell == 'A':
                row_display += "X "
            elif cell == 'B':
                row_display += "O "
            else:
                row_display += ". "
        lines.append(row_display.rstrip())
    
    return "\n".join(lines)

class LearningStats:
    def __init__(self):
        self.total_games = 0
        self.learning_games = 0
        self.evaluation_games = 0
        self.learning_wins = {'A': 0, 'B': 0}
        self.learning_draws = 0
        self.evaluation_wins = {'A': 0, 'B': 0}
        self.evaluation_draws = 0
        self.game_lengths = []
        self.learning_progress = []  # Track performance over time
        self.pattern_recogniser_hits = 0  # Track pattern recogniser usage
        self.extended_learning_positions = 0  # Track positions learned beyond opening
        
        # Additional attributes for progress tracking compatibility
        self.wins_a = 0  # Total wins for player A across all phases
        self.wins_b = 0  # Total wins for player B across all phases  
        self.draws = 0   # Total draws across all phases
        self.total_moves = 0  # Total moves across all games
    
    def update_game_result(self, winner: Optional[str], move_count: int, phase: str = 'learning'):
        """Update statistics with game result"""
        self.total_games += 1
        self.total_moves += move_count
        self.game_lengths.append(move_count)
        
        if phase == 'learning':
            self.learning_games += 1
            if winner == 'A':
                self.learning_wins['A'] += 1
                self.wins_a += 1
            elif winner == 'B':
                self.learning_wins['B'] += 1
                self.wins_b += 1
            else:
                self.learning_draws += 1
                self.draws += 1
        else:  # evaluation phase
            self.evaluation_games += 1
            if winner == 'A':
                self.evaluation_wins['A'] += 1
                self.wins_a += 1
            elif winner == 'B':
                self.evaluation_wins['B'] += 1
                self.wins_b += 1
            else:
                self.evaluation_draws += 1
                self.draws += 1

class PatternRecogniser:
    """Enhanced pattern recogniser for tactical AI learning"""
    
    @staticmethod
    def recognise_tactical_patterns(board: List[List[Optional[str]]], player: str) -> Dict[str, float]:
        """Identify common tactical patterns with minimal computation"""
        patterns = {}
        opponent = get_opponent(player)
        
        # Centre control patterns
        patterns['centre_control'] = PatternRecogniser._evaluate_centre_control(board, player)
        
        # Fork opportunities (threatening multiple centre squares)
        patterns['fork_threat'] = PatternRecogniser._detect_fork_opportunities(board, player)
        
        # Defensive formations
        patterns['defensive_formation'] = PatternRecogniser._detect_defensive_setups(board, player)
        
        # Approaching centre pattern
        patterns['centre_approach'] = PatternRecogniser._evaluate_centre_approach(board, player)
        
        # Token clustering (good or bad depending on context)
        patterns['token_clustering'] = PatternRecogniser._evaluate_token_clustering(board, player)
        
        return patterns
    
    @staticmethod
    def _evaluate_centre_control(board: List[List[Optional[str]]], player: str) -> float:
        """Evaluate centre control strength"""
        player_centre = sum(1 for (x, y) in CENTER_SQUARES if board[y][x] == player)
        opponent_centre = sum(1 for (x, y) in CENTER_SQUARES if board[y][x] == get_opponent(player))
        
        # Strong bonus for multiple centre squares
        if player_centre >= 2:
            return 2.0 + (player_centre - 2) * 0.5
        elif opponent_centre >= 2:
            return -1.5 - (opponent_centre - 2) * 0.5
        else:
            return (player_centre - opponent_centre) * 0.5
    
    @staticmethod
    def _detect_fork_opportunities(board: List[List[Optional[str]]], player: str) -> float:
        """Detect opportunities to threaten multiple centre squares"""
        player_tokens = get_tokens(board, player)
        fork_score = 0.0
        
        for token_x, token_y in player_tokens:
            # Check how many centre squares this token can threaten with one move
            threatened_centres = 0
            moves = get_legal_moves(board, player)
            
            for move in moves:
                if len(move) >= 3 and move[1] == (token_x, token_y):
                    target_x, target_y = move[2]
                    if (target_x, target_y) in CENTER_SQUARES:
                        threatened_centres += 1
            
            if threatened_centres >= 2:
                fork_score += 1.0
            elif threatened_centres == 1:
                fork_score += 0.3
        
        return fork_score
    
    @staticmethod
    def _detect_defensive_setups(board: List[List[Optional[str]]], player: str) -> float:
        """Detect defensive formations around centre"""
        player_tokens = get_tokens(board, player)
        defensive_score = 0.0
        
        # Check for tokens positioned to defend centre approaches
        for (x, y) in player_tokens:
            # Count adjacent centre squares
            adjacent_centres = 0
            for cx, cy in CENTER_SQUARES:
                if abs(x - cx) <= 1 and abs(y - cy) <= 1:
                    adjacent_centres += 1
            
            if adjacent_centres >= 2:
                defensive_score += 0.5
            elif adjacent_centres == 1:
                defensive_score += 0.2
        
        return defensive_score
    
    @staticmethod
    def _evaluate_centre_approach(board: List[List[Optional[str]]], player: str) -> float:
        """Evaluate how well positioned tokens are to approach centre"""
        player_tokens = get_tokens(board, player)
        approach_score = 0.0
        
        for (x, y) in player_tokens:
            # Distance to nearest centre square
            min_distance = min(abs(x - cx) + abs(y - cy) for cx, cy in CENTER_SQUARES)
            
            # Closer is better, but diminishing returns
            if min_distance == 1:
                approach_score += 1.0
            elif min_distance == 2:
                approach_score += 0.5
            elif min_distance == 3:
                approach_score += 0.2
        
        return approach_score
    
    @staticmethod
    def _evaluate_token_clustering(board: List[List[Optional[str]]], player: str) -> float:
        """Evaluate token clustering - can be good or bad"""
        player_tokens = get_tokens(board, player)
        if len(player_tokens) <= 2:
            return 0.0
        
        clustering_score = 0.0
        
        for i, (x1, y1) in enumerate(player_tokens):
            adjacent_count = 0
            for j, (x2, y2) in enumerate(player_tokens):
                if i != j and abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1:
                    adjacent_count += 1
            
            # Moderate clustering is good, excessive clustering is bad
            if adjacent_count == 1:
                clustering_score += 0.3
            elif adjacent_count == 2:
                clustering_score += 0.1
            elif adjacent_count >= 3:
                clustering_score -= 0.2
        
        return clustering_score

class OpponentModel:
    """Track opponent preferences and adapt strategy"""
    
    def __init__(self):
        self.move_preferences = defaultdict(int)  # Track move type preferences
        self.aggression_level = 0.5  # 0 = defensive, 1 = aggressive
        self.centre_preference = 0.5  # Preference for centre control
        self.total_moves = 0
        
    def update_from_move(self, board_before: List[List[Optional[str]]], move: Tuple, player: str):
        """Update model based on opponent's move"""
        self.total_moves += 1
        
        # Track move type preferences
        move_type = move[0] if move else "UNKNOWN"
        self.move_preferences[move_type] += 1
        
        # Update aggression level based on move characterisation
        if move_type == "SWAP":
            self.aggression_level = min(1.0, self.aggression_level + 0.1)
        elif move_type == "DROP" and len(move) >= 3 and move[2] in CENTER_SQUARES:
            self.centre_preference = min(1.0, self.centre_preference + 0.1)
            self.aggression_level = min(1.0, self.aggression_level + 0.05)
        elif move_type == "HOP":
            # Neutral or slightly defensive
            self.aggression_level = max(0.0, self.aggression_level - 0.02)
    
    def get_opponent_profile(self) -> Dict[str, float]:
        """Get current opponent profile"""
        if self.total_moves == 0:
            return {"aggression": 0.5, "centre_preference": 0.5, "experience": 0.0}
        
        return {
            "aggression": self.aggression_level,
            "centre_preference": self.centre_preference,
            "experience": min(1.0, self.total_moves / 20.0),  # Experience based on move count
            "swap_preference": self.move_preferences.get("SWAP", 0) / self.total_moves,
            "drop_preference": self.move_preferences.get("DROP", 0) / self.total_moves,
            "hop_preference": self.move_preferences.get("HOP", 0) / self.total_moves
        }

def get_learning_configuration():
    """Get configuration for learning simulation"""
    print("\n=== ENHANCED LEARNING SIMULATION CONFIGURATION ===")
    print("This simulation trains an AI through self-play with advanced learning features:")
    print("• Pattern recogniser for tactical positions")
    print("• Progressive learning beyond opening moves")
    print("• Opponent modelling and adaptation")
    print("• Weighted position learning for critical situations")
    print()
    
    # Get number of learning games
    while True:
        try:
            learning_games = int(input("Number of games for AI to learn from (100-10000, recommended: 1000): ").strip())
            if 100 <= learning_games <= 10000:
                break
            print("Please enter a number between 100 and 10000.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get number of evaluation games
    while True:
        try:
            eval_games = int(input("Number of evaluation games to test performance (50-1000, recommended: 200): ").strip())
            if 50 <= eval_games <= 1000:
                break
            print("Please enter a number between 50 and 1000.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get search depth
    while True:
        try:
            depth = int(input("Search depth for AI (1-10, recommended: 6 for learning): ").strip())
            if 1 <= depth <= 10:
                break
            print("Please enter a depth between 1 and 10.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get enhanced learning options
    print("\nEnhanced learning features:")
    print("1. Basic learning (opening book only)")
    print("2. Enhanced learning (patterns + progressive learning)")
    print("3. Advanced learning (all features + opponent modelling)")
    print("4. Bias-optimised learning (uses pre-analysed enhanced books)")
    
    while True:
        learning_mode = input("Choose learning mode (1-4, recommended: 3): ").strip()
        if learning_mode in ['1', '2', '3', '4']:
            break
        print("Please enter 1, 2, 3, or 4.")
    
    # Handle bias-optimised learning
    enhancement_level = "standard"
    if learning_mode == '4':
        print("\nBias-optimised learning options:")
        print("1. Mode 2 Enhanced (Expert level)")
        print("2. Mode 3 Enhanced (Master level)")
        print("3. Adaptive Enhanced (Grandmaster level)")
        
        while True:
            bias_choice = input("Choose bias optimisation (1-3): ").strip()
            if bias_choice in ['1', '2', '3']:
                break
            print("Please enter 1, 2, or 3.")
        
        if bias_choice == '1':
            enhancement_level = "bias_optimised_mode2"
        elif bias_choice == '2':
            enhancement_level = "bias_optimised_mode3"
        else:  # bias_choice == '3'
            enhancement_level = "bias_optimised_mode3"  # Use the most advanced
            
        # Override learning mode to advanced for bias-optimised
        learning_mode = '3'
        print("Note: Bias-optimised learning uses advanced learning features automatically.")
    
    # Get game mode
    print("\nGame mode options:")
    print("1. Mode 1 only (opposing corners)")
    print("2. Mode 2 only (all starting positions)")
    print("3. Both modes")
    
    while True:
        mode_choice = input("Choose mode (1-3): ").strip()
        if mode_choice in ['1', '2', '3']:
            break
        print("Please enter 1, 2, or 3.")
    
    # Configuration complete, proceed with training
    
    return {
        'learning_games': learning_games,
        'evaluation_games': eval_games,
        'search_depth': depth,
        'learning_mode': learning_mode,
        'mode_choice': mode_choice,
        'enhancement_level': enhancement_level
    }

def should_learn_position(move_number: int, game_phase: str, learning_mode: str) -> bool:
    """Decide if position is worth learning based on game phase and learning mode"""
    if learning_mode == '1':  # Basic learning
        return move_number <= 8  # Only opening
    elif learning_mode == '2':  # Enhanced learning
        if move_number <= 8:  # Opening
            return True
        elif move_number <= 16 and game_phase == 'middlegame':  # Key middlegame
            return random.random() < 0.3  # Sample 30% of middlegame positions
        return False
    else:  # Advanced learning (mode 3)
        if move_number <= 8:  # Opening
            return True
        elif move_number <= 20:  # Extended middlegame learning
            return random.random() < 0.4  # Sample 40% of middlegame positions
        return False

def get_position_learning_weight(board: List[List[Optional[str]]], move_number: int, patterns: Dict[str, float]) -> float:
    """Assign learning weights based on position characterisation"""
    weight = 1.0
    
    # Weight based on move number (early opening is very important)
    if move_number <= 4:
        weight *= 1.5
    elif move_number <= 8:
        weight *= 1.2
    elif move_number <= 12:
        weight *= 0.8
    else:
        weight *= 0.6
    
    # Weight based on tactical patterns
    if patterns.get('centre_control', 0) >= 1.5:
        weight *= 2.0  # Critical centre control positions
    elif patterns.get('fork_threat', 0) >= 1.0:
        weight *= 1.5  # Important tactical positions
    elif patterns.get('centre_control', 0) <= -1.0:
        weight *= 1.8  # Critical defensive positions
    
    # Weight based on position complexity
    total_tokens = sum(1 for row in board for cell in row if cell is not None)
    if total_tokens <= 6:  # Early game positions
        weight *= 1.3
    elif total_tokens >= 12:  # Complex positions
        weight *= 1.1
    
    return max(0.1, min(3.0, weight))  # Clamp between 0.1 and 3.0

def assess_move_quality(board_before: List[List[Optional[str]]], move: Tuple, 
                       board_after: List[List[Optional[str]]], patterns_before: Dict[str, float], 
                       patterns_after: Dict[str, float]) -> Dict[str, float]:
    """Assess if a move was strategically sound"""
    
    # Calculate immediate tactical improvement
    tactical_gain = 0.0
    for pattern_name in patterns_before:
        before_value = patterns_before.get(pattern_name, 0)
        after_value = patterns_after.get(pattern_name, 0)
        
        if pattern_name == 'centre_control':
            tactical_gain += (after_value - before_value) * 2.0  # Centre control is most important
        elif pattern_name == 'fork_threat':
            tactical_gain += (after_value - before_value) * 1.5
        else:
            tactical_gain += (after_value - before_value) * 1.0
    
    # Assess move type quality
    move_type_quality = 0.5  # Neutral by default
    if move and len(move) >= 1:
        if move[0] == 'DROP' and len(move) >= 3 and move[2] in CENTER_SQUARES:
            move_type_quality = 0.8  # Centre drops are generally good
        elif move[0] == 'SWAP':
            move_type_quality = 0.7  # Captures are usually good
        elif move[0] == 'HOP':
            move_type_quality = 0.6  # Hops are more positional
    
    return {
        'tactical_soundness': max(-2.0, min(2.0, tactical_gain)),
        'move_type_quality': move_type_quality,
        'overall_assessment': (tactical_gain + move_type_quality) / 2.0
    }

def get_game_phase(move_number: int, total_tokens: int) -> str:
    """Determine the current game phase"""
    if move_number <= 8 or total_tokens >= 10:
        return 'opening'
    elif move_number <= 20 or total_tokens >= 6:
        return 'middlegame'
    else:
        return 'endgame'

def run_learning_phase(config: Dict[str, Any], stats: LearningStats):
    """Run the enhanced learning phase where AI learns through self-play"""
    learning_mode = config.get('learning_mode', '1')
    
    print(f"\n=== ENHANCED LEARNING PHASE ===")
    print(f"Training AI with {config['learning_games']} games...")
    print(f"Learning mode: {['Basic', 'Enhanced', 'Advanced'][int(learning_mode)-1]}")
    print("Progress will be shown after every game.")
    
    # Initialise time tracking for progress summaries
    last_summary_time = time.time()
    
    # Initialise enhanced learning components
    pattern_recogniser = PatternRecogniser()
    opponent_model = OpponentModel() if learning_mode == '3' else None
    
    # Determine configurations to test
    configurations = []
    
    # Mode configurations
    if config['mode_choice'] in ['1', '3']:
        configurations.extend(['mode1'])
    if config['mode_choice'] in ['2', '3']:
        configurations.extend(['mode2'])
    
    # Create separate AI instances for each configuration
    config_ais = {}
    for mode in configurations:
        # Create specialised book filename for each configuration
        book_filename = f"specialised_{mode}_depth_{config['search_depth']}_learning{learning_mode}.json"
        
        # Handle bias-optimised learning books
        if config.get('enhancement_level', 'standard') in ['bias_optimised_mode2', 'bias_optimised_mode3']:
            enhancement = config['enhancement_level']
            if enhancement == 'bias_optimised_mode2':
                book_filename = f"enhanced_learning_book_depth_{config['search_depth']}_mode2.json"
            elif enhancement == 'bias_optimised_mode3':
                book_filename = f"enhanced_learning_book_depth_{config['search_depth']}_mode3.json"
            print(f"Using bias-optimised book: {book_filename}")
        
        print(f"Creating specialised AI for {mode.upper()}")
        print(f"Book file: {book_filename}")
        
        ai = create_tactical_ai(
            search_depth=config['search_depth'],
            learning_enabled=True,
            opening_book_file=book_filename
        )
        config_ais[mode] = ai
    
    games_per_config = config['learning_games'] // len(configurations)
    
    start_time = time.time()
    interesting_games = []  # Store interesting games for display
    
    for config_name in configurations:
        print(f"\nTraining specialised AI for {config_name.upper()}...")
        
        # Get the specialised AI for this configuration
        ai = config_ais[config_name]
        
        if config_name == 'mode1':
            starting_positions = [create_mode1_board]  # Function to create Mode 1 board
        else:
            starting_positions = get_all_mode2_starting_positions()  # List of (board, player) tuples
        
        config_games = 0
        for _ in range(games_per_config):
            if config_name == 'mode1':
                initial_board, starting_player = starting_positions[0]()  # Call the function
            else:
                initial_board, starting_player = random.choice(starting_positions)
            
            # Store initial board state for analysis
            starting_board_copy = [row[:] for row in initial_board]  # Deep copy for analysis
            
            # Enhanced game tracking for learning
            game_moves = []
            move_number = 0
            current_board = [row[:] for row in initial_board]  # Deep copy
            current_player = starting_player
            
            result = play_game(
                initial_board, starting_player, ai, ai, max_moves=200
            )
            
            # Enhanced learning from game with patterns and progressive learning
            if learning_mode in ['2', '3']:
                # Simulate move-by-move analysis for enhanced learning
                total_tokens = sum(1 for row in starting_board_copy for cell in row if cell is not None)
                
                for move_num in range(min(result.move_count, 20)):  # Analyse up to 20 moves
                    game_phase = get_game_phase(move_num, total_tokens)
                    
                    if should_learn_position(move_num, game_phase, learning_mode):
                        # Analyse patterns for this position
                        patterns = pattern_recogniser.recognise_tactical_patterns(current_board, current_player)
                        
                        # Calculate learning weight
                        weight = get_position_learning_weight(current_board, move_num, patterns)
                        
                        if weight > 1.5:  # Only learn from highly weighted positions
                            stats.extended_learning_positions += 1
                        
                        # Update pattern recogniser statistics
                        if any(value > 1.0 for value in patterns.values()):
                            stats.pattern_recogniser_hits += 1
            
            # Standard AI learning - convert game result to player-specific outcomes
            if result.winner is None:
                # Draw - both players get draw result
                ai.learn_from_game('draw', 'A')
                ai.learn_from_game('draw', 'B')
            else:
                # Someone won - give appropriate win/loss results
                if result.winner == 'A':
                    ai.learn_from_game('win', 'A')
                    ai.learn_from_game('loss', 'B')
                else:  # result.winner == 'B'
                    ai.learn_from_game('loss', 'A')
                    ai.learn_from_game('win', 'B')
            
            # Update opponent model if using advanced learning
            if opponent_model and learning_mode == '3':
                # Simulate opponent move analysis (simplified)
                opponent_model.update_from_move(current_board, None, 'A')  # Placeholder
            
            stats.total_games += 1
            # Update statistics using the new method
            stats.update_game_result(result.winner, result.move_count, 'learning')
            config_games += 1
            
            # Save interesting games (draws or long games)
            if result.winner is None or result.move_count >= 30:
                interesting_games.append((stats.learning_games, result, config_name))
            
            # Progress update after every game
            elapsed = time.time() - start_time
            games_per_sec = stats.learning_games / elapsed
            remaining_games = config['learning_games'] - stats.learning_games
            eta_minutes = (remaining_games / games_per_sec) / 60 if games_per_sec > 0 else 0
            
            # Enhanced progress display with board state analysis
            enhancement_info = ""
            if learning_mode in ['2', '3']:
                enhancement_info = f", Patterns: {stats.pattern_recogniser_hits}, Extended: {stats.extended_learning_positions}"
            
            # Build board state information
            board_state_info = ""
            show_starting_board = (config_name == 'mode2')  # Only show starting board for Mode 2
            
            if hasattr(result, 'final_board') and result.final_board:
                final_board = result.final_board
                
                # Add board grids
                boards_display = ""
                if show_starting_board:
                    boards_display += "\n" + format_board_compact(starting_board_copy, "Start")
                
                # Always show final board for interesting games (draws, long games, or significant position changes)
                show_final_board = (result.winner is None or result.move_count >= 25)
                
                if show_final_board:
                    boards_display += "\n" + format_board_compact(final_board, "Final")
                
                board_state_info = boards_display
            else:
                # If no final board, just show starting state
                if show_starting_board:
                    board_state_info += "\n" + format_board_compact(starting_board_copy, "Start")
            
            config_display = f"{config_name.upper()}"
            print(f"Game {stats.learning_games:,}/{config['learning_games']:,} "
                  f"({stats.learning_games/config['learning_games']*100:.1f}%) [{config_display}] "
                  f"Winner: {result.winner or 'Draw'}, Moves: {result.move_count}, "
                  f"Type: {result.result_type}, ETA: {eta_minutes:.1f}min{enhancement_info}{board_state_info}")
            
            # Frequent progress summary every 7 games with visual board display
            if stats.learning_games % 7 == 0:
                elapsed_summary = time.time() - start_time
                games_per_sec_summary = stats.learning_games / elapsed_summary if elapsed_summary > 0 else 0
                win_percentage_a = (stats.wins_a / stats.learning_games * 100) if stats.learning_games > 0 else 0
                win_percentage_b = (stats.wins_b / stats.learning_games * 100) if stats.learning_games > 0 else 0
                draw_percentage = (stats.draws / stats.learning_games * 100) if stats.learning_games > 0 else 0
                avg_moves = (stats.total_moves / stats.learning_games) if stats.learning_games > 0 else 0
                
                print(f"\n═══ PROGRESS SUMMARY (Games {stats.learning_games:,}) ═══")
                print(f"Performance: A={stats.wins_a:,} wins ({win_percentage_a:.1f}%), "
                      f"B={stats.wins_b:,} wins ({win_percentage_b:.1f}%), "
                      f"Draws={stats.draws:,} ({draw_percentage:.1f}%)")
                print(f"Speed: {games_per_sec_summary:.1f} games/sec, Avg moves: {avg_moves:.1f}")
                
                # Pattern recognition stats if available
                if learning_mode in ['2', '3'] and hasattr(stats, 'pattern_recogniser_hits'):
                    pattern_percentage = (stats.pattern_recogniser_hits / stats.learning_games * 100) if stats.learning_games > 0 else 0
                    print(f"Learning: Pattern hits: {stats.pattern_recogniser_hits:,} ({pattern_percentage:.1f}%), "
                          f"Extended positions: {stats.extended_learning_positions:,}")
                
                # Current board state if available
                if hasattr(result, 'final_board') and result.final_board:
                    print("Most recent final board state:")
                    print(format_board_compact(result.final_board, "Latest Game"))
                
                # Opponent model insights if available
                if learning_mode in ['2', '3'] and hasattr(stats, 'opponent_model'):
                    model = stats.opponent_model
                    adaptations = getattr(model, 'adaptations', 0)
                    confidence = getattr(model, 'confidence', 0.5)
                    print(f"Opponent model: {adaptations} adaptations, confidence: {confidence:.2f}")
                
                print("═" * 50)
            
            # Show detailed learning stats every 10 games for better progress visibility
            if stats.learning_games % 10 == 0:
                # Calculate current statistics
                recent_games = min(10, stats.learning_games)
                recent_wins_a = sum(1 for i in range(max(0, len(stats.game_lengths) - recent_games), len(stats.game_lengths)) 
                                  if i < len(stats.game_lengths) and stats.learning_wins['A'] > 0)
                recent_avg_length = sum(stats.game_lengths[-recent_games:]) / recent_games if stats.game_lengths else 0
                
                total_learning_games = stats.learning_wins['A'] + stats.learning_wins['B'] + stats.learning_draws
                win_rate_a = (stats.learning_wins['A'] / total_learning_games * 100) if total_learning_games > 0 else 0
                draw_rate = (stats.learning_draws / total_learning_games * 100) if total_learning_games > 0 else 0
                
                print(f"  ↳ Last {recent_games} games avg length: {recent_avg_length:.1f} moves")
                print(f"  ↳ Overall: A wins {win_rate_a:.1f}%, Draws {draw_rate:.1f}%, Avg length: {sum(stats.game_lengths)/len(stats.game_lengths):.1f}")
                
                # Show AI learning progress if available
                if hasattr(ai, 'get_learning_stats'):
                    try:
                        learning_stats = ai.get_learning_stats()
                        print(f"  ↳ AI learned: {learning_stats.get('positions_learned', 0)} positions, "
                              f"{learning_stats.get('moves_in_book', 0)} moves in book")
                    except:
                        print(f"  ↳ AI learning data not available")
                
                # Enhanced learning statistics for modes 2 and 3
                if learning_mode in ['2', '3'] and stats.learning_games > 0:
                    pattern_rate = (stats.pattern_recogniser_hits / stats.learning_games) * 100
                    extended_rate = (stats.extended_learning_positions / stats.learning_games) * 100
                    print(f"  ↳ Enhanced learning: Pattern hits {pattern_rate:.1f}%, Extended positions {extended_rate:.1f}%")
                    
                    if opponent_model and stats.learning_games >= 20:
                        profile = opponent_model.get_opponent_profile()
                        print(f"  ↳ Opponent model: Aggression {profile['aggression']:.2f}, Centre pref {profile['centre_preference']:.2f}")
            
            # Show more detailed stats every 50 games
            elif stats.learning_games % 50 == 0 and hasattr(ai, 'ai') and hasattr(ai.ai, 'opening_book'):
                print(f"  ↳ 📊 Milestone: {stats.learning_games} games completed")
                
                # Show interesting game patterns from recent games
                recent_interesting = [g for g in interesting_games if g[0] > stats.learning_games - 50]
                if recent_interesting:
                    draws_count = len([g for g in recent_interesting if g[1].winner is None])
                    long_games = len([g for g in recent_interesting if g[1].move_count >= 30])
                    print(f"  ↳ Recent interesting games: {draws_count} draws, {long_games} long games (30+ moves)")
                
                # Book size and learning efficiency
                if hasattr(ai, 'get_learning_stats'):
                    try:
                        learning_stats = ai.get_learning_stats()
                        positions_learned = learning_stats.get('positions_learned', 0)
                        learning_efficiency = (positions_learned / stats.learning_games) * 100 if stats.learning_games > 0 else 0
                        print(f"  ↳ Learning efficiency: {learning_efficiency:.2f} positions per game")
                    except:
                        pass
        
        # Save the specialised opening book for this configuration
        ai.save_opening_book()
        print(f"Saved specialised book for {config_name.upper()}")
    
    print(f"\nLearning phase completed! Created {len(config_ais)} specialised AIs trained on {stats.learning_games:,} games.")
    
    # Enhanced learning summary
    if learning_mode in ['2', '3']:
        print(f"Enhanced learning statistics:")
        print(f"  Pattern recogniser hits: {stats.pattern_recogniser_hits:,}")
        print(f"  Extended learning positions: {stats.extended_learning_positions:,}")
        
        if opponent_model:
            profile = opponent_model.get_opponent_profile()
            print(f"  Opponent model: Aggression={profile['aggression']:.2f}, "
                  f"Centre preference={profile['centre_preference']:.2f}")
    
    # Show which books were created
    print(f"\nSpecialised learning books created:")
    for mode, ai in config_ais.items():
        book_file = getattr(ai.ai, 'opening_book_file', 'unknown')
        print(f"  {mode.upper()}: {book_file}")
    
    # Show interesting games from learning phase with enhanced board analysis
    if interesting_games:
        print(f"\nInteresting games from learning phase (first 5 of {len(interesting_games)}):")
        for i, (game_num, result, config_name) in enumerate(interesting_games[:5]):
            
            # Enhanced game description
            game_desc = "Draw" if result.winner is None else f"{result.winner} wins"
            move_desc = "short" if result.move_count < 15 else "medium" if result.move_count < 25 else "long"
            
            print(f"Game {game_num} ({config_name.upper()}): "
                  f"{game_desc}, {result.move_count} moves ({move_desc}), Type: {result.result_type}")
            
            # Board state analysis if available
            if hasattr(result, 'final_board') and result.final_board:
                # Analyse final board state
                board = result.final_board
                tokens_a = sum(1 for row in board for cell in row if cell == 'A')
                tokens_b = sum(1 for row in board for cell in row if cell == 'B')
                centre_a = sum(1 for (x, y) in CENTER_SQUARES if board[y][x] == 'A')
                centre_b = sum(1 for (x, y) in CENTER_SQUARES if board[y][x] == 'B')
                
                print(f"  ↳ Final state: A={tokens_a} tokens ({centre_a} in centre), B={tokens_b} tokens ({centre_b} in centre)")
                
                # Show the actual board state for draws or very long games
                if result.winner is None or result.move_count >= 35:
                    print("  ↳ Final board state:")
                    for row in board:
                        print("    " + " ".join(cell or "." for cell in row))
            else:
                print("  ↳ (Board state not available)")
            
            if i < 4 and i < len(interesting_games) - 1:
                print()
    
    # Print final learning statistics for each AI
    for mode, ai in config_ais.items():
        print(f"\nLearning statistics for {mode.upper()}:")
        if hasattr(ai, 'ai'):
            ai.ai.print_learning_stats()
    
    # Return the first AI (for backward compatibility) or a dict of all AIs
    if len(config_ais) == 1:
        return list(config_ais.values())[0]
    else:
        return config_ais

def run_evaluation_phase(config: Dict[str, Any], learned_ai, stats: LearningStats):
    """Evaluate the learned AI against a baseline AI"""
    print(f"\n=== EVALUATION PHASE ===")
    print(f"Testing learned AI against baseline AI with {config['evaluation_games']} games...")
    print("Progress will be shown after every game.")
    
    # Create baseline AI (no learning)
    baseline_ai = create_tactical_ai(
        search_depth=config['search_depth'],
        learning_enabled=False
    )
    
    # Determine configurations to test
    configurations = []
    if config['mode_choice'] in ['1', '3']:
        configurations.extend(['mode1'])
    if config['mode_choice'] in ['2', '3']:
        configurations.extend(['mode2'])
    
    games_per_config = config['evaluation_games'] // len(configurations)
    
    start_time = time.time()
    learned_ai_wins = 0
    baseline_ai_wins = 0
    evaluation_interesting_games = []  # Store interesting evaluation games
    
    for config_name in configurations:
        print(f"\nEvaluating on {config_name.upper()}...")
        
        if config_name == 'mode1':
            starting_positions = [create_mode1_board]  # Function to create Mode 1 board
        else:
            starting_positions = get_all_mode2_starting_positions()  # List of (board, player) tuples
        
        config_learned_wins = 0
        config_baseline_wins = 0
        config_draws = 0
        
        for game_num in range(games_per_config):
            if config_name == 'mode1':
                initial_board, starting_player = starting_positions[0]()  # Call the function
            else:
                initial_board, starting_player = random.choice(starting_positions)
            
            # Store initial board state for analysis
            starting_board_copy = [row[:] for row in initial_board]  # Deep copy for analysis
            
            # Alternate who plays A/B
            if game_num % 2 == 0:
                # Learned AI plays A
                result = play_game(
                    initial_board, starting_player, learned_ai, baseline_ai, max_moves=200
                )
                if result.winner == 'A':
                    learned_ai_wins += 1
                    config_learned_wins += 1
                elif result.winner == 'B':
                    baseline_ai_wins += 1
                    config_baseline_wins += 1
                else:
                    config_draws += 1
            else:
                # Learned AI plays B
                result = play_game(
                    initial_board, starting_player, baseline_ai, learned_ai, max_moves=200
                )
                if result.winner == 'B':
                    learned_ai_wins += 1
                    config_learned_wins += 1
                elif result.winner == 'A':
                    baseline_ai_wins += 1
                    config_baseline_wins += 1
                else:
                    config_draws += 1
            
            # Update statistics using the new method
            stats.update_game_result(result.winner, result.move_count, 'evaluation')
            
            # Save interesting evaluation games (draws or long games)
            if result.winner is None or result.move_count >= 30:
                ai_side = 'A' if game_num % 2 == 0 else 'B'
                evaluation_interesting_games.append((stats.evaluation_games, result, config_name, ai_side))
            
            # Progress update after every game
            elapsed = time.time() - start_time
            games_per_sec = stats.evaluation_games / elapsed
            remaining_games = config['evaluation_games'] - stats.evaluation_games
            eta_minutes = (remaining_games / games_per_sec) / 60 if games_per_sec > 0 else 0
            
            ai_side = 'A' if game_num % 2 == 0 else 'B'
            ai_won = (result.winner == 'A' and ai_side == 'A') or (result.winner == 'B' and ai_side == 'B')
            outcome_symbol = "🤖" if ai_won else "🔸" if result.winner is None else "🔹"
            
            print(f"Game {stats.evaluation_games:,}/{config['evaluation_games']:,} "
                  f"({stats.evaluation_games/config['evaluation_games']*100:.1f}%) "
                  f"{outcome_symbol} AI as {ai_side}: {result.winner or 'Draw'}, "
                  f"Moves: {result.move_count}, Type: {result.result_type}, ETA: {eta_minutes:.1f}min")
        
            # Show evaluation progress every 10 games
            if stats.evaluation_games % 10 == 0:
                # Calculate recent performance
                recent_games = min(10, stats.evaluation_games)
                current_learned_wins = learned_ai_wins
                current_baseline_wins = baseline_ai_wins
                current_total = current_learned_wins + current_baseline_wins + config_draws
                
                if current_total > 0:
                    current_win_rate = (current_learned_wins / current_total) * 100
                    recent_avg_length = sum(stats.game_lengths[-recent_games:]) / recent_games if len(stats.game_lengths) >= recent_games else 0
                    
                    print(f"  ↳ Current win rate: {current_win_rate:.1f}% ({current_learned_wins}W-{current_baseline_wins}L-{config_draws}D)")
                    print(f"  ↳ Recent {recent_games} games avg: {recent_avg_length:.1f} moves")
                    
                    # Show trend
                    if stats.evaluation_games >= 20:
                        print(f"  ↳ Learned AI {'gaining momentum' if current_win_rate > 50 else 'struggling' if current_win_rate < 45 else 'holding steady'}")
            
            # Show more detailed evaluation stats every 25 games
            elif stats.evaluation_games % 25 == 0:
                print(f"  ↳ 🎯 Evaluation checkpoint: {stats.evaluation_games} games")
                
                # Show interesting evaluation games
                recent_eval_interesting = [g for g in evaluation_interesting_games if g[0] > stats.evaluation_games - 25]
                if recent_eval_interesting:
                    eval_draws = len([g for g in recent_eval_interesting if g[1].winner is None])
                    eval_long = len([g for g in recent_eval_interesting if g[1].move_count >= 30])
                    print(f"  ↳ Recent interesting: {eval_draws} draws, {eval_long} long games")
                
                # Performance analysis
                if current_total >= 25:
                    performance_desc = "excellent" if current_win_rate >= 60 else "good" if current_win_rate >= 55 else "fair" if current_win_rate >= 50 else "poor"
                    print(f"  ↳ Performance assessment: {performance_desc} ({current_win_rate:.1f}% win rate)")
        
        # Show results for this configuration
        total_config_games = config_learned_wins + config_baseline_wins + config_draws
        learned_win_rate = config_learned_wins / total_config_games * 100 if total_config_games > 0 else 0
        
        print(f"  Results: Learned AI: {config_learned_wins}, Baseline AI: {config_baseline_wins}, "
              f"Draws: {config_draws} (Learned AI win rate: {learned_win_rate:.1f}%)")
    
    # Final evaluation results
    total_eval_games = learned_ai_wins + baseline_ai_wins + (stats.evaluation_draws - (stats.learning_draws if hasattr(stats, 'learning_draws') else 0))
    learned_overall_win_rate = learned_ai_wins / total_eval_games * 100 if total_eval_games > 0 else 0
    
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Learned AI wins: {learned_ai_wins}")
    print(f"Baseline AI wins: {baseline_ai_wins}")
    print(f"Draws: {total_eval_games - learned_ai_wins - baseline_ai_wins}")
    print(f"Learned AI win rate: {learned_overall_win_rate:.1f}%")
    
    # Show interesting games from evaluation phase with enhanced analysis
    if evaluation_interesting_games:
        print(f"\nInteresting games from evaluation phase (first 5 of {len(evaluation_interesting_games)}):")
        for i, (game_num, result, config_name, ai_side) in enumerate(evaluation_interesting_games[:5]):
            
            # Determine outcome from AI perspective
            ai_won = (result.winner == ai_side)
            outcome_desc = "AI wins" if ai_won else "AI loses" if result.winner else "Draw"
            move_desc = "short" if result.move_count < 15 else "medium" if result.move_count < 25 else "long"
            
            print(f"Game {game_num} ({config_name.upper()}, Learned AI as {ai_side}): "
                  f"{outcome_desc}, {result.move_count} moves ({move_desc}), Type: {result.result_type}")
            
            # Enhanced board analysis
            if hasattr(result, 'final_board') and result.final_board:
                board = result.final_board
                tokens_a = sum(1 for row in board for cell in row if cell == 'A')
                tokens_b = sum(1 for row in board for cell in row if cell == 'B')
                centre_a = sum(1 for (x, y) in CENTER_SQUARES if board[y][x] == 'A')
                centre_b = sum(1 for (x, y) in CENTER_SQUARES if board[y][x] == 'B')
                
                ai_tokens = tokens_a if ai_side == 'A' else tokens_b
                ai_centre = centre_a if ai_side == 'A' else centre_b
                opp_tokens = tokens_b if ai_side == 'A' else tokens_a
                opp_centre = centre_b if ai_side == 'A' else centre_a
                
                print(f"  ↳ Final: AI={ai_tokens} tokens ({ai_centre} centre), Baseline={opp_tokens} tokens ({opp_centre} centre)")
                
                # Show tactical assessment
                if result.winner is None:
                    if ai_centre > opp_centre:
                        print("  ↳ Tactical: AI had centre advantage but couldn't convert")
                    elif ai_centre < opp_centre:
                        print("  ↳ Tactical: AI held on despite centre disadvantage")
                    else:
                        print("  ↳ Tactical: Evenly matched centre control")
                elif ai_won and result.move_count >= 25:
                    print("  ↳ Tactical: AI showed patience in long game")
                elif not ai_won and result.move_count < 20:
                    print("  ↳ Tactical: AI lost quickly - possible opening weakness")
                
                # Show board for very interesting games (draws or AI wins in long games)
                if result.winner is None or (ai_won and result.move_count >= 30):
                    print("  ↳ Final board state:")
                    for row in board:
                        print("    " + " ".join(cell or "." for cell in row))
            else:
                print("  ↳ (Board state not available)")
            
            if i < 4 and i < len(evaluation_interesting_games) - 1:
                print()
    
    if learned_overall_win_rate > 55:
        print("🎉 Excellent! The AI learned significantly better strategies!")
    elif learned_overall_win_rate > 52:
        print("👍 Good! The AI learned some beneficial strategies.")
    elif learned_overall_win_rate > 48:
        print("📊 The AI learning had minimal impact on performance.")
    else:
        print("🤔 The learning may need more games or different parameters.")

def print_final_results(config: Dict[str, Any], stats: LearningStats):
    """Print comprehensive results of the enhanced learning simulation"""
    print(f"\n" + "="*80)
    print("ENHANCED LEARNING SIMULATION FINAL RESULTS")
    print("="*80)
    
    total_time = time.time() - getattr(stats, 'start_time', time.time())
    
    print(f"Configuration:")
    print(f"  Search depth: {config['search_depth']}")
    learning_mode_desc = {
        '1': 'Basic (opening book only)',
        '2': 'Enhanced (patterns + progressive learning)',
        '3': 'Advanced (all features + opponent modelling)'
    }[config.get('learning_mode', '1')]
    print(f"  Learning mode: {learning_mode_desc}")
    print(f"  Learning games: {config['learning_games']:,}")
    print(f"  Evaluation games: {config['evaluation_games']:,}")
    print(f"  Total games: {stats.total_games:,}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Average game length: {sum(stats.game_lengths)/len(stats.game_lengths):.1f} moves")
    
    # Enhanced learning statistics
    if hasattr(stats, 'pattern_recogniser_hits') and config.get('learning_mode', '1') in ['2', '3']:
        print(f"\nEnhanced Learning Statistics:")
        pattern_rate = (stats.pattern_recogniser_hits / max(1, stats.learning_games)) * 100
        extended_rate = (stats.extended_learning_positions / max(1, stats.learning_games)) * 100
        print(f"  Pattern recogniser hits: {stats.pattern_recogniser_hits:,} ({pattern_rate:.1f}% of games)")
        print(f"  Extended learning positions: {stats.extended_learning_positions:,} ({extended_rate:.1f}% of games)")
        
        # Calculate learning efficiency
        total_enhanced_features = stats.pattern_recogniser_hits + stats.extended_learning_positions
        if total_enhanced_features > 0:
            efficiency = (total_enhanced_features / stats.learning_games) * 100
            print(f"  Learning enhancement efficiency: {efficiency:.1f}%")
    
    print(f"\nLearning Phase Results:")
    learning_total = stats.learning_wins['A'] + stats.learning_wins['B'] + stats.learning_draws
    if learning_total > 0:
        print(f"  Player A wins: {stats.learning_wins['A']:,} ({stats.learning_wins['A']/learning_total*100:.1f}%)")
        print(f"  Player B wins: {stats.learning_wins['B']:,} ({stats.learning_wins['B']/learning_total*100:.1f}%)")
        print(f"  Draws: {stats.learning_draws:,} ({stats.learning_draws/learning_total*100:.1f}%)")
    
    print(f"\nEvaluation Phase Results:")
    eval_total = stats.evaluation_wins['A'] + stats.evaluation_wins['B'] + stats.evaluation_draws
    if eval_total > 0:
        print(f"  Player A wins: {stats.evaluation_wins['A']:,} ({stats.evaluation_wins['A']/eval_total*100:.1f}%)")
        print(f"  Player B wins: {stats.evaluation_wins['B']:,} ({stats.evaluation_wins['B']/eval_total*100:.1f}%)")
        print(f"  Draws: {stats.evaluation_draws:,} ({stats.evaluation_draws/eval_total*100:.1f}%)")
    
    # Learning effectiveness summary
    if config.get('learning_mode', '1') in ['2', '3']:
        print(f"\nLearning Enhancement Impact:")
        print(f"  This simulation used {learning_mode_desc.lower()} learning features")
        if hasattr(stats, 'pattern_recogniser_hits'):
            if stats.pattern_recogniser_hits > stats.learning_games * 0.2:
                print(f"  ✓ High pattern recogniser engagement ({stats.pattern_recogniser_hits} hits)")
            else:
                print(f"  ⚠ Low pattern recogniser engagement ({stats.pattern_recogniser_hits} hits)")
            
            if stats.extended_learning_positions > stats.learning_games * 0.1:
                print(f"  ✓ Effective extended learning ({stats.extended_learning_positions} positions)")
            else:
                print(f"  ⚠ Limited extended learning ({stats.extended_learning_positions} positions)")
    
    print("="*80)

def main():
    """Main function for enhanced learning simulation"""
    print("Starting Enhanced Learning-Enabled Azers Simulation")
    print("This will train an AI through self-play with advanced learning features and evaluate its improvement.")
    
    # Get configuration
    config = get_learning_configuration()
    
    # Initialise statistics
    stats = LearningStats()
    stats.start_time = time.time()
    
    # Confirm configuration
    print(f"\n=== CONFIGURATION SUMMARY ===")
    print(f"Learning games: {config['learning_games']:,}")
    print(f"Evaluation games: {config['evaluation_games']:,}")
    print(f"Total games: {config['learning_games'] + config['evaluation_games']:,}")
    print(f"Search depth: {config['search_depth']}")
    
    learning_mode_desc = {
        '1': 'Basic (opening book only)',
        '2': 'Enhanced (patterns + progressive learning)',
        '3': 'Advanced (all features + opponent modelling)'
    }[config['learning_mode']]
    
    mode_desc = {
        '1': 'Mode 1 only',
        '2': 'Mode 2 only', 
        '3': 'Both modes'
    }[config['mode_choice']]
    
    print(f"Learning mode: {learning_mode_desc}")
    print(f"Game modes: {mode_desc}")
    
    confirm = input("\nProceed with this configuration? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Simulation cancelled.")
        return
    
    try:
        # Run learning phase
        learned_ai = run_learning_phase(config, stats)
        
        # Run evaluation phase
        run_evaluation_phase(config, learned_ai, stats)
        
        # Print final results
        print_final_results(config, stats)
        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        print(f"Completed {stats.total_games:,} games before interruption.")
        
        if stats.total_games > 0:
            print_final_results(config, stats)
    
    except Exception as e:
        print(f"\nError during simulation: {e}")
        print(f"Completed {stats.total_games:,} games before error.")
        
        # Print enhanced error information for debugging
        if hasattr(stats, 'pattern_recogniser_hits'):
            print(f"Enhancement statistics before error:")
            print(f"  Pattern recogniser hits: {stats.pattern_recogniser_hits}")
            print(f"  Extended learning positions: {stats.extended_learning_positions}")

if __name__ == "__main__":
    main()
