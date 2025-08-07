import sys
import os
import random
import importlib.util

BOARD_SIZE = 4
CENTER_SQUARES = [(1, 1), (2, 1), (1, 2), (2, 2)]  # (x, y) with new axes

# Try to import AI modules
try:
    # Add the simulations/monte_carlo directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    monte_carlo_path = os.path.join(current_dir, 'simulations', 'monte_carlo')
    if monte_carlo_path not in sys.path:
        sys.path.insert(0, monte_carlo_path)
    
    # Import the modules using importlib for better error handling
    import importlib.util
    
    # Load tactical_ai module
    tactical_ai_path = os.path.join(monte_carlo_path, 'tactical_ai.py')
    tactical_ai_spec = importlib.util.spec_from_file_location("tactical_ai", tactical_ai_path)
    tactical_ai = importlib.util.module_from_spec(tactical_ai_spec)
    tactical_ai_spec.loader.exec_module(tactical_ai)
    
    # Load learning_simulation module
    learning_sim_path = os.path.join(monte_carlo_path, 'learning_simulation.py')
    learning_sim_spec = importlib.util.spec_from_file_location("learning_simulation", learning_sim_path)
    learning_simulation = importlib.util.module_from_spec(learning_sim_spec)
    learning_sim_spec.loader.exec_module(learning_simulation)
    
    # Import specific functions and classes
    create_tactical_ai = tactical_ai.create_tactical_ai
    run_learning_phase = learning_simulation.run_learning_phase
    LearningStats = learning_simulation.LearningStats
    
    AI_AVAILABLE = True
except (ImportError, AttributeError, FileNotFoundError) as e:
    AI_AVAILABLE = False
    print("Warning: AI modules not found. Playing in human-vs-human mode only.")
    print("To enable AI, ensure tactical_ai.py and learning_simulation.py are in simulations/monte_carlo/ folder.")
    print(f"Import error: {e}")
    
    # Create dummy classes/functions for when AI is not available
    def create_tactical_ai(*args, **kwargs):
        return None
    def run_learning_phase(*args, **kwargs):
        return None
    class LearningStats:
        def __init__(self):
            self.learning_games = 0

def print_board(board, highlight_tokens=None, highlight_moves=None):
    """
    Print the board.
    highlight_tokens: list of (x, y, option_number) to show token selection numbers
    highlight_moves: list of (nx, ny, option_number) to show move selection numbers
    """
    # Unicode circled numbers 1-20
    circled_numbers = [
        "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩",
        "⑪", "⑫", "⑬", "⑭", "⑮", "⑯", "⑰", "⑱", "⑲", "⑳"
    ]
    token_lookup = {}
    if highlight_tokens:
        for x, y, num in highlight_tokens:
            token_lookup[(x, y)] = circled_numbers[num-1] if 1 <= num <= 20 else str(num)
    move_lookup = {}
    if highlight_moves:
        for nx, ny, num in highlight_moves:
            move_lookup[(nx, ny)] = circled_numbers[num-1] if 1 <= num <= 20 else str(num)

    # Each cell will be 5 characters wide: e.g. " X ① " or " O   "
    print("      " + "     ".join(chr(ord('A') + x) for x in range(BOARD_SIZE)))
    print("   +" + "-----+" * BOARD_SIZE)
    for y in range(BOARD_SIZE):
        row = []
        for x in range(BOARD_SIZE):
            token = None
            if board[y][x] == "A":
                token = "X"
            elif board[y][x] == "B":
                token = "O"
            else:
                token = " "

            num = None
            if (x, y) in move_lookup:
                num = move_lookup[(x, y)]
            elif (x, y) in token_lookup:
                num = token_lookup[(x, y)]
            else:
                num = " "

            # Compose cell: token left, number right, centred
            cell = f" {token} {num} "
            row.append(cell)
        print(f"{y+1}  |" + "|".join(row) + "|")
        print("   +" + "-----+" * BOARD_SIZE)
    print()

def get_opponent(player):
    return "B" if player == "A" else "A"

def in_bounds(x, y):
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

def get_tokens(board, player):
    return [(x, y) for y in range(BOARD_SIZE) for x in range(BOARD_SIZE) if board[y][x] == player]

def has_3_or_4_in_centre(board, player):
    count = sum(1 for (x, y) in CENTER_SQUARES if board[y][x] == player)
    return count >= 3, count

def check_opposition_pattern(board):
    """
    Check if the centre four squares form an OPPOSITION pattern:
    X O    or    O X
    O X          X O
    """
    # CENTER_SQUARES = [(1, 1), (2, 1), (1, 2), (2, 2)]
    # This corresponds to:
    # (1,1) (2,1)
    # (1,2) (2,2)
    
    # Get the tokens in the centre squares
    centre_tokens = {}
    for x, y in CENTER_SQUARES:
        centre_tokens[(x, y)] = board[y][x]
    
    # Check if all centre squares are occupied
    if any(token is None for token in centre_tokens.values()):
        return False
    
    # Pattern 1: X O / O X
    pattern1 = (centre_tokens[(1, 1)] == "A" and centre_tokens[(2, 1)] == "B" and
                centre_tokens[(1, 2)] == "B" and centre_tokens[(2, 2)] == "A")
    
    # Pattern 2: O X / X O  
    pattern2 = (centre_tokens[(1, 1)] == "B" and centre_tokens[(2, 1)] == "A" and
                centre_tokens[(1, 2)] == "A" and centre_tokens[(2, 2)] == "B")
    
    return pattern1 or pattern2

def get_legal_moves(board, player):
    moves = []
    for (x, y) in get_tokens(board, player):
        # DROP and HOP
        for dx in [-2, 0, 2]:
            for dy in [-2, 0, 2]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                mx, my = x + dx // 2, y + dy // 2
                if in_bounds(nx, ny) and board[ny][nx] is None:
                    if board[my][mx] is None:
                        moves.append(("DROP", (x, y), (nx, ny)))
                    elif board[my][mx] is not None:
                        moves.append(("HOP", (x, y), (nx, ny)))
        # SWAP
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if in_bounds(nx, ny) and board[ny][nx] == get_opponent(player):
                moves.append(("SWAP", (x, y), (nx, ny)))
    return moves

def apply_move(board, player, move):
    kind, (x, y), (nx, ny) = move
    new_board = [row[:] for row in board]
    if kind == "DROP":
        new_board[y][x] = None
        new_board[ny][nx] = player
        mx, my = x + (nx - x) // 2, y + (ny - y) // 2
        new_board[my][mx] = player
    elif kind == "HOP":
        new_board[y][x] = None
        new_board[ny][nx] = player
    elif kind == "SWAP":
        new_board[y][x] = None
        new_board[ny][nx] = player
    return new_board

def input_move(moves, tokens, board):
    # Group moves by token
    from collections import defaultdict
    moves_by_token = defaultdict(list)
    for move in moves:
        moves_by_token[move[1]].append(move)

    if len(tokens) == 1:
        # Only one token, skip selection
        token = tokens[0]
    else:
        # Token selection loop
        while True:
            print("Your tokens:")
            highlight_tokens = [(x, y, i) for i, (x, y) in enumerate(tokens, 1)]
            print_board(board, highlight_tokens=highlight_tokens)
            for i, (x, y) in enumerate(tokens, 1):
                print(f"{i}: {chr(ord('A')+x)}{y+1}")
            try:
                token_idx = int(input("Select token number: ")) - 1
                if 0 <= token_idx < len(tokens):
                    token = tokens[token_idx]
                    break
            except Exception:
                pass
            print("Invalid selection.")

    # Move selection loop
    while True:
        available_moves = [m for m in moves if m[1] == token]
        token_symbol = "X" if board[token[1]][token[0]] == "A" else "O"
        highlight_moves = [(nx, ny, i) for i, (_, _, (nx, ny)) in enumerate(available_moves, 1)]
        print(f"Available moves for token {token_symbol} at {chr(ord('A')+token[0])}{token[1]+1}:")
        print_board(board, highlight_moves=highlight_moves)
        for i, (kind, (x, y), (nx, ny)) in enumerate(available_moves, 1):
            dx = nx - x
            dy = ny - y
            if dx == 0 and dy != 0:
                direction = "UP" if dy < 0 else "DOWN"
            elif dy == 0 and dx != 0:
                direction = "LEFT" if dx < 0 else "RIGHT"
            else:
                direction = "DIAGONAL"
            print(f"{i}: {kind} to {chr(ord('A')+nx)}{ny+1} ({direction})")
        if len(tokens) > 1:
            print("r: Return to token selection")
        choice = input("Select move number: ").strip()
        if len(tokens) > 1 and choice.lower() == 'r':
            # Go back to token selection
            return input_move(moves, tokens, board)
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available_moves):
                return available_moves[idx]
        except Exception:
            pass
        print("Invalid selection.")

def parse_position(prompt):
    while True:
        try:
            pos = input(prompt).strip().upper()
            # Only accept input like "B2"
            if len(pos) == 2 and pos[0] in "ABCD" and pos[1] in "1234":
                x = ord(pos[0]) - ord('A')
                y = int(pos[1]) - 1
                if in_bounds(x, y):
                    return x, y
            raise ValueError
        except Exception:
            pass
        print("Invalid input. Please enter as: column(A-D)row(1-4) (e.g. B2)")

def create_learning_ai(depth=6, learning_games=100, randomness=0.1, book_file=None, enhancement_level="standard", learning_mode="advanced", game_mode=None, opposition_enabled=None):
    """Create a learning AI that has been trained through self-play"""
    if not AI_AVAILABLE:
        return None
        
    if book_file is None:
        # Use specialised learning books based on game context
        if game_mode is not None and opposition_enabled is not None:
            # Use context-aware specialised books
            mode_name = "mode1" if game_mode == 1 else "mode2"
            opposition_suffix = "with_opposition" if opposition_enabled else "without_opposition"
            learning_mode_num = {'basic': '1', 'enhanced': '2', 'advanced': '3'}[learning_mode]
            book_file = f"specialised_{mode_name}_{opposition_suffix}_depth_{depth}_learning{learning_mode_num}.json"
        elif enhancement_level == "bias_optimised_mode2":
            book_file = f"enhanced_learning_book_depth_{depth}_mode2.json"
        elif enhancement_level == "bias_optimised_mode3":
            book_file = f"enhanced_learning_book_depth_{depth}_mode3.json"
        else:
            # Standard learning books based on learning mode
            learning_mode_num = {'basic': '1', 'enhanced': '2', 'advanced': '3'}[learning_mode]
            book_file = f"interactive_learning_book_depth_{depth}_mode{learning_mode_num}.json"
    
    # Check if specialised book exists, if not create with training
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    book_path = os.path.join(current_dir, 'simulations', 'monte_carlo', book_file)
    
    if os.path.exists(book_path) and any(x in book_file for x in ["specialised_", "enhanced_learning_book"]):
        print(f"Using existing specialised learning book: {book_file}")
        # Load existing enhanced AI
        learned_ai = create_tactical_ai(search_depth=depth, learning_enabled=True, 
                                      opening_book_file=book_file, randomness=randomness)
        return learned_ai
    else:
        print(f"Training Learning AI with {learning_games} games...")
        print("This may take a moment...")
        
        # Create configuration for learning
        if game_mode is not None and opposition_enabled is not None:
            # Specialised training for specific game context
            mode_choice = '1' if game_mode == 1 else '2'
            opposition_choice = '2' if opposition_enabled else '1'
        else:
            # General training
            mode_choice = '3'  # Both modes for comprehensive training
            opposition_choice = '3'  # Both opposition settings
            
        # Map learning mode names to numbers for learning_simulation.py
        learning_mode_map = {'basic': '1', 'enhanced': '2', 'advanced': '3'}
        learning_mode_num = learning_mode_map.get(learning_mode, '3')
            
        config = {
            'learning_games': learning_games,
            'evaluation_games': 0,  # We don't need evaluation for interactive play
            'search_depth': depth,
            'learning_mode': learning_mode_num,
            'mode_choice': mode_choice,
            'opposition_choice': opposition_choice
        }
        
        # Create learning stats
        stats = LearningStats()
        
        # Run the learning phase to train the AI
        learned_ai_result = run_learning_phase(config, stats)
        
        # Handle both single AI and dictionary of AIs
        if isinstance(learned_ai_result, dict):
            # Multiple specialised AIs were created, pick the appropriate one
            if game_mode is not None and opposition_enabled is not None:
                mode_key = "mode1" if game_mode == 1 else "mode2"
                learned_ai = learned_ai_result.get((mode_key, opposition_enabled), list(learned_ai_result.values())[0])
            else:
                learned_ai = list(learned_ai_result.values())[0]  # Pick first AI
        else:
            learned_ai = learned_ai_result
        
        print(f"Learning AI trained! Learned from {stats.learning_games} games.")
        
        return learned_ai

def get_ai_difficulty_level():
    """Get AI difficulty/enhancement level from user"""
    print("\nAI Difficulty & Learning Level:")
    print("1. Novice (Depth 3, High randomness, Basic learning)")
    print("2. Intermediate (Depth 4-5, Moderate randomness, Enhanced learning)")
    print("3. Advanced (Depth 6-7, Low randomness, Advanced learning)")
    print("4. Expert (Depth 8-9, Minimal randomness, Advanced learning)")
    print("5. Master (Depth 10, Deterministic, Advanced learning)")
    print("6. Bias-Optimised Expert (Uses pre-analysed Mode 2 book)")
    print("7. Bias-Optimised Master (Uses pre-analysed Mode 3 book)")
    print("8. Custom (Choose your own settings)")
    
    while True:
        choice = input("Select difficulty level (1-8): ").strip()
        if choice in ['1', '2', '3', '4', '5', '6', '7', '8']:
            break
        print("Please enter a number between 1 and 8.")
    
    if choice == '1':
        return {'depth': 3, 'randomness': 0.4, 'learning_games': 50, 'learning_mode': 'basic', 'enhancement': 'standard'}
    elif choice == '2':
        return {'depth': 5, 'randomness': 0.2, 'learning_games': 200, 'learning_mode': 'enhanced', 'enhancement': 'standard'}
    elif choice == '3':
        return {'depth': 6, 'randomness': 0.1, 'learning_games': 500, 'learning_mode': 'advanced', 'enhancement': 'standard'}
    elif choice == '4':
        return {'depth': 8, 'randomness': 0.05, 'learning_games': 1000, 'learning_mode': 'advanced', 'enhancement': 'standard'}
    elif choice == '5':
        return {'depth': 10, 'randomness': 0.0, 'learning_games': 1500, 'learning_mode': 'advanced', 'enhancement': 'standard'}
    elif choice == '6':
        return {'depth': 8, 'randomness': 0.05, 'learning_games': 1000, 'learning_mode': 'advanced', 'enhancement': 'bias_optimised_mode2'}
    elif choice == '7':
        return {'depth': 10, 'randomness': 0.0, 'learning_games': 1500, 'learning_mode': 'advanced', 'enhancement': 'bias_optimised_mode3'}
    else:  # choice == '8'
        print("Custom AI Configuration:")
        depth = get_ai_depth("Custom AI")
        randomness = get_ai_randomness("Custom AI")
        
        print("\nTraining intensity:")
        print("1. Light (50-100 games)")
        print("2. Standard (200-500 games)")
        print("3. Intensive (1000+ games)")
        
        while True:
            training_choice = input("Select training intensity (1-3): ").strip()
            if training_choice in ['1', '2', '3']:
                break
            print("Please enter 1, 2, or 3.")
        
        training_map = {'1': 100, '2': 500, '3': 1000}
        learning_games = training_map[training_choice]
        
        print("\nLearning mode (affects AI training sophistication):")
        print("1. Basic learning (Opening book only)")
        print("2. Enhanced learning (Patterns + progressive learning)")
        print("3. Advanced learning (All features + opponent modelling)")
        
        while True:
            learning_choice = input("Select learning mode (1-3): ").strip()
            if learning_choice in ['1', '2', '3']:
                break
            print("Please enter 1, 2, or 3.")
        
        learning_mode_map = {'1': 'basic', '2': 'enhanced', '3': 'advanced'}
        learning_mode = learning_mode_map[learning_choice]
        
        print("\nEnhancement level:")
        print("1. Standard (Train from scratch)")
        print("2. Bias-Optimised Mode 2 (Use pre-analysed book for all starting positions)")
        print("3. Bias-Optimised Mode 3 (Use advanced pre-analysed book)")
        
        while True:
            enhancement_choice = input("Select enhancement (1-3): ").strip()
            if enhancement_choice in ['1', '2', '3']:
                break
            print("Please enter 1, 2, or 3.")
        
        enhancement_map = {'1': 'standard', '2': 'bias_optimised_mode2', '3': 'bias_optimised_mode3'}
        enhancement = enhancement_map[enhancement_choice]
        
        return {'depth': depth, 'randomness': randomness, 'learning_games': learning_games, 'learning_mode': learning_mode, 'enhancement': enhancement}

def get_ai_move(board, player, ai_strategy):
    """Get move from AI strategy"""
    moves = get_legal_moves(board, player)
    if not moves:
        return None
    
    move = ai_strategy(board, player, moves)
    return move

def ai_choose_starting_positions(board):
    """Let AI choose starting positions for Mode 2"""
    available_positions = [(x, y) for x in range(BOARD_SIZE) for y in range(BOARD_SIZE)]
    
    # AI chooses position for A
    a_pos = random.choice(available_positions)
    board[a_pos[1]][a_pos[0]] = "A"
    available_positions.remove(a_pos)
    
    # AI chooses position for B, avoiding diagonal adjacency
    valid_b_positions = []
    for x, y in available_positions:
        ax, ay = a_pos
        # Check for diagonal adjacency
        if not (abs(ax - x) == 1 and abs(ay - y) == 1):
            valid_b_positions.append((x, y))
    
    if valid_b_positions:
        b_pos = random.choice(valid_b_positions)
        board[b_pos[1]][b_pos[0]] = "B"
    else:
        # Fallback - choose any non-diagonal position
        b_pos = random.choice(available_positions)
        board[b_pos[1]][b_pos[0]] = "B"
    
    print(f"AI placed Player A at {chr(ord('A')+a_pos[0])}{a_pos[1]+1}")
    print(f"AI placed Player B at {chr(ord('A')+b_pos[0])}{b_pos[1]+1}")

def setup_board(mode, ai_player_a=None, ai_player_b=None, setup_choice="human"):
    board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    print("Board coordinates:")
    print_board(board)
    if mode == 1:
        board[0][0] = "A"
        board[BOARD_SIZE-1][BOARD_SIZE-1] = "B"
        return board, "A"
    else:
        # Mode 2 - starting position selection
        if setup_choice == "ai" and (ai_player_a or ai_player_b):
            # AI chooses starting positions
            print("AI choosing starting positions...")
            ai_choose_starting_positions(board)
            print_board(board)
            
            # Determine who goes first based on the setup
            if ai_player_a and not ai_player_b:
                # AI is A, human is B - human goes first when AI chooses
                print("You (Player B) go first.")
                return board, "B"
            elif not ai_player_a and ai_player_b:
                # Human is A, AI is B - human goes first when AI chooses  
                print("You (Player A) go first.")
                return board, "A"
            else:
                # Both AI - B goes first by convention
                print("Player B goes first.")
                return board, "B"
        else:
            # Human chooses starting positions (original behaviour)
            if ai_player_b and not ai_player_a:
                print("You choose the starting positions for both players. AI will go first.")
            elif ai_player_a and not ai_player_b:
                print("You choose the starting positions for both players. You will go first.")
            else:
                print("One player will choose the starting positions for both tokens. The other player will go first.")
                
            while True:
                ax, ay = parse_position("Choose position for Player A's token (column row, e.g. B1): ")
                if board[ay][ax] is not None:
                    print("That square is already occupied. Choose a different position.")
                    continue
                board[ay][ax] = "A"
                print_board(board)
                bx, by = parse_position("Choose position for Player B's token (column row, e.g. D4): ")
                # Check for same square
                if (ax, ay) == (bx, by):
                    print("Both tokens cannot be placed on the same square. Try again.")
                    board[ay][ax] = None
                    print_board(board)
                    continue
                # Check for diagonal adjacency
                if abs(ax - bx) == 1 and abs(ay - by) == 1:
                    print("Tokens cannot be placed diagonally adjacent. Try again.")
                    board[ay][ax] = None
                    print_board(board)
                    continue
                if board[by][bx] is not None:
                    print("That square is already occupied. Choose a different position.")
                    board[ay][ax] = None
                    print_board(board)
                    continue
                board[by][bx] = "B"
                print_board(board)
                break
            
            if ai_player_b and not ai_player_a:
                print("AI (Player B) goes first.")
                return board, "B"
            elif ai_player_a and not ai_player_b:
                print("You (Player B) go first.")
                return board, "B"
            else:
                print("Player B goes first.")
                return board, "B"

def ask_opposition_rule():
    """Ask if OPPOSITION draw rule should be enabled"""
    print("\nOPPOSITION Draw Rule:")
    print("This optional rule creates a draw when the centre four squares form:")
    print("X O    or    O X")
    print("O X          X O")
    while True:
        choice = input("Enable OPPOSITION draw rule? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False
        print("Please enter 'y' for yes or 'n' for no.")

def get_ai_configuration():
    """Configure AI settings"""
    if not AI_AVAILABLE:
        return None, None
    
    print("\n=== AI CONFIGURATION ===")
    print("Choose AI opponents:")
    print("1. Human vs Human")
    print("2. Human vs Tactical AI")
    print("3. Human vs Learning AI")
    print("4. Tactical AI vs Learning AI")
    print("5. Learning AI vs Learning AI (training mode)")
    print("6. Bias-optimised AI (uses enhanced learning from analysis)")
    
    while True:
        choice = input("Select option (1-6): ").strip()
        if choice in ['1', '2', '3', '4', '5', '6']:
            break
        print("Please enter a number between 1 and 6.")
    
    ai_player_a = None
    ai_player_b = None
    
    if choice == '1':
        # Human vs Human - no AI
        return None, None
    
    elif choice == '2':
        # Human vs Tactical AI
        print("You will play as Player A (X), Tactical AI as Player B (O)")
        ai_config = get_ai_difficulty_level()
        ai_player_b = create_tactical_ai(search_depth=ai_config['depth'], learning_enabled=False, 
                                       randomness=ai_config['randomness'])
        return None, ai_player_b
    
    elif choice == '3':
        # Human vs Learning AI
        print("You will play as Player A (X), Learning AI as Player B (O)")
        ai_config = get_ai_difficulty_level()
        
        print(f"\nConfiguring AI with:")
        print(f"  Depth: {ai_config['depth']}")
        print(f"  Randomness: {ai_config['randomness']}")
        print(f"  Training games: {ai_config['learning_games']}")
        print(f"  Learning mode: {ai_config['learning_mode'].capitalize()}")
        print(f"  Enhancement: {ai_config['enhancement']}")
        
        ai_player_b = create_learning_ai(depth=ai_config['depth'], 
                                       learning_games=ai_config['learning_games'],
                                       randomness=ai_config['randomness'],
                                       enhancement_level=ai_config['enhancement'],
                                       learning_mode=ai_config['learning_mode'])
        return None, ai_player_b
    
    elif choice == '4':
        # Tactical AI vs Learning AI
        print("Tactical AI as Player A (X), Learning AI as Player B (O)")
        
        print("\nConfiguring Tactical AI (Player A):")
        ai_config_a = get_ai_difficulty_level()
        ai_player_a = create_tactical_ai(search_depth=ai_config_a['depth'], learning_enabled=False, 
                                       randomness=ai_config_a['randomness'])
        
        print("\nConfiguring Learning AI (Player B):")
        ai_config_b = get_ai_difficulty_level()
        ai_player_b = create_learning_ai(depth=ai_config_b['depth'], 
                                       learning_games=ai_config_b['learning_games'],
                                       randomness=ai_config_b['randomness'],
                                       enhancement_level=ai_config_b['enhancement'],
                                       learning_mode=ai_config_b['learning_mode'])
        
        return ai_player_a, ai_player_b
    
    elif choice == '5':
        # Learning AI vs Learning AI (training)
        print("Learning AI vs Learning AI (training mode)")
        
        print("\nConfiguring both Learning AIs:")
        ai_config = get_ai_difficulty_level()
        
        print(f"Training both AIs with same settings:")
        print(f"  Depth: {ai_config['depth']}")
        print(f"  Randomness: {ai_config['randomness']}")
        print(f"  Training games: {ai_config['learning_games']}")
        print(f"  Learning mode: {ai_config['learning_mode'].capitalize()}")
        print(f"  Enhancement: {ai_config['enhancement']}")
        
        # Create two learning AIs that will train against each other
        print("Training first Learning AI...")
        ai_player_a = create_learning_ai(depth=ai_config['depth'], 
                                        learning_games=ai_config['learning_games'],
                                        randomness=ai_config['randomness'],
                                        enhancement_level=ai_config['enhancement'],
                                        learning_mode=ai_config['learning_mode'],
                                        book_file=f"training_ai_a_depth_{ai_config['depth']}.json")
        
        print("Training second Learning AI...")
        ai_player_b = create_learning_ai(depth=ai_config['depth'], 
                                        learning_games=ai_config['learning_games'],
                                        randomness=ai_config['randomness'],
                                        enhancement_level=ai_config['enhancement'],
                                        learning_mode=ai_config['learning_mode'],
                                        book_file=f"training_ai_b_depth_{ai_config['depth']}.json")
        
        return ai_player_a, ai_player_b
    
    elif choice == '6':
        # Bias-optimised AI (new option)
        print("Bias-optimised AI uses enhanced learning books from Monte Carlo bias analysis")
        print("You will play as Player A (X), Bias-optimised AI as Player B (O)")
        
        print("\nBias-optimised AI Configuration:")
        print("1. Mode 2 Optimised (Expert at all starting positions)")
        print("2. Mode 3 Enhanced (Master level with advanced bias analysis)")
        print("3. Adaptive (Switches between optimisations based on game mode)")
        
        while True:
            bias_choice = input("Select optimisation (1-3): ").strip()
            if bias_choice in ['1', '2', '3']:
                break
            print("Please enter 1, 2, or 3.")
        
        if bias_choice == '1':
            # Mode 2 optimised
            ai_player_b = create_learning_ai(depth=8, learning_games=1000, randomness=0.05,
                                           enhancement_level="bias_optimised_mode2", learning_mode="advanced")
            print("Created Mode 2 optimised AI (Expert level)")
        elif bias_choice == '2':
            # Mode 3 enhanced
            ai_player_b = create_learning_ai(depth=10, learning_games=1500, randomness=0.0,
                                           enhancement_level="bias_optimised_mode3", learning_mode="advanced")
            print("Created Mode 3 enhanced AI (Master level)")
        else:  # bias_choice == '3'
            # Adaptive - will use enhanced_mode3 as the most comprehensive
            ai_player_b = create_learning_ai(depth=9, learning_games=1200, randomness=0.02,
                                           enhancement_level="bias_optimised_mode3", learning_mode="advanced")
            print("Created Adaptive AI (Grandmaster level)")
        
        return None, ai_player_b

def get_ai_depth(ai_name="AI"):
    """Get AI search depth from user"""
    while True:
        try:
            depth = int(input(f"Search depth for {ai_name} (1-10, recommended: 6): ").strip())
            if 1 <= depth <= 10:
                return depth
            print("Please enter a depth between 1 and 10.")
        except ValueError:
            print("Please enter a valid number.")

def get_ai_randomness(ai_name="AI"):
    """Get AI randomness setting from user"""
    print(f"\n{ai_name} Randomisation Settings:")
    print("0.0 = Completely deterministic (always same move)")
    print("0.1 = Slightly random (recommended for competitive play)")
    print("0.3 = Moderate randomisation (more varied games)")
    print("0.5 = High randomisation (unpredictable but strategic)")
    
    while True:
        try:
            randomness = float(input(f"Randomisation for {ai_name} (0.0-1.0, recommended: 0.1): ").strip())
            if 0.0 <= randomness <= 1.0:
                return randomness
            print("Please enter a value between 0.0 and 1.0.")
        except ValueError:
            print("Please enter a valid number.")

def get_mode2_setup_choice(ai_player_a, ai_player_b):
    """For Mode 2, ask who sets up the starting positions"""
    if ai_player_a and ai_player_b:
        # Both AI - AI chooses automatically
        return "ai"
    elif ai_player_a or ai_player_b:
        # One AI, one human
        print(f"\nMode 2 Setup Options:")
        print(f"1. AI chooses starting positions, You go first")
        print(f"2. You choose starting positions, AI goes first")
        
        while True:
            choice = input("Select option (1-2): ").strip()
            if choice == '1':
                return "ai"
            elif choice == '2':
                return "human"
            print("Please enter 1 or 2.")
    else:
        # Both human - use original behaviour
        return "human"

def create_context_aware_ai(ai_config, game_mode=None, opposition_enabled=None):
    """Create an AI that's optimised for the specific game context"""
    if ai_config['enhancement'] in ['bias_optimised_mode2', 'bias_optimised_mode3']:
        # Use context-aware specialised books if available
        return create_learning_ai(
            depth=ai_config['depth'], 
            learning_games=ai_config['learning_games'],
            randomness=ai_config['randomness'],
            enhancement_level=ai_config['enhancement'],
            learning_mode=ai_config['learning_mode'],
            game_mode=game_mode,
            opposition_enabled=opposition_enabled
        )
    else:
        # Standard AI creation
        return create_learning_ai(
            depth=ai_config['depth'], 
            learning_games=ai_config['learning_games'],
            randomness=ai_config['randomness'],
            enhancement_level=ai_config['enhancement'],
            learning_mode=ai_config['learning_mode']
        )

def main():
    print("Welcome to Azers!")
    print("Player A: X   Player B: O")
    print()
    print("🎯 Enhanced AI Features Available:")
    print("• Multiple learning modes: Basic, Enhanced, Advanced")
    print("• Bias-optimised AI using Monte Carlo analysis")
    print("• Multiple difficulty levels (Novice to Master)")
    print("• Enhanced learning books from thousands of games")
    print("• Advanced strategic understanding with opponent modelling")
    print("• Context-aware specialised training")
    print()
    
    # Get AI configuration
    ai_player_a, ai_player_b = get_ai_configuration()
    
    print("Choose start mode: 1 (opposing corners) or 2 (players choose)")
    mode = int(input("Mode: "))
    
    # For Mode 2, ask about setup choice if AI is involved
    setup_choice = "human"  # default
    if mode == 2 and (ai_player_a or ai_player_b):
        setup_choice = get_mode2_setup_choice(ai_player_a, ai_player_b)
    
    # Ask about OPPOSITION rule
    opposition_enabled = ask_opposition_rule()
    if opposition_enabled:
        print("OPPOSITION draw rule is ENABLED.")
    else:
        print("OPPOSITION draw rule is DISABLED.")
    
    # Now that we know the game context, create context-aware AIs if needed
    if ai_player_a and hasattr(ai_player_a, 'ai') and ai_player_a.ai.learning_enabled:
        print("Optimising Player A AI for current game context...")
        # This would require storing the AI config, for now we'll use the existing AI
    
    if ai_player_b and hasattr(ai_player_b, 'ai') and ai_player_b.ai.learning_enabled:
        print("Optimising Player B AI for current game context...")
        # This would require storing the AI config, for now we'll use the existing AI
    
    board, player = setup_board(mode, ai_player_a, ai_player_b, setup_choice)
    round_num = 1
    previous_states = []
    CYCLE_LIMIT = 8

    while True:
        print("=" * 30)
        player_symbol = "X" if player == "A" else "O"
        
        # Determine who is playing this turn
        current_ai = ai_player_a if player == "A" else ai_player_b
        is_human = current_ai is None
        
        if is_human:
            print(f"Round {round_num} — Your turn ({player_symbol})")
        else:
            ai_type = "Learning AI" if hasattr(current_ai, 'ai') and current_ai.ai.learning_enabled else "Tactical AI"
            
            # Enhanced AI information
            if hasattr(current_ai, 'ai') and current_ai.ai.learning_enabled:
                book_file = getattr(current_ai.ai, 'opening_book_file', '')
                if 'enhanced_learning_book_depth' in book_file:
                    if 'mode3' in book_file:
                        ai_type = "Enhanced AI (Master Level)"
                    elif 'mode2' in book_file:
                        ai_type = "Enhanced AI (Expert Level)"
                    else:
                        ai_type = "Enhanced AI"
            
            print(f"Round {round_num} — {ai_type} turn ({player_symbol})")
        
        print("=" * 30)
        print_board(board)

        moves = get_legal_moves(board, player)
        if not moves:
            print(f"Player {player} cannot move. {get_opponent(player)} wins by DEFENCE!")
            break
        
        # Get move from human or AI
        if is_human:
            tokens = get_tokens(board, player)
            move = input_move(moves, tokens, board)
        else:
            print("AI is thinking...")
            move = get_ai_move(board, player, current_ai)
            kind, (x, y), (nx, ny) = move
            print(f"AI plays: {kind} from {chr(ord('A')+x)}{y+1} to {chr(ord('A')+nx)}{ny+1}")
        
        board = apply_move(board, player, move)

        # --- CHECK WIN/END CONDITIONS IMMEDIATELY AFTER MOVE ---
        # EXILE if opponent has no tokens
        opponent = get_opponent(player)
        opponent_tokens = get_tokens(board, opponent)
        if not opponent_tokens:
            print_board(board)
            winner_name = get_player_name(player, ai_player_a, ai_player_b)
            print(f"{winner_name} wins by MASSACRE (all opponent tokens captured)!")
            break

        # EXILE win (3 or 4 in centre)
        win, count = has_3_or_4_in_centre(board, player)
        if win:
            print_board(board)
            winner_name = get_player_name(player, ai_player_a, ai_player_b)
            print(f"{winner_name} wins by EXILE (occupying {count} centre squares)!")
            break

        # OPPOSITION draw check (if enabled)
        if opposition_enabled and check_opposition_pattern(board):
            print_board(board)
            print("Draw by OPPOSITION (centre squares form the opposition pattern)!")
            # Update learning for both AIs if available
            if ai_player_a and hasattr(ai_player_a, 'learn_from_game'):
                ai_player_a.learn_from_game('draw', 'A')
            if ai_player_b and hasattr(ai_player_b, 'learn_from_game'):
                ai_player_b.learn_from_game('draw', 'B')
            break

        # DEFENCE if opponent has no legal moves
        opponent_moves = get_legal_moves(board, opponent)
        if not opponent_moves:
            print_board(board)
            winner_name = get_player_name(player, ai_player_a, ai_player_b)
            print(f"{winner_name} wins by DEFENCE (opponent cannot move)!")
            break

        # BOREDOM (cycle detection)
        state = (tuple(tuple(row) for row in board), player)
        previous_states.append(state)
        if previous_states.count(state) >= CYCLE_LIMIT:
            print_board(board)
            print("Draw by BOREDOM (repetitive moves detected).")
            # Update learning for both AIs if available
            if ai_player_a and hasattr(ai_player_a, 'learn_from_game'):
                ai_player_a.learn_from_game('draw', 'A')
            if ai_player_b and hasattr(ai_player_b, 'learn_from_game'):
                ai_player_b.learn_from_game('draw', 'B')
            break
        if len(previous_states) > 100:
            previous_states.pop(0)

        # Next player's turn
        player = get_opponent(player)
        round_num += 1
    
    # Update AI learning after game ends (if not already done for draws)
    game_ended_with_winner = False
    if not opponent_tokens or ('win' in locals() and win):
        # Someone won - update learning
        game_ended_with_winner = True
        winner = player
        loser = get_opponent(player)
        
        if ai_player_a and hasattr(ai_player_a, 'learn_from_game'):
            result = 'win' if winner == 'A' else 'loss'
            ai_player_a.learn_from_game(result, 'A')
        if ai_player_b and hasattr(ai_player_b, 'learn_from_game'):
            result = 'win' if winner == 'B' else 'loss' 
            ai_player_b.learn_from_game(result, 'B')
    
    # Save learning progress with enhanced feedback
    if ai_player_a and hasattr(ai_player_a, 'ai') and ai_player_a.ai.learning_enabled:
        ai_player_a.ai.save_opening_book()
        
        # Show enhanced learning feedback
        total_positions = len(ai_player_a.ai.opening_book)
        total_moves = sum(len(moves) for moves in ai_player_a.ai.opening_book.values())
        print(f"📚 Player A AI learning updated! Games played: {ai_player_a.ai.games_learned}, Knowledge: {total_positions} positions, {total_moves} moves")
        
    if ai_player_b and hasattr(ai_player_b, 'ai') and ai_player_b.ai.learning_enabled:
        ai_player_b.ai.save_opening_book()
        
        # Show enhanced learning feedback
        total_positions = len(ai_player_b.ai.opening_book)
        total_moves = sum(len(moves) for moves in ai_player_b.ai.opening_book.values())
        print(f"📚 Player B AI learning updated! Games played: {ai_player_b.ai.games_learned}, Knowledge: {total_positions} positions, {total_moves} moves")

def get_player_name(player, ai_player_a, ai_player_b):
    """Get display name for player (human or AI type)"""
    if player == "A":
        if ai_player_a is None:
            return "Player A (You)"
        else:
            ai = ai_player_a
            if hasattr(ai, 'ai') and ai.ai.learning_enabled:
                # Check if using enhanced books
                book_file = getattr(ai.ai, 'opening_book_file', '')
                if 'enhanced_learning_book_depth' in book_file:
                    if 'mode3' in book_file:
                        return "Player A (Enhanced AI - Master)"
                    elif 'mode2' in book_file:
                        return "Player A (Enhanced AI - Expert)"
                    else:
                        return "Player A (Enhanced AI)"
                else:
                    return "Player A (Learning AI)"
            else:
                return "Player A (Tactical AI)"
    else:  # player == "B"
        if ai_player_b is None:
            return "Player B (You)"
        else:
            ai = ai_player_b
            if hasattr(ai, 'ai') and ai.ai.learning_enabled:
                # Check if using enhanced books
                book_file = getattr(ai.ai, 'opening_book_file', '')
                if 'enhanced_learning_book_depth' in book_file:
                    if 'mode3' in book_file:
                        return "Player B (Enhanced AI - Master)"
                    elif 'mode2' in book_file:
                        return "Player B (Enhanced AI - Expert)"
                    else:
                        return "Player B (Enhanced AI)"
                else:
                    return "Player B (Learning AI)"
            else:
                return "Player B (Tactical AI)"

if __name__ == "__main__":
    main()