"""
Script to find opposition draw    centre_tok    centre_tokens = {}
    for x, y in CENTER_SQUARES:
        centre_tokens[(x, y)] = board[y][x]
    
    # Check if all centre squares are occupied  
    if any(token is None for token in centre_tokens.values()):{}
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
                centre_tokens[(1, 2)] == "A" and centre_tokens[(2, 2)] == "B") reasonable play from both sides.
This version balances finding opposition draws with ensuring moves are not obviously bad.

A draw by opposition occurs when the centre four squares form one of these patterns:
X O    or    O X
O X          X O
"""
import sys
from collections import deque

# Import the game logic from azers.py
BOARD_SIZE = 4
CENTER_SQUARES = [(1, 1), (2, 1), (1, 2), (2, 2)]

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
    # Get the tokens in the centre squares
    center_tokens = {}
    for x, y in CENTER_SQUARES:
        center_tokens[(x, y)] = board[y][x]
    
    # Check if all centre squares are occupied
    if any(token is None for token in center_tokens.values()):
        return False
    
    # Pattern 1: X O / O X
    pattern1 = (center_tokens[(1, 1)] == "A" and center_tokens[(2, 1)] == "B" and
                center_tokens[(1, 2)] == "B" and center_tokens[(2, 2)] == "A")
    
    # Pattern 2: O X / X O  
    pattern2 = (center_tokens[(1, 1)] == "B" and center_tokens[(2, 1)] == "A" and
                center_tokens[(1, 2)] == "A" and center_tokens[(2, 2)] == "B")
    
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

def print_board_simple(board):
    """Simple board printer for analysis"""
    print("   A B C D")
    for y in range(BOARD_SIZE):
        row_str = f"{y+1}  "
        for x in range(BOARD_SIZE):
            if board[y][x] == "A":
                row_str += "X "
            elif board[y][x] == "B":
                row_str += "O "
            else:
                row_str += ". "
        print(row_str)
    print()

def format_move(move):
    """Format a move for human reading"""
    kind, (x, y), (nx, ny) = move
    from_pos = f"{chr(ord('A')+x)}{y+1}"
    to_pos = f"{chr(ord('A')+nx)}{ny+1}"
    return f"{kind} {from_pos} to {to_pos}"

def board_to_tuple(board):
    """Convert board to tuple for hashing"""
    return tuple(tuple(row) for row in board)

def evaluate_position_simple(board, player):
    """
    Simple position evaluation to avoid obviously bad moves.
    Returns a score where higher is better for the player.
    """
    opponent = get_opponent(player)
    
    # Check for immediate win conditions
    win, count = has_3_or_4_in_centre(board, player)
    if win:
        return 1000
    
    # Check for immediate loss conditions
    opp_win, opp_count = has_3_or_4_in_centre(board, opponent)
    if opp_win:
        return -1000
    
    # Check if no tokens (massacre)
    player_tokens = get_tokens(board, player)
    opponent_tokens = get_tokens(board, opponent)
    if not player_tokens:
        return -1000
    if not opponent_tokens:
        return 1000
    
    # Simple positional evaluation
    score = 0
    
    # Prefer having tokens in centre
    player_centre = sum(1 for (x, y) in CENTER_SQUARES if board[y][x] == player)
    opponent_centre = sum(1 for (x, y) in CENTER_SQUARES if board[y][x] == opponent)
    score += (player_centre - opponent_centre) * 15
    
    # Token count
    score += (len(player_tokens) - len(opponent_tokens)) * 10
    
    # Mobility
    player_moves = len(get_legal_moves(board, player))
    opponent_moves = len(get_legal_moves(board, opponent))
    score += (player_moves - opponent_moves) * 2
    
    return score

def get_reasonable_moves(board, player, max_moves=5):
    """
    Get the best few moves according to simple evaluation.
    This helps avoid obviously terrible moves while still exploring.
    """
    moves = get_legal_moves(board, player)
    if not moves:
        return []
    
    # Evaluate each move
    move_scores = []
    for move in moves:
        new_board = apply_move(board, player, move)
        score = evaluate_position_simple(new_board, player)
        move_scores.append((score, move))
    
    # Sort by score (best first) and return top moves
    move_scores.sort(reverse=True)
    return [move for score, move in move_scores[:max_moves]]

def search_opposition_draw_reasonable(max_moves=15):
    """
    Search for opposition draw where both players make reasonably good moves.
    """
    # Mode 1 setup: A at (0,0), B at (3,3), A goes first
    initial_board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    initial_board[0][0] = "A"
    initial_board[3][3] = "B"
    
    print("Starting position (Mode 1):")
    print_board_simple(initial_board)
    print("Searching for opposition draw with reasonable play from both sides...")
    print(f"Each player considers top 5 moves, max game length: {max_moves} moves\\n")
    
    # BFS queue: (board, current_player, move_sequence, move_count)
    queue = deque([(initial_board, "A", [], 0)])
    visited = set()
    visited.add((board_to_tuple(initial_board), "A"))
    
    positions_checked = 0
    
    while queue:
        board, player, move_sequence, move_count = queue.popleft()
        positions_checked += 1
        
        if positions_checked % 5000 == 0:
            print(f"Checked {positions_checked} positions...")
        
        if move_count > max_moves:
            continue
        
        # Check if we found an opposition draw
        if check_opposition_pattern(board):
            print(f"Found Opposition Draw in {move_count} moves!")
            print("Move sequence:")
            for i, (p, m) in enumerate(move_sequence, 1):
                player_symbol = "X" if p == "A" else "O"
                print(f"{i}. {player_symbol}: {format_move(m)}")
            
            print("\\nFinal board position:")
            print_board_simple(board)
            print("Opposition pattern achieved!")
            
            # Show evaluation of the final position
            eval_a = evaluate_position_simple(board, "A")
            eval_b = evaluate_position_simple(board, "B")
            print(f"Position evaluation - A: {eval_a}, B: {eval_b}")
            
            return move_sequence
        
        # Get reasonable moves for current player (not all moves)
        reasonable_moves = get_reasonable_moves(board, player, max_moves=5)
        
        for move in reasonable_moves:
            new_board = apply_move(board, player, move)
            new_move_sequence = move_sequence + [(player, move)]
            
            # Check if opponent has moves after this move
            next_player = get_opponent(player)
            next_moves = get_legal_moves(new_board, next_player)
            
            # Skip if this move leads to immediate loss of opponent's mobility
            if not next_moves:
                continue
            
            # Check for immediate wins that would end the game
            win, count = has_3_or_4_in_centre(new_board, player)
            if win:
                continue  # Skip immediate wins as we want draws
            
            opp_win, opp_count = has_3_or_4_in_centre(new_board, next_player)
            if opp_win:
                continue  # Skip positions where opponent already won
            
            # Continue search
            state = (board_to_tuple(new_board), next_player)
            if state not in visited and move_count + 1 <= max_moves:
                visited.add(state)
                queue.append((new_board, next_player, new_move_sequence, move_count + 1))
    
    print(f"No Opposition Draw found within {max_moves} moves with reasonable play.")
    print(f"Checked {positions_checked} positions.")
    return None

def demo_found_sequence():
    """Demonstrate the opposition draw sequence we found earlier"""
    moves = [
        ("A", ("DROP", (0, 0), (2, 2))),  # DROP A1 to C3
        ("B", ("SWAP", (3, 3), (2, 2))),  # SWAP D4 to C3
        ("A", ("DROP", (1, 1), (1, 3))),  # DROP B2 to B4
        ("B", ("DROP", (2, 2), (0, 0))),  # DROP C3 to A1
        ("A", ("DROP", (1, 2), (3, 0))),  # DROP B3 to D1
        ("B", ("HOP", (0, 0), (2, 2))),   # HOP A1 to C3
        ("A", ("HOP", (3, 0), (1, 2)))    # HOP D1 to B3
    ]
    
    print("=== Demonstrating Previously Found Opposition Draw ===")
    
    # Mode 1 setup
    board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    board[0][0] = "A"
    board[3][3] = "B"
    
    print("Starting position:")
    print_board_simple(board)
    
    for i, (player, move) in enumerate(moves, 1):
        board = apply_move(board, player, move)
        player_symbol = "X" if player == "A" else "O"
        print(f"Move {i}. {player_symbol}: {format_move(move)}")
        print_board_simple(board)
        
        if check_opposition_pattern(board):
            print("Opposition pattern achieved!")
            break
    
    # Analyse the reasonableness of each move
    print("\\n=== Move Analysis ===")
    board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    board[0][0] = "A"
    board[3][3] = "B"
    
    for i, (player, move) in enumerate(moves, 1):
        print(f"\\nMove {i} - Player {player}:")
        reasonable_moves = get_reasonable_moves(board, player, max_moves=10)
        move_rank = None
        for rank, rmove in enumerate(reasonable_moves, 1):
            if rmove == move:
                move_rank = rank
                break
        
        if move_rank:
            print(f"  Chosen move ranks #{move_rank} out of {len(reasonable_moves)} reasonable moves")
        else:
            print(f"  Chosen move is not in top 10 reasonable moves")
        
        print(f"  Move: {format_move(move)}")
        board = apply_move(board, player, move)

if __name__ == "__main__":
    print("=== Azers Opposition Draw Finder (Reasonable Play) ===\\n")
    
    # First demonstrate the sequence we found
    demo_found_sequence()
    
    print("\\n\\n=== Searching for New Opposition Draws ===")
    opposition_sequence = search_opposition_draw_reasonable(12)
    
    if not opposition_sequence:
        print("\\n=== Trying longer search ===")
        opposition_sequence = search_opposition_draw_reasonable(15)
