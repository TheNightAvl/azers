"""
Script to find a sequence of moves in Azers that leads from Mode 1 start to a Defence Win.
A Defence Win occurs when the opponent has no legal moves available.
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

def has_3_or_4_in_center(board, player):
    count = sum(1 for (x, y) in CENTER_SQUARES if board[y][x] == player)
    return count >= 3, count

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

def is_game_over(board, player):
    """Check if game is over and return winner if any"""
    # Check if current player has no tokens (MASSACRE)
    player_tokens = get_tokens(board, player)
    if not player_tokens:
        return True, get_opponent(player), "MASSACRE"
    
    # Check if current player achieved EXILE (3+ in centre)
    win, count = has_3_or_4_in_center(board, player)
    if win:
        return True, player, f"EXILE ({count} in centre)"
    
    # Check if current player has no legal moves (DEFENCE for opponent)
    moves = get_legal_moves(board, player)
    if not moves:
        return True, get_opponent(player), "DEFENCE"
    
    return False, None, None

def search_defence_win(max_moves=20):
    """Search for a sequence leading to Defence Win using BFS"""
    # Mode 1 setup: A at (0,0), B at (3,3), A goes first
    initial_board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    initial_board[0][0] = "A"
    initial_board[3][3] = "B"
    
    print("Starting position (Mode 1):")
    print_board_simple(initial_board)
    
    # BFS queue: (board, current_player, move_sequence, move_count)
    queue = deque([(initial_board, "A", [], 0)])
    visited = set()
    visited.add((board_to_tuple(initial_board), "A"))
    
    while queue:
        board, player, move_sequence, move_count = queue.popleft()
        
        if move_count > max_moves:
            continue
            
        # Get legal moves for current player
        moves = get_legal_moves(board, player)
        
        for move in moves:
            new_board = apply_move(board, player, move)
            new_move_sequence = move_sequence + [(player, move)]
            
            # Check game over conditions after the move
            next_player = get_opponent(player)
            game_over, winner, win_type = is_game_over(new_board, next_player)
            
            if game_over and win_type == "DEFENCE" and winner == player:
                print(f"Found Defence Win in {move_count + 1} moves!")
                print(f"Player {player} wins by DEFENCE")
                print("\nMove sequence:")
                for i, (p, m) in enumerate(new_move_sequence, 1):
                    player_symbol = "X" if p == "A" else "O"
                    print(f"{i}. {player_symbol}: {format_move(m)}")
                
                print("\nFinal board position:")
                print_board_simple(new_board)
                print(f"Player {next_player} ({'X' if next_player == 'A' else 'O'}) has no legal moves.")
                return new_move_sequence
            
            # Continue search if game not over
            if not game_over:
                state = (board_to_tuple(new_board), next_player)
                if state not in visited and move_count + 1 <= max_moves:
                    visited.add(state)
                    queue.append((new_board, next_player, new_move_sequence, move_count + 1))
    
    print(f"No Defence Win found within {max_moves} moves.")
    return None

def search_shorter_wins():
    """Search for other types of wins that might be shorter"""
    initial_board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    initial_board[0][0] = "A"
    initial_board[3][3] = "B"
    
    print("Searching for any win condition...")
    
    queue = deque([(initial_board, "A", [], 0)])
    visited = set()
    visited.add((board_to_tuple(initial_board), "A"))
    
    wins_found = []
    
    while queue:
        board, player, move_sequence, move_count = queue.popleft()
        
        if move_count > 10:  # Keep search shorter for other wins
            continue
            
        moves = get_legal_moves(board, player)
        
        for move in moves:
            new_board = apply_move(board, player, move)
            new_move_sequence = move_sequence + [(player, move)]
            
            next_player = get_opponent(player)
            game_over, winner, win_type = is_game_over(new_board, next_player)
            
            if game_over:
                wins_found.append((move_count + 1, winner, win_type, new_move_sequence, new_board))
                if len(wins_found) >= 5:  # Find a few examples
                    break
            
            if not game_over:
                state = (board_to_tuple(new_board), next_player)
                if state not in visited and move_count + 1 <= 10:
                    visited.add(state)
                    queue.append((new_board, next_player, new_move_sequence, move_count + 1))
        
        if len(wins_found) >= 5:
            break
    
    print(f"\nFound {len(wins_found)} wins:")
    for moves, winner, win_type, sequence, final_board in wins_found:
        print(f"- {moves} moves: Player {winner} wins by {win_type}")
    
    return wins_found

if __name__ == "__main__":
    print("=== Searching for Defence Win ===")
    defence_sequence = search_defence_win(15)
    
    if not defence_sequence:
        print("\n=== Searching for other wins ===")
        other_wins = search_shorter_wins()
