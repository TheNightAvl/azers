"""
Script to find the shortest Exile Win sequences in Azers and display the best examples.
"""
import sys
from collections import deque

# Game logic
BOARD_SIZE = 4
CENTER_SQUARES = [(1, 1), (2, 1), (1, 2), (2, 2)]  # B2, C2, B3, C3

def get_opponent(player):
    return "B" if player == "A" else "A"

def in_bounds(x, y):
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

def get_tokens(board, player):
    return [(x, y) for y in range(BOARD_SIZE) for x in range(BOARD_SIZE) if board[y][x] == player]

def has_3_or_4_in_centre(board, player):
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

def print_board_with_centre_highlight(board):
    """Board printer that highlights centre squares"""
    print("   A B C D")
    for y in range(BOARD_SIZE):
        row_str = f"{y+1}  "
        for x in range(BOARD_SIZE):
            is_center = (x, y) in CENTER_SQUARES
            if board[y][x] == "A":
                token = "⊗" if is_center else "X"  # Highlighted X in centre
            elif board[y][x] == "B":
                token = "⊙" if is_center else "O"  # Highlighted O in centre
            else:
                token = "·" if is_center else "."  # Centre squares marked
            row_str += token + " "
        print(row_str)
    print()

def format_move(move):
    """Format a move for human reading"""
    kind, (x, y), (nx, ny) = move
    from_pos = f"{chr(ord('A')+x)}{y+1}"
    to_pos = f"{chr(ord('A')+nx)}{ny+1}"
    return f"{kind} {from_pos} → {to_pos}"

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
    win, count = has_3_or_4_in_centre(board, player)
    if win:
        return True, player, f"EXILE ({count} in centre)"
    
    # Check if current player has no legal moves (DEFENCE for opponent)
    moves = get_legal_moves(board, player)
    if not moves:
        return True, get_opponent(player), "DEFENCE"
    
    return False, None, None

def count_centre_tokens(board, player):
    """Count how many tokens a player has in centre squares"""
    return sum(1 for (x, y) in CENTER_SQUARES if board[y][x] == player)

def find_shortest_exile_wins():
    """Find the shortest Exile Win sequences"""
    # Mode 1 setup: A at (0,0), B at (3,3), A goes first
    initial_board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    initial_board[0][0] = "A"
    initial_board[3][3] = "B"
    
    print("Searching for shortest Exile Win sequences...")
    print("Starting position (Mode 1):")
    print_board_with_centre_highlight(initial_board)
    
    # BFS queue: (board, current_player, move_sequence, move_count)
    queue = deque([(initial_board, "A", [], 0)])
    visited = set()
    visited.add((board_to_tuple(initial_board), "A"))
    
    shortest_wins = {"A": None, "B": None}
    min_moves = {"A": float('inf'), "B": float('inf')}
    
    while queue:
        board, player, move_sequence, move_count = queue.popleft()
        
        # Skip if we already found shorter wins for both players
        if min(min_moves.values()) < move_count:
            continue
            
        if move_count > 12:  # Reasonable upper bound
            continue
            
        # Get legal moves for current player
        moves = get_legal_moves(board, player)
        
        for move in moves:
            new_board = apply_move(board, player, move)
            new_move_sequence = move_sequence + [(player, move)]
            
            # Check game over conditions after the move
            next_player = get_opponent(player)
            game_over, winner, win_type = is_game_over(new_board, next_player)
            
            if game_over and "EXILE" in win_type:
                if move_count + 1 < min_moves[winner]:
                    min_moves[winner] = move_count + 1
                    center_count = count_centre_tokens(new_board, winner)
                    shortest_wins[winner] = {
                        'moves': move_count + 1,
                        'center_count': center_count,
                        'sequence': new_move_sequence,
                        'board': new_board
                    }
                    print(f"New shortest for Player {winner}: {move_count + 1} moves ({center_count} in centre)")
                
                # Don't continue this branch
                continue
            
            # Continue search if game not over
            if not game_over:
                state = (board_to_tuple(new_board), next_player)
                if state not in visited:
                    visited.add(state)
                    queue.append((new_board, next_player, new_move_sequence, move_count + 1))
    
    return shortest_wins

def demonstrate_exile_win(win_data, player):
    """Demonstrate an exile win step by step"""
    if not win_data:
        print(f"No exile win found for Player {player}")
        return
    
    player_symbol = "X" if player == "A" else "O"
    print(f"\n{'='*60}")
    print(f"EXILE WIN DEMONSTRATION - Player {player} ({player_symbol})")
    print(f"{'='*60}")
    print(f"Achieves {win_data['center_count']} tokens in centre in {win_data['moves']} moves")
    
    # Initial setup
    board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    board[0][0] = "A"  # X at A1
    board[3][3] = "B"  # O at D4
    
    print("\nStarting Position:")
    print_board_with_centre_highlight(board)
    
    # Execute each move
    for i, (p, move) in enumerate(win_data['sequence'], 1):
        ps = "X" if p == "A" else "O"
        print(f"Move {i}: {ps} plays {format_move(move)}")
        board = apply_move(board, p, move)
        print_board_with_centre_highlight(board)
        
        # Check centre count
        center_a = count_centre_tokens(board, "A")
        center_b = count_centre_tokens(board, "B")
        print(f"Centre control: X={center_a}, O={center_b}")
        print()
    
    print(f"🎉 EXILE WIN! Player {player} ({player_symbol}) wins!")
    print(f"Final centre count: {win_data['center_count']} tokens")

if __name__ == "__main__":
    shortest_wins = find_shortest_exile_wins()
    
    # Show results summary
    print(f"\n{'='*50}")
    print("SHORTEST EXILE WINS FOUND:")
    print(f"{'='*50}")
    
    for player in ["A", "B"]:
        symbol = "X" if player == "A" else "O"
        if shortest_wins[player]:
            moves = shortest_wins[player]['moves']
            centre = shortest_wins[player]['center_count']
            print(f"Player {player} ({symbol}): {moves} moves ({centre} tokens in centre)")
        else:
            print(f"Player {player} ({symbol}): No exile win found")
    
    # Demonstrate the shortest wins
    for player in ["A", "B"]:
        if shortest_wins[player]:
            demonstrate_exile_win(shortest_wins[player], player)
