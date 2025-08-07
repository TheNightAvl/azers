"""
Core game engine for Azers Monte Carlo simulations.
Contains all the game logic needed for automated gameplay.
"""
import random
from typing import List, Tuple, Optional, Set

BOARD_SIZE = 4
CENTER_SQUARES = [(1, 1), (2, 1), (1, 2), (2, 2)]  # (x, y) coordinates

def get_opponent(player: str) -> str:
    return "B" if player == "A" else "A"

def in_bounds(x: int, y: int) -> bool:
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

def get_tokens(board: List[List[Optional[str]]], player: str) -> List[Tuple[int, int]]:
    return [(x, y) for y in range(BOARD_SIZE) for x in range(BOARD_SIZE) if board[y][x] == player]

def has_3_or_4_in_centre(board: List[List[Optional[str]]], player: str) -> Tuple[bool, int]:
    count = sum(1 for (x, y) in CENTER_SQUARES if board[y][x] == player)
    return count >= 3, count

def check_opposition_pattern(board: List[List[Optional[str]]]) -> bool:
    """
    Check if the centre four squares form an OPPOSITION pattern:
    X O    or    O X
    O X          X O
    """
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

def get_legal_moves(board: List[List[Optional[str]]], player: str) -> List[Tuple[str, Tuple[int, int], Tuple[int, int]]]:
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

def apply_move(board: List[List[Optional[str]]], player: str, move: Tuple[str, Tuple[int, int], Tuple[int, int]]) -> List[List[Optional[str]]]:
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

def board_to_tuple(board: List[List[Optional[str]]]) -> Tuple:
    """Convert board to tuple for hashing"""
    return tuple(tuple(row) for row in board)

def create_mode1_board() -> Tuple[List[List[Optional[str]]], str]:
    """Create Mode 1 starting position: A at (0,0), B at (3,3), A goes first"""
    board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    board[0][0] = "A"
    board[BOARD_SIZE-1][BOARD_SIZE-1] = "B"
    return board, "A"

def get_all_mode2_starting_positions() -> List[Tuple[List[List[Optional[str]]], str]]:
    """
    Generate all possible Mode 2 starting positions excluding symmetries.
    Returns list of (board, starting_player) tuples.
    """
    positions = []
    
    # Generate all possible pairs of starting positions
    for ax in range(BOARD_SIZE):
        for ay in range(BOARD_SIZE):
            for bx in range(BOARD_SIZE):
                for by in range(BOARD_SIZE):
                    # Skip same position
                    if (ax, ay) == (bx, by):
                        continue
                    
                    # Skip diagonal adjacency
                    if abs(ax - bx) == 1 and abs(ay - by) == 1:
                        continue
                    
                    board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
                    board[ay][ax] = "A"
                    board[by][bx] = "B"
                    
                    # B goes first in Mode 2
                    positions.append((board, "B"))
    
    # Remove rotational and reflectional symmetries
    unique_positions = []
    seen_boards = set()
    
    for board, starting_player in positions:
        # Generate all symmetric variants
        symmetric_boards = get_all_symmetries(board)
        
        # Check if we've seen any symmetric variant
        board_tuple = board_to_tuple(board)
        if not any(sym_board in seen_boards for sym_board in symmetric_boards):
            unique_positions.append((board, starting_player))
            # Add all symmetric variants to seen set
            seen_boards.update(symmetric_boards)
    
    return unique_positions

def get_all_symmetries(board: List[List[Optional[str]]]) -> Set[Tuple]:
    """Get all rotational and reflectional symmetries of a board"""
    symmetries = set()
    
    # Original
    symmetries.add(board_to_tuple(board))
    
    # 90 degree rotations
    for rotations in range(1, 4):
        rotated = board
        for _ in range(rotations):
            rotated = rotate_90(rotated)
        symmetries.add(board_to_tuple(rotated))
    
    # Reflections (horizontal, vertical, and both diagonals)
    # Horizontal reflection
    h_reflected = [row[::-1] for row in board]
    symmetries.add(board_to_tuple(h_reflected))
    
    # Vertical reflection
    v_reflected = board[::-1]
    symmetries.add(board_to_tuple(v_reflected))
    
    # Main diagonal reflection (transpose)
    d1_reflected = [[board[j][i] for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)]
    symmetries.add(board_to_tuple(d1_reflected))
    
    # Anti-diagonal reflection
    d2_reflected = [[board[BOARD_SIZE-1-j][BOARD_SIZE-1-i] for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)]
    symmetries.add(board_to_tuple(d2_reflected))
    
    return symmetries

def rotate_90(board: List[List[Optional[str]]]) -> List[List[Optional[str]]]:
    """Rotate board 90 degrees clockwise"""
    return [[board[BOARD_SIZE-1-j][i] for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)]

class GameResult:
    def __init__(self, winner: Optional[str], result_type: str, move_count: int, final_board: Optional[List[List[Optional[str]]]] = None):
        self.winner = winner  # "A", "B", or None for draw
        self.result_type = result_type  # "EXILE", "MASSACRE", "DEFENCE", "OPPOSITION", "BOREDOM"
        self.move_count = move_count
        self.final_board = final_board  # Final board state when game ended

def play_game(board: List[List[Optional[str]]], starting_player: str, 
              opposition_enabled: bool, player_a_strategy, player_b_strategy,
              max_moves: int = 200) -> GameResult:
    """
    Play a complete game using the given strategies.
    Returns GameResult with winner and game ending type.
    """
    current_board = [row[:] for row in board]
    current_player = starting_player
    move_count = 0
    
    # Track game states for boredom detection
    game_states = []
    CYCLE_LIMIT = 8
    
    while move_count < max_moves:
        # Get legal moves
        moves = get_legal_moves(current_board, current_player)
        
        # Check for DEFENCE (no legal moves)
        if not moves:
            return GameResult(get_opponent(current_player), "DEFENCE", move_count, current_board)
        
        # Get move from appropriate strategy
        if current_player == "A":
            move = player_a_strategy(current_board, current_player, moves)
        else:
            move = player_b_strategy(current_board, current_player, moves)
        
        # Apply move
        current_board = apply_move(current_board, current_player, move)
        move_count += 1
        
        # Check win conditions after move
        
        # MASSACRE (opponent has no tokens)
        opponent = get_opponent(current_player)
        opponent_tokens = get_tokens(current_board, opponent)
        if not opponent_tokens:
            return GameResult(current_player, "MASSACRE", move_count, current_board)
        
        # EXILE (3 or 4 in centre)
        win, count = has_3_or_4_in_centre(current_board, current_player)
        if win:
            return GameResult(current_player, "EXILE", move_count, current_board)
        
        # OPPOSITION draw (if enabled)
        if opposition_enabled and check_opposition_pattern(current_board):
            return GameResult(None, "OPPOSITION", move_count, current_board)
        
        # BOREDOM (cycle detection)
        state = (board_to_tuple(current_board), current_player)
        game_states.append(state)
        if game_states.count(state) >= CYCLE_LIMIT:
            return GameResult(None, "BOREDOM", move_count, current_board)
        
        # Limit state tracking to avoid memory issues
        if len(game_states) > 100:
            game_states.pop(0)
        
        # Switch players
        current_player = get_opponent(current_player)
    
    # Game exceeded max moves - call it boredom
    return GameResult(None, "BOREDOM", move_count, current_board)

def random_strategy(board: List[List[Optional[str]]], player: str, moves: List) -> Tuple:
    """Random move selection strategy"""
    return random.choice(moves)

def format_board_simple(board: List[List[Optional[str]]]) -> str:
    """Simple board representation for debugging"""
    result = "   A B C D\n"
    for y in range(BOARD_SIZE):
        row_str = f"{y+1}  "
        for x in range(BOARD_SIZE):
            if board[y][x] == "A":
                row_str += "X "
            elif board[y][x] == "B":
                row_str += "O "
            else:
                row_str += ". "
        result += row_str + "\n"
    return result
