"""
Analyse the probability of outcomes and bias in Azers for Mode 1 and Mode 2
using Monte Carlo simulations with both players playing tactically.

Mode 1: Opposing corners
Mode 2: All possible valid starting positions
"""
import os
import sys
import itertools
import random

# Ensure tactical_ai is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
monte_carlo_path = current_dir
if monte_carlo_path not in sys.path:
    sys.path.insert(0, monte_carlo_path)

import tactical_ai

def create_board(size=4):
    return [[None for _ in range(size)] for _ in range(size)]

def get_opponent(player):
    return "B" if player == "A" else "A"

def get_tokens(board, player):
    return [(x, y) for y in range(len(board)) for x in range(len(board)) if board[y][x] == player]

def in_bounds(x, y, size=4):
    return 0 <= x < size and 0 <= y < size

def get_legal_moves(board, player):
    moves = []
    size = len(board)
    for (x, y) in get_tokens(board, player):
        for dx in [-2, 0, 2]:
            for dy in [-2, 0, 2]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                mx, my = x + dx // 2, y + dy // 2
                if in_bounds(nx, ny, size) and board[ny][nx] is None:
                    if board[my][mx] is None:
                        moves.append(("DROP", (x, y), (nx, ny)))
                    elif board[my][mx] is not None:
                        moves.append(("HOP", (x, y), (nx, ny)))
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if in_bounds(nx, ny, size) and board[ny][nx] == get_opponent(player):
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

def has_3_or_4_in_centre(board, player):
    centre = [(1, 1), (2, 1), (1, 2), (2, 2)]
    count = sum(1 for (x, y) in centre if board[y][x] == player)
    return count >= 3, count

def run_game(mode, tactical_a, tactical_b, a_pos=None, b_pos=None):
    board = create_board()
    if mode == 1:
        board[0][0] = "A"
        board[3][3] = "B"
        player = "A"
    else:
        board[a_pos[1]][a_pos[0]] = "A"
        board[b_pos[1]][b_pos[0]] = "B"
        player = "B"  # Mode 2: B goes first
    previous_states = []
    CYCLE_LIMIT = 8
    while True:
        current_ai = tactical_a if player == "A" else tactical_b
        moves = get_legal_moves(board, player)
        if not moves:
            return get_opponent(player)  # Defence win
        move = current_ai.get_move(board, player, moves)
        board = apply_move(board, player, move)
        opponent = get_opponent(player)
        opponent_tokens = get_tokens(board, opponent)
        if not opponent_tokens:
            return player  # Massacre win
        win, count = has_3_or_4_in_centre(board, player)
        if win:
            return player  # Exile win
        opponent_moves = get_legal_moves(board, opponent)
        if not opponent_moves:
            return player  # Defence win
        state = (tuple(tuple(row) for row in board), player)
        previous_states.append(state)
        if previous_states.count(state) >= CYCLE_LIMIT:
            return "Draw"  # Boredom draw
        if len(previous_states) > 100:
            previous_states.pop(0)
        player = opponent

def all_valid_mode2_positions():
    size = 4
    positions = []
    for ax in range(size):
        for ay in range(size):
            for bx in range(size):
                for by in range(size):
                    if (ax, ay) == (bx, by):
                        continue
                    if abs(ax - bx) == 1 and abs(ay - by) == 1:
                        continue
                    positions.append(((ax, ay), (bx, by)))
    return positions

def run_mode_analysis(mode, tactical_a, tactical_b, N, positions=None):
    """Run analysis for a specific mode"""
    import time
    results = {"A": 0, "B": 0, "Draw": 0}
    
    if mode == 1:
        total_games = N
        print(f"Simulating Mode 1 ({N} games)...")
        start_time = time.time()
        for i in range(N):
            outcome = run_game(1, tactical_a, tactical_b)
            results[outcome] += 1
            elapsed = time.time() - start_time
            games_done = i + 1
            games_left = N - games_done
            if games_done > 0:
                per_game = elapsed / games_done
                eta = elapsed / games_done * games_left
                eta_h = eta // 3600
                eta_m = eta / 60 - eta_h * 60
                eta_s = eta % 60
                print(f"  Progress: {games_done}/{N} games ({games_done/N:.1%}), ETA: {int(eta_h)}h{int(eta_m)}m{int(eta_s)}s (avg. {per_game:.2f}s/game)", end='\r')
        print()
    else:  # mode == 2
        total_to_simulate = (N // len(positions)) * len(positions)
        total_games = total_to_simulate
        print(f"Simulating Mode 2 ({total_to_simulate} games)...")
        start_time = time.time()
        games_done = 0
        for idx, (a_pos, b_pos) in enumerate(positions):
            for j in range(N // len(positions)):
                outcome = run_game(2, tactical_a, tactical_b, a_pos, b_pos)
                results[outcome] += 1
                games_done += 1
                elapsed = time.time() - start_time
                games_left = total_to_simulate - games_done
                if games_done > 0:
                    per_game = elapsed / games_done
                    eta = elapsed / games_done * games_left
                    eta_h = eta // 3600
                    eta_m = eta / 60 - eta_h * 60
                    eta_s = eta % 60
                    print(f"  Progress: {games_done}/{total_to_simulate} games ({games_done/total_to_simulate:.1%}), ETA: {int(eta_h)}h{int(eta_m)}m{int(eta_s)}s (avg. {per_game:.2f}s/game)", end='\r')
        print()
    
    return results, total_games

def main():
    print("Analysing Azers outcome probabilities and bias using AI (Monte Carlo simulation)")
    print("Mode 1: Opposing corners")
    print("Mode 2: All valid starting positions")
    print()
    
    # AI selection
    print("AI Configuration:")
    print("1. Tactical AI only (Alpha-Beta search)")
    print("2. Learning AI only (with opening book)")
    print("3. Tactical AI vs Learning AI")
    
    while True:
        ai_choice = input("Select AI configuration (1-3): ").strip()
        if ai_choice in ['1', '2', '3']:
            break
        print("Please enter 1, 2, or 3.")
    
    if ai_choice == '1':
        print("Using Tactical AI for both players")
        tactical_a = tactical_ai.TacticalAI(max_depth=6, learning_enabled=False, randomness=0.1)
        tactical_b = tactical_ai.TacticalAI(max_depth=6, learning_enabled=False, randomness=0.1)
    elif ai_choice == '2':
        print("Using Learning AI for both players")
        tactical_a = tactical_ai.TacticalAI(max_depth=6, learning_enabled=True, randomness=0.1)
        tactical_b = tactical_ai.TacticalAI(max_depth=6, learning_enabled=True, randomness=0.1)
    else:  # ai_choice == '3'
        print("Using Tactical AI (Player A) vs Learning AI (Player B)")
        tactical_a = tactical_ai.TacticalAI(max_depth=6, learning_enabled=False, randomness=0.1)
        tactical_b = tactical_ai.TacticalAI(max_depth=6, learning_enabled=True, randomness=0.1)
    
    # Get number of simulations from user
    print()
    while True:
        try:
            N = int(input("Enter number of simulations per position (recommended: 1000): ").strip())
            if N <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"Running {N} simulations per position...")
    positions = all_valid_mode2_positions()  # For Mode 2
    
    # Store all results for comprehensive analysis
    all_results = {}
    
    # Run analysis for all modes
    for mode in [1, 2]:
        key = f"Mode {mode}"
        if mode == 1:
            results, total_games = run_mode_analysis(mode, tactical_a, tactical_b, N)
        else:
            results, total_games = run_mode_analysis(mode, tactical_a, tactical_b, N, positions)
        
        all_results[key] = (results, total_games)
        
        # Print results for this configuration
        print(f"{key} ({total_games} games):")
        print(f"  Player A wins: {results['A']} ({results['A']/total_games:.1%})")
        print(f"  Player B wins: {results['B']} ({results['B']/total_games:.1%})")
        print(f"  Draws: {results['Draw']} ({results['Draw']/total_games:.1%})")
        print()
    
    # Comprehensive bias analysis
    print("=" * 60)
    print("COMPREHENSIVE BIAS ANALYSIS")
    print("=" * 60)
    
    for key, (results, total_games) in all_results.items():
        bias = results['A']/total_games - results['B']/total_games
        print(f"{key}:")
        print(f"  Player A win rate: {results['A']/total_games:.1%}")
        print(f"  Player B win rate: {results['B']/total_games:.1%}")
        print(f"  Bias (A - B): {bias:.2%}")
        if abs(bias) < 0.05:
            print("  Assessment: Well balanced")
        elif bias > 0:
            print(f"  Assessment: Favours Player A by {bias:.1%}")
        else:
            print(f"  Assessment: Favours Player B by {abs(bias):.1%}")
        print()

if __name__ == "__main__":
    main()
