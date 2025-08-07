"""
Monte Carlo simulation of Azers with tactical play using Alpha-Beta pruning.
Simulates 50,000 games for each configuration and reports statistics.
"""
import time
import random
from collections import defaultdict
from game_engine import (
    create_mode1_board, get_all_mode2_starting_positions,
    play_game, GameResult
)
from tactical_ai import create_tactical_ai

class SimulationStats:
    def __init__(self):
        self.total_games = 0
        self.wins_a = 0
        self.wins_b = 0
        self.draws = 0
        self.result_types = defaultdict(int)
        self.total_moves = 0
        self.max_moves = 0
        self.min_moves = float('inf')
        self.total_nodes_evaluated = 0
    
    def add_result(self, result: GameResult, nodes_evaluated: int = 0):
        self.total_games += 1
        self.total_moves += result.move_count
        self.total_nodes_evaluated += nodes_evaluated
        self.max_moves = max(self.max_moves, result.move_count)
        self.min_moves = min(self.min_moves, result.move_count)
        
        if result.winner == "A":
            self.wins_a += 1
        elif result.winner == "B":
            self.wins_b += 1
        else:
            self.draws += 1
        
        self.result_types[result.result_type] += 1
    
    def get_summary(self):
        if self.total_games == 0:
            return "No games played"
        
        avg_moves = self.total_moves / self.total_games
        avg_nodes = self.total_nodes_evaluated / self.total_games if self.total_games > 0 else 0
        win_rate_a = (self.wins_a / self.total_games) * 100
        win_rate_b = (self.wins_b / self.total_games) * 100
        draw_rate = (self.draws / self.total_games) * 100
        
        summary = f"""
=== SIMULATION RESULTS ===
Total Games: {self.total_games:,}
Player A (X) Wins: {self.wins_a:,} ({win_rate_a:.2f}%)
Player B (O) Wins: {self.wins_b:,} ({win_rate_b:.2f}%)
Draws: {self.draws:,} ({draw_rate:.2f}%)

Average Game Length: {avg_moves:.1f} moves
Shortest Game: {self.min_moves} moves
Longest Game: {self.max_moves} moves
Average Nodes per Game: {avg_nodes:,.0f}

Result Types:
"""
        for result_type, count in sorted(self.result_types.items()):
            percentage = (count / self.total_games) * 100
            summary += f"  {result_type}: {count:,} ({percentage:.2f}%)\n"
        
        return summary

def run_simulation(name: str, board_generator, 
                  search_depth: int = 8, num_games: int = 50000, display_boards: bool = False):
    """Run Monte Carlo simulation for a specific configuration"""
    print(f"\n{'='*60}")
    print(f"STARTING SIMULATION: {name}")
    print(f"Search Depth: {search_depth}")
    print(f"Target Games: {num_games:,}")
    print(f"Board Display: {'ENABLED' if display_boards else 'DISABLED'}")
    print(f"{'='*60}")
    
    stats = SimulationStats()
    start_time = time.time()
    
    # Create tactical AI strategies
    tactical_a = create_tactical_ai(search_depth)
    tactical_b = create_tactical_ai(search_depth)
    
    # Get all starting positions
    if callable(board_generator):
        starting_positions = [board_generator()]
    else:
        starting_positions = board_generator
    
    games_per_position = num_games // len(starting_positions)
    remaining_games = num_games % len(starting_positions)
    
    print(f"Starting positions: {len(starting_positions)}")
    print(f"Games per position: {games_per_position}")
    if remaining_games > 0:
        print(f"Additional games for first {remaining_games} positions")
    
    print(f"\nStarting tactical simulation with search depth {search_depth}...")
    print("Progress updates every 100 games (or 25 games for very slow depths)")
    print("=" * 60)
    
    position_count = 0
    for position_idx, (board, starting_player) in enumerate(starting_positions):
        position_count += 1
        
        # Determine number of games for this position
        pos_games = games_per_position
        if position_idx < remaining_games:
            pos_games += 1
        
        for game_num in range(pos_games):
            # Display starting board for Mode 2 games only if enabled
            is_mode2 = not callable(board_generator)  # Mode 2 uses a list of positions, Mode 1 uses a function
            if display_boards and is_mode2 and (stats.total_games < 5 or stats.total_games % 100 == 0):
                from game_engine import format_board_simple
                print(f"\n--- Tactical Game {stats.total_games + 1} Starting Position (Mode 2) ---")
                print(format_board_simple(board))
            
            result = play_game(board, starting_player, 
                             tactical_a, tactical_b, max_moves=100)  # Tactical games may be shorter
            
            # Note: We're not tracking individual AI node counts in this version
            # as it would require modifying the AI interface
            stats.add_result(result, 0)
            
            # Display final board for interesting games if enabled
            if display_boards and (stats.total_games <= 5 or result.move_count < 10 or result.result_type in ['MASSACRE', 'EXILE']):
                from game_engine import format_board_simple
                print(f"\n--- Tactical Game {stats.total_games} Final Board ---")
                if result.final_board:
                    print(format_board_simple(result.final_board))
                print(f"Game ended after {result.move_count} moves: {result.result_type}")
                if result.winner:
                    print(f"Winner: Player {result.winner}")
                else:
                    print("Result: Draw")
                print()
            
            # Progress update every 100 games for tactical (more frequent updates)
            if stats.total_games % 100 == 0:
                elapsed = time.time() - start_time
                games_per_sec = stats.total_games / elapsed if elapsed > 0 else 0
                eta_seconds = (num_games - stats.total_games) / games_per_sec if games_per_sec > 0 else 0
                eta_minutes = eta_seconds / 60
                eta_hours = eta_minutes / 60
                
                # Format ETA appropriately
                if eta_hours >= 1:
                    eta_str = f"{eta_hours:.1f} hours"
                else:
                    eta_str = f"{eta_minutes:.1f} min"
                
                print(f"Progress: {stats.total_games:,}/{num_games:,} games "
                      f"({stats.total_games/num_games*100:.1f}%) "
                      f"- {games_per_sec:.2f} games/sec "
                      f"- ETA: {eta_str}")
            
            # Also provide updates every 25 games for very slow runs (depth 10+)
            elif stats.total_games % 25 == 0 and stats.total_games > 0:
                elapsed = time.time() - start_time
                current_games_per_sec = stats.total_games / elapsed if elapsed > 0 else 0
                
                # Only show frequent updates if we're running very slowly
                if current_games_per_sec < 0.5:
                    eta_seconds = (num_games - stats.total_games) / current_games_per_sec if current_games_per_sec > 0 else 0
                    eta_hours = eta_seconds / 3600
                    
                    print(f"Progress: {stats.total_games:,}/{num_games:,} games "
                          f"({stats.total_games/num_games*100:.1f}%) "
                          f"- {current_games_per_sec:.3f} games/sec "
                          f"- ETA: {eta_hours:.1f} hours")
    
    elapsed_time = time.time() - start_time
    games_per_second = stats.total_games / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\nSIMULATION COMPLETED in {elapsed_time:.1f} seconds")
    print(f"Average speed: {games_per_second:.2f} games/second")
    print(stats.get_summary())
    
    return stats

def get_configuration_selection():
    """Let user choose which configurations to run"""
    print("\nCONFIGURATION SELECTION:")
    print("="*40)
    
    configurations = []
    
    # Mode selection
    while True:
        print("\nWhich game modes do you want to test?")
        print("1. Mode 1 only (opposing corners: A1 vs D4)")
        print("2. Mode 2 only (all possible starting positions)")  
        print("3. Both Mode 1 and Mode 2")
        
        mode_choice = input("\nEnter choice (1-3): ").strip()
        
        if mode_choice == "1":
            modes = ["mode1"]
            break
        elif mode_choice == "2":
            modes = ["mode2"] 
            break
        elif mode_choice == "3":
            modes = ["mode1", "mode2"]
            break
        else:
            print("Please enter 1, 2, or 3.")
    
    # Build configuration list
    for mode in modes:
        if mode == "mode1":
            configurations.append(("Mode 1", create_mode1_board))
        else:  # mode2
            configurations.append(("Mode 2", "mode2_positions"))
    
    return configurations

def get_user_parameters():
    """Get search depth and number of games from user with time estimates"""
    print("="*80)
    print("TACTICAL SIMULATION CONFIGURATION")
    print("="*80)
    
    # Get configuration selection
    configurations = get_configuration_selection()
    
    # Get search depth
    while True:
        try:
            print("\nSearch Depth Configuration:")
            print("  1-3: Very fast, weak play (~0.5-2 sec/game)")
            print("  4-6: Fast, decent play (~2-10 sec/game)")
            print("  7-9: Medium, good play (~10-60 sec/game)")
            print("  10+: Slow, strong play (~60+ sec/game)")
            
            depth_input = input("\nEnter search depth (1-15, recommended 6-8): ").strip()
            search_depth = int(depth_input)
            
            if 1 <= search_depth <= 15:
                break
            else:
                print("Please enter a value between 1 and 15.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get number of games
    while True:
        try:
            print(f"\nNumber of Games per Configuration:")
            print("  1,000: Quick test (~30min - 24hrs depending on depth)")
            print("  10,000: Good sample (~5hrs - 10 days)")
            print("  50,000: Full analysis (~24hrs - 50 days)")
            print("  100,000: Maximum precision (~48hrs - 100 days)")
            
            games_input = input("\nEnter number of games per configuration (100-100000): ").strip()
            num_games = int(games_input)
            
            if 100 <= num_games <= 100000:
                break
            else:
                print("Please enter a value between 100 and 100,000.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get board display preference
    while True:
        display_choice = input("\nDisplay board states for interesting games? (y/n, default n): ").strip().lower()
        if display_choice in ['y', 'yes']:
            display_boards = True
            break
        elif display_choice in ['n', 'no', '']:
            display_boards = False
            break
        else:
            print("Please enter 'y' for yes or 'n' for no.")
    
    # Calculate time estimates
    # Rough estimates based on search depth (games per second)
    time_estimates = {
        1: 50, 2: 30, 3: 20, 4: 10, 5: 5,
        6: 2, 7: 1, 8: 0.5, 9: 0.2, 10: 0.1,
        11: 0.05, 12: 0.02, 13: 0.01, 14: 0.005, 15: 0.002
    }
    
    games_per_sec = time_estimates.get(search_depth, 0.001)
    total_games = num_games * len(configurations)
    estimated_seconds = total_games / games_per_sec
    estimated_hours = estimated_seconds / 3600
    estimated_days = estimated_hours / 24
    
    print(f"\n{'='*60}")
    print("CONFIGURATION SUMMARY")
    print(f"{'='*60}")
    print(f"Search Depth: {search_depth}")
    print(f"Selected Configurations: {len(configurations)}")
    for name, _ in configurations:
        print(f"  - {name}")
    print(f"Games per Configuration: {num_games:,}")
    print(f"Total Games: {total_games:,}")
    print(f"Board Display: {'Enabled' if display_boards else 'Disabled'}")
    print(f"Estimated Speed: ~{games_per_sec} games/second")
    
    if estimated_hours < 1:
        print(f"Estimated Time: ~{estimated_seconds/60:.1f} minutes")
    elif estimated_days < 1:
        print(f"Estimated Time: ~{estimated_hours:.1f} hours")
    else:
        print(f"Estimated Time: ~{estimated_days:.1f} days")
    
    print(f"\nNote: Actual time may vary based on CPU performance and game complexity.")
    if display_boards:
        print("Note: Board display will show final game states for interesting games.")
        print("      Starting positions will only be shown for Mode 2 games.")
    
    return configurations, search_depth, num_games, display_boards

def main():
    """Run all Monte Carlo simulations for tactical play"""
    print("="*80)
    print("AZERS MONTE CARLO SIMULATION - TACTICAL PLAY")
    print("="*80)
    print("This simulation runs tactical games for selected configurations:")
    print("You can choose which modes to test.")
    print()
    print("Tactical play: Both players use Alpha-Beta search AI.")
    print("Note: Higher search depths will be significantly slower.")
    print()
    
    # Get user parameters
    configurations, search_depth, num_games, display_boards = get_user_parameters()
    
    response = input(f"\nProceed with tactical simulation? (y/n): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Simulation cancelled.")
        return
    
    # Set random seed for reproducibility
    random.seed(42)
    
    all_results = {}
    
    # Prepare Mode 2 positions if needed
    mode2_positions = None
    for config_name, board_gen in configurations:
        if "Mode 2" in config_name:
            if mode2_positions is None:
                print("Generating all unique starting positions for Mode 2 (excluding symmetries)...")
                mode2_positions = get_all_mode2_starting_positions()
                print(f"Found {len(mode2_positions)} unique starting positions for Mode 2")
            break
    
    # Run selected configurations
    for config_name, board_generator in configurations:
        if "Mode 2" in config_name:
            # Use pre-generated Mode 2 positions
            all_results[config_name] = run_simulation(
                config_name, mode2_positions, search_depth, num_games, display_boards)
        else:
            # Mode 1
            all_results[config_name] = run_simulation(
                config_name, board_generator, search_depth, num_games, display_boards)
    
    # Final summary
    print(f"\\n{'='*80}")
    print("FINAL SUMMARY - TACTICAL PLAY SIMULATIONS")
    print(f"{'='*80}")
    
    for config_name, stats in all_results.items():
        win_rate_a = (stats.wins_a / stats.total_games) * 100
        win_rate_b = (stats.wins_b / stats.total_games) * 100
        draw_rate = (stats.draws / stats.total_games) * 100
        avg_moves = stats.total_moves / stats.total_games
        
        print(f"\\n{config_name}:")
        print(f"  Player A: {win_rate_a:.2f}% | Player B: {win_rate_b:.2f}% | Draws: {draw_rate:.2f}%")
        print(f"  Average game length: {avg_moves:.1f} moves")
        
        # Show most common result types
        top_results = sorted(stats.result_types.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top result types: {', '.join([f'{rt}({c})' for rt, c in top_results])}")
    
    # Bias analysis
    print(f"\\n{'='*50}")
    print("BIAS ANALYSIS")
    print(f"{'='*50}")
    
    for config_name, stats in all_results.items():
        win_rate_a = (stats.wins_a / stats.total_games) * 100
        win_rate_b = (stats.wins_b / stats.total_games) * 100
        bias = abs(win_rate_a - win_rate_b)
        favoured = "A" if win_rate_a > win_rate_b else "B" if win_rate_b > win_rate_a else "NONE"
        
        print(f"{config_name}:")
        print(f"  Bias: {bias:.2f}% favouring Player {favoured}")
        if bias < 2:
            print("  -> WELL BALANCED")
        elif bias < 5:
            print("  -> SLIGHTLY BIASED")
        elif bias < 10:
            print("  -> MODERATELY BIASED")
        else:
            print("  -> HEAVILY BIASED")
    
    print(f"\\nTactical simulation completed! All results saved above.")
    print(f"\\nNote: Tactical play typically results in:")
    print("- Shorter games (better move selection)")
    print("- More decisive outcomes (fewer draws)")
    print("- Different bias patterns compared to random play")

if __name__ == "__main__":
    main()
