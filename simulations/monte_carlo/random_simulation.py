"""
Monte Carlo simulation of Azers with random play.
Simulates 50,000 games for each configuration and reports statistics.
"""
import time
import random
from collections import defaultdict
from game_engine import (
    create_mode1_board, get_all_mode2_starting_positions,
    play_game, random_strategy, GameResult
)

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
    
    def add_result(self, result: GameResult):
        self.total_games += 1
        self.total_moves += result.move_count
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

Result Types:
"""
        for result_type, count in sorted(self.result_types.items()):
            percentage = (count / self.total_games) * 100
            summary += f"  {result_type}: {count:,} ({percentage:.2f}%)\n"
        
        return summary

def run_simulation(name: str, board_generator, num_games: int = 50000, display_boards: bool = False):
    """Run Monte Carlo simulation for a specific configuration"""
    print(f"\n{'='*60}")
    print(f"STARTING SIMULATION: {name}")
    print(f"Target Games: {num_games:,}")
    print(f"Board Display: {'ENABLED' if display_boards else 'DISABLED'}")
    print(f"{'='*60}")
    
    stats = SimulationStats()
    start_time = time.time()
    
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
            if display_boards and is_mode2 and (stats.total_games < 5 or stats.total_games % 1000 == 0):
                from game_engine import format_board_simple
                print(f"\n--- Game {stats.total_games + 1} Starting Position (Mode 2) ---")
                print(format_board_simple(board))
            
            result = play_game(board, starting_player, 
                             random_strategy, random_strategy)
            stats.add_result(result)
            
            # Display final board for interesting games if enabled
            if display_boards and (stats.total_games <= 5 or result.move_count < 15 or result.move_count > 100 or result.result_type in ['MASSACRE', 'EXILE']):
                from game_engine import format_board_simple
                print(f"\n--- Game {stats.total_games} Final Board ---")
                if result.final_board:
                    print(format_board_simple(result.final_board))
                print(f"Game ended after {result.move_count} moves: {result.result_type}")
                if result.winner:
                    print(f"Winner: Player {result.winner}")
                else:
                    print("Result: Draw")
                print()
            
            # Progress update every 1000 games
            if stats.total_games % 1000 == 0:
                elapsed = time.time() - start_time
                games_per_sec = stats.total_games / elapsed if elapsed > 0 else 0
                eta_seconds = (num_games - stats.total_games) / games_per_sec if games_per_sec > 0 else 0
                eta_minutes = eta_seconds / 60
                
                print(f"Progress: {stats.total_games:,}/{num_games:,} games "
                      f"({stats.total_games/num_games*100:.1f}%) "
                      f"- {games_per_sec:.1f} games/sec "
                      f"- ETA: {eta_minutes:.1f} min")
    
    elapsed_time = time.time() - start_time
    games_per_second = stats.total_games / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\nSIMULATION COMPLETED in {elapsed_time:.1f} seconds")
    print(f"Average speed: {games_per_second:.1f} games/second")
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
    """Get number of games from user with time estimates"""
    print("="*80)
    print("RANDOM SIMULATION CONFIGURATION")
    print("="*80)
    
    # Get configuration selection
    configurations = get_configuration_selection()
    
    # Get number of games
    while True:
        try:
            print(f"\nNumber of Games per Configuration:")
            print("  1,000: Quick test (~1-2 minutes per config)")
            print("  10,000: Good sample (~10-20 minutes per config)")
            print("  50,000: Full analysis (~50-100 minutes per config)")
            print("  100,000: Maximum precision (~2-3 hours per config)")
            
            games_input = input("\nEnter number of games per configuration (100-1000000): ").strip()
            num_games = int(games_input)
            
            if 100 <= num_games <= 1000000:
                break
            else:
                print("Please enter a value between 100 and 1,000,000.")
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
    
    # Calculate time estimates for random play (much faster than tactical)
    # Random games are typically 1000-3000 games per second
    games_per_sec = 2000  # Conservative estimate
    total_games = num_games * len(configurations)
    estimated_seconds = total_games / games_per_sec
    estimated_hours = estimated_seconds / 3600
    
    print(f"\n{'='*60}")
    print("CONFIGURATION SUMMARY")
    print(f"{'='*60}")
    print(f"Selected Configurations: {len(configurations)}")
    for name, _, _ in configurations:
        print(f"  - {name}")
    print(f"Games per Configuration: {num_games:,}")
    print(f"Total Games: {total_games:,}")
    print(f"Board Display: {'Enabled' if display_boards else 'Disabled'}")
    print(f"Estimated Speed: ~{games_per_sec:,} games/second")
    
    if estimated_hours < 1:
        print(f"Estimated Time: ~{estimated_seconds/60:.1f} minutes")
    else:
        print(f"Estimated Time: ~{estimated_hours:.1f} hours")
    
    print(f"\nNote: Actual time may vary based on CPU performance.")
    if display_boards:
        print("Note: Board display will show final game states for interesting games.")
        print("      Starting positions will only be shown for Mode 2 games.")
    
    return configurations, num_games, display_boards

def main():
    """Run all Monte Carlo simulations for random play"""
    print("="*80)
    print("AZERS MONTE CARLO SIMULATION - RANDOM PLAY")
    print("="*80)
    print("This simulation runs random games for selected configurations:")
    print("You can choose which modes to test.")
    print()
    print("Random play: Both players select moves completely at random.")
    print()
    
    # Get user parameters
    configurations, num_games, display_boards = get_user_parameters()
    
    response = input(f"\nProceed with random simulation? (y/n): ").strip().lower()
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
                config_name, mode2_positions, num_games, display_boards)
        else:
            # Mode 1
            all_results[config_name] = run_simulation(
                config_name, board_generator, num_games, display_boards)
    
    # Final summary
    print(f"\\n{'='*80}")
    print("FINAL SUMMARY - RANDOM PLAY SIMULATIONS")
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
    
    print(f"\\nSimulation completed! All results saved above.")

if __name__ == "__main__":
    main()
