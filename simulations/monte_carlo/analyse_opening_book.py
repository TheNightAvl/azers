"""
Opening Book Analyser - Examine learned AI strategies.
"""
import json
import os
from collections import defaultdict

def analyse_opening_book(filename: str = "opening_book.json"):
    """Analyse an opening book file and show interesting statistics"""
    
    if not os.path.exists(filename):
        print(f"Opening book file '{filename}' not found.")
        return
    
    try:
        with open(filename, 'r') as f:
            book = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading opening book: {e}")
        return
    
    if not book:
        print("Opening book is empty.")
        return
    
    print(f"=== OPENING BOOK ANALYSIS: {filename} ===")
    print()
    
    # Basic statistics
    total_positions = len(book)
    total_moves = sum(len(moves) for moves in book.values())
    total_games = sum(sum(move_data['total'] for move_data in moves.values()) 
                     for moves in book.values())
    
    print(f"Total positions: {total_positions:,}")
    print(f"Total moves: {total_moves:,}")
    print(f"Total games recorded: {total_games:,}")
    print(f"Average moves per position: {total_moves/total_positions:.1f}")
    print()
    
    # Find most experienced positions
    position_games = []
    for pos_hash, moves in book.items():
        pos_total = sum(move_data['total'] for move_data in moves.values())
        position_games.append((pos_total, pos_hash, moves))
    
    position_games.sort(reverse=True)
    
    print("=== MOST EXPERIENCED POSITIONS ===")
    for i, (games, pos_hash, moves) in enumerate(position_games[:5]):
        print(f"{i+1}. Position with {games} games:")
        print(f"   Board: {format_board_hash(pos_hash)}")
        
        # Show best moves for this position
        move_scores = []
        for move_str, stats in moves.items():
            win_rate = (stats['wins'] + 0.5 * stats['draws']) / stats['total']
            move_scores.append((win_rate, move_str, stats))
        
        move_scores.sort(reverse=True)
        print(f"   Best moves:")
        for j, (win_rate, move_str, stats) in enumerate(move_scores[:3]):
            print(f"     {j+1}. {move_str}: {win_rate:.1%} win rate "
                  f"({stats['wins']}W-{stats['draws']}D-{stats['losses']}L)")
        print()
    
    # Find best and worst performing moves overall
    all_moves = []
    for moves in book.values():
        for move_str, stats in moves.items():
            if isinstance(stats, dict) and 'total' in stats and stats['total'] >= 5:  # Only consider moves with enough data
                win_rate = (stats['wins'] + 0.5 * stats['draws']) / stats['total']
                all_moves.append((win_rate, move_str, stats))
    
    if all_moves:
        all_moves.sort(key=lambda x: x[0], reverse=True)
        
        print("=== BEST PERFORMING MOVES (min 5 games) ===")
        for i, (win_rate, move_str, stats) in enumerate(all_moves[:10]):
            print(f"{i+1}. {move_str}: {win_rate:.1%} win rate "
                  f"({stats['wins']}W-{stats['draws']}D-{stats['losses']}L in {stats['total']} games)")
        
        print(f"\n=== WORST PERFORMING MOVES (min 5 games) ===")
        for i, (win_rate, move_str, stats) in enumerate(all_moves[-5:]):
            print(f"{i+1}. {move_str}: {win_rate:.1%} win rate "
                  f"({stats['wins']}W-{stats['draws']}D-{stats['losses']}L in {stats['total']} games)")
    
    # Move type analysis
    move_types = defaultdict(lambda: {'count': 0, 'total_games': 0, 'total_wins': 0})
    
    for moves in book.values():
        for move_str, stats in moves.items():
            try:
                move = eval(move_str)
                move_type = move[0]  # First element is move type
                move_types[move_type]['count'] += 1
                move_types[move_type]['total_games'] += stats['total']
                move_types[move_type]['total_wins'] += stats['wins'] + 0.5 * stats['draws']
            except:
                continue
    
    print(f"\n=== MOVE TYPE ANALYSIS ===")
    for move_type, data in sorted(move_types.items()):
        if data['total_games'] > 0:
            win_rate = data['total_wins'] / data['total_games']
            print(f"{move_type}: {data['count']} moves, {data['total_games']} games, "
                  f"{win_rate:.1%} avg win rate")

def format_board_hash(board_hash: str) -> str:
    """Format a board hash string for display"""
    if len(board_hash) != 16:  # 4x4 board
        return board_hash
    
    formatted = ""
    for i in range(0, 16, 4):
        row = board_hash[i:i+4]
        row = row.replace('.', '·')  # Use middle dot for empty squares
        formatted += row + " "
    return formatted.strip()

def compare_opening_books(file1: str, file2: str):
    """Compare two opening books to see differences"""
    
    print(f"=== COMPARING OPENING BOOKS ===")
    print(f"File 1: {file1}")
    print(f"File 2: {file2}")
    print()
    
    # Load both books
    try:
        with open(file1, 'r') as f:
            book1 = json.load(f)
        with open(file2, 'r') as f:
            book2 = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading files: {e}")
        return
    
    # Find common positions
    common_positions = set(book1.keys()) & set(book2.keys())
    unique_to_1 = set(book1.keys()) - set(book2.keys())
    unique_to_2 = set(book2.keys()) - set(book1.keys())
    
    print(f"Positions in {file1}: {len(book1)}")
    print(f"Positions in {file2}: {len(book2)}")
    print(f"Common positions: {len(common_positions)}")
    print(f"Unique to {file1}: {len(unique_to_1)}")
    print(f"Unique to {file2}: {len(unique_to_2)}")
    print()
    
    # Compare move preferences in common positions
    significant_differences = []
    
    for pos in common_positions:
        moves1 = book1[pos]
        moves2 = book2[pos]
        
        # Find moves that exist in both books
        common_moves = set(moves1.keys()) & set(moves2.keys())
        
        for move in common_moves:
            stats1 = moves1[move]
            stats2 = moves2[move]
            
            if stats1['total'] >= 5 and stats2['total'] >= 5:
                win_rate1 = (stats1['wins'] + 0.5 * stats1['draws']) / stats1['total']
                win_rate2 = (stats2['wins'] + 0.5 * stats2['draws']) / stats2['total']
                
                diff = abs(win_rate1 - win_rate2)
                if diff > 0.2:  # 20% difference
                    significant_differences.append((diff, move, win_rate1, win_rate2, pos))
    
    if significant_differences:
        significant_differences.sort(reverse=True)
        
        print("=== SIGNIFICANT DIFFERENCES IN MOVE EVALUATIONS ===")
        for i, (diff, move, rate1, rate2, pos) in enumerate(significant_differences[:5]):
            print(f"{i+1}. Move {move}:")
            print(f"   {file1}: {rate1:.1%} win rate")
            print(f"   {file2}: {rate2:.1%} win rate")
            print(f"   Difference: {diff:.1%}")
            print(f"   Position: {format_board_hash(pos)}")
            print()

def main():
    """Main function for opening book analysis"""
    print("Opening Book Analyser")
    print("="*50)
    print()
    
    while True:
        print("Options:")
        print("1. Analyse an opening book")
        print("2. Compare two opening books")
        print("3. List available opening book files")
        print("4. Exit")
        
        choice = input("\nChoose an option (1-4): ").strip()
        
        if choice == "1":
            filename = input("Enter opening book filename (default: opening_book.json): ").strip()
            if not filename:
                filename = "opening_book.json"
            analyse_opening_book(filename)
            
        elif choice == "2":
            file1 = input("Enter first opening book filename: ").strip()
            file2 = input("Enter second opening book filename: ").strip()
            if file1 and file2:
                compare_opening_books(file1, file2)
            else:
                print("Please provide both filenames.")
                
        elif choice == "3":
            print("Available opening book files:")
            json_files = [f for f in os.listdir('.') if f.endswith('.json') and 'book' in f.lower()]
            if json_files:
                for i, filename in enumerate(json_files, 1):
                    print(f"  {i}. {filename}")
            else:
                print("  No opening book files found in current directory.")
                
        elif choice == "4":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
        
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()
