# Azers Game

A Python implementation of the Azers board game with advanced AI and comprehensive Monte Carlo simulation capabilities.

For detailed rules and game information, visit the [Azers Wiki page](https://wiki.laenkea.si/wiki/Appendix:World/Azers).

## Features

- **Interactive Game Play**: Human vs Human or Human vs AI
- **Advanced AI**: Alpha-Beta pruning with configurable search depth and learning capabilities
- **Machine Learning AI**: Self-improving AI with pattern recognition and opponent modelling
- **Monte Carlo Simulations**: Comprehensive game balance analysis with visual board displays
- **Statistical Analysis**: Detailed bias and performance metrics with real-time progress tracking
- **Enhanced Learning System**: Progressive learning with pattern recognition and adaptive strategy
- **Visual Board Analysis**: Compact grid format displays of game states and position analysis

## Quick Start

### Play the Game
```bash
python azers.py
```

### Run Simulations
```bash
cd simulations/monte_carlo
python run_simulations.py
```

## Game Rules

- Two players (A and B) start with one token each
- Players can DROP, HOP, or SWAP tokens
- Win conditions:
  - **EXILE**: Occupy 3 or 4 centre squares
  - **MASSACRE**: Capture all opponent tokens
  - **DEFENCE**: Force opponent into a position with no legal moves
- Draw conditions:
  - **BOREDOM**: Repetitive moves detected
  - **OPPOSITION** (optional): Centre four squares form an alternating pattern:
    ```
    A B    or    B A
    B A          A B
    ```

## Controls

- Select tokens by number when you have multiple
- Choose moves by number from the available options
- Enter positions as column-row (e.g., B2, A1, D4)

## Monte Carlo Simulations

The simulation suite provides comprehensive analysis of game balance and AI performance:

### Simulation Types

1. **Random Play Simulation**
   - Both players make completely random moves
   - Fast execution for baseline statistics
   - Reveals inherent game biases

2. **Tactical Play Simulation**
   - Both players use Alpha-Beta search AI
   - Configurable search depth (1-15)
   - Strategic gameplay analysis
   - Time estimates and progress tracking

3. **Learning AI Simulation**
   - Advanced AI learns through self-play training with machine learning features
   - **Pattern Recognition**: Identifies 5 types of tactical patterns (centre control, token advantage, mobility, defensive, aggressive)
   - **Opponent Modelling**: Adapts strategy based on opponent behaviour patterns
   - **Progressive Learning**: Enhanced learning algorithms with weighted position evaluation
   - Builds comprehensive opening book from experience
   - Evaluates learning effectiveness with detailed performance analysis
   - Compares learned AI vs baseline AI with statistical significance testing
   - **Real-time progress**: Shows detailed progress after every game with visual board displays
   - **Board state analysis**: Shows final boards in compact grid format for interesting games
   - **Performance indicators**: Visual symbols for AI wins, draws, and losses
   - **Frequent updates**: Progress summaries every 5-10 games with comprehensive statistics

4. **Enhanced Simulation**
   - All simulation types with comprehensive board visualisation
   - Automatic detection and display of interesting games
   - Final board states for draws and long games
   - **Smart board display**: Shows starting positions only for Mode 2 games (Mode 1 always starts identically)
   - **Compact grid format**: 3-line visual board representation showing token positions

### Configuration Options

- **Game Modes**: Mode 1 (opposing corners) and/or Mode 2 (all positions)
- **Opposition Rule**: With/without opposition draw condition
- **Game Count**: 100 to 1,000,000 games per configuration
- **Search Depth**: 1-15 for tactical simulations
- **Selective Testing**: Choose specific configurations to reduce computation

### Usage Examples

#### Quick Learning Test
#### Full Analysis
```bash
cd simulations/monte_carlo
python run_simulations.py
# Choose option 4 for comprehensive analysis with board display
# Choose option 3 for learning AI with game-by-game progress
```

#### Analyse AI Learning
```bash
cd simulations/monte_carlo
python analyse_opening_book.py
```

## Advanced AI Learning System

### Machine Learning Features ⭐ **ENHANCED**

The learning AI now includes sophisticated machine learning capabilities:

#### Pattern Recognition System
- **Centre Control Patterns**: Identifies and values centre square dominance
- **Token Advantage Patterns**: Recognises numerical superiority positions
- **Mobility Patterns**: Evaluates movement freedom and piece flexibility  
- **Defensive Patterns**: Detects protective and blocking formations
- **Aggressive Patterns**: Identifies attacking opportunities and threats

#### Opponent Modelling
- **Adaptive Strategy**: Adjusts tactics based on opponent's playing style
- **Confidence Tracking**: Builds confidence in predictions over time
- **Behavioural Analysis**: Learns from opponent's move preferences and patterns

#### Progressive Learning
- **Weighted Position Evaluation**: More sophisticated position scoring
- **Extended Learning Depth**: Enhanced learning beyond opening moves
- **Pattern-Weighted Decisions**: Move selection influenced by recognised patterns

### Learning Modes

1. **Mode 1 (Basic)**: Traditional opening book learning
2. **Mode 2 (Enhanced)**: Adds pattern recognition system
3. **Mode 3 (Advanced)**: Full ML with pattern recognition + opponent modelling

### How It Works

The enhanced learning AI improves through experience by:

1. **Opening Book Creation**: Records board positions and move outcomes
2. **Pattern Analysis**: Identifies tactical patterns and their success rates
3. **Opponent Adaptation**: Models opponent behaviour and adapts strategy
4. **Progressive Enhancement**: Weighted learning with position importance
5. **Win Rate Tracking**: Calculates success rates for moves and patterns
6. **Strategic Selection**: Prefers moves with higher historical performance
7. **Continuous Learning**: Improves with more training games
8. **Persistent Storage**: Saves knowledge to JSON files

### Real-Time Progress Tracking

The learning system provides comprehensive feedback:

- **Game-by-game updates**: Shows progress after every single game with visual board grids
- **Learning phase**: Displays game number, winner, move count, pattern hits, and ETA
- **Evaluation phase**: Shows which side AI played with performance indicators
- **Visual feedback**: 🤖 for AI wins, 🔸 for draws, 🔹 for baseline wins
- **Frequent summaries**: Detailed statistics every 5-10 games with board analysis
- **Board states**: Automatically displays final positions in compact grid format
- **Pattern statistics**: Real-time tracking of pattern recognition effectiveness
- **Opponent insights**: Adaptation statistics and confidence levels

### Learning Configuration

- **Training Games**: 100-10,000 (recommended: 1000+)
- **Evaluation Games**: 50-1000 (recommended: 200)
- **Search Depth**: 1-10 (recommended: 6 for learning)
- **Learning Modes**: 3 modes from basic to advanced ML
- **Opening Book**: Learns first 8 moves of games
- **Performance Threshold**: Uses moves with >40% success rate
- **Pattern Recognition**: 5 tactical pattern types with success tracking
- **Opponent Modelling**: Adaptive strategy based on opponent analysis
- **Interesting Games**: Automatically identifies draws and long games (≥30 moves)
- **Board Display**: Shows final positions in compact grid format for interesting games
- **Smart Display Logic**: Starting positions shown only for Mode 2 games (Mode 1 starts identically)

### Expected Results

- **Typical Improvement**: 8-20% win rate increase with 1000+ training games (enhanced from 5-15%)
- **Pattern Recognition Impact**: Additional 3-8% improvement in strategic positions
- **Opponent Adaptation**: 2-5% further improvement against predictable opponents
- **Learning Speed**: Most improvement in first 500 games, with pattern learning continuing longer
- **Best Depth**: Depth 6-8 provides good balance of speed and strategy
- **File Persistence**: Opening books saved as `enhanced_learning_book_depth_X_modeY.json`
- **Progress Feedback**: Real-time ETA and performance tracking with visual board analysis
- **Strategic Insights**: Board state analysis with pattern recognition for interesting games

### Learning Files

- `opening_book.json`: Default opening book
- `learning_book_depth_X.json`: Basic depth-specific opening books  
- `enhanced_learning_book_depth_X_modeY.json`: Enhanced ML opening books with pattern data

## Visual Board Display System

### Compact Grid Format

All simulations now use a compact 3-line visual board representation:

```
--- Final Board State ---
   A B C D
1  A B A B   ← Row 1: Tokens or dots (.) for empty
2  B . . A   ← Row 2: Clear visual layout  
3  . A B .   ← Row 3: Easy pattern recognition
4  B A A B
Winner: Player A (EXILE)
```

### Smart Display Logic

- **Mode 1 Games**: Shows final board states only (opposing corners start A1 vs D4 is standard)
- **Mode 2 Games**: Shows both starting and final boards (starting positions vary significantly)
- **Interesting Games**: Final boards automatically shown for draws, long games (≥25 moves), or significant position changes
- **Progress Summaries**: Visual board grids included in frequent progress updates (every 5-10 games)

### Display Benefits

- **Visual Pattern Recognition**: Easy to spot centre control and token distributions
- **Compact Format**: Minimal screen space while maintaining clarity
- **Educational Value**: Learn from AI games by seeing final positions
- **Strategic Analysis**: Understand how games develop and conclude

## Performance Characterisation

### Simulation Speed (approximate)

- **Random Play**: 1000-5000 games/second
- **Tactical Depth 4**: 10-50 games/second
- **Tactical Depth 8**: 1-5 games/second
- **Tactical Depth 12**: 0.1-0.5 games/second
- **Learning AI**: Varies by depth, shows real-time ETA

### Memory Usage

- **Game Engine**: Minimal (< 1 MB)
- **Opening Books**: 1-10 MB for extensive learning
- **Simulation Data**: Temporary, cleared after each run

## File Structure

```
Azers/
├── azers.py                    # Main game file with enhanced AI integration
├── README.md                   # This file
├── LICENSE                     # Licence information
└── simulations/
    └── monte_carlo/
        ├── run_simulations.py      # Main simulation interface with board display
        ├── game_engine.py          # Core game logic
        ├── tactical_ai.py          # Alpha-Beta AI with learning capabilities
        ├── random_simulation.py    # Random play analysis
        ├── tactical_simulation.py  # Tactical play analysis
        ├── learning_simulation.py  # Enhanced AI learning system with ML features
        └── analyse_opening_book.py # Opening book analyser with pattern insights
```