"""
Main runner for Azers Monte Carlo simulations.
Allows running both random and tactical simulations.
"""
import sys
import os
import time

def print_banner():
    print("="*80)
    print("AZERS GAME - MONTE CARLO SIMULATION SUITE")
    print("="*80)
    print("This tool runs comprehensive Monte Carlo simulations to analyse")
    print("game balance and bias in the Azers board game.")
    print()
    print("Four types of simulations are available:")
    print("1. RANDOM PLAY - Both players select moves randomly")
    print("2. TACTICAL PLAY - Both players use Alpha-Beta search AI")
    print("3. LEARNING AI - Train an AI through self-play and test improvement")
    print("   • Standard learning: builds knowledge from scratch")
    print("   • Bias-optimised learning: starts with enhanced opening books")
    print("Available configurations:")
    print("- Mode 1: Opposing corners start (A1 vs D4)")
    print("- Mode 2: All possible starting positions")
    print("="*80)

def main():
    print_banner()
    
    while True:
        print("\nChoose simulation type:")
        print("1. Random Play Simulation")
        print("2. Tactical Play Simulation")
        print("3. Learning AI Simulation")
        print("4. Run Multiple Simulations")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            print("\nStarting Random Play Simulation...")
            print("You'll be able to select which configurations to test.")
            try:
                from random_simulation import main as run_random
                run_random()
            except ImportError as e:
                print(f"Error importing random simulation: {e}")
            except Exception as e:
                print(f"Error running random simulation: {e}")
        
        elif choice == "2":
            print("\nStarting Tactical Play Simulation...")
            print("You'll be able to select configurations and configure search depth.")
            print("Time estimates will be provided before starting.")
            try:
                from tactical_simulation import main as run_tactical
                run_tactical()
            except ImportError as e:
                print(f"Error importing tactical simulation: {e}")
            except Exception as e:
                print(f"Error running tactical simulation: {e}")
        
        elif choice == "3":
            print("\nStarting Learning AI Simulation...")
            print("This will train an AI through self-play and evaluate learning.")
            print("The AI will build an opening book and improve its strategy.")
            print("You can choose standard learning or bias-optimised training.")
            try:
                from learning_simulation import main as run_learning
                run_learning()
            except ImportError as e:
                print(f"Error importing learning simulation: {e}")
            except Exception as e:
                print(f"Error running learning simulation: {e}")
        
        elif choice == "4":
            print("\nRunning multiple simulations...")
            print("You'll configure each simulation separately.")
            print("You can select different configurations for each simulation type.")
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                try:
                    print("\n" + "="*60)
                    print("PHASE 1: RANDOM PLAY SIMULATION")
                    print("="*60)
                    from random_simulation import main as run_random
                    run_random()
                    
                    print("\n" + "="*60)
                    print("PHASE 2: TACTICAL PLAY SIMULATION")
                    print("="*60)
                    from tactical_simulation import main as run_tactical
                    run_tactical()
                    
                    print("\n" + "="*60)
                    print("PHASE 3: LEARNING AI SIMULATION")
                    print("="*60)
                    from learning_simulation import main as run_learning
                    run_learning()
                    
                    print("\n" + "="*60)
                    print("ALL SIMULATIONS COMPLETED!")
                    print("="*60)
                    
                except ImportError as e:
                    print(f"Error importing simulation modules: {e}")
                except Exception as e:
                    print(f"Error running simulations: {e}")
            else:
                print("Multiple simulation cancelled.")
        

        
        elif choice == "5":
            print("Exiting simulation suite. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
        
        # Ask if user wants to run another simulation
        if choice in ["1", "2", "3", "4"]:
            while True:
                another = input("\nRun another simulation? (y/n): ").strip().lower()
                if another in ['y', 'yes']:
                    break
                elif another in ['n', 'no']:
                    print("Exiting simulation suite. Goodbye!")
                    return
                else:
                    print("Please enter 'y' for yes or 'n' for no.")

if __name__ == "__main__":
    main()
