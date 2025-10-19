import argparse
from collection.collect_sequences import collect_sequences
from models.model_trainer import main_trainer
from app.inference import run_inference


def main():
    """
    Main function to run the hand gesture recognition project pipeline.

    This script uses 'argparse' to create a command-line tool.
    You can run this script with different "actions" or "commands".

    Examples:
    - python main.py collect
    - python main.py train
    - python main.py inference
    """

    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="Hand Gesture Recognition for Japanese Sign Language."
    )

    # Add a *positional argument* named 'action'.
    # 'choices' restricts the user input to only these strings.
    parser.add_argument(
        'action',
        choices=['collect', 'train', 'inference'],
        help="""
        Action to perform:
        'collect': Start collecting images for the dataset.
        'train': Train the classifier model.
        'inference': Run real-time gesture inference.
        """
    )

    # Parse the arguments provided by the user in the command line
    args = parser.parse_args()

    # Get the specific action the user typed (e.g., 'collect')
    action = args.action

    # --- Route the action to the correct function ---
    if action == 'collect':
        print("Starting sequence collection and compiling...")
        collect_sequences()

    elif action == 'train':
        print("Training the model...")
        main_trainer()

    elif action == 'inference':
        print("Running inference... (Press 'q' to quit)")
        run_inference()


# This is a standard Python guard.
# It checks if the script is being run directly (not imported).
# If it is, it calls the main() function.
if __name__ == "__main__":
    main()