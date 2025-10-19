import argparse
from src.collection.collect_image import collect_images
from src.data.create_dataset import create_dataset
from src.models.train_classifier import train_model
from src.app.inference_classifier import run_inference


def main():
    """
    Main function to run the hand gesture recognition project pipeline.

    This script uses 'argparse' to create a command-line tool.
    You can run this script with different "actions" or "commands".

    Examples:
    - python rf_main.py collect
    - python rf_main.py dataset
    - python rf_main.py train
    - python rf_main.py inference
    """

    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="Hand Gesture Recognition for Japanese Sign Language."
    )

    # Add a *positional argument* named 'action'.
    # 'choices' restricts the user input to only these strings.
    parser.add_argument(
        'action',
        choices=['collect', 'dataset', 'train', 'inference'],
        help="""
        Action to perform:
        'collect': Start collecting images for the dataset.
        'dataset': Create the dataset from collected images.
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
        print("Starting image collection...")
        collect_images()

    elif action == 'dataset':
        print("Creating dataset from images...")
        create_dataset()

    elif action == 'train':
        print("Training the model...")
        train_model()

    elif action == 'inference':
        print("Running inference... (Press 'q' to quit)")
        run_inference()


# This is a standard Python guard.
# It checks if the script is being run directly (not imported).
# If it is, it calls the main() function.
if __name__ == "__main__":
    main()