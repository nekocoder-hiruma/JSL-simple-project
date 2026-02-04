import sys
from core.config_loader import load_system_config, load_specialized_config
from core.resource_manager import ResourceManager
from core.mediapipe_utils import MediaPipeHandler
from data.collection.static_collector import StaticCollector
from data.collection.dynamic_collector import DynamicCollector
from data.processing.dataset_creator import DatasetCreator
from models.random_forest.rf_model import RFModel
from models.lstm.lstm_model import LSTMModel

class JSLApp:
    """
    Main controller for the Japanese Sign Language study project.
    Provides an interactive menu to collect data, train models, and run predictions.
    """
    def __init__(self):
        self.rm = ResourceManager()
        self.system_config = load_system_config()

    def run_menu(self):
        """Displays the main options and routes the user to their desired activity."""
        while True:
            print("\n--- Project Management Menu ---")
            print("1. Gather Training Data")
            print("2. Train Decision Models")
            print("3. Run Live Gesture Recognition")
            print("q. Exit Program")
            
            choice = input("Select an option: ").strip().lower()
            if choice == '1': self.data_collection_flow()
            elif choice == '2': self.training_flow()
            elif choice == '3': self.inference_flow()
            elif choice == 'q': break
            else: print("Invalid selection. Please try again.")

    def data_collection_flow(self):
        """Guides the user through capturing static images or movement sequences."""
        print("\n--- Data Collection ---")
        print("1. Photo Capture (for static signs)")
        print("2. Motion Recording (for dynamic signs)")
        
        choice = input("Select capture type: ").strip().lower()
        if choice == '1':
            collector = StaticCollector(self.system_config, self.rm)
            collector.collect()
            if input("Process images into a dataset now? (y/n): ").lower() == 'y':
                handler = MediaPipeHandler(is_video_stream=False)
                creator = DatasetCreator(self.system_config, handler)
                creator.create()
                handler.close()
        elif choice == '2':
            handler = MediaPipeHandler(is_video_stream=True)
            collector = DynamicCollector(self.system_config, self.rm, handler)
            collector.collect()
            handler.close()

    def training_flow(self):
        """Standardizes the process of teaching the models based on collected data."""
        print("\n--- Model Training ---")
        print("1. Random Forest (Image-based)")
        print("2. LSTM (Sequence-based)")
        
        choice = input("Select model to train: ").strip().lower()
        if choice == '1':
            config = load_specialized_config('base')
            model = RFModel(config)
            model.train()
            model.save()
        elif choice == '2':
            config = load_specialized_config('keras')
            model = LSTMModel(config)
            model.train()
            model.save()

    def inference_flow(self):
        """Loads a trained model and starts the real-time recognition interface."""
        print("\n--- Gesture Recognition ---")
        print("1. Recognize Static Signs")
        print("2. Recognize Dynamic Signs")
        
        choice = input("Select mode: ").strip().lower()
        if choice == '1':
            from app.inference_classifier import run_inference
            run_inference() # Note: Refactoring these legacy app modules is next
        elif choice == '2':
            from app.inference import run_inference
            run_inference()

if __name__ == "__main__":
    app = JSLApp()
    app.run_menu()