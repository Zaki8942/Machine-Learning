import torch
# Import the class from your other file
from ModelStructure import DigitCNN 

def load_model(weights_path):
    # 1. Instantiate the model class first (Empty brain)
    # This creates the structure in memory
    model = DigitCNN()

    # 2. Load the dictionary of parameters (The learned knowledge)
    # 'map_location' ensures it loads on CPU if you don't have a GPU available
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    
    # 3. Apply the weights to the model
    model.load_state_dict(state_dict)

    # 4. Set to Evaluation Mode (CRITICAL STEP)
    # This freezes Dropout and tells BatchNorm to use saved statistics
    model.eval()
    
    print(f"Success! Model loaded from {weights_path}")
    return model

# --- usage ---
if __name__ == "__main__":
    # path to where you saved your .pth file during training
    MODEL_PATH = "my_model_weights.pth" 
    
    try:
        loaded_model = load_model(MODEL_PATH)
        
        # Test with a dummy input to verify shape
        # (Batch_Size=1, Channels=1, Height=28, Width=28)
        dummy_input = torch.randn(1, 1, 28, 28)
        
        # Make a prediction
        with torch.no_grad(): # Disable gradient calculation for inference
            output = loaded_model(dummy_input)
            print("Model output shape:", output.shape) # Should be [1, 10]
            
    except FileNotFoundError:
        print(f"Error: Could not find the file '{MODEL_PATH}'. Check your path.")