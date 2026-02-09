import torch
import numpy as np
from ModelStructure import DigitCNN

def load_model(model_path, model_class, device):
    """Loads the trained weights into the model architecture."""
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode (disables Dropout, BatchNorm updates)
    return model

def predict_single_input(model, input_tensor, device):
    """Runs inference on a single input."""
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = model(input_tensor)
        prediction = torch.argmax(outputs, dim=1).item()
        
    return prediction

if __name__ == "__main__":
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = 'mnist_model.pth'
    
    # 2. Load Model (Replace 'MyModel' with your actual class name)
    model = load_model(MODEL_PATH, DigitCNN, device)
    print("Model loaded successfully.")

    # 3. Create Dummy Input (Simulating a 28x28 grayscale image)
    # We mimic the logic from your data_loader.py:
    # It expects float32, normalized 0-1, shape (1, 1, 28, 28)
    
    # Example: Random pixel data (0-255)
    dummy_pixels = np.random.randint(0, 255, (28, 28), dtype='uint8')
    
    # Preprocessing matching your data_loader.py:
    # Convert to tensor -> Float -> Normalize / 255.0 -> Add Batch Dimension (unsqueeze)
    input_tensor = torch.from_numpy(dummy_pixels).float().div(255.0).unsqueeze(0).unsqueeze(0)
    
    # 4. Run Prediction
    result = predict_single_input(model, input_tensor, device)
    print(f"Predicted Digit: {result}")