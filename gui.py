import pygame
import torch
import numpy as np
import sys
from predict import load_model, predict_single_input
from ModelStructure import DigitCNN

# Constants
WINDOW_SIZE = 560  # 28 * 20 (Scale up 20x for easier drawing)
GRID_SIZE = 28
BRUSH_SIZE = 25
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

def get_tensor_from_surface(surface):
    """Converts the Pygame surface to a PyTorch tensor (1, 1, 28, 28)."""
    # 1. Resize the 560x560 surface down to 28x28
    small_surface = pygame.transform.smoothscale(surface, (GRID_SIZE, GRID_SIZE))
    
    # 2. Get pixel data. Pygame returns (Width, Height, Channels)
    pixel_array = pygame.surfarray.array3d(small_surface)
    
    # 3. Transpose to (Height, Width, Channels) for numpy/torch conventions
    pixel_array = pixel_array.transpose(1, 0, 2)
    
    # 4. Convert to grayscale (take the first channel, as R=G=B for white on black)
    grayscale = pixel_array[:, :, 0]
    
    # 5. Normalize to 0-1 range
    normalized = grayscale.astype(np.float32) / 255.0
    
    # 6. Convert to Tensor and add Batch & Channel dimensions
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    return tensor

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 60))
    pygame.display.set_caption("MNIST Digit Predictor")
    font = pygame.font.SysFont(None, 36)
    
    # Drawing Canvas
    canvas = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
    canvas.fill(BLACK)
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        # Ensure 'mnist_model.pth' is in the same folder
        model = load_model('mnist_model.pth', DigitCNN, device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure 'mnist_model.pth' exists and 'DigitCNN' class matches the saved model.")
        pygame.quit()
        return

    drawing = False
    prediction_text = "Draw & Press ENTER"
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Mouse Interactions
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left Click
                    drawing = True
                elif event.button == 3: # Right Click
                    canvas.fill(BLACK)
                    prediction_text = "Canvas Cleared"
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False
            
            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    mouse_x, mouse_y = event.pos
                    if mouse_y < WINDOW_SIZE:
                        pygame.draw.circle(canvas, WHITE, (mouse_x, mouse_y), BRUSH_SIZE)
            
            # Keyboard Interactions
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    # Run Prediction
                    input_tensor = get_tensor_from_surface(canvas)
                    try:
                        digit = predict_single_input(model, input_tensor, device)
                        prediction_text = f"Prediction: {digit}"
                        print(f"Predicted: {digit}")
                    except Exception as e:
                        print(f"Prediction Error: {e}")
                        prediction_text = "Error"
                
                elif event.key == pygame.K_c:
                    canvas.fill(BLACK)
                    prediction_text = "Canvas Cleared"

        # Rendering
        screen.fill(BLACK)
        screen.blit(canvas, (0, 0))
        
        # Draw Text Area
        pygame.draw.rect(screen, (50, 50, 50), (0, WINDOW_SIZE, WINDOW_SIZE, 60))
        text_surface = font.render(prediction_text, True, WHITE)
        screen.blit(text_surface, (20, WINDOW_SIZE + 15))
        
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()