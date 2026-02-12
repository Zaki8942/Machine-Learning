"""
MNIST-style digit drawing GUI (28x28 output) using Pygame + PyTorch.

Layout:
- Left Column: Drawing Canvas
- Right Column: Dashboard (Preview + Confidence Scores)

Features:
- Cleaner, wider dashboard (300px)
- Right-aligned percentage text
- Tighter vertical spacing
- Fixed 'main' function
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import pygame
except Exception as e:
    raise RuntimeError(
        "Pygame is required for this GUI.\n"
        "Install it with: pip install pygame\n"
        f"Original import error: {e}"
    )

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------
# 1. MODEL DEFINITION
# ------------------------------------------------------------------------------
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

# ------------------------------------------------------------------------------
# 2. LOAD WEIGHTS
# ------------------------------------------------------------------------------
my_model = DigitCNN()
try:
    my_model.load_state_dict(torch.load("my_model_weights.pth", map_location="cpu"))
    my_model.eval()
    print("Successfully loaded DigitCNN weights!")
except Exception as e:
    print(f"Could not load weights: {e}")
    print("Ensure 'my_model_weights.pth' is in the same folder.")

# ------------------------------------------------------------------------------
# 3. GUI CONFIGURATION
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class GUIConfig:
    out_w: int = 28
    out_h: int = 28
    canvas_w: int = 280
    canvas_h: int = 280
    padding: int = 20
    toolbar_h: int = 60
    preview_scale: int = 5

    brush_radius: int = 14
    brush_hardness: float = 0.65
    erase_strength: float = 0.75
    
    show_grid: bool = True
    grid_line_every: int = 10
    fps: int = 120
    smoothing_passes: int = 1
    stamp_spacing_px: float = 2.0


class Button:
    def __init__(self, rect: pygame.Rect, text: str):
        self.rect = rect
        self.text = text
        self.enabled = True

    def is_hovered(self, pos: Tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)

    def is_clicked(self, pos: Tuple[int, int]) -> bool:
        return self.enabled and self.rect.collidepoint(pos)

    def draw(self, screen: pygame.Surface, font: pygame.font.Font, *, hovered: bool) -> None:
        bg = (60, 60, 60) if not hovered else (80, 80, 80)
        pygame.draw.rect(screen, bg, self.rect, border_radius=8)
        pygame.draw.rect(screen, (150, 150, 150), self.rect, width=1, border_radius=8)
        
        label = font.render(self.text, True, (240, 240, 240))
        label_rect = label.get_rect(center=self.rect.center)
        screen.blit(label, label_rect)


# ------------------------------------------------------------------------------
# 4. MAIN GUI CLASS
# ------------------------------------------------------------------------------
class MNISTDrawGUI:
    def __init__(self, cfg: GUIConfig = GUIConfig()):
        self.cfg = cfg
        pygame.init()
        pygame.display.set_caption("MNIST Dashboard")

        pad = cfg.padding
        preview_px = int(cfg.out_w * cfg.preview_scale) # ~140px

        # --- LAYOUT ---
        # 1. Left: Canvas
        self.canvas_origin = (pad, cfg.toolbar_h + pad)
        canvas_rect = pygame.Rect(self.canvas_origin[0], self.canvas_origin[1], cfg.canvas_w, cfg.canvas_h)

        # 2. Right: Dashboard
        # WIDENED RIGHT COLUMN to 300px for better spacing
        right_col_w = 300 
        right_col_x = canvas_rect.right + pad
        
        # A. Preview (Top Right)
        # Center the preview in the right column
        preview_offset_x = (right_col_w - preview_px) // 2
        self.preview_origin = (right_col_x + preview_offset_x, cfg.toolbar_h + pad)
        self.preview_rect = pygame.Rect(self.preview_origin[0], self.preview_origin[1], preview_px, preview_px)

        # B. Probability Box (Bottom Right)
        prob_y = self.preview_rect.bottom + pad
        # Make it tall enough to align with canvas bottom
        prob_h = (canvas_rect.bottom - prob_y) 
        
        # Ensure minimum height for the list
        if prob_h < 220: prob_h = 220

        self.prob_origin = (right_col_x, prob_y)
        self.prob_rect = pygame.Rect(right_col_x, prob_y, right_col_w, prob_h)

        # Window Size
        self.window_w = int(right_col_x + right_col_w + pad)
        self.window_h = int(max(canvas_rect.bottom, self.prob_rect.bottom) + pad)

        self.screen = pygame.display.set_mode((self.window_w, self.window_h))
        self.clock = pygame.time.Clock()

        # Fonts
        self.font = pygame.font.SysFont("consolas", 18) or pygame.font.Font(None, 18)
        self.font_small = pygame.font.SysFont("consolas", 14) or pygame.font.Font(None, 14)
        self.font_tiny = pygame.font.SysFont("consolas", 12) or pygame.font.Font(None, 12)

        # State
        self.ink = np.zeros((cfg.canvas_h, cfg.canvas_w), dtype=np.float32)
        self.brush_mask, self.brush_size = self._make_brush_mask(cfg.brush_radius, cfg.brush_hardness)
        
        self.running = True
        self.left_down = False
        self.right_down = False
        self.last_canvas_pos = None 
        self.last_tensor_28 = None
        self.last_probs_10 = None 

        # Clear Button
        self.btn_clear = Button(pygame.Rect(pad, 10, 100, 40), "CLEAR")

    def _make_brush_mask(self, radius, hardness):
        radius = int(max(1, radius))
        size = radius * 2 + 1
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        dist = np.sqrt(x*x + y*y)
        edge = max(1e-6, (1.0 - hardness)*radius)
        mask = np.clip(1.0 - (dist - (radius-edge))/edge, 0.0, 1.0)
        mask[dist > radius] = 0.0
        return mask.astype(np.float32), size

    def _window_to_canvas_coords(self, mx, my):
        ox, oy = self.canvas_origin
        cx, cy = mx - ox, my - oy
        if 0 <= cx < self.cfg.canvas_w and 0 <= cy < self.cfg.canvas_h:
            return cx, cy
        return None

    def center_tensor_by_mass(self, t28):
        img = t28[0, 0].numpy()
        total_mass = img.sum()
        if total_mass <= 1.0: return t28 

        h, w = img.shape
        y_grid, x_grid = np.indices((h, w))
        cy = (y_grid * img).sum() / total_mass
        cx = (x_grid * img).sum() / total_mass
        
        dy, dx = int(round(14 - cy)), int(round(14 - cx))
        new_img = np.zeros_like(img)
        
        # Slicing logic
        sy_s, sy_e = max(0, -dy), min(h, h - dy)
        sx_s, sx_e = max(0, -dx), min(w, w - dx)
        dy_s, dy_e = max(0, dy), min(h, h + dy)
        dx_s, dx_e = max(0, dx), min(w, w + dx)
        
        try:
            if (sy_e > sy_s) and (sx_e > sx_s):
                new_img[dy_s:dy_s+(sy_e-sy_s), dx_s:dx_s+(sx_e-sx_s)] = img[sy_s:sy_e, sx_s:sx_e]
        except: pass
        return torch.from_numpy(new_img).unsqueeze(0).unsqueeze(0)

    # ----------------------------
    # Logic
    # ----------------------------
    def clear(self):
        self.ink.fill(0.0)
        self.last_tensor_28 = None
        self.last_probs_10 = None
        self.last_canvas_pos = None

    def on_predict(self):
        x = torch.from_numpy(self.ink).float().unsqueeze(0).unsqueeze(0)
        if self.cfg.smoothing_passes > 0:
            x = F.avg_pool2d(x, 3, 1, 1)
        t28 = F.interpolate(x, (28,28), mode="bilinear", align_corners=False).clamp(0,1)
        t28_c = self.center_tensor_by_mass(t28)
        t28_norm = (t28_c - 0.1307) / 0.3081
        
        with torch.no_grad():
            output = my_model(t28_norm)
            probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()

        self.last_tensor_28 = t28_c
        self.last_probs_10 = probs

    def _apply_brush(self, cx, cy, add):
        r = self.cfg.brush_radius
        x0, y0 = max(0, cx-r), max(0, cy-r)
        x1, y1 = min(self.cfg.canvas_w, cx+r+1), min(self.cfg.canvas_h, cy+r+1)
        if x0>=x1 or y0>=y1: return
        
        mask = self.brush_mask[y0-(cy-r):y0-(cy-r)+(y1-y0), x0-(cx-r):x0-(cx-r)+(x1-x0)]
        if add: self.ink[y0:y1, x0:x1] = 1.0 - (1.0 - self.ink[y0:y1, x0:x1]) * (1.0 - mask)
        else: self.ink[y0:y1, x0:x1] *= (1.0 - mask * self.cfg.erase_strength)

    def _stamp_line(self, p0, p1, add):
        dist = np.hypot(p1[0]-p0[0], p1[1]-p0[1])
        if dist < 1e-6:
            self._apply_brush(*p1, add)
            return
        n = int(dist / self.cfg.stamp_spacing_px) + 1
        for i in range(n+1):
            t = i / max(1, n)
            cx = int(p0[0] + (p1[0]-p0[0])*t)
            cy = int(p0[1] + (p1[1]-p0[1])*t)
            self._apply_brush(cx, cy, add)

    # ----------------------------
    # Rendering
    # ----------------------------
    def _render_all(self):
        self.screen.fill((20, 20, 20))
        
        # Toolbar
        pygame.draw.rect(self.screen, (30, 30, 30), (0, 0, self.window_w, self.cfg.toolbar_h))
        pygame.draw.line(self.screen, (60, 60, 60), (0, self.cfg.toolbar_h), (self.window_w, self.cfg.toolbar_h), 1)
        self.btn_clear.draw(self.screen, self.font, hovered=self.btn_clear.is_hovered(pygame.mouse.get_pos()))
        
        # Instructions
        help_t = self.font_small.render("Draw: Left Click | Erase: C", True, (120, 120, 120))
        self.screen.blit(help_t, (140, 22))

        # Canvas
        ox, oy = self.canvas_origin
        gray = (255.0 * (1.0 - self.ink)).astype(np.uint8)
        rgb = np.dstack([gray]*3).transpose(1, 0, 2)
        surf = pygame.surfarray.make_surface(rgb)
        self.screen.blit(surf, (ox, oy))
        pygame.draw.rect(self.screen, (100, 100, 100), (ox, oy, self.cfg.canvas_w, self.cfg.canvas_h), 1)

        if self.cfg.show_grid:
            st = self.cfg.grid_line_every
            for i in range(st, self.cfg.canvas_w, st):
                c = (60,60,60) if i%(st*5)==0 else (35,35,35)
                pygame.draw.line(self.screen, c, (ox+i, oy), (ox+i, oy+self.cfg.canvas_h))
            for i in range(st, self.cfg.canvas_h, st):
                c = (60,60,60) if i%(st*5)==0 else (35,35,35)
                pygame.draw.line(self.screen, c, (ox, oy+i), (ox+self.cfg.canvas_w, oy+i))

        # Preview
        pr = self.preview_rect
        pygame.draw.rect(self.screen, (30, 30, 30), pr, border_radius=5)
        pygame.draw.rect(self.screen, (80, 80, 80), pr, width=1, border_radius=5)
        
        plabel = self.font_tiny.render("MODEL INPUT (28x28)", True, (150, 150, 150))
        self.screen.blit(plabel, (pr.x, pr.y - 15))

        if self.last_tensor_28 is not None:
            img = self.last_tensor_28[0,0].numpy()
            gray = (255.0 * (1.0-img)).astype(np.uint8)
            rgb = np.repeat(np.repeat(np.dstack([gray]*3), self.cfg.preview_scale, 0), self.cfg.preview_scale, 1)
            self.screen.blit(pygame.surfarray.make_surface(rgb.transpose(1,0,2)), pr.inflate(-2,-2).topleft)

        # Probabilities
        self._render_prob_box()
        
        pygame.display.flip()

    def _render_prob_box(self):
        rect = self.prob_rect
        # Background
        pygame.draw.rect(self.screen, (25, 25, 25), rect, border_radius=5)
        pygame.draw.rect(self.screen, (60, 60, 60), rect, width=1, border_radius=5)

        if self.last_probs_10 is None:
            return

        probs = self.last_probs_10
        winner_idx = np.argmax(probs)
        
        # Layout metrics
        margin_top = 15
        margin_bottom = 15
        available_h = rect.height - margin_top - margin_bottom
        
        # TIGHTER SPACING:
        # We calculate row_h based on space, but clamp it so it doesn't get too spread out
        row_h = min(28, available_h / 10) 
        
        bar_h = 6 # Thin bar
        
        # Alignments
        col_digit_x = rect.x + 15
        col_bar_x = rect.x + 40
        # Right align the text to the edge of the box minus padding
        col_text_right_edge = rect.right - 15 
        
        # Max bar width must stop before the text area
        # We reserve ~45px for "100%" text on the right
        max_bar_w = (col_text_right_edge - 45) - col_bar_x

        for i in range(10):
            p = probs[i]
            cy = int(rect.y + margin_top + (i * row_h) + (row_h/2))
            
            # 1. Digit
            col = (255, 255, 255) if i == winner_idx else (100, 100, 100)
            lbl = self.font.render(str(i), True, col)
            self.screen.blit(lbl, (col_digit_x, cy - 8))

            # 2. Track
            track_rect = pygame.Rect(col_bar_x, cy - bar_h//2, max_bar_w, bar_h)
            pygame.draw.rect(self.screen, (40, 40, 40), track_rect, border_radius=2)

            # 3. Bar
            if p > 0.005:
                w = int(max_bar_w * p)
                bar_rect = pygame.Rect(col_bar_x, cy - bar_h//2, w, bar_h)
                c = (0, 255, 120) if i == winner_idx else (60, 100, 160)
                pygame.draw.rect(self.screen, c, bar_rect, border_radius=2)

            # 4. Percent (Right Aligned)
            if p > 0.01:
                txt_str = f"{int(p*100)}%"
                txt = self.font_small.render(txt_str, True, col)
                # Align right edge of text to col_text_right_edge
                txt_rect = txt.get_rect(midright=(col_text_right_edge, cy))
                self.screen.blit(txt, txt_rect)

    def run(self):
        while self.running:
            self.clock.tick(self.cfg.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c: self.clear()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.btn_clear.is_clicked(event.pos): self.clear()
                        else: 
                            self.left_down = True
                            self.last_canvas_pos = None
                            self.on_predict()
                    elif event.button == 3:
                        self.right_down = True
                        self.last_canvas_pos = None
                        self.on_predict()
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.left_down = False
                    self.right_down = False
                    self.last_canvas_pos = None

            if self.left_down or self.right_down:
                mx, my = pygame.mouse.get_pos()
                coord = self._window_to_canvas_coords(mx, my)
                if coord:
                    if self.last_canvas_pos: self._stamp_line(self.last_canvas_pos, coord, self.left_down)
                    else: self._apply_brush(*coord, self.left_down)
                    self.last_canvas_pos = coord
                    self.on_predict()
            
            self._render_all()
        pygame.quit()

# ------------------------------------------------------------------------------
# 5. EXECUTION
# ------------------------------------------------------------------------------
def main():
    gui = MNISTDrawGUI()
    gui.run()

if __name__ == "__main__":
    main()