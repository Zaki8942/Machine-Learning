"""
MNIST-style digit drawing GUI (28x28 output) using Pygame + PyTorch.

Features:
- Cleaner GUI layout (toolbar with clickable buttons)
- CLEAR button (clickable)
- Preview box showing the *28x28* image after PREDICT
- Probability box showing top-3 probabilities (placeholder zeros for now)
- NOT wired to your model yet, but code is structured so you can plug it in easily.

Ink values:
- Background (white) = 0.0
- Black ink = 1.0
- Output tensor values are ALWAYS clamped to [0, 1]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List

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
import torch.nn.functional as F


@dataclass(frozen=True)
class GUIConfig:
    # Output size (MNIST)
    out_w: int = 28
    out_h: int = 28

    # High-res canvas for smoother drawing (recommended divisible by 28)
    canvas_w: int = 280  # 28 * 10
    canvas_h: int = 280  # 28 * 10

    # Layout / styling
    padding: int = 12
    toolbar_h: int = 70
    preview_scale: int = 5  # preview shown as (28*scale)x(28*scale)

    # Brush settings (UNCHANGED)
    brush_radius: int = 14
    brush_hardness: float = 0.65
    erase_strength: float = 0.75

    # Grid overlay
    show_grid: bool = True
    grid_line_every: int = 10  # 280/28

    # Performance / feel
    fps: int = 120
    smoothing_passes: int = 1  # used only for conversion to tensor (preview)

    # Stroke interpolation (fixes gaps when moving fast)
    stamp_spacing_px: float = 2.0  # smaller = denser stamping


class Button:
    """Simple clickable button (no external UI libs)."""

    def __init__(self, rect: pygame.Rect, text: str):
        self.rect = rect
        self.text = text
        self.enabled = True

    def is_hovered(self, pos: Tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)

    def is_clicked(self, pos: Tuple[int, int]) -> bool:
        return self.enabled and self.rect.collidepoint(pos)

    def draw(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        *,
        hovered: bool,
    ) -> None:
        if not self.enabled:
            bg = (45, 45, 45)
            border = (90, 90, 90)
            text_color = (140, 140, 140)
        else:
            bg = (55, 55, 55) if not hovered else (75, 75, 75)
            border = (140, 140, 140)
            text_color = (245, 245, 245)

        pygame.draw.rect(screen, bg, self.rect, border_radius=10)
        pygame.draw.rect(screen, border, self.rect, width=2, border_radius=10)

        label = font.render(self.text, True, text_color)
        label_rect = label.get_rect(center=self.rect.center)
        screen.blit(label, label_rect)


class MNISTDrawGUI:
    def __init__(self, cfg: GUIConfig = GUIConfig()):
        self.cfg = cfg

        # --- Validate config (edge cases)
        if cfg.canvas_w <= 0 or cfg.canvas_h <= 0:
            raise ValueError("Canvas dimensions must be positive.")
        if cfg.out_w <= 0 or cfg.out_h <= 0:
            raise ValueError("Output (MNIST) dimensions must be positive.")
        if cfg.canvas_w < cfg.out_w or cfg.canvas_h < cfg.out_h:
            raise ValueError("Canvas must be >= output size.")

        pygame.init()
        pygame.display.set_caption("MNIST Draw (28x28)")

        pad = cfg.padding
        preview_px = cfg.out_w * cfg.preview_scale

        # Layout
        self.canvas_origin = (pad, cfg.toolbar_h + pad)

        # Toolbar right panel: preview on top, prob box next to it
        self.preview_origin = (pad + cfg.canvas_w + pad, pad + 18)
        self.preview_rect = pygame.Rect(self.preview_origin[0], self.preview_origin[1], preview_px, preview_px)

        # Probability box to the right of preview
        prob_w = 190
        prob_h = preview_px
        self.prob_origin = (self.preview_rect.right + pad, self.preview_rect.top)
        self.prob_rect = pygame.Rect(self.prob_origin[0], self.prob_origin[1], prob_w, prob_h)

        self.window_w = pad + cfg.canvas_w + pad + preview_px + pad + prob_w + pad
        self.window_h = cfg.toolbar_h + pad + cfg.canvas_h + pad

        self.screen = pygame.display.set_mode((self.window_w, self.window_h))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("consolas", 18) or pygame.font.Font(None, 18)
        self.font_small = pygame.font.SysFont("consolas", 16) or pygame.font.Font(None, 16)

        # High-res greyscale ink canvas float32 [0..1]
        # 0.0 = background (white), 1.0 = black ink
        self.ink = np.zeros((cfg.canvas_h, cfg.canvas_w), dtype=np.float32)

        # Precompute brush (UNCHANGED logic)
        self.brush_mask, self.brush_size = self._make_brush_mask(
            radius=cfg.brush_radius, hardness=cfg.brush_hardness
        )

        # State
        self.running = True
        self.left_down = False
        self.right_down = False
        self.last_canvas_pos: Optional[Tuple[int, int]] = None  # for stroke interpolation

        # Preview tensor: [1,1,28,28] on CPU
        self.last_tensor_28: Optional[torch.Tensor] = None

        # Flattened vector: [1,784] on CPU (helpful for MLP models)
        self.last_vector_784: Optional[torch.Tensor] = None

        # Probabilities (placeholder)
        self.last_probs_10: Optional[np.ndarray] = None  # shape (10,), values in [0,1]

        # Buttons
        self.btn_clear = Button(pygame.Rect(pad, pad + 18, 120, 40), "CLEAR")
        self.btn_predict = Button(pygame.Rect(pad + 130, pad + 18, 140, 40), "PREDICT")

    # ----------------------------
    # Brush creation and painting
    # ----------------------------
    def _make_brush_mask(self, radius: int, hardness: float) -> Tuple[np.ndarray, int]:
        radius = int(max(1, radius))
        hardness = float(np.clip(hardness, 0.0, 1.0))
        size = radius * 2 + 1

        y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
        dist = np.sqrt(x * x + y * y)

        edge = max(1e-6, (1.0 - hardness) * radius)
        mask = np.clip(1.0 - (dist - (radius - edge)) / edge, 0.0, 1.0)
        mask[dist > radius] = 0.0
        return mask.astype(np.float32), size

    def _apply_brush(self, cx: int, cy: int, add_ink: bool) -> None:
        cfg = self.cfg
        r = cfg.brush_radius

        x0 = max(0, cx - r)
        y0 = max(0, cy - r)
        x1 = min(cfg.canvas_w, cx + r + 1)
        y1 = min(cfg.canvas_h, cy + r + 1)

        mx0 = x0 - (cx - r)
        my0 = y0 - (cy - r)
        mx1 = mx0 + (x1 - x0)
        my1 = my0 + (y1 - y0)

        if x0 >= x1 or y0 >= y1:
            return

        mask = self.brush_mask[my0:my1, mx0:mx1]
        region = self.ink[y0:y1, x0:x1]

        if add_ink:
            # UNCHANGED draw logic
            region[:] = 1.0 - (1.0 - region) * (1.0 - mask)
        else:
            # UNCHANGED erase logic
            strength = float(np.clip(cfg.erase_strength, 0.0, 1.0))
            region[:] = region * (1.0 - mask * strength)

    def _stamp_line(self, p0: Tuple[int, int], p1: Tuple[int, int], add_ink: bool) -> None:
        """Interpolate between points and stamp brush to avoid gaps."""
        cfg = self.cfg
        x0, y0 = p0
        x1, y1 = p1
        dx = x1 - x0
        dy = y1 - y0
        dist = float(np.hypot(dx, dy))

        if dist < 1e-6:
            self._apply_brush(x1, y1, add_ink=add_ink)
            return

        step = max(0.5, float(cfg.stamp_spacing_px))
        n = int(dist / step) + 1

        for i in range(n + 1):
            t = i / max(1, n)
            cx = int(round(x0 + dx * t))
            cy = int(round(y0 + dy * t))
            self._apply_brush(cx, cy, add_ink=add_ink)

    # ----------------------------
    # Coordinate helpers
    # ----------------------------
    def _window_to_canvas_coords(self, mx: int, my: int) -> Optional[Tuple[int, int]]:
        ox, oy = self.canvas_origin
        cx = mx - ox
        cy = my - oy
        if 0 <= cx < self.cfg.canvas_w and 0 <= cy < self.cfg.canvas_h:
            return cx, cy
        return None

    # ----------------------------
    # MNIST tensor conversion
    # ----------------------------
    def get_mnist_tensor(self, add_smoothing: bool = True) -> torch.Tensor:
        """
        Returns tensor [1,1,28,28] values in [0,1]:
            background white = 0.0
            black ink = 1.0

        This "breaks down" your drawing into per-cell brightness like your screenshot,
        except we keep it strictly in [0, 1].
        """
        cfg = self.cfg

        x = torch.from_numpy(self.ink).to(dtype=torch.float32)  # [H,W], already 0..1
        x = x.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        if add_smoothing and cfg.smoothing_passes > 0:
            for _ in range(cfg.smoothing_passes):
                x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        # Downsample to 28x28 (brightness-aware): bilinear blends values smoothly
        x = F.interpolate(x, size=(cfg.out_h, cfg.out_w), mode="bilinear", align_corners=False)

        # Ensure exact range
        return x.clamp(0.0, 1.0)

    def get_flat_vector_784(self, t28: torch.Tensor) -> torch.Tensor:
        """Convert [1,1,28,28] -> [1,784] (common for non-CNN models)."""
        return t28.view(1, -1).contiguous()

    # ----------------------------
    # Probabilities (model hook)
    # ----------------------------
    def set_prediction_probabilities(self, probs_10: Optional[np.ndarray]) -> None:
        """
        Set prediction probabilities for digits 0..9.

        Expected:
            probs_10 shape: (10,)
            values: in [0,1]
            (Later you can also normalize to sum=1 if your model doesn't already.)
        """
        if probs_10 is None:
            self.last_probs_10 = None
            return

        p = np.asarray(probs_10, dtype=np.float32).reshape(-1)
        if p.shape[0] != 10:
            raise ValueError("probs_10 must have shape (10,).")

        # Clamp for safety
        p = np.clip(p, 0.0, 1.0)
        self.last_probs_10 = p

    def _top3_from_probs(self) -> List[Tuple[int, float]]:
        """
        Returns list of (digit, prob) for top-3.
        If probs are missing (not linked yet), returns zeros.
        """
        if self.last_probs_10 is None:
            return [(0, 0.0), (1, 0.0), (2, 0.0)]

        p = self.last_probs_10
        idx = np.argsort(-p)[:3]
        return [(int(i), float(p[i])) for i in idx]

    # ----------------------------
    # Button actions
    # ----------------------------
    def clear(self) -> None:
        self.ink.fill(0.0)
        self.last_tensor_28 = None
        self.last_vector_784 = None
        self.last_probs_10 = None
        self.last_canvas_pos = None

    def on_predict_clicked(self) -> None:
        """
        For now:
        - generate the 28x28 tensor and store it for preview
        - create a flattened 784 vector (easy model integration)
        - set placeholder probabilities (zeros) to show the GUI box works

        Later, you will replace the placeholder with your model call.
        """
        t28 = self.get_mnist_tensor(add_smoothing=True).cpu()      # [1,1,28,28]
        v784 = self.get_flat_vector_784(t28).cpu()                # [1,784]

        self.last_tensor_28 = t28
        self.last_vector_784 = v784

        # --- PLACEHOLDER probabilities until model is wired:
        # Later you will do: probs = model(t28) or model(v784) -> 10 probabilities
        self.set_prediction_probabilities(np.zeros(10, dtype=np.float32))

        print("[Predict] t28:", tuple(t28.shape), "min/max:", float(t28.min()), float(t28.max()))
        print("[Predict] v784:", tuple(v784.shape))

        # If you want to see the 28x28 values like your screenshot, uncomment:
        # grid = t28[0, 0].numpy()
        # np.set_printoptions(precision=2, suppress=True, linewidth=200)
        # print(grid)

    # ----------------------------
    # Rendering
    # ----------------------------
    def _render_canvas(self) -> None:
        cfg = self.cfg
        ox, oy = self.canvas_origin

        # Convert ink (0=white background,1=black ink) to display grayscale:
        # display wants 255=white,0=black
        gray = (255.0 * (1.0 - self.ink)).astype(np.uint8)  # [H,W]
        rgb = np.stack([gray, gray, gray], axis=-1)          # [H,W,3]

        # pygame.surfarray expects (width, height, channels)
        rgb = np.transpose(rgb, (1, 0, 2))                   # [W,H,3]
        canvas_surf = pygame.surfarray.make_surface(rgb)

        self.screen.blit(canvas_surf, (ox, oy))

        # Border
        pygame.draw.rect(
            self.screen,
            (140, 140, 140),
            pygame.Rect(ox, oy, cfg.canvas_w, cfg.canvas_h),
            width=2,
            border_radius=6,
        )

        if cfg.show_grid:
            line_color = (55, 55, 55)
            thick_color = (80, 80, 80)

            for i in range(0, cfg.canvas_w + 1, cfg.grid_line_every):
                pygame.draw.line(self.screen, line_color, (ox + i, oy), (ox + i, oy + cfg.canvas_h), 1)
            for j in range(0, cfg.canvas_h + 1, cfg.grid_line_every):
                pygame.draw.line(self.screen, line_color, (ox, oy + j), (ox + cfg.canvas_w, oy + j), 1)

            step = cfg.grid_line_every * 5
            for i in range(0, cfg.canvas_w + 1, step):
                pygame.draw.line(self.screen, thick_color, (ox + i, oy), (ox + i, oy + cfg.canvas_h), 2)
            for j in range(0, cfg.canvas_h + 1, step):
                pygame.draw.line(self.screen, thick_color, (ox, oy + j), (ox + cfg.canvas_w, oy + j), 2)

    def _render_preview(self) -> None:
        """Shows the last 28x28 tensor (after Predict) in a small box."""
        cfg = self.cfg
        rect = self.preview_rect

        pygame.draw.rect(self.screen, (30, 30, 30), rect, border_radius=8)
        pygame.draw.rect(self.screen, (140, 140, 140), rect, width=2, border_radius=8)

        title = self.font_small.render("28x28 preview", True, (230, 230, 230))
        self.screen.blit(title, (rect.x, rect.y - 18))

        if self.last_tensor_28 is None:
            hint = self.font_small.render("Press PREDICT to update", True, (160, 160, 160))
            self.screen.blit(hint, (rect.x + 8, rect.y + 8))
            return

        img = self.last_tensor_28[0, 0].numpy()  # [28,28] in [0,1]
        gray = (255.0 * (1.0 - img)).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)  # [28,28,3]

        scale = cfg.preview_scale
        up = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)  # [(28*scale),(28*scale),3]
        up = np.transpose(up, (1, 0, 2))  # [W,H,3]
        surf = pygame.surfarray.make_surface(up)

        # Inset blit so the border stays visible
        inner = rect.inflate(-4, -4)
        self.screen.blit(surf, inner.topleft)

    def _render_prob_box(self) -> None:
        """Shows top-3 probabilities (placeholder zeros until model is wired)."""
        rect = self.prob_rect

        pygame.draw.rect(self.screen, (30, 30, 30), rect, border_radius=8)
        pygame.draw.rect(self.screen, (140, 140, 140), rect, width=2, border_radius=8)

        title = self.font_small.render("Top-3 probabilities", True, (230, 230, 230))
        self.screen.blit(title, (rect.x, rect.y - 18))

        top3 = self._top3_from_probs()

        # Render as lines: "digit : prob"
        y = rect.y + 14
        for rank, (d, p) in enumerate(top3, start=1):
            line = f"{rank}) {d} : {p:.3f}"
            surf = self.font.render(line, True, (240, 240, 240))
            self.screen.blit(surf, (rect.x + 12, y))
            y += 28

        # Hint for wiring
        hint = self.font_small.render("Wire model -> set_prediction_probabilities()", True, (160, 160, 160))
        self.screen.blit(hint, (rect.x + 12, rect.bottom - 26))

    def _render_toolbar(self) -> None:
        cfg = self.cfg
        pad = cfg.padding

        pygame.draw.rect(
            self.screen,
            (25, 25, 25),
            pygame.Rect(0, 0, self.window_w, cfg.toolbar_h),
        )
        pygame.draw.line(self.screen, (70, 70, 70), (0, cfg.toolbar_h), (self.window_w, cfg.toolbar_h), 2)

        mouse_pos = pygame.mouse.get_pos()
        self.btn_clear.draw(self.screen, self.font, hovered=self.btn_clear.is_hovered(mouse_pos))
        self.btn_predict.draw(self.screen, self.font, hovered=self.btn_predict.is_hovered(mouse_pos))

        info = "LMB draw   RMB erase   C clear   Enter predict   Esc quit"
        info_surf = self.font_small.render(info, True, (200, 200, 200))
        self.screen.blit(info_surf, (pad + 290, pad + 28))

        self._render_preview()
        self._render_prob_box()

    # ----------------------------
    # Main loop
    # ----------------------------
    def run(self) -> None:
        cfg = self.cfg

        while self.running:
            self.clock.tick(cfg.fps)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_c:
                        self.clear()
                    elif event.key == pygame.K_RETURN:
                        self.on_predict_clicked()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # left click
                        if self.btn_clear.is_clicked(event.pos):
                            self.clear()
                        elif self.btn_predict.is_clicked(event.pos):
                            self.on_predict_clicked()
                        else:
                            self.left_down = True
                            self.last_canvas_pos = None  # reset stroke anchor

                    elif event.button == 3:  # right click => erase
                        self.right_down = True
                        self.last_canvas_pos = None  # reset stroke anchor

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.left_down = False
                        self.last_canvas_pos = None
                    elif event.button == 3:
                        self.right_down = False
                        self.last_canvas_pos = None

            # Continuous drawing/erasing with interpolation
            if self.left_down or self.right_down:
                mx, my = pygame.mouse.get_pos()
                coords = self._window_to_canvas_coords(mx, my)
                if coords is not None:
                    cx, cy = coords
                    current = (cx, cy)

                    if self.last_canvas_pos is None:
                        # first stamp
                        if self.left_down:
                            self._apply_brush(cx, cy, add_ink=True)
                        if self.right_down:
                            self._apply_brush(cx, cy, add_ink=False)
                    else:
                        if self.left_down:
                            self._stamp_line(self.last_canvas_pos, current, add_ink=True)
                        if self.right_down:
                            self._stamp_line(self.last_canvas_pos, current, add_ink=False)

                    self.last_canvas_pos = current

            # Render
            self.screen.fill((18, 18, 18))
            self._render_toolbar()
            self._render_canvas()
            pygame.display.flip()

        pygame.quit()


def main() -> None:
    gui = MNISTDrawGUI()
    gui.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Fatal Error] {e}")
        sys.exit(1)
