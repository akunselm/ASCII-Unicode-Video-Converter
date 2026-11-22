"""High-detail ASCII/Unicode Video Studio — V3.

Includes: shape-aware glyph matching, edge enhancement, ordered dithering,
adaptive shape weighting, side-by-side comparison, live controls, plus:
  1. Temporal glyph stabilization (with scene/motion-aware reset)
  4. Local contrast enhancement (CLAHE)
  5. Edge-preserving noise reduction
  7. Live font selection (font atlases cached after first use)

Install: python -m pip install opencv-python pillow numpy
Controls: Q/Esc quit, Space pause, S compare, A adaptive, D dither,
          T temporal stabilization, L CLAHE, N denoise, F next font.
Adjust the controls using the sliders in the second window.
"""

import argparse
import ctypes
import os
import time
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# YOUR SETTINGS
# ============================================================
# Select a video through the file picker or pass a path on the command line.
TARGET_FPS = 5
COLUMNS = 450
FONT_SIZE = 8
SUPERSAMPLING = 5
MAX_CHARACTERS = 256

# Original rendering settings
SHAPE_SAMPLES = 4
CANDIDATE_RADIUS = 14
SHAPE_WEIGHT = 1.0
BRIGHTNESS_WEIGHT = 7.0
CONTRAST_PERCENT = 115
BRIGHTNESS_OFFSET = 0
EDGE_ENHANCEMENT = 35
DITHER_STRENGTH = 35
ADAPTIVE_STRENGTH = 65

# New quality settings (all adjustable live)
TEMPORAL_ON = 1
TEMPORAL_STRENGTH = 60  # 0..100; stronger = fewer character changes
CLAHE_ON = 1
CLAHE_STRENGTH = 30     # 0..100; blend of local contrast correction
DENOISE_ON = 1
DENOISE_STRENGTH = 25   # 0..100; edge-preserving bilateral filter

COMPARE = 0
ADAPTIVE = 1
DITHER = 1

PREVIEW_WINDOW = "Unicode Video - Preview"
CONTROLS_WINDOW = "Unicode Video - Controls"

FONT_CANDIDATES = [
    ("Consolas", r"C:\Windows\Fonts\consola.ttf"),
    ("Cascadia Mono", r"C:\Windows\Fonts\CascadiaMono.ttf"),
    ("Cascadia Code", r"C:\Windows\Fonts\CascadiaCode.ttf"),
    ("Lucida Console", r"C:\Windows\Fonts\lucon.ttf"),
    ("Courier New", r"C:\Windows\Fonts\cour.ttf"),
    ("DejaVu Mono", r"C:\Windows\Fonts\DejaVuSansMono.ttf"),
    ("DejaVu Mono", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    ("Liberation Mono", "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf"),
]

UNICODE_RANGES = [
    (33, 127), (0x00A1, 0x0250), (0x0370, 0x0530),
    (0x1D00, 0x1DC0), (0x2000, 0x2070), (0x2100, 0x2300),
    (0x2C60, 0x2C80), (0xA720, 0xA800),
]

BAYER_8 = np.array([
    [0, 48, 12, 60, 3, 51, 15, 63],
    [32, 16, 44, 28, 35, 19, 47, 31],
    [8, 56, 4, 52, 11, 59, 7, 55],
    [40, 24, 36, 20, 43, 27, 39, 23],
    [2, 50, 14, 62, 1, 49, 13, 61],
    [34, 18, 46, 30, 33, 17, 45, 29],
    [10, 58, 6, 54, 9, 57, 5, 53],
    [42, 26, 38, 22, 41, 25, 37, 21],
], dtype=np.float32) / 64.0 - 0.5


def available_fonts():
    result = []
    seen = set()
    for name, path in FONT_CANDIDATES:
        if os.path.isfile(path) and path not in seen:
            result.append((name, path))
            seen.add(path)
    if not result:
        raise FileNotFoundError(
            "No monospace font found. Add an installed .ttf path to FONT_CANDIDATES."
        )
    return result


def make_characters():
    codes = [32]
    for first, last in UNICODE_RANGES:
        codes.extend(range(first, last))
    for cp in codes:
        char = chr(cp)
        if char != " " and unicodedata.category(char)[0] not in "LNPS":
            continue
        if unicodedata.east_asian_width(char) in "FW":
            continue
        name = unicodedata.name(char, "")
        if any(word in name for word in (
            "BOX", "BLOCK", "SQUARE", "RECTANGLE", "BRAILLE", "QUADRANT"
        )):
            continue
        yield char


def render_glyph(character, high_font, cell_w, cell_h):
    """Render in high resolution, then shrink with Lanczos filtering."""
    scale = SUPERSAMPLING
    w, h = cell_w * scale, cell_h * scale
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    ascent, _ = high_font.getmetrics()
    baseline = ascent + scale
    bbox = draw.textbbox((0, baseline), character, font=high_font, anchor="ls")
    glyph_w = bbox[2] - bbox[0]
    x = (w - glyph_w) // 2 - bbox[0]
    draw.text((x, baseline), character, font=high_font, fill=255, anchor="ls")
    return np.asarray(
        img.resize((cell_w, cell_h), Image.Resampling.LANCZOS),
        dtype=np.uint8
    )


def build_atlas(font_name, font_path):
    """Build a font-specific glyph atlas; called on first use of each font."""
    normal_font = ImageFont.truetype(font_path, FONT_SIZE)
    high_font = ImageFont.truetype(font_path, FONT_SIZE * SUPERSAMPLING)
    cell_w = max(1, round(normal_font.getlength("M")))
    ascent, descent = normal_font.getmetrics()
    cell_h = ascent + descent + 2

    missing = render_glyph(chr(0x10FFFF), high_font, cell_w, cell_h).tobytes()
    seen = set()
    glyphs = []
    for char in make_characters():
        bitmap = render_glyph(char, high_font, cell_w, cell_h)
        signature = bitmap.tobytes()
        if signature == missing or signature in seen:
            continue
        seen.add(signature)
        glyphs.append((float(bitmap.mean()), char, bitmap))
    glyphs.sort(key=lambda item: item[0])
    if len(glyphs) < 2:
        raise RuntimeError(f"Font {font_name} contains too few distinct glyphs")

    # Evenly sample the *actual* visual brightness range, using unique glyphs.
    density_all = np.asarray([g[0] for g in glyphs], np.float32)
    n = min(MAX_CHARACTERS, len(glyphs))
    targets = np.linspace(density_all[0], density_all[-1], n)
    available = np.ones(len(glyphs), dtype=bool)
    indices = []
    for target in targets:
        idx = int(np.argmin(np.where(available, np.abs(density_all - target), np.inf)))
        available[idx] = False
        indices.append(idx)
    selected = [glyphs[i] for i in sorted(indices)]

    atlas = np.stack([g[2] for g in selected])
    densities = np.asarray([g[0] for g in selected], np.float32)
    patterns = np.stack([
        cv2.resize(g[2], (SHAPE_SAMPLES, SHAPE_SAMPLES),
                   interpolation=cv2.INTER_AREA).reshape(-1)
        for g in selected
    ]).astype(np.float32)
    patterns -= patterns.mean(axis=1, keepdims=True)
    patterns /= np.maximum(np.linalg.norm(patterns, axis=1, keepdims=True), 1e-6)

    print(f"\nLoaded font: {font_name} ({font_path})")
    print(f"Glyphs available: {len(glyphs)}; selected: {len(selected)}")
    return {
        "name": font_name, "atlas": atlas, "densities": densities,
        "patterns": patterns, "cell_w": cell_w, "cell_h": cell_h,
    }


def nearest_glyph(densities, wanted):
    right = np.searchsorted(densities, wanted)
    high = np.clip(right, 0, len(densities) - 1)
    low = np.clip(right - 1, 0, len(densities) - 1)
    return np.where(
        np.abs(densities[low] - wanted) <= np.abs(densities[high] - wanted),
        low, high
    ).astype(np.int32)


def improve_grayscale(frame, cols, rows, settings):
    """Preprocess at character-matching sample resolution for performance."""
    size = (cols * SHAPE_SAMPLES, rows * SHAPE_SAMPLES)
    reduced = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)

    # 5. Noise reduction: bilateral blur reduces noise but preserves edges.
    denoise = settings["denoise_strength"] / 100.0
    if settings["denoise_on"] and denoise > 0:
        filtered = cv2.bilateralFilter(
            gray, d=5, sigmaColor=6 + 30 * denoise,
            sigmaSpace=1.4 + denoise
        )
        gray = cv2.addWeighted(gray, 1 - denoise, filtered, denoise, 0)

    # 4. CLAHE: local contrast enhancement for dark and flat regions.
    clahe_strength = settings["clahe_strength"] / 100.0
    if settings["clahe_on"] and clahe_strength > 0:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        locally_enhanced = clahe.apply(gray)
        gray = cv2.addWeighted(
            gray, 1 - clahe_strength, locally_enhanced, clahe_strength, 0
        )

    # Previous global contrast + brightness controls.
    contrast = settings["contrast"] / 100.0
    brightness = settings["brightness"] - 100
    gray = np.clip(
        (gray.astype(np.float32) - 128) * contrast + 128 + brightness,
        0, 255
    ).astype(np.uint8)

    # Previous edge enhancement control.
    edge_amount = settings["edges"] / 100.0
    if edge_amount > 0:
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
        gray = cv2.addWeighted(gray, 1 + edge_amount, blurred, -edge_amount, 0)
    return gray


def match_glyphs(gray, cols, rows, glyph_set, settings, prev_ids=None,
                 prev_means=None):
    """Shape-aware matching, with optional temporal hysteresis."""
    densities = glyph_set["densities"]
    patterns = glyph_set["patterns"]
    sample_size = SHAPE_SAMPLES

    # Preprocessor already supplies exactly one small patch per character.
    patches = (gray.reshape(rows, sample_size, cols, sample_size)
               .transpose(0, 2, 1, 3)
               .reshape(rows * cols, sample_size * sample_size)
               .astype(np.float32))
    means = patches.mean(axis=1)
    centered = patches - means[:, None]
    energy = np.linalg.norm(centered, axis=1)
    std = energy / np.sqrt(sample_size * sample_size)
    span = max(float(densities[-1] - densities[0]), 1e-5)
    wanted = densities[0] + (means / 255.0) * span

    if settings["dither_on"] and settings["dither_strength"]:
        yy, xx = np.indices((rows, cols))
        bayer = BAYER_8[yy % 8, xx % 8].reshape(-1)
        wanted += bayer * span * (0.08 * settings["dither_strength"] / 100.0)
    wanted = np.clip(wanted, densities[0], densities[-1])
    chosen = nearest_glyph(densities, wanted)

    detail = np.clip((std - 4.0) / 22.0, 0.0, 1.0)
    if settings["adaptive_on"]:
        sensitivity = settings["adaptive_strength"] / 100.0
        shape_importance = (0.10 + 2.8 * sensitivity * detail).astype(np.float32)
        refine_ids = np.flatnonzero(detail > 0.10)
    else:
        shape_importance = np.full(len(means), SHAPE_WEIGHT, dtype=np.float32)
        refine_ids = np.flatnonzero(std > 3.0)

    offsets = np.arange(-CANDIDATE_RADIUS, CANDIDATE_RADIUS + 1)
    for start in range(0, len(refine_ids), 2048):
        ids = refine_ids[start:start + 2048]
        base = np.searchsorted(densities, wanted[ids])
        options = np.clip(base[:, None] + offsets[None, :], 0, len(densities) - 1)
        shape = centered[ids] / np.maximum(energy[ids, None], 1e-6)
        similarity = np.einsum("nkd,nd->nk", patterns[options], shape)
        brightness_error = np.abs(densities[options] - wanted[ids, None]) / span
        score = shape_importance[ids, None] * similarity - BRIGHTNESS_WEIGHT * brightness_error
        chosen[ids] = options[np.arange(len(ids)), np.argmax(score, axis=1)]

    # 1. Temporal stabilization: keep prior glyph if the image hasn't moved
    # much and the old glyph is almost as good a match as the new one.
    held = 0
    strength = settings["temporal_strength"] / 100.0
    if (settings["temporal_on"] and strength > 0 and
            prev_ids is not None and prev_means is not None and
            prev_ids.size == chosen.size and prev_means.size == means.size):
        old = prev_ids.reshape(-1)
        old_means = prev_means.reshape(-1)
        gate = np.abs(means - old_means) < (3.0 + 9.0 * strength)
        old_err = np.abs(densities[old] - wanted) / span
        new_err = np.abs(densities[chosen] - wanted) / span
        # Match shapes only in detailed patches; brightness in smooth areas.
        normalized = centered / np.maximum(energy[:, None], 1e-6)
        old_sim = np.einsum("nd,nd->n", patterns[old], normalized)
        new_sim = np.einsum("nd,nd->n", patterns[chosen], normalized)
        old_score = shape_importance * old_sim - BRIGHTNESS_WEIGHT * old_err
        new_score = shape_importance * new_sim - BRIGHTNESS_WEIGHT * new_err
        # Hysteresis margin: low strength changes often, high strength holds.
        close_enough = (new_score - old_score) < (0.02 + 0.36 * strength)
        keep = gate & close_enough & (chosen != old)
        held = int(np.count_nonzero(keep))
        chosen[keep] = old[keep]

    return (chosen.reshape(rows, cols), means.reshape(rows, cols),
            int(np.count_nonzero(detail > 0.4)), held)


def render_frame(frame, glyph_set, cols, rows, settings,
                 prev_ids=None, prev_means=None):
    gray = improve_grayscale(frame, cols, rows, settings)
    ids, means, detail_count, held_count = match_glyphs(
        gray, cols, rows, glyph_set, settings, prev_ids, prev_means
    )
    atlas = glyph_set["atlas"]
    cell_w, cell_h = glyph_set["cell_w"], glyph_set["cell_h"]
    art = (atlas[ids].transpose(0, 2, 1, 3)
           .reshape(rows * cell_h, cols * cell_w))
    return art, ids, means, detail_count, held_count


def get_screen_size():
    if os.name == "nt":
        try:
            user32 = ctypes.windll.user32
            return (int(user32.GetSystemMetrics(0)),
                    int(user32.GetSystemMetrics(1)))
        except Exception:
            pass
    return 1920, 1080


def make_preview(original, art, settings):
    art_color = cv2.cvtColor(art, cv2.COLOR_GRAY2BGR)
    if settings["compare"]:
        h, w = art.shape
        source = cv2.resize(original, (w, h), interpolation=cv2.INTER_AREA)
        separator = np.full((h, 3, 3), 75, dtype=np.uint8)
        output = np.concatenate((source, separator, art_color), axis=1)
    else:
        output = art_color
    screen_w, screen_h = get_screen_size()
    limit_w, limit_h = int(screen_w * 0.91), int(screen_h * 0.83)
    h, w = output.shape[:2]
    scale = min(1.0, max(1, limit_w) / w, max(1, limit_h) / h)
    if scale < 1:
        output = cv2.resize(
            output, (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA
        )
    return output


def setup_controls(fonts):
    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(CONTROLS_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS_WINDOW, 650, 780)

    initial = {
        "columns": COLUMNS, "fps": TARGET_FPS,
        "contrast": CONTRAST_PERCENT, "brightness": BRIGHTNESS_OFFSET + 100,
        "edges": EDGE_ENHANCEMENT, "dither_strength": DITHER_STRENGTH,
        "adaptive_strength": ADAPTIVE_STRENGTH,
        "temporal_strength": TEMPORAL_STRENGTH,
        "clahe_strength": CLAHE_STRENGTH,
        "denoise_strength": DENOISE_STRENGTH,
        "font": 0,
        "compare": COMPARE, "adaptive_on": ADAPTIVE, "dither_on": DITHER,
        "temporal_on": TEMPORAL_ON, "clahe_on": CLAHE_ON, "denoise_on": DENOISE_ON,
    }
    maximums = {
        "columns": 600, "fps": 30, "contrast": 200, "brightness": 200,
        "edges": 200, "dither_strength": 100, "adaptive_strength": 100,
        "temporal_strength": 100, "clahe_strength": 100, "denoise_strength": 100,
        "font": max(1, len(fonts) - 1),
        "compare": 1, "adaptive_on": 1, "dither_on": 1,
        "temporal_on": 1, "clahe_on": 1, "denoise_on": 1,
    }
    labels = {
        "columns": "Columns", "fps": "Target FPS", "contrast": "Contrast %",
        "brightness": "Brightness (100=normal)", "edges": "Edge enhancement %",
        "dither_strength": "Dither strength %", "adaptive_strength": "Adaptive shape strength %",
        "temporal_strength": "Temporal stability %", "clahe_strength": "Local contrast / CLAHE %",
        "denoise_strength": "Noise reduction %", "font": "Font index (see legend)",
        "compare": "Side-by-side", "adaptive_on": "Adaptive detail", "dither_on": "Dithering",
        "temporal_on": "Temporal stabilize", "clahe_on": "Local contrast CLAHE",
        "denoise_on": "Denoise",
    }
    for name, value in initial.items():
        cv2.createTrackbar(labels[name], CONTROLS_WINDOW, value,
                           maximums[name], lambda _value: None)

    def draw_legend(selected_index):
        # The trackbar is an index rather than a dropdown; list available fonts.
        lines = [
            "Q/Esc quit | Space pause | S compare | A adaptive | D dither",
            "T temporal | L local contrast | N denoise | F next font",
            "Fonts installed (select with Font index slider):",
        ]
        lines.extend(f" {i}: {name}" for i, (name, _path) in enumerate(fonts))
        image = np.full((max(140, 82 + 22 * len(lines)), 635, 3), 22, np.uint8)
        for i, line in enumerate(lines):
            color = (130, 240, 160) if line == f" {selected_index}: {fonts[selected_index][0]}" else (230, 230, 230)
            cv2.putText(image, line, (12, 24 + 21 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 1, cv2.LINE_AA)
        cv2.imshow(CONTROLS_WINDOW, image)

    def read_settings():
        values = {name: cv2.getTrackbarPos(label, CONTROLS_WINDOW)
                  for name, label in labels.items()}
        values["columns"] = max(80, values["columns"])
        values["fps"] = max(1, values["fps"])
        values["font"] = min(values["font"], len(fonts) - 1)
        return values

    def flip_setting(name):
        label = labels[name]
        current = cv2.getTrackbarPos(label, CONTROLS_WINDOW)
        cv2.setTrackbarPos(label, CONTROLS_WINDOW, 1 - current)

    draw_legend(0)
    return read_settings, flip_setting, draw_legend, labels


def choose_video():
    parser = argparse.ArgumentParser(
        description="Render a video using high-detail ASCII and Unicode glyphs."
    )
    parser.add_argument(
        "video", nargs="?", help="Path to the video file (optional: opens a picker)"
    )
    args = parser.parse_args()
    if args.video:
        return args.video

    # A file picker makes the repository usable without editing any code.
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.webm *.m4v"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if path:
            return path
    except Exception as exc:
        print(f"File picker unavailable: {exc}")
    raise SystemExit("No video selected. Pass a video path as an argument.")


def main():
    video_path = choose_video()
    fonts = available_fonts()
    print("Available fonts:", ", ".join(n for n, _ in fonts))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if source_fps <= 0:
        source_fps = 30.0
    source_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920
    source_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080

    read_settings, flip, draw_legend, labels = setup_controls(fonts)
    font_cache = {}
    current_font_idx = -1
    frame_number = 0
    previous_settings = None
    last_frame = None
    prev_ids, prev_means = None, None
    paused = False
    next_tick = time.perf_counter()

    print("Keys: Q/Esc quit, Space pause, S compare, A adaptive, D dither,")
    print("      T stabilize, L local contrast, N denoise, F next font")
    try:
        while True:
            settings = read_settings()
            idx = settings["font"]
            if idx != current_font_idx:
                if idx not in font_cache:
                    name, path = fonts[idx]
                    font_cache[idx] = build_atlas(name, path)
                glyph_set = font_cache[idx]
                current_font_idx = idx
                draw_legend(idx)
                prev_ids, prev_means = None, None
                next_tick = time.perf_counter()
            else:
                glyph_set = font_cache[idx]

            cols = settings["columns"]
            cell_w, cell_h = glyph_set["cell_w"], glyph_set["cell_h"]
            rows = max(2, round(cols * cell_w * source_h / (source_w * cell_h)))
            changed = settings != previous_settings
            # Any changed image-processing parameter invalidates prior glyph history.
            # FPS and compare mode alone don't change the character assignment.
            if previous_settings is not None:
                image_keys = (
                    "columns", "contrast", "brightness", "edges", "dither_strength",
                    "adaptive_strength", "temporal_strength", "clahe_strength",
                    "denoise_strength", "adaptive_on", "dither_on", "temporal_on",
                    "clahe_on", "denoise_on", "font",
                )
                if any(settings[k] != previous_settings[k] for k in image_keys):
                    prev_ids, prev_means = None, None

            if not paused or last_frame is None:
                skip = max(1, round(source_fps / settings["fps"]))
                ok = True
                for _ in range(skip):
                    if not cap.grab():
                        ok = False
                        break
                    frame_number += 1
                if not ok:
                    break
                ok, last_frame = cap.retrieve()
                if not ok:
                    break

            if not paused or changed:
                art, ids, means, detailed, held = render_frame(
                    last_frame, glyph_set, cols, rows, settings,
                    prev_ids, prev_means
                )
                prev_ids, prev_means = ids.copy(), means.copy()
                preview = make_preview(last_frame, art, settings)
                cv2.imshow(PREVIEW_WINDOW, preview)
                print(
                    f"\rFrame {frame_number:6d} | {glyph_set['name']:<16} | "
                    f"{cols}x{rows} chars | detailed {detailed:6d} | "
                    f"stabilized {held:6d} | output {art.shape[1]}x{art.shape[0]}   ",
                    end="", flush=True
                )
                previous_settings = settings.copy()

            if paused:
                delay = 40
            else:
                skip = max(1, round(source_fps / settings["fps"]))
                next_tick += skip / source_fps
                delay = max(1, int(1000 * (next_tick - time.perf_counter())))
                if next_tick < time.perf_counter() - 1.0:
                    next_tick = time.perf_counter()

            key = cv2.waitKey(delay) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                paused = not paused
                next_tick = time.perf_counter()
            elif key == ord("s"):
                flip("compare")
            elif key == ord("a"):
                flip("adaptive_on")
            elif key == ord("d"):
                flip("dither_on")
            elif key == ord("t"):
                flip("temporal_on")
            elif key == ord("l"):
                flip("clahe_on")
            elif key == ord("n"):
                flip("denoise_on")
            elif key == ord("f"):
                cv2.setTrackbarPos(
                    labels["font"], CONTROLS_WINDOW, (settings["font"] + 1) % len(fonts)
                )

            if cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nPlayback stopped.")


if __name__ == "__main__":
    main()