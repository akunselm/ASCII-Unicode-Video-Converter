# ASCII/Unicode Video Converter

A Python program that converts videos into high-resolution ASCII and Unicode character art. Built with OpenCV, Pillow, and NumPy, with a live control panel for tuning image quality.

## Features

- Shape-aware glyph matching based on brightness and local stroke patterns
- Up to 256 distinct ASCII and Unicode characters, excluding block and box characters
- Supersampled antialiased glyph rendering
- Live controls for contrast, brightness, resolution, FPS, and edge enhancement
- Temporal stabilization to reduce flicker
- Local contrast enhancement (CLAHE) and edge-preserving denoising
- Ordered dithering and adaptive shape matching
- Font selection from installed monospace fonts
- Side-by-side comparison with original video

## Installation

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

## Run

Launch a file picker:

```bash
python unicode_video_studio.py
```

Or specify a video file directly:

```bash
python unicode_video_studio.py "path/to/video.mp4"
```

Default quality settings: **450 columns, 8-pixel glyph font size, 5× supersampling, up to 256 glyphs, and target 5 FPS**. Change these in the source file or through live sliders where available.

## Controls

| Key | Action |
| --- | --- |
| Q / Esc | Quit |
| Space | Pause/resume |
| S | Side-by-side preview |
| A | Adaptive shape matching |
| D | Dithering |
| T | Temporal stabilization |
| L | CLAHE local contrast |
| N | Noise reduction |
| F | Cycle fonts |

The settings window includes live sliders for rendering quality and font selection.

For a deeper explanation of the glyph selection process and processing pipeline, see [Rendering architecture](docs/RENDERING_PIPELINE.md).

## How it works

1. Open a video using OpenCV and sample each frame at character-grid resolution.
2. Optionally apply denoising, local/global contrast adjustments, and edge enhancement.
3. Render font glyphs with Pillow at high resolution; downsample and measure their brightness and stroke patterns.
4. Select glyphs by brightness and local image-shape similarity, with optional dithering and temporal stabilization.
5. Compose chosen glyphs into a grayscale image and show it in an OpenCV preview window.

## Notes

- At 450 columns, glyph matching can be CPU intensive. Actual speed may be below the chosen target FPS.
- Available fonts and Unicode support depend on the machine.
- This version previews processed video; it does not currently save output videos or synchronize audio.
- The `.gitignore` excludes video files to avoid accidentally committing large or copyrighted source footage. Use footage you have permission to share.
- Add screenshots or a sample GIF to this README to show the effect.
- No open-source license is included yet; add one if you decide to grant reuse permissions.
