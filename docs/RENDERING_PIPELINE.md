# How the ASCII/Unicode renderer works

The renderer converts each video frame into a grid of **rendered Unicode glyphs**, matching both local brightness and local stroke shape. Unlike terminal-based ASCII art, it draws its own antialiased glyph bitmaps into an OpenCV preview window.

## Processing pipeline

```mermaid
flowchart LR
    A[Video frame] --> B[Resize and grayscale]
    B --> C[Denoise and contrast]
    C --> D[Edge enhancement]
    D --> E[Character-grid patches]
    E --> F[Brightness and shape matching]
    F --> G[Temporal stabilization]
    G --> H[Assemble glyph bitmaps]
    H --> I[OpenCV preview]
```

### 1. Build a font-specific glyph atlas

For each supported mono-spaced font, the program:

- Enumerates printable ASCII and selected Unicode ranges (while excluding box and block glyphs).
- Renders glyphs with Pillow at **5×** the desired font resolution and downsamples with Lanczos interpolation.
- Discards missing glyphs and duplicate rendered bitmaps.
- Measures average pixel brightness and selects up to **256** glyphs spanning the actual density range.
- Precomputes normalized **4×4 shape patterns** for use in matching.

Font atlases are cached after the first time a font is selected. More Unicode characters do not automatically mean better results; the supported shapes and actual font rendering matter.

### 2. Prepare each frame

OpenCV downsamples the source to the character-grid sampling resolution. Optional stages include:

- **Bilateral filtering** to reduce grain while preserving edges.
- **CLAHE** to boost contrast in local image regions.
- **Global brightness and contrast** adjustments.
- **Unsharp-style edge enhancement** using a Gaussian-blurred reference.

The default character grid is **450 columns**, with a small sampled patch per glyph position. The corresponding row count depends on the video aspect ratio and glyph dimensions.

### 3. Match a glyph to each image patch

Each patch is summarized by:

- Its **mean brightness**, used to identify a base glyph by measured ink coverage.
- A centered and normalized **4×4 shape pattern**, used to identify similar lines and textures.
- Its local intensity variation, used for adaptive detail weighting.

Instead of comparing against every glyph at every position, the matcher evaluates a window of candidates around the closest brightness match (default radius **14**, or at most 29 candidates).

A candidate's score combines pattern similarity with a penalty for brightness mismatch. This keeps the rendering visually recognizable while preserving the grayscale composition.

### 4. Optional visual refinements

- **Ordered dithering:** an 8×8 Bayer matrix makes subtle brightness differences easier to represent.
- **Adaptive shape weighting:** detailed regions prioritize stroke similarity more strongly; flat areas emphasize brightness. The spatial grid stays the same size.
- **Temporal stabilization:** the previous glyph may be retained if the brightness change is small and the old glyph is nearly as good a match, reducing visible flicker.

### 5. Compose and display

The chosen glyph IDs index a pre-rendered NumPy atlas. The corresponding character bitmaps are tiled into a grayscale image, which is displayed in a dedicated OpenCV window. The preview can be switched to show the original video alongside the rendered result.

## Default quality settings

| Setting | Default |
| --- | ---: |
| Target FPS | 5 |
| Character columns | 450 |
| Font size | 8 px |
| Supersampling | 5× |
| Maximum selected glyphs | 256 |
| Shape samples per glyph | 4×4 |
| Candidate radius | 14 |

These settings favor fine spatial detail, although actual throughput depends on CPU and source video resolution.

## Limitations and tradeoffs

- **Character shapes are discrete.** More spatial samples or Unicode characters cannot exactly reproduce the source image.
- **Font availability varies by computer.** Unsupported Unicode characters are filtered, and two fonts may produce different results.
- **Small glyphs sacrifice readability for image detail.** This is intentional at the default 450 columns.
- **Temporal stabilization can introduce persistence.** Higher stability settings reduce flicker but can delay visual changes.
- **Preview is not an export.** The current implementation does not preserve audio or save a processed video.

See the [README](../README.md) for installation, playback controls, and feature highlights.
