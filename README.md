# TooFlashy

Python tools for screening video against photosensitive epilepsy (PSE) flash-risk
thresholds described in `docs/3694790.pdf`, Jordan and Vanderheiden (2024),
"International Guidelines for Photosensitive Epilepsy: Gap Analysis and
Recommendations."

The implementation uses `ffprobe`/`ffmpeg` for video metadata and decoding,
OpenCV for area-window operations, and NumPy for color and frame math.

> [!WARNING]
> THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT WARRANTY OF ANY KIND. SPECIFICALLY,
> THIS SOFTWARE IS NOT CERTIFIED FOR MEDICAL USE AND SHOULD NOT BE RELIED UPON
> AS A GUARANTEE OF SAFETY FOR INDIVIDUALS WITH PHOTOSENSITIVE EPILEPSY.


## Implemented Algorithms

- sRGB gamma expansion and relative luminance.
- Michelson contrast and luminance transition thresholds.
- Saturated-red detection and WCAG 2.0 / WCAG 2.2 red-transition math.
- WCAG local area threshold, proposed 416 x 416 px area threshold, and 25%-screen broadcast thresholds.
- Proposed 66 ms qualifying transition duration.
- Proposed 20 ms synchronized area summing.
- Proposed fast-flash merging for 65 Hz or faster flashing.
- One-second alternating transition count failure rule.
- Appendix viewing-angle, solid-angle, visual-cortex stimulation, and response-proportion equations.
- NHK/JBA SDR and HDR moderate/intermediate/scene-change classifiers from Table 4.

The paper discusses regular pattern detection and motion compensation, but it does
not provide complete implementation thresholds for those algorithms. This package
does not claim pattern or motion-compensated conformance.

## Usage

Install and run with `uv`:

```bash
uv sync --dev
uv run tooflashy path/to/video.mp4
uv run tooflashy --json path/to/video.mp4
```

Python API:

```python
from tooflashy import analyze_video, profile_for_standard

result = analyze_video("video.mp4", profile=profile_for_standard("wcag2_2"))
print(result.passes, result.failures)
```

Available standard profiles include `wcag2_2`, `trace24`/`proposed`, `iso`,
`itu_r1702_4`, and `ofcom2017`.

## Tests

The test suite uses `uv`, a local virtual environment, `ffmpeg`, OpenCV, and
generated media from `https://github.com/traceRERC/pse-test-media`.

```bash
uv run pytest
```

Set `PSE_TEST_MEDIA=/path/to/pse-test-media` to reuse an existing clone instead
of letting the tests clone the benchmark repository.
