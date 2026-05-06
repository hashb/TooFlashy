# Test Suite

This directory contains the regression tests for the PSE analysis algorithms
implemented in `tooflashy`. The tests are split between small synthetic unit
tests and larger end-to-end checks against the optional PSE media repository.

## Running Tests

Run the regular test suite:

```sh
uv run pytest
```

Run with the PSE media cases enabled:

```sh
PSE_TEST_MEDIA=/path/to/pse-test-media uv run pytest
```

The PSE media run is much slower because it decodes many full-HD videos. When
`pytest-xdist` is available, parallel execution is useful:

```sh
PSE_TEST_MEDIA=/path/to/pse-test-media uv run pytest -n auto
```

## Test Files

- `test_paper_algorithms.py` covers core threshold logic from the Jordan and
  Vanderheiden paper, including luminance transitions, red transitions, area
  thresholds, and event counting.
- `test_transition_duration.py` and `test_transition_timing.py` cover the
  proposed transition-duration window and one-second event-count boundary.
- `test_synchronization.py` covers combining same-direction areas that transition
  within the synchronization window.
- `test_standard_profiles.py` checks that named standards map to the expected
  threshold profiles.
- `test_nhk_jba_thresholds.py` covers the NHK/JBA SDR and HDR classification
  helper logic.
- `test_ffmpeg_reader.py` covers frame decoding behavior.
- `test_video_analysis.py` covers end-to-end video analysis, including optional
  PSE media fixtures when `PSE_TEST_MEDIA` is set.
- `pse_media_cases.py` discovers PSE media JSON/video pairs and normalizes which
  standards apply to each media directory.

## PSE Media Expectations

The PSE media repository contains JSON files with expected outcomes for many
standards. The test harness filters those expectations in two ways:

- Only standards listed for a media directory in
  `APPLICABLE_STANDARDS_BY_DIRECTORY` are tested. This avoids asserting
  standards that the media directory was not intended to cover.
- `KNOWN_UNVERIFIABLE_VIDEO_CASES` are excluded from the parametrized encoded
  video expectation test. Discovery tests still verify that all JSON cases and
  prebuilt videos exist.

The exclusion list is deliberately in `pse_media_cases.py`, next to the
collection code, because those IDs are part of the fixture policy for the
encoded-video test.

## Unverifiable Encoded-Video Cases

Some PSE media cases are useful as source-pattern examples but are not reliable
as end-to-end assertions against the distributed `.avi` files. The current
unverifiable cases fall into three groups:

- Threshold-collapse cases: source CSV/PNG inputs sit just above or just below a
  threshold, but the distributed I420 video encoding collapses the relevant
  luminance, chroma, or area distinction. The encoded frames no longer preserve
  the condition the source fixture was designed to test.
- Count-rule conflicts: some expected-pass JSON fixtures contain more than six
  alternating qualifying transitions within one second in the encoded video.
  That contradicts the paper's stated failure condition for seven alternating
  transitions in a one-second span.
- Red-flash fixture conflicts: some JSON expectations disagree with the set
  manifest, or encoded chroma lands on the wrong side of the saturated-red or
  UCS-distance threshold.

These cases are excluded only from the PSE encoded-video expectation check. The
algorithmic behavior they touch should be covered by focused synthetic tests
where frame values, areas, and timings are controlled exactly.

