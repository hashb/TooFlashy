from tooflashy.thresholds import classify_nhk_jba_hdr, classify_nhk_jba_sdr


def test_nhk_jba_sdr_table4_categories() -> None:
    assert classify_nhk_jba_sdr(brightness_change_percent=15, area_fraction=0.30).category == "moderate"
    assert classify_nhk_jba_sdr(brightness_change_percent=25, area_fraction=0.30).category == "intermediate"
    assert classify_nhk_jba_sdr(brightness_change_percent=25, area_fraction=0.85).category == "scene-change"
    assert classify_nhk_jba_sdr(brightness_change_percent=5, area_fraction=0.30) is None
    assert classify_nhk_jba_sdr(brightness_change_percent=25, area_fraction=0.10) is None


def test_nhk_jba_hdr_table4_categories_below_160_cd_m2() -> None:
    assert classify_nhk_jba_hdr(darker_luminance=100, brighter_luminance=125, area_fraction=0.30).category == "moderate"
    assert classify_nhk_jba_hdr(darker_luminance=100, brighter_luminance=145, area_fraction=0.30).category == "intermediate"
    assert classify_nhk_jba_hdr(darker_luminance=100, brighter_luminance=145, area_fraction=0.85).category == "scene-change"


def test_nhk_jba_hdr_table4_categories_at_or_above_160_cd_m2() -> None:
    moderate = classify_nhk_jba_hdr(darker_luminance=180, brighter_luminance=210, area_fraction=0.30)
    intermediate = classify_nhk_jba_hdr(darker_luminance=180, brighter_luminance=240, area_fraction=0.30)
    scene = classify_nhk_jba_hdr(darker_luminance=180, brighter_luminance=240, area_fraction=0.85)

    assert moderate is not None
    assert moderate.category == "moderate"
    assert moderate.max_flashes_per_second == 5
    assert moderate.max_duration_seconds == 2
    assert intermediate is not None
    assert intermediate.category == "intermediate"
    assert scene is not None
    assert scene.category == "scene-change"
