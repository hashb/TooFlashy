from tooflashy.analysis import FlashEvent, count_failure_in_events, merge_fast_flash_events


def _event(frame: int, time: float, direction: int) -> FlashEvent:
    return FlashEvent("luminance", frame, time, direction, 1000)


def test_more_than_six_alternating_transition_counts_fail() -> None:
    events = [_event(i, i / 30, 1 if i % 2 else -1) for i in range(7)]
    assert count_failure_in_events(events)


def test_slow_or_non_alternating_transitions_do_not_fail() -> None:
    non_alternating = [_event(i, i / 30, 1) for i in range(7)]
    assert not count_failure_in_events(non_alternating)

    slow = [_event(i, i * 0.2, 1 if i % 2 else -1) for i in range(7)]
    assert not count_failure_in_events(slow)


def test_transition_exactly_one_second_later_is_outside_count_window() -> None:
    events = [_event(i, i / 30, 1 if i % 2 else -1) for i in (0, 1, 10, 11, 20, 21, 30)]

    assert not count_failure_in_events(events)


def test_fast_flashes_merge_to_two_transition_counts() -> None:
    events = [
        _event(0, 0.000, 1),
        _event(1, 0.004, -1),
        _event(2, 0.008, 1),
        _event(3, 0.012, -1),
        _event(4, 0.016, 1),
        _event(5, 0.020, -1),
        _event(6, 0.024, 1),
        _event(7, 0.028, -1),
    ]

    merged = merge_fast_flash_events(events)

    assert [(event.time_seconds, event.direction) for event in merged] == [(0.0, 1), (0.004, -1)]
    assert not count_failure_in_events(events)
