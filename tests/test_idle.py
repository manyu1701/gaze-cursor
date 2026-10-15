from gaze_cursor.idle import IdleTimer


def test_awake_processes_every_frame_and_a_face_resets_the_timer() -> None:
    t = IdleTimer(0.0)
    assert all(t.process_this_frame(s * 0.033) for s in range(100))
    t.update(20.0, False)
    assert not t.sleeping(20.0) and t.idle_s(20.0) == 20.0
    t.update(25.0, True)
    assert t.idle_s(40.0) == 15.0 and not t.sleeping(40.0)


def test_sleeps_after_30_s_runs_one_frame_in_ten_and_wakes_on_a_face() -> None:
    t = IdleTimer(0.0)
    assert t.sleeping(30.0) and not t.sleeping(29.9)
    ran = [t.process_this_frame(31.0 + i * 0.033) for i in range(30)]
    assert sum(ran) == 3 and ran[9] and ran[19] and ran[29]
    t.update(32.0, True)
    assert not t.sleeping(32.1) and t.process_this_frame(32.1)


def test_quits_after_five_minutes_without_a_face() -> None:
    t = IdleTimer(100.0)
    assert not t.should_quit(399.9) and t.should_quit(400.0)
    t.update(399.0, True)
    assert not t.should_quit(500.0)


def test_busy_keeps_it_awake_so_a_live_test_is_never_cut_off() -> None:
    t = IdleTimer(0.0)
    for s in range(0, 600, 10):
        t.update(float(s), False, busy=True)
    assert not t.sleeping(600.0) and not t.should_quit(600.0)


def test_message_is_constant_while_asleep_then_counts_down_before_quitting() -> None:
    t = IdleTimer(0.0)
    assert t.message(29.0) is None
    assert t.message(31.0) == t.message(200.0) == "no face: low power until you are back"
    assert t.message(240.5) == "no face: closing in 60 s to save battery"
    assert t.message(299.2) == "no face: closing in 1 s to save battery"


def test_chip_shows_on_change_reminds_for_lasting_warnings_and_obeys_the_mode() -> None:
    from gaze_cursor.sinks.cursor_view import ChipTimer
    c = ChipTimer("auto", show_s=4.0, remind_s=60.0)
    a, b = (("on", "ok"),), (("moved", "warn"),)
    assert c.visible(0.0, a) and c.visible(3.9, a) and not c.visible(4.1, a)
    assert not c.visible(100.0, a)                          # nothing changed, nothing lasting: stays hidden
    assert c.visible(101.0, b, remind=True) and not c.visible(106.0, b, remind=True)
    assert c.visible(161.0, b, remind=True)                 # a lasting warning comes back after a minute
    assert not c.visible(170.0, b, remind=False)
    assert ChipTimer("always").visible(0.0, a) and not ChipTimer("never").visible(0.0, b, remind=True)
