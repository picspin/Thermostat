"""Left control column scrolls instead of squashing window-level controls.

Uses the offscreen Qt platform from conftest.py. No visible display server.
"""

from PyQt5.QtWidgets import QApplication, QGroupBox, QLabel, QScrollArea, QSplitter

from mvc.views.thermometry_view import ThermometryView


if QApplication.instance() is None:
    _QT_APP = QApplication([])
else:
    _QT_APP = QApplication.instance()

_SLACK = 2


def _show(view, width, height):
    view.resize(width, height)
    view.show()
    _QT_APP.processEvents()
    # Fonts and styles can change layout minimums after the first show.
    view._freeze_left_panel_minimum()
    view.left_panel.updateGeometry()
    view.left_panel_scroll.updateGeometry()
    _QT_APP.processEvents()


def _left_scroll(view):
    scrolls = [
        area
        for area in view.findChildren(QScrollArea)
        if area.objectName() == "left_panel_scroll"
    ]
    assert len(scrolls) == 1
    return scrolls[0]


def _nearest_group(widget):
    parent = widget.parent()
    while parent is not None:
        if isinstance(parent, QGroupBox):
            return parent
        parent = parent.parent()
    return None


def _box_in(widget, ancestor):
    """Return top, bottom, left, right of widget mapped into ancestor."""
    top_left = widget.mapTo(ancestor, widget.rect().topLeft())
    return (
        top_left.y(),
        top_left.y() + widget.height(),
        top_left.x(),
        top_left.x() + widget.width(),
    )


def _assert_fully_in_viewport(widget, viewport):
    top, bottom, left, right = _box_in(widget, viewport)
    name = widget.objectName() or type(widget).__name__
    assert top >= -_SLACK, (name, top)
    assert bottom <= viewport.height() + _SLACK, (
        name,
        bottom,
        viewport.height(),
    )
    assert left >= -_SLACK, (name, left)
    assert right <= viewport.width() + _SLACK, (name, right, viewport.width())


def _assert_visible_without_scrolling(view, scroll, controls):
    bar = scroll.verticalScrollBar()
    assert bar.value() == 0
    # The column is wide enough for its rows, so nothing is clipped sideways.
    assert scroll.horizontalScrollBar().maximum() == 0
    viewport = scroll.viewport()
    for widget in controls:
        assert widget.height() >= 20
        assert widget.width() >= 20
        _assert_fully_in_viewport(widget, viewport)


def test_left_panel_scroll_keeps_window_controls_visible():
    view = ThermometryView()
    try:
        scroll = _left_scroll(view)
        assert scroll.widget() is view.left_panel
        assert scroll.minimumWidth() >= 320

        splitters = [
            splitter
            for splitter in view.findChildren(QSplitter)
            if splitter.widget(0) is scroll
        ]
        assert len(splitters) == 1

        controls = (
            view.window_center_input,
            view.window_width_input,
            view.auto_window_button,
        )

        _show(view, 1200, 800)
        for widget in controls:
            assert widget.isEnabled()
            assert widget.height() >= 20

        window_group = _nearest_group(view.window_center_input)
        assert window_group is not None and window_group.title() == "Window"
        assert _nearest_group(view.window_width_input) is window_group
        assert _nearest_group(view.auto_window_button) is window_group

        options_groups = [
            group
            for group in view.findChildren(QGroupBox)
            if group.title() == "Processing Options"
        ]
        assert len(options_groups) == 1
        options_group = options_groups[0]
        for widget in controls:
            assert not options_group.isAncestorOf(widget)
        assert options_group.isAncestorOf(view.original_temp_input)

        two_frame_groups = [
            group
            for group in view.findChildren(QGroupBox)
            if group.title() == "Two-frame temperature"
        ]
        assert len(two_frame_groups) == 1
        two_top, _, _, _ = _box_in(two_frame_groups[0], view.left_panel)
        window_top, _, _, _ = _box_in(window_group, view.left_panel)
        options_top, _, _, _ = _box_in(options_group, view.left_panel)
        assert two_top < window_top < options_top

        paired_buddies = set()
        for label in window_group.findChildren(QLabel):
            buddy = label.buddy()
            if buddy is None:
                continue
            label_top, label_bottom, label_left, label_right = _box_in(
                label, window_group
            )
            spin_top, _, spin_left, spin_right = _box_in(buddy, window_group)
            assert label_bottom <= spin_top + _SLACK
            assert label_left < spin_right and spin_left < label_right
            assert label_top < spin_top
            paired_buddies.add(buddy)
        assert view.window_center_input in paired_buddies
        assert view.window_width_input in paired_buddies

        _, button_bottom, _, _ = _box_in(view.auto_window_button, window_group)
        assert button_bottom <= window_group.height() + _SLACK
        assert view.auto_window_button.height() >= 20

        _assert_visible_without_scrolling(view, scroll, controls)

        _show(view, 1280, 800)
        _assert_visible_without_scrolling(view, scroll, controls)

        _show(view, 1200, 480)
        for widget in controls:
            assert widget.isEnabled()
            assert widget.height() >= 20
        viewport = scroll.viewport()
        content_height = max(
            view.left_panel.minimumHeight(),
            view.left_panel.sizeHint().height(),
        )
        if content_height > viewport.height() + _SLACK:
            assert scroll.verticalScrollBar().maximum() > 0

        for name in (
            "load_real1_button",
            "calc_button",
            "cinema_button",
            "coarse_liver_checkbox",
            "registration_site_combo",
            "original_temp_input",
        ):
            widget = getattr(view, name)
            assert widget is not None
            assert scroll.widget().isAncestorOf(widget)
    finally:
        view.close()
        _QT_APP.processEvents()
