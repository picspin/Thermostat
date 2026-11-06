"""Run one callable off the GUI thread and return its result to that thread.

The registration defaults are not involved here.  This only moves work that
already has a result so the Qt event loop can keep painting and can accept
cancel.  Callbacks are delivered with a queued connection, so they run on the
thread that created the ``BackgroundCall`` (the GUI thread).
"""

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal


class _FunctionThread(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, function):
        super().__init__()
        self._function = function

    def run(self):
        try:
            value = self._function()
        except Exception as exc:  # noqa: BLE001 - the GUI thread reports it
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(value)


class BackgroundCall(QObject):
    """One in-flight call.  Create this object on the GUI thread."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._on_success = None
        self._on_failure = None

    def running(self):
        return self._thread is not None and self._thread.isRunning()

    def start(self, function, on_success, on_failure):
        if self.running():
            raise RuntimeError("background work is already running")
        thread = _FunctionThread(function)
        self._on_success = on_success
        self._on_failure = on_failure
        thread.succeeded.connect(self._deliver_success, Qt.QueuedConnection)
        thread.failed.connect(self._deliver_failure, Qt.QueuedConnection)
        self._thread = thread
        thread.start()

    def _deliver_success(self, value):
        callback = self._on_success
        if callback is not None:
            callback(value)

    def _deliver_failure(self, message):
        callback = self._on_failure
        if callback is not None:
            callback(message)
