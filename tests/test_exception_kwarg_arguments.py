from io import BytesIO

from offline_debug import load_traceback, save_traceback


class TestError(Exception):
    def __init__(self, *, message: str) -> None:
        self.message = message


def test_exception_kwargs_arguments():
    try:
        raise TestError(message="hello")
    except TestError as e:
        buffer = BytesIO()
        save_traceback(e, buffer)
        buffer.seek(0)
        _loaded_exc = load_traceback(buffer, should_raise=False)
