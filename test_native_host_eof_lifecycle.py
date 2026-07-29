import contextlib
import io
import threading
import unittest

from native_host.jarvis_native_host import PipeRelay


class FakePipe:
    def __init__(self, eof=False):
        self.eof = eof
        self.recv_calls = 0
        self.close_calls = 0
        self.sent = []

    def send(self, message):
        self.sent.append(message)

    def recv(self):
        self.recv_calls += 1
        if self.eof:
            raise EOFError
        threading.Event().wait(5)

    def close(self):
        self.close_calls += 1


class NativeHostEofLifecycleTests(unittest.TestCase):
    def test_eof_stops_reader_and_owner_reconnects_once(self):
        old_pipe = FakePipe(eof=True)
        new_pipe = FakePipe()
        pipes = iter((old_pipe, new_pipe))
        factory_calls = 0

        def factory(*args, **kwargs):
            nonlocal factory_calls
            factory_calls += 1
            return next(pipes)

        relay = PipeRelay(b"secret", client_factory=factory, max_reconnects=3)
        stderr = io.StringIO()
        thread_errors = []
        old_hook = threading.excepthook
        threading.excepthook = lambda args: thread_errors.append(args.exc_value)
        try:
            with contextlib.redirect_stderr(stderr):
                relay.ensure_connected()
                relay.start_reader()
                old_reader = relay._reader
                old_reader.join(1.0)

                self.assertTrue(relay.stop_event.is_set())
                self.assertFalse(old_reader.is_alive())
                self.assertEqual(old_pipe.recv_calls, 1)
                self.assertTrue(relay.reconnect(join_timeout=0.1))
                self.assertIsNot(relay._connection, old_pipe)
                self.assertIsNot(relay._reader, old_reader)
                self.assertEqual(factory_calls, 2)
                self.assertEqual(old_pipe.close_calls, 1)
                self.assertEqual(old_pipe.recv_calls, 1)
                relay.close()
        finally:
            threading.excepthook = old_hook

        output = stderr.getvalue()
        self.assertEqual(thread_errors, [])
        self.assertIn("[JARVIS NATIVE HOST] pipe_disconnected reason=eof", output)
        self.assertNotIn("Traceback", output)

    def test_shutdown_eof_does_not_retry(self):
        pipe = FakePipe(eof=True)
        factory_calls = 0

        def factory(*args, **kwargs):
            nonlocal factory_calls
            factory_calls += 1
            return pipe

        relay = PipeRelay(b"secret", client_factory=factory, max_reconnects=3)
        with contextlib.redirect_stderr(io.StringIO()):
            relay.ensure_connected()
            relay.start_reader()
            relay._reader.join(1.0)
            relay.close()
        self.assertEqual(factory_calls, 1)
        self.assertEqual(pipe.recv_calls, 1)
        self.assertEqual(pipe.close_calls, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
