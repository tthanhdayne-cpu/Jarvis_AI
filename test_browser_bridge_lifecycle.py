import threading
import unittest
from pathlib import Path

from actions.browser_tab_bridge import NamedPipeBridge
from native_host.jarvis_native_host import PipeRelay


class Runtime:
    expected_generation = 4
    cancellation_event = threading.Event()

    @staticmethod
    def generation_getter():
        return 4


class FakeConnection:
    def __init__(self, connection_id="connection-test", poll_result=True):
        self.connection_id = connection_id
        self.poll_result = poll_result
        self.sent = []
        self.close_calls = 0

    def send(self, message):
        self.sent.append(message)

    def poll(self, timeout):
        return self.poll_result

    def recv(self):
        request = self.sent[-1]
        return {
            "request_id": request["request_id"],
            "connection_id": request["connection_id"],
            "session_generation": request["session_generation"],
            "success": True,
            "status": "completed",
            "data": {"tabs": []},
        }

    def close(self):
        self.close_calls += 1


class FakeNativePipe:
    def __init__(self):
        self.sent = []
        self.close_calls = 0

    def send(self, message):
        self.sent.append(message)

    def close(self):
        self.close_calls += 1


class BrowserBridgeLifecycleTests(unittest.TestCase):
    def test_multiple_requests_reuse_same_connection(self):
        bridge = NamedPipeBridge(timeout=0.01)
        connection = FakeConnection()
        bridge._started = True
        bridge._connection = connection
        bridge.connection_id = connection.connection_id
        bridge._connected.set()

        first = bridge.request("list_tabs", {}, Runtime())
        second = bridge.request("list_tabs", {}, Runtime())

        self.assertTrue(first["success"])
        self.assertTrue(second["success"])
        self.assertEqual(len(connection.sent), 2)
        self.assertEqual(connection.close_calls, 0)

    def test_stale_connection_is_replaced_once(self):
        bridge = NamedPipeBridge(timeout=0.01)
        old = FakeConnection("old")
        new = FakeConnection("new")
        bridge._connection = old
        bridge.connection_id = "old"
        bridge._connected.set()

        bridge._install_connection(new, "new")

        self.assertEqual(old.close_calls, 1)
        self.assertIs(bridge._connection, new)
        self.assertEqual(bridge.connection_id, "new")

    def test_timed_out_connection_is_not_reused(self):
        bridge = NamedPipeBridge(timeout=0.01)
        connection = FakeConnection(poll_result=False)
        bridge._started = True
        bridge._connection = connection
        bridge.connection_id = connection.connection_id
        bridge._connected.set()

        result = bridge.request("list_tabs", {}, Runtime())

        self.assertEqual(result["status"], "request_timeout")
        self.assertEqual(connection.close_calls, 1)
        self.assertIsNone(bridge._connection)

    def test_native_host_survives_multiple_responses(self):
        pipe = FakeNativePipe()
        factory_calls = 0

        def factory(*args, **kwargs):
            nonlocal factory_calls
            factory_calls += 1
            return pipe

        relay = PipeRelay(b"secret", client_factory=factory, max_reconnects=3)
        self.assertTrue(relay.send({"response": 1}))
        self.assertTrue(relay.send({"response": 2}))

        self.assertEqual(factory_calls, 1)
        self.assertEqual(len(pipe.sent), 3)  # hello + two responses
        self.assertEqual(pipe.close_calls, 0)

    def test_native_host_reconnect_is_bounded(self):
        attempts = 0

        def failing_factory(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            raise OSError("pipe unavailable")

        relay = PipeRelay(
            b"secret", client_factory=failing_factory, max_reconnects=3
        )
        self.assertIsNone(relay.ensure_connected())
        self.assertEqual(attempts, 3)

    def test_gemini_and_sleep_do_not_close_bridge(self):
        source = Path(__file__).with_name("main.py").read_text(encoding="utf-8")
        active = source[source.index("async def _run_active_session"):source.index("async def run(self)")]
        sleep = source[source.index("def _prepare_sleep_state"):source.index("def _complete_sleep_transition")]
        self.assertNotIn("BROWSER_TAB_SERVICE.shutdown", active)
        self.assertNotIn("WINDOWS_ACTION_REGISTRY.shutdown", active)
        self.assertNotIn("BROWSER_TAB_SERVICE.shutdown", sleep)
        self.assertNotIn("WINDOWS_ACTION_REGISTRY.shutdown", sleep)

    def test_extension_has_single_port_guard_and_disconnect_reason(self):
        source = (
            Path(__file__).parent
            / "chrome_extension" / "jarvis_youtube" / "service_worker.js"
        ).read_text(encoding="utf-8")
        self.assertIn("if (port || connecting) return;", source)
        self.assertIn("if (port !== nativePort) return;", source)
        self.assertIn("chrome.runtime.lastError?.message", source)
        self.assertIn("reconnects >= MAX_RECONNECTS", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
