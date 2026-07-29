import threading
import time
import unittest

from actions.browser_tab_bridge import NamedPipeBridge


class Runtime:
    expected_generation = 7
    cancellation_event = threading.Event()

    @staticmethod
    def generation_getter():
        return 7


class FakeConnection:
    closed = False

    def __init__(self):
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)

    def poll(self, timeout):
        return True

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
        self.closed = True


class BridgeRuntimeReadinessTests(unittest.TestCase):
    def make_bridge(self):
        bridge = NamedPipeBridge(timeout=0.1, readiness_timeout=0.15)
        bridge._started = True
        bridge._server_ready.set()
        return bridge

    def test_server_started_without_client_reports_no_client(self):
        bridge = self.make_bridge()
        result = bridge.request("list_tabs", {}, Runtime())
        self.assertEqual(result["status"], "bridge_unavailable")
        self.assertEqual(result["unavailable_reason"], "no_client")

    def test_client_arriving_inside_readiness_window_succeeds(self):
        bridge = self.make_bridge()
        connection = FakeConnection()
        timer = threading.Timer(
            0.03, bridge._install_connection, args=(connection, "new")
        )
        timer.start()
        started = time.monotonic()
        result = bridge.request("list_tabs", {}, Runtime())
        timer.join()
        self.assertTrue(result["success"])
        self.assertLess(time.monotonic() - started, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
