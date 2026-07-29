import threading
import unittest

from actions.browser_tab_bridge import NamedPipeBridge
from actions.browser_tab_actions import BrowserTabService, BROWSER_TAB_SERVICE


def tearDownModule():
    BROWSER_TAB_SERVICE.shutdown()


class Runtime:
    expected_generation = 2
    cancellation_event = threading.Event()
    source_turn = 1

    @staticmethod
    def state_getter():
        return "ACTIVE"

    @staticmethod
    def sleep_intent_getter():
        return False

    @staticmethod
    def generation_getter():
        return 2


class FakeConnection:
    def __init__(self, connection_id):
        self.connection_id = connection_id
        self.closed = False
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)

    def poll(self, timeout):
        return True

    def recv(self):
        request = self.sent[-1]
        return {
            "request_id": request["request_id"],
            "connection_id": self.connection_id,
            "session_generation": request["session_generation"],
            "success": True,
            "status": "completed",
            "data": {"tabs": []},
        }

    def close(self):
        self.closed = True


class BridgeConnectionReplacementTests(unittest.TestCase):
    def test_new_connection_survives_late_old_disconnect(self):
        bridge = NamedPipeBridge(timeout=0.1, readiness_timeout=0.05)
        bridge._started = True
        bridge._server_ready.set()
        old = FakeConnection("A")
        new = FakeConnection("B")
        bridge._install_connection(old, "A")
        bridge._install_connection(new, "B")

        bridge._disconnect("late_old_callback", expected=old)

        self.assertIs(bridge._connection, new)
        self.assertTrue(bridge._connected.is_set())
        registry_service = BrowserTabService(transport=bridge)
        result = registry_service.list_tabs(runtime=Runtime())
        self.assertTrue(result["success"])
        self.assertEqual(len(new.sent), 1)
        self.assertNotEqual(result.get("status"), "bridge_unavailable")
        registry_service.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
