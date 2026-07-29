import unittest

from actions.local_intent_gate import LOCAL_INTENT_GATE


class ListTabsLocalRoutingTests(unittest.TestCase):
    POSITIVES = (
        "Liệt kê các tab đang mở", "Liệt kê tab hiện tại",
        "Cho tôi xem các tab đang mở", "Tôi đang mở những tab nào?",
        "Có những tab nào đang mở?", "Đọc danh sách tab",
        "Những tab nào đang bật?", "Xem giúp tôi các tab hiện tại",
        "List my open tabs", "Show current browser tabs",
        "Liệt kê những tab đang mở", "Cho tôi xem tab hiện tại",
        "Các tab nào đang mở?", "Hiện tại tôi mở tab nào?",
        "Đọc danh sách các tab đang mở", "Xem các tab đang bật",
        "List open browser tabs", "What tabs are open?",
        "Which tabs are currently open?", "Show my open tabs",
    )
    NEGATIVES = (
        "Tab trình duyệt là gì?", "Chrome có hỗ trợ nhiều tab không?",
        "Tôi vừa đóng một tab.", "Tab kia.", "Tôi thích dùng tab.",
        "Hướng dẫn tôi tạo tab trong Unity.",
    )

    def test_twenty_vietnamese_and_english_variants(self):
        for phrase in self.POSITIVES:
            with self.subTest(phrase=phrase):
                intent = LOCAL_INTENT_GATE.classify(phrase)
                self.assertIsNotNone(intent)
                self.assertEqual(intent.action, "list_browser_tabs")
                self.assertEqual(intent.arguments, {})

    def test_false_positives_and_ambiguous_phrase_do_not_route(self):
        false_positive_count = 0
        for phrase in self.NEGATIVES:
            if LOCAL_INTENT_GATE.classify(phrase) is not None:
                false_positive_count += 1
        self.assertEqual(false_positive_count, 0)

    def test_count_question_routes_without_arguments(self):
        intent = LOCAL_INTENT_GATE.classify("Tôi đang mở bao nhiêu tab?")
        self.assertEqual(intent.action, "list_browser_tabs")
        self.assertEqual(intent.arguments, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
