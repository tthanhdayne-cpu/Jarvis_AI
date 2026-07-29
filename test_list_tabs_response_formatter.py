import unittest

from actions.browser_tab_response_formatter import format_browser_tab_list_response


class ListTabsResponseFormatterTests(unittest.TestCase):
    def test_four_titles_are_spoken_with_count(self):
        tabs = [{"title": name, "hostname": "safe.example"} for name in (
            "YouTube", "Gmail", "GitHub", "ChatGPT"
        )]
        text = format_browser_tab_list_response(
            {"status": "completed", "data": {"tabs": tabs}},
            "Liệt kê các tab đang mở", logger=lambda _: None,
        )
        self.assertEqual(
            text, "Bạn đang mở 4 tab: YouTube, Gmail, GitHub và ChatGPT."
        )

    def test_nine_tabs_speaks_at_most_five_titles(self):
        tabs = [{"title": f"Tab {index}", "hostname": "safe.example"}
                for index in range(1, 10)]
        text = format_browser_tab_list_response(
            {"data": {"tabs": tabs}}, "Show tabs", logger=lambda _: None
        )
        self.assertIn("Bạn đang mở 9 tab. Năm tab đầu là", text)
        self.assertIn("Tab 5", text)
        self.assertNotIn("Tab 6", text)

    def test_count_only_does_not_speak_titles(self):
        result = {"data": {"tabs": [{"title": "YouTube"}]}}
        self.assertEqual(
            format_browser_tab_list_response(
                result, "Có bao nhiêu tab đang mở?", logger=lambda _: None
            ),
            "Bạn đang mở 1 tab.",
        )

    def test_url_and_raw_identifiers_are_never_spoken_and_hostname_is_fallback(self):
        result = {"data": {"tabs": [{
            "title": "https://example.com/?secret=1",
            "hostname": "example.com", "tab_ref": "raw-secret", "index": 99,
        }]}}
        text = format_browser_tab_list_response(
            result, "List tabs", logger=lambda _: None
        )
        self.assertIn("example.com", text)
        self.assertNotIn("https://", text)
        self.assertNotIn("secret", text)
        self.assertNotIn("raw-secret", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
