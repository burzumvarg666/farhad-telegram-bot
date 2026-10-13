import unittest

from app.main import MAX_MESSAGES, main_menu, memory, remember, reset_memory, route, history


class BotBehaviorTests(unittest.TestCase):
    def tearDown(self):
        for chat_id in (987654321, 987654322):
            reset_memory(chat_id)

    def test_menu_has_expected_actions(self):
        keyboard = main_menu()["keyboard"]
        actions = [button["text"] for row in keyboard for button in row]
        self.assertEqual(
            actions,
            ["گفتگو با هوش مصنوعی", "وضعیت حافظه", "راهنما", "وضعیت سرویس", "درباره ربات"],
        )

    def test_normal_text_keeps_fast_free_route(self):
        self.assertEqual(route("سلام، حالت چطوره؟"), ["deepseek_fast", "gemini", "groq"])

    def test_complex_text_keeps_existing_route(self):
        self.assertEqual(route("لطفاً این کد Python را بررسی کن"), ["deepseek", "groq"])

    def test_images_keep_existing_vision_route(self):
        self.assertEqual(route("این عکس چیست؟", has_image=True), ["deepseek", "groq"])

    def test_memory_is_capped(self):
        chat_id = 987654321
        for index in range(MAX_MESSAGES + 4):
            remember(chat_id, "user", f"message-{index}")
        messages = history(chat_id)
        self.assertEqual(len(messages), MAX_MESSAGES)
        self.assertEqual(messages[-1]["content"], f"message-{MAX_MESSAGES + 3}")


if __name__ == "__main__":
    unittest.main()
