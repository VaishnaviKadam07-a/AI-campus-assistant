"""Unit tests for the CampusAI assistant (stdlib unittest, no network, no API key).

Run from the project root:  python -m unittest discover -s tests -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import assistant as a  # noqa: E402


class FakeClient:
    """Mimics client.chat.completions.create(...) without any network call."""

    def __init__(self, text="AI says hello", error=None):
        self.calls, self._text, self._error = [], text, error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._text))])


class KnowledgeBaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = a.load_knowledge_base()

    def test_loads_all_categories(self):
        self.assertGreaterEqual(len(self.kb.entries), 16)
        self.assertEqual(len(self.kb.categories), 8)

    def test_every_entry_is_labelled_demo_data(self):
        for e in self.kb.entries:
            self.assertIn("[Demo data]", e["answer"], e["id"])

    def test_related_ids_exist(self):
        for e in self.kb.entries:
            for rid in e["related"]:
                self.assertIn(rid, self.kb.by_id, f"{e['id']} -> {rid}")

    def test_missing_file_raises(self):
        with self.assertRaises(a.KnowledgeBaseError):
            a.load_knowledge_base("does_not_exist.json")

    def test_bad_json_and_bad_entries_raise(self):
        cases = ["{not json", json.dumps({"entries": []}),
                 json.dumps({"entries": [{"id": "x", "category": "c", "question": "q"}]})]
        for content in cases:
            with self.subTest(content=content[:30]):
                with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
                    fh.write(content)
                try:
                    with self.assertRaises(a.KnowledgeBaseError):
                        a.load_knowledge_base(fh.name)
                finally:
                    os.unlink(fh.name)


class MatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = a.load_knowledge_base()

    def test_different_phrasings_find_the_right_entry(self):
        cases = {
            "What time does the library open?": "lib-hours",
            "library timings": "lib-hours",
            "libary hours": "lib-hours",  # typo
            "how can I issue a book": "lib-borrow",
            "minimum attendance needed?": "att-minimum",
            "am I detained if my attendance is low": "att-minimum",
            "where do I get my hall ticket": "exam-hall-ticket",
            "how to apply for a scholarship": "sch-apply",
            "how do I prepare for placements": "plc-prep",
            "what is the wifi password": "fac-wifi",
            "how do I get my paper rechecked": "exam-revaluation",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                answer = a.answer_question(question, self.kb)
                self.assertEqual(answer.mode, "faq")
                self.assertEqual(answer.sources[0], expected)

    def test_unknown_questions_are_not_invented(self):
        for q in ["What is the capital of France?", "Who is the principal?", "what is the fee structure",
                  "tell me about the college", "asdfghjkl"]:
            with self.subTest(q=q):
                answer = a.answer_question(q, self.kb)
                self.assertEqual(answer.mode, "unknown")
                self.assertFalse(answer.found)
                self.assertIn("won't guess", answer.text)
                self.assertTrue(answer.related)  # offers something useful instead

    def test_related_questions_exclude_the_answered_one(self):
        answer = a.answer_question("minimum attendance requirement", self.kb)
        self.assertNotIn(self.kb.by_id["att-minimum"]["question"], answer.related)
        self.assertTrue(answer.related)


class InputHandlingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = a.load_knowledge_base()

    def test_invalid_input(self):
        for bad in ["", "   ", "!!! ???", None, 123, "x" * (a.MAX_QUESTION_LENGTH + 1)]:
            with self.subTest(bad=str(bad)[:20]):
                self.assertEqual(a.answer_question(bad, self.kb).mode, "invalid")

    def test_control_characters_are_cleaned(self):
        clean, err = a.validate_question("library\x00 hours\n\t please")
        self.assertIsNone(err)
        self.assertEqual(clean, "library hours please")

    def test_small_talk(self):
        self.assertEqual(a.answer_question("hello", self.kb).mode, "smalltalk")
        self.assertEqual(a.answer_question("Thanks!", self.kb).mode, "smalltalk")


class FallbackAndAITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = a.load_knowledge_base()

    def test_works_without_api_key(self):
        answer = a.answer_question("minimum attendance", self.kb)
        self.assertEqual(answer.mode, "faq")
        self.assertEqual(answer.text, self.kb.by_id["att-minimum"]["answer"])

    def test_ai_answer_used_when_client_available(self):
        client = FakeClient("Grounded AI answer")
        answer = a.answer_question("library hours", self.kb, client=client)
        self.assertEqual(answer.mode, "ai")
        self.assertEqual(answer.text, "Grounded AI answer")
        self.assertEqual(len(client.calls), 1)

    def test_prompt_contains_context_and_guardrails(self):
        client = FakeClient()
        a.answer_question("library hours", self.kb, client=client)
        system, user = client.calls[0]["messages"]
        self.assertIn("Never invent", system["content"])
        self.assertIn("library opening hours", user["content"].lower())

    def test_ai_failure_falls_back_to_faq_with_notice(self):
        client = FakeClient(error=RuntimeError("boom"))
        answer = a.answer_question("library hours", self.kb, client=client)
        self.assertEqual(answer.mode, "faq")
        self.assertIsNotNone(answer.notice)
        self.assertEqual(answer.text, self.kb.by_id["lib-hours"]["answer"])

    def test_empty_ai_response_falls_back(self):
        answer = a.answer_question("library hours", self.kb, client=FakeClient(text="   "))
        self.assertEqual(answer.mode, "faq")

    def test_ai_is_not_called_for_unknown_questions(self):
        client = FakeClient()
        answer = a.answer_question("capital of France", self.kb, client=client)
        self.assertEqual(answer.mode, "unknown")
        self.assertEqual(client.calls, [])

    def test_missing_openai_package_or_bad_key_does_not_crash(self):
        answer = a.answer_question("library hours", self.kb, api_key="sk-test-not-real")
        self.assertIn(answer.mode, {"ai", "faq"})  # "faq" when openai isn't installed / call fails
        self.assertTrue(answer.text)

    def test_api_key_helper_ignores_placeholders(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "your_api_key_here"}):
            self.assertIsNone(a.get_api_key())
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            self.assertIsNone(a.get_api_key())
            self.assertEqual(a.get_api_key("sk-real-looking"), "sk-real-looking")


if __name__ == "__main__":
    unittest.main()
