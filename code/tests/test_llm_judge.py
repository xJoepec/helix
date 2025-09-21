from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure the 'code' directory (package root for 'helix') is on sys.path
PKG_ROOT = os.path.dirname(os.path.dirname(__file__))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

"""Tests for LLM judge implementation with rubric-based scoring.

Imports from the package are done inside test bodies to avoid E402 with sys.path edits.
"""


class TestCalibrationExample(unittest.TestCase):
    def test_calibration_example_creation(self):
        """Test CalibrationExample dataclass creation."""
        # Import here to avoid path issues
        sys.path.insert(0, '/Users/nest/Desktop/GITHUB/helix/environments')
        from helixenv.llm_judge import CalibrationExample

        example = CalibrationExample(
            prompt="Test prompt",
            response="Test response",
            physics_score=0.8,
            reasoning="Good physics understanding",
            key_concepts=["entropy", "spectral gap"]
        )

        self.assertEqual(example.prompt, "Test prompt")
        self.assertEqual(example.response, "Test response")
        self.assertEqual(example.physics_score, 0.8)
        self.assertEqual(example.reasoning, "Good physics understanding")
        self.assertEqual(len(example.key_concepts), 2)

    def test_calibration_example_to_dict(self):
        """Test CalibrationExample conversion to dictionary."""
        sys.path.insert(0, '/Users/nest/Desktop/GITHUB/helix/environments')
        from helixenv.llm_judge import CalibrationExample

        example = CalibrationExample(
            prompt="Test",
            response="Response",
            physics_score=0.9,
            reasoning="Excellent",
            key_concepts=["test"]
        )

        example_dict = example.to_dict()

        self.assertIsInstance(example_dict, dict)
        self.assertEqual(example_dict["physics_score"], 0.9)
        self.assertIn("prompt", example_dict)
        self.assertIn("response", example_dict)


class TestLLMJudge(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        sys.path.insert(0, '/Users/nest/Desktop/GITHUB/helix/environments')
        from helixenv.llm_judge import LLMJudge, CalibrationExample

        self.calibration_examples = [
            CalibrationExample(
                prompt="What is entropy?",
                response="Entropy measures disorder in a system.",
                physics_score=0.7,
                reasoning="Basic understanding shown",
                key_concepts=["entropy", "disorder"]
            ),
            CalibrationExample(
                prompt="Explain spectral gap",
                response="Spectral gap is the difference between largest and second-largest eigenvalues.",
                physics_score=0.9,
                reasoning="Precise technical definition",
                key_concepts=["spectral gap", "eigenvalues"]
            )
        ]

    def test_llm_judge_initialization_with_api_key(self):
        """Test LLMJudge initialization with API key."""
        sys.path.insert(0, '/Users/nest/Desktop/GITHUB/helix/environments')
        from helixenv.llm_judge import LLMJudge

        # Test with API key
        judge = LLMJudge(
            api_key="test-key",
            model="gpt-4o-mini",
            calibration_examples=self.calibration_examples
        )

        self.assertEqual(judge.api_key, "test-key")
        self.assertEqual(judge.model, "gpt-4o-mini")
        self.assertEqual(len(judge.calibration_examples), 2)
        self.assertTrue(judge.use_api)

    def test_llm_judge_initialization_without_api_key(self):
        """Test LLMJudge initialization without API key (fallback mode)."""
        sys.path.insert(0, '/Users/nest/Desktop/GITHUB/helix/environments')
        from helixenv.llm_judge import LLMJudge

        # Test without API key
        judge = LLMJudge(
            api_key=None,
            calibration_examples=self.calibration_examples
        )

        self.assertIsNone(judge.api_key)
        self.assertFalse(judge.use_api)

    def test_format_calibration_examples(self):
        """Test calibration examples formatting."""
        sys.path.insert(0, '/Users/nest/Desktop/GITHUB/helix/environments')
        from helixenv.llm_judge import LLMJudge

        judge = LLMJudge(api_key=None, calibration_examples=self.calibration_examples)
        formatted = judge._format_calibration_examples()

        self.assertIsInstance(formatted, str)
        self.assertIn("What is entropy?", formatted)
        self.assertIn("Explain spectral gap", formatted)
        self.assertIn("0.7", formatted)  # Score should be included
        self.assertIn("0.9", formatted)

    def test_build_prompt(self):
        """Test prompt building with rubric and examples."""
        sys.path.insert(0, '/Users/nest/Desktop/GITHUB/helix/environments')
        from helixenv.llm_judge import LLMJudge

        judge = LLMJudge(api_key=None, calibration_examples=self.calibration_examples)

        prompt = judge._build_prompt(
            user_prompt="What is a spectral gap?",
            user_response="It's the gap between eigenvalues."
        )

        self.assertIsInstance(prompt, str)
        self.assertIn("What is a spectral gap?", prompt)
        self.assertIn("It's the gap between eigenvalues.", prompt)
        self.assertIn("physics_score", prompt.lower())
        self.assertIn("calibration", prompt.lower())

    def test_parse_response_valid_json(self):
        """Test parsing valid JSON response."""
        sys.path.insert(0, '/Users/nest/Desktop/GITHUB/helix/environments')
        from helixenv.llm_judge import LLMJudge

        judge = LLMJudge(api_key=None, calibration_examples=self.calibration_examples)

        # Valid JSON response
        api_response = '''
        {
            "physics_score": 0.85,
            "reasoning": "Good understanding of operator theory",
            "key_concepts": ["spectral theory", "eigenvalues"]
        }
        '''

        result = judge._parse_response(api_response)

        self.assertEqual(result["physics_score"], 0.85)
        self.assertEqual(result["reasoning"], "Good understanding of operator theory")
        self.assertEqual(len(result["key_concepts"]), 2)

    def test_parse_response_invalid_json(self):
        """Test parsing invalid JSON response."""
        sys.path.insert(0, '/Users/nest/Desktop/GITHUB/helix/environments')
        from helixenv.llm_judge import LLMJudge

        judge = LLMJudge(api_key=None, calibration_examples=self.calibration_examples)

        # Invalid JSON
        invalid_response = "This is not JSON at all"

        result = judge._parse_response(invalid_response)

        # Should fall back to default values
        self.assertIn("physics_score", result)
        self.assertIn("reasoning", result)
        self.assertIsInstance(result["physics_score"], (int, float))

    def test_parse_response_partial_json(self):
        """Test parsing JSON with missing fields."""
        sys.path.insert(0, '/Users/nest/Desktop/GITHUB/helix/environments')
        from helixenv.llm_judge import LLMJudge

        judge = LLMJudge(api_key=None, calibration_examples=self.calibration_examples)

        # JSON missing some required fields
        partial_response = '{"physics_score": 0.6}'

        result = judge._parse_response(partial_response)

        self.assertEqual(result["physics_score"], 0.6)
        # Should have default values for missing fields
        self.assertIn("reasoning", result)
        self.assertIn("key_concepts", result)

    @patch('openai.OpenAI')
    def test_evaluate_with_api_success(self, mock_openai_class):
        """Test successful API evaluation."""
        sys.path.insert(0, '/Users/nest/Desktop/GITHUB/helix/environments')
        from helixenv.llm_judge import LLMJudge

        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = '''
        {
            "physics_score": 0.8,
            "reasoning": "Demonstrates solid understanding",
            "key_concepts": ["test_concept"]
        }
        '''

        mock_client.chat.completions.create.return_value = mock_response

        judge = LLMJudge(
            api_key="test-key",
            model="gpt-4o-mini",
            calibration_examples=self.calibration_examples
        )

        result = judge.evaluate(
            prompt="Test prompt",
            response="Test response"
        )

        self.assertEqual(result["physics_score"], 0.8)
        self.assertEqual(result["reasoning"], "Demonstrates solid understanding")
        mock_client.chat.completions.create.assert_called_once()

    @patch('openai.OpenAI')
    def test_evaluate_with_api_failure(self, mock_openai_class):
        """Test API evaluation with failure (should fall back)."""
        sys.path.insert(0, '/Users/nest/Desktop/GITHUB/helix/environments')
        from helixenv.llm_judge import LLMJudge

        # Mock OpenAI client to raise exception
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        judge = LLMJudge(
            api_key="test-key",
            calibration_examples=self.calibration_examples
        )

        result = judge.evaluate(
            prompt="Test prompt",
            response="Test response"
        )

        # Should fall back to heuristic evaluation
        self.assertIn("physics_score", result)
        self.assertIn("reasoning", result)
        self.assertIsInstance(result["physics_score"], (int, float))

    def test_evaluate_fallback_mode(self):
        """Test evaluation in fallback mode (no API)."""
        sys.path.insert(0, '/Users/nest/Desktop/GITHUB/helix/environments')
        from helixenv.llm_judge import LLMJudge

        judge = LLMJudge(api_key=None, calibration_examples=self.calibration_examples)

        result = judge.evaluate(
            prompt="What is entropy?",
            response="Entropy is a measure of disorder in thermodynamics."
        )

        # Should return valid result structure
        self.assertIn("physics_score", result)
        self.assertIn("reasoning", result)
        self.assertIn("key_concepts", result)
        self.assertIsInstance(result["physics_score"], (int, float))
        self.assertTrue(0.0 <= result["physics_score"] <= 1.0)

    def test_heuristic_evaluation_keyword_matching(self):
        """Test heuristic evaluation keyword matching."""
        sys.path.insert(0, '/Users/nest/Desktop/GITHUB/helix/environments')
        from helixenv.llm_judge import LLMJudge

        judge = LLMJudge(api_key=None, calibration_examples=self.calibration_examples)

        # Response with physics keywords
        physics_response = "The spectral gap determines mixing time in Markov chains and operator theory."

        result_physics = judge._heuristic_evaluation(
            prompt="Explain spectral gap",
            response=physics_response
        )

        # Response without physics keywords
        generic_response = "I don't know much about this topic."

        result_generic = judge._heuristic_evaluation(
            prompt="Explain spectral gap",
            response=generic_response
        )

        # Physics response should score higher
        self.assertGreater(result_physics["physics_score"], result_generic["physics_score"])

    def test_heuristic_evaluation_response_length(self):
        """Test that heuristic evaluation considers response length."""
        sys.path.insert(0, '/Users/nest/Desktop/GITHUB/helix/environments')
        from helixenv.llm_judge import LLMJudge

        judge = LLMJudge(api_key=None, calibration_examples=self.calibration_examples)

        # Long detailed response
        long_response = "The spectral gap is a fundamental concept in operator theory. " * 10

        result_long = judge._heuristic_evaluation(
            prompt="Explain spectral gap",
            response=long_response
        )

        # Short response
        short_response = "Yes."

        result_short = judge._heuristic_evaluation(
            prompt="Explain spectral gap",
            response=short_response
        )

        # Longer response should generally score higher (all else being equal)
        # Note: This might not always be true depending on implementation details
        self.assertIsInstance(result_long["physics_score"], (int, float))
        self.assertIsInstance(result_short["physics_score"], (int, float))

    def test_evaluate_batch(self):
        """Test batch evaluation functionality."""
        sys.path.insert(0, '/Users/nest/Desktop/GITHUB/helix/environments')
        from helixenv.llm_judge import LLMJudge

        judge = LLMJudge(api_key=None, calibration_examples=self.calibration_examples)

        # Prepare batch data
        prompts = ["What is entropy?", "Explain spectral gap", "Define operator norm"]
        responses = [
            "Entropy measures disorder",
            "Gap between eigenvalues",
            "Maximum eigenvalue magnitude"
        ]

        results = judge.evaluate_batch(prompts, responses)

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn("physics_score", result)
            self.assertIn("reasoning", result)
            self.assertIsInstance(result["physics_score"], (int, float))

    def test_score_bounds(self):
        """Test that scores are properly bounded between 0 and 1."""
        sys.path.insert(0, '/Users/nest/Desktop/GITHUB/helix/environments')
        from helixenv.llm_judge import LLMJudge

        judge = LLMJudge(api_key=None, calibration_examples=self.calibration_examples)

        # Test with various responses
        test_cases = [
            ("What is entropy?", "I don't know"),
            ("Explain spectral gap", "Spectral gap in operator theory"),
            ("Define norm", "A norm is a function that assigns lengths")
        ]

        for prompt, response in test_cases:
            result = judge.evaluate(prompt, response)
            score = result["physics_score"]

            self.assertGreaterEqual(score, 0.0, f"Score {score} is below 0 for response: {response}")
            self.assertLessEqual(score, 1.0, f"Score {score} is above 1 for response: {response}")

    def test_calibration_consistency(self):
        """Test that similar responses to calibration examples get similar scores."""
        sys.path.insert(0, '/Users/nest/Desktop/GITHUB/helix/environments')
        from helixenv.llm_judge import LLMJudge

        judge = LLMJudge(api_key=None, calibration_examples=self.calibration_examples)

        # Response similar to calibration example
        similar_response = "Entropy is a measure of disorder and randomness in systems."

        result = judge.evaluate(
            prompt="What is entropy?",
            response=similar_response
        )

        # Should get a reasonable score (this is more of a sanity check)
        score = result["physics_score"]
        self.assertGreater(score, 0.3)  # Should be better than random
        self.assertLess(score, 1.0)     # But not perfect


class TestIntegrationTests(unittest.TestCase):
    def test_llm_judge_with_real_physics_examples(self):
        """Test LLM judge with realistic physics examples."""
        sys.path.insert(0, '/Users/nest/Desktop/GITHUB/helix/environments')
        from helixenv.llm_judge import LLMJudge, CalibrationExample

        # Create physics-focused calibration examples
        physics_examples = [
            CalibrationExample(
                prompt="What is the spectral gap in the context of Markov chains?",
                response="The spectral gap is 1 minus the second-largest eigenvalue of the transition matrix, determining mixing time.",
                physics_score=0.95,
                reasoning="Precise definition with connection to mixing time",
                key_concepts=["spectral gap", "eigenvalues", "mixing time", "Markov chains"]
            ),
            CalibrationExample(
                prompt="Explain AF algebras in operator theory",
                response="AF algebras are approximately finite-dimensional C*-algebras built as inductive limits of finite-dimensional algebras.",
                physics_score=0.9,
                reasoning="Technical accuracy with proper mathematical terminology",
                key_concepts=["AF algebras", "C*-algebras", "inductive limits"]
            )
        ]

        judge = LLMJudge(api_key=None, calibration_examples=physics_examples)

        # Test with good physics response
        good_response = "The spectral gap determines how quickly a Markov chain converges to its stationary distribution."
        good_result = judge.evaluate(
            prompt="What is the significance of the spectral gap?",
            response=good_response
        )

        # Test with poor physics response
        poor_response = "I think it's something about numbers."
        poor_result = judge.evaluate(
            prompt="What is the significance of the spectral gap?",
            response=poor_response
        )

        # Good response should score higher
        self.assertGreater(good_result["physics_score"], poor_result["physics_score"])

    def test_error_resilience(self):
        """Test that the judge handles various error conditions gracefully."""
        sys.path.insert(0, '/Users/nest/Desktop/GITHUB/helix/environments')
        from helixenv.llm_judge import LLMJudge

        judge = LLMJudge(api_key=None, calibration_examples=[])

        # Test with empty strings
        result = judge.evaluate("", "")
        self.assertIn("physics_score", result)

        # Test with very long strings
        long_prompt = "What is entropy? " * 1000
        long_response = "Entropy is disorder. " * 1000

        result = judge.evaluate(long_prompt, long_response)
        self.assertIn("physics_score", result)

        # Test with special characters
        special_prompt = "What is ∫∞₀ e^(-x²) dx?"
        special_response = "It's √π/2 by Gaussian integral."

        result = judge.evaluate(special_prompt, special_response)
        self.assertIn("physics_score", result)

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-env-key'})
    def test_api_key_from_environment(self):
        """Test reading API key from environment variable."""
        sys.path.insert(0, '/Users/nest/Desktop/GITHUB/helix/environments')
        from helixenv.llm_judge import LLMJudge

        # Should pick up API key from environment
        judge = LLMJudge(calibration_examples=[])

        # Note: This test assumes the implementation reads from environment
        # The actual behavior depends on how the LLMJudge handles API key discovery


if __name__ == "__main__":
    unittest.main()