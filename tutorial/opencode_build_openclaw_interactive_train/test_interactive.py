# -*- coding: utf-8 -*-
"""Tests for opencode_build_openclaw_interactive_train features."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from on_compute_relative_reward import (
    detect_user_opinion,
    parse_agentjet_command,
    update_judge_prompt_with_feedback,
    compute_user_feedback_scores,
    compute_diversity_scores,
    compute_quality_scores,
    on_compute_relative_reward,
    UserFeedback,
    AgentJetCommand,
    _dynamic_listwise_judge_prompt,
    _user_preference_history,
)


class TestDetectUserOpinion:
    """Tests for user opinion detection."""

    @pytest.mark.anyio
    async def test_detect_opinion_humor_request(self):
        """Test detecting user asking for humor."""
        mock_response = MagicMock()
        mock_response.content = '{"has_opinion": true, "feedback_type": "style", "feedback_content": "be more humorous", "explanation": "User is requesting humor"}'

        with patch('on_compute_relative_reward.qwen_max_model') as mock_model:
            mock_model.achat = AsyncMock(return_value=mock_response)
            result = await detect_user_opinion("请幽默一点")

            assert result.has_opinion is True
            assert result.feedback_type == "style"
            assert "humorous" in result.feedback_content.lower()

    @pytest.mark.anyio
    async def test_detect_opinion_criticism(self):
        """Test detecting user criticizing the model."""
        mock_response = MagicMock()
        mock_response.content = '{"has_opinion": true, "feedback_type": "tone", "feedback_content": "be smarter and more intelligent", "explanation": "User is criticizing intelligence"}'

        with patch('on_compute_relative_reward.qwen_max_model') as mock_model:
            mock_model.achat = AsyncMock(return_value=mock_response)
            result = await detect_user_opinion("你太傻了")

            assert result.has_opinion is True
            assert result.feedback_type == "tone"

    @pytest.mark.anyio
    async def test_no_opinion_simple_question(self):
        """Test that simple questions don't trigger opinion detection."""
        mock_response = MagicMock()
        mock_response.content = '{"has_opinion": false, "feedback_type": "none", "feedback_content": "", "explanation": "User is just asking a question"}'

        with patch('on_compute_relative_reward.qwen_max_model') as mock_model:
            mock_model.achat = AsyncMock(return_value=mock_response)
            result = await detect_user_opinion("今天天气怎么样？")

            assert result.has_opinion is False


class TestParseAgentjetCommand:
    """Tests for /agentjet command parsing."""

    def test_parse_model_switch_command(self):
        """Test parsing model switch command."""
        result = parse_agentjet_command("/agentjet: 切换 '/path/to/model' 模型")

        assert result is not None
        assert result.command_type == "switch_model"
        assert result.parameters["model"] == "/path/to/model"

    def test_parse_config_update_command(self):
        """Test parsing config update command."""
        result = parse_agentjet_command("/agentjet: 更新 batch_size 为 64")

        assert result is not None
        assert result.command_type == "update_config"
        assert result.parameters["batch_size"] == 64

    def test_parse_n_gpu_update_command(self):
        """Test parsing n_gpu update command."""
        result = parse_agentjet_command("/agentjet: 更新 n_gpu 为 4")

        assert result is not None
        assert result.command_type == "update_config"
        assert result.parameters["n_gpu"] == 4

    def test_no_command_regular_input(self):
        """Test that regular input returns None."""
        result = parse_agentjet_command("今天天气怎么样？")
        assert result is None

    def test_parse_command_case_insensitive(self):
        """Test that command parsing is case insensitive."""
        result = parse_agentjet_command("/AGENTJET: 切换 '/path/to/model' 模型")
        assert result is not None
        assert result.command_type == "switch_model"


class TestComputeDiversityScores:
    """Tests for diversity score computation."""

    def test_diversity_unique_responses(self):
        """Test that unique responses get high diversity scores."""
        contents = ["This is response one", "This is response two", "This is response three"]
        scores = compute_diversity_scores(contents, [])

        assert len(scores) == 3
        assert all(s > 0.5 for s in scores)  # All should be relatively unique

    def test_diversity_duplicate_responses(self):
        """Test that duplicate responses get low diversity scores."""
        contents = ["Same text", "Same text", "Same text"]
        scores = compute_diversity_scores(contents, [])

        assert len(scores) == 3
        assert all(s < 0.1 for s in scores)  # Should have very low diversity

    def test_diversity_with_history(self):
        """Test diversity computation with historical context."""
        contents = ["New response"]
        history = ["Old response one", "Old response two"]
        scores = compute_diversity_scores(contents, history)

        assert len(scores) == 1
        assert 0 <= scores[0] <= 1

    def test_diversity_single_response(self):
        """Test with single response."""
        contents = ["Only one response"]
        scores = compute_diversity_scores(contents, [])

        assert len(scores) == 1
        assert scores[0] == 1.0  # No overlap possible with single item


class TestComputeQualityScores:
    """Tests for quality score computation."""

    @pytest.mark.anyio
    async def test_quality_clean_text(self):
        """Test quality score for clean text."""
        contents = ["This is a normal response without any issues."]
        scores = await compute_quality_scores(contents)

        assert len(scores) == 1
        assert scores[0] > 0.5  # Should be high quality

    @pytest.mark.anyio
    async def test_quality_with_special_tokens(self):
        """Test that special tokens trigger low quality."""
        contents = ["Hello <|im_start|> how are you"]
        scores = await compute_quality_scores(contents)

        assert len(scores) == 1
        assert scores[0] == 0.0  # Should be penalized

    @pytest.mark.anyio
    async def test_quality_multiple_responses(self):
        """Test quality with multiple responses."""
        contents = [
            "First clean response",
            "Second clean response",
            "<|im_start|> bad response"
        ]
        scores = await compute_quality_scores(contents)

        assert len(scores) == 3
        assert scores[2] == 0.0  # Third has special token


class TestComputeUserFeedbackScores:
    """Tests for listwise user feedback scoring."""

    @pytest.mark.anyio
    async def test_single_answer_gets_neutral(self):
        """Test that single answer gets neutral score."""
        all_answers = [{"content": "Only one answer"}]
        scores = await compute_user_feedback_scores("test question", all_answers)

        assert scores == [0.5]

    @pytest.mark.anyio
    async def test_empty_answers_returns_empty(self):
        """Test that empty answers list returns empty list."""
        scores = await compute_user_feedback_scores("test question", [])
        assert scores == []

    @pytest.mark.anyio
    async def test_listwise_grader_called(self):
        """Test that listwise grader is properly called."""
        mock_result = MagicMock()
        mock_result.rank = [2, 1, 3]  # answer_2 is best, answer_1 is second, answer_3 is worst

        all_answers = [
            {"content": "Answer one"},
            {"content": "Answer two"},
            {"content": "Answer three"}
        ]

        with patch('on_compute_relative_reward.get_dynamic_listwise_grader') as mock_get_grader:
            mock_grader = MagicMock()
            mock_grader.aevaluate = AsyncMock(return_value=mock_result)
            mock_get_grader.return_value = mock_grader

            scores = await compute_user_feedback_scores("test question", all_answers)

            assert len(scores) == 3
            # rank = [2, 1, 3] means:
            # position 0 -> rank 2 (best), position 1 -> rank 1 (second), position 2 -> rank 3 (worst)
            # After conversion with 1-indexed idx: scores[idx-1] = 1.0 - (position/(n-1))
            # position 0, idx 2 -> scores[1] = 1.0 - 0/2 = 1.0 (best)
            # position 1, idx 1 -> scores[0] = 1.0 - 1/2 = 0.5 (second)
            # position 2, idx 3 -> scores[2] = 1.0 - 2/2 = 0.0 (worst)
            # So: scores = [0.5, 1.0, 0.0]
            assert scores[1] > scores[0] > scores[2]


class TestUserFeedbackClass:
    """Tests for UserFeedback dataclass."""

    def test_user_feedback_defaults(self):
        """Test UserFeedback default values."""
        feedback = UserFeedback()

        assert feedback.has_opinion is False
        assert feedback.feedback_type == ""
        assert feedback.feedback_content == ""
        assert feedback.raw_input == ""

    def test_user_feedback_with_values(self):
        """Test UserFeedback with specific values."""
        feedback = UserFeedback(
            has_opinion=True,
            feedback_type="style",
            feedback_content="be more formal",
            raw_input="请正式一点"
        )

        assert feedback.has_opinion is True
        assert feedback.feedback_type == "style"
        assert feedback.feedback_content == "be more formal"
        assert feedback.raw_input == "请正式一点"


class TestAgentJetCommandClass:
    """Tests for AgentJetCommand dataclass."""

    def test_agentjet_command_with_values(self):
        """Test AgentJetCommand with specific values."""
        cmd = AgentJetCommand(
            command_type="switch_model",
            parameters={"model": "/path/to/model"},
            raw_command="切换 '/path/to/model' 模型"
        )

        assert cmd.command_type == "switch_model"
        assert cmd.parameters["model"] == "/path/to/model"
        assert "切换" in cmd.raw_command


class TestDynamicJudgePrompt:
    """Tests for dynamic judge prompt management."""

    @pytest.mark.anyio
    async def test_update_judge_prompt(self):
        """Test that judge prompt gets updated with feedback."""
        feedback = UserFeedback(
            has_opinion=True,
            feedback_type="style",
            feedback_content="prefer concise answers",
            raw_input="请简洁一点"
        )

        mock_response = MagicMock()
        mock_response.content = """You are ranking multiple responses based on user preferences.

Current evaluation criteria:
- Respond to the question accurately and completely
- Prefer concise answers
- Be helpful and clear

Question: {question}

Responses to rank:
{answers_block}

Rank these responses from best to worst based on user preferences.
Return a json object with exactly two fields:
- "rank": list of integers (1-indexed) ordered from best to worst, e.g. [2, 1, 3]
- "reason": brief explanation of the ranking"""

        original_history_len = len(_user_preference_history)

        with patch('on_compute_relative_reward.qwen_max_model') as mock_model:
            mock_model.achat = AsyncMock(return_value=mock_response)
            result = await update_judge_prompt_with_feedback(feedback)

            assert "concise" in result.lower()
            assert len(_user_preference_history) == original_history_len + 1


class TestOnComputeRelativeReward:
    """Tests for the main on_compute_relative_reward function."""

    @pytest.mark.anyio
    async def test_reward_computation_basic(self):
        """Test basic reward computation with listwise scoring."""
        valid_results = [MagicMock(), MagicMock()]
        all_answers = [
            {"content": "This is response one with some content"},
            {"content": "This is response two with different content"}
        ]
        question = "What is the capital of France?"

        with patch('on_compute_relative_reward.detect_user_opinion') as mock_detect, \
             patch('on_compute_relative_reward.parse_agentjet_command_with_llm') as mock_parse, \
             patch('on_compute_relative_reward.compute_quality_scores') as mock_quality, \
             patch('on_compute_relative_reward.compute_user_feedback_scores') as mock_feedback, \
             patch('on_compute_relative_reward.compute_relevance_scores') as mock_relevance, \
             patch('on_compute_relative_reward.compute_diversity_scores') as mock_diversity:

            mock_detect.return_value = UserFeedback(has_opinion=False)
            mock_parse.return_value = None
            mock_quality.return_value = [1.0, 1.0]
            mock_feedback.return_value = [0.8, 0.6]
            mock_relevance.return_value = [0.9, 0.7]
            mock_diversity.return_value = [0.5, 0.5]

            rewards = await on_compute_relative_reward(valid_results, all_answers, question)

            assert len(rewards) == 2
            assert all(isinstance(r, float) for r in rewards)
            assert all(0 <= r <= 1 for r in rewards)
            # First answer should have higher reward (better feedback and relevance scores)
            assert rewards[0] >= rewards[1]

    @pytest.mark.anyio
    async def test_reward_with_user_opinion(self):
        """Test that user opinion triggers judge prompt update."""
        valid_results = [MagicMock()]
        all_answers = [{"content": "Response with humor"}]
        question = "Tell me a joke"

        feedback = UserFeedback(
            has_opinion=True,
            feedback_type="style",
            feedback_content="prefer humorous responses",
            raw_input="请幽默一点"
        )

        with patch('on_compute_relative_reward.detect_user_opinion') as mock_detect, \
             patch('on_compute_relative_reward.update_judge_prompt_with_feedback') as mock_update, \
             patch('on_compute_relative_reward.parse_agentjet_command_with_llm') as mock_parse, \
             patch('on_compute_relative_reward.compute_quality_scores') as mock_quality, \
             patch('on_compute_relative_reward.compute_user_feedback_scores') as mock_feedback, \
             patch('on_compute_relative_reward.compute_relevance_scores') as mock_relevance, \
             patch('on_compute_relative_reward.compute_diversity_scores') as mock_diversity:

            mock_detect.return_value = feedback
            mock_update.return_value = _dynamic_listwise_judge_prompt
            mock_parse.return_value = None
            mock_quality.return_value = [1.0]
            mock_feedback.return_value = [0.7]
            mock_relevance.return_value = [0.8]
            mock_diversity.return_value = [0.6]

            rewards = await on_compute_relative_reward(valid_results, all_answers, question)

            mock_update.assert_called_once_with(feedback)
            assert len(rewards) == 1

    @pytest.mark.anyio
    async def test_reward_with_agentjet_command(self):
        """Test that /agentjet command triggers config update."""
        valid_results = [MagicMock()]
        all_answers = [{"content": "Response"}]
        question = "/agentjet: 切换 '/new/model' 模型"

        callback_called = []
        def mock_callback(updates):
            callback_called.append(updates)

        with patch('on_compute_relative_reward._agentjet_command_callback', mock_callback), \
             patch('on_compute_relative_reward.detect_user_opinion') as mock_detect, \
             patch('on_compute_relative_reward.parse_agentjet_command_with_llm') as mock_parse, \
             patch('on_compute_relative_reward.compute_quality_scores') as mock_quality, \
             patch('on_compute_relative_reward.compute_user_feedback_scores') as mock_feedback, \
             patch('on_compute_relative_reward.compute_relevance_scores') as mock_relevance, \
             patch('on_compute_relative_reward.compute_diversity_scores') as mock_diversity:

            mock_detect.return_value = UserFeedback(has_opinion=False)
            mock_parse.return_value = {"model": "/new/model"}
            mock_quality.return_value = [1.0]
            mock_feedback.return_value = [0.5]
            mock_relevance.return_value = [0.5]
            mock_diversity.return_value = [0.5]

            rewards = await on_compute_relative_reward(valid_results, all_answers, question)

            assert len(callback_called) == 1
            assert callback_called[0]["model"] == "/new/model"

    @pytest.mark.anyio
    async def test_reward_quality_gate_multiplies(self):
        """Test that quality score acts as a multiplier."""
        valid_results = [MagicMock(), MagicMock()]
        all_answers = [
            {"content": "Good clean response"},
            {"content": "Bad <|im_start|> response"}
        ]
        question = "Hello"

        with patch('on_compute_relative_reward.detect_user_opinion') as mock_detect, \
             patch('on_compute_relative_reward.parse_agentjet_command_with_llm') as mock_parse, \
             patch('on_compute_relative_reward.compute_quality_scores') as mock_quality, \
             patch('on_compute_relative_reward.compute_user_feedback_scores') as mock_feedback, \
             patch('on_compute_relative_reward.compute_relevance_scores') as mock_relevance, \
             patch('on_compute_relative_reward.compute_diversity_scores') as mock_diversity:

            mock_detect.return_value = UserFeedback(has_opinion=False)
            mock_parse.return_value = None
            # Quality: 1.0 for good, 0.0 for bad
            mock_quality.return_value = [1.0, 0.0]
            mock_feedback.return_value = [0.5, 0.5]
            mock_relevance.return_value = [0.5, 0.5]
            mock_diversity.return_value = [0.5, 0.5]

            rewards = await on_compute_relative_reward(valid_results, all_answers, question)

            # Second answer should have 0 reward due to quality gate
            assert rewards[1] == 0.0


class TestListwiseScoringConversion:
    """Tests for listwise rank to score conversion."""

    @pytest.mark.anyio
    async def test_rank_to_score_conversion(self):
        """Test that listwise ranks are converted to proper scores."""
        mock_result = MagicMock()
        # Rank: answer_2 is best (pos 0), answer_1 is second (pos 1), answer_3 is worst (pos 2)
        mock_result.rank = [2, 1, 3]

        all_answers = [
            {"content": "Answer one"},
            {"content": "Answer two"},
            {"content": "Answer three"}
        ]

        with patch('on_compute_relative_reward.get_dynamic_listwise_grader') as mock_get_grader:
            mock_grader = MagicMock()
            mock_grader.aevaluate = AsyncMock(return_value=mock_result)
            mock_get_grader.return_value = mock_grader

            scores = await compute_user_feedback_scores("test question", all_answers)

            assert len(scores) == 3
            # Best (rank 2, at index 1) should get highest score
            # Second (rank 1, at index 0) should get middle score
            # Worst (rank 3, at index 2) should get lowest score
            # With n=3: best = 1.0 - 0/2 = 1.0, second = 1.0 - 1/2 = 0.5, worst = 1.0 - 2/2 = 0.0
            assert scores[1] == 1.0  # Best
            assert scores[0] == 0.5  # Second
            assert scores[2] == 0.0  # Worst


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
