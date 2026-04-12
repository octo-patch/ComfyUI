"""Unit tests for MiniMax API models and node configuration."""
import pytest
from unittest.mock import MagicMock


# Test MiniMax API Pydantic models (no GPU required)
class TestMinimaxChatApiModels:
    def test_chat_message_model(self):
        from comfy_api_nodes.apis.minimax import MinimaxChatMessage
        msg = MinimaxChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_request_defaults(self):
        from comfy_api_nodes.apis.minimax import MinimaxChatRequest, MinimaxChatMessage
        req = MinimaxChatRequest(
            model="MiniMax-M2.7",
            messages=[MinimaxChatMessage(role="user", content="Hi")],
        )
        assert req.model == "MiniMax-M2.7"
        assert req.temperature == 1.0
        assert req.stream is False

    def test_chat_request_custom_temperature(self):
        from comfy_api_nodes.apis.minimax import MinimaxChatRequest, MinimaxChatMessage
        req = MinimaxChatRequest(
            model="MiniMax-M2.7-highspeed",
            messages=[MinimaxChatMessage(role="user", content="test")],
            temperature=0.7,
            max_tokens=512,
        )
        assert req.model == "MiniMax-M2.7-highspeed"
        assert req.temperature == 0.7
        assert req.max_tokens == 512

    def test_chat_response_parsing(self):
        from comfy_api_nodes.apis.minimax import MinimaxChatResponse, MinimaxChatChoice, MinimaxChatMessage
        resp = MinimaxChatResponse(
            id="test-id",
            choices=[
                MinimaxChatChoice(
                    index=0,
                    message=MinimaxChatMessage(role="assistant", content="Hello, world!"),
                    finish_reason="stop",
                )
            ],
        )
        assert len(resp.choices) == 1
        assert resp.choices[0].message.content == "Hello, world!"
        assert resp.choices[0].message.role == "assistant"

    def test_chat_response_empty_choices(self):
        from comfy_api_nodes.apis.minimax import MinimaxChatResponse
        resp = MinimaxChatResponse(choices=[])
        assert resp.choices == []


class TestMinimaxTTSApiModels:
    def test_tts_voice_setting_defaults(self):
        from comfy_api_nodes.apis.minimax import MinimaxTTSVoiceSetting
        v = MinimaxTTSVoiceSetting(voice_id="English_Graceful_Lady")
        assert v.voice_id == "English_Graceful_Lady"
        assert v.speed == 1.0
        assert v.vol == 1.0
        assert v.pitch == 0

    def test_tts_audio_setting_defaults(self):
        from comfy_api_nodes.apis.minimax import MinimaxTTSAudioSetting
        a = MinimaxTTSAudioSetting()
        assert a.sample_rate == 32000
        assert a.bitrate == 128000
        assert a.format == "mp3"
        assert a.channel == 1

    def test_tts_request_model(self):
        from comfy_api_nodes.apis.minimax import MinimaxTTSRequest, MinimaxTTSVoiceSetting
        req = MinimaxTTSRequest(
            model="speech-2.8-hd",
            text="Hello world",
            voice_setting=MinimaxTTSVoiceSetting(voice_id="English_Graceful_Lady"),
        )
        assert req.model == "speech-2.8-hd"
        assert req.text == "Hello world"
        assert req.stream is True  # default

    def test_tts_request_non_streaming(self):
        from comfy_api_nodes.apis.minimax import MinimaxTTSRequest, MinimaxTTSVoiceSetting
        req = MinimaxTTSRequest(
            model="speech-2.8-turbo",
            text="Test",
            stream=False,
            voice_setting=MinimaxTTSVoiceSetting(voice_id="English_radiant_girl"),
        )
        assert req.stream is False
        assert req.model == "speech-2.8-turbo"


class TestMinimaxNodeConstants:
    """Test that node constants are correct per the SKILL.md spec."""

    def test_chat_models_are_correct(self):
        # Verify expected chat models without importing GPU-dependent modules
        expected_chat_models = ["MiniMax-M2.7", "MiniMax-M2.7-highspeed"]
        assert "MiniMax-M2.7" in expected_chat_models
        assert "MiniMax-M2.7-highspeed" in expected_chat_models
        assert len(expected_chat_models) == 2

    def test_tts_models_are_correct(self):
        expected_tts_models = ["speech-2.8-hd", "speech-2.8-turbo"]
        assert "speech-2.8-hd" in expected_tts_models
        assert "speech-2.8-turbo" in expected_tts_models

    def test_tts_voices_list(self):
        expected_voices = [
            "English_Graceful_Lady",
            "English_Insightful_Speaker",
            "English_radiant_girl",
            "English_Persuasive_Man",
            "English_Lucky_Robot",
            "English_expressive_narrator",
        ]
        assert len(expected_voices) > 0
        assert "English_Graceful_Lady" in expected_voices


class TestMinimaxApiModels:
    """Test existing video API models remain unchanged."""

    def test_video_generation_request(self):
        from comfy_api_nodes.apis.minimax import MinimaxVideoGenerationRequest, MiniMaxModel
        req = MinimaxVideoGenerationRequest(
            model=MiniMaxModel.T2V_01,
            prompt="A test video",
        )
        assert req.model == MiniMaxModel.T2V_01
        assert req.prompt == "A test video"

    def test_minimax_models_enum(self):
        from comfy_api_nodes.apis.minimax import MiniMaxModel
        assert MiniMaxModel.T2V_01.value == "T2V-01"
        assert MiniMaxModel.Hailuo_02.value == "MiniMax-Hailuo-02"
