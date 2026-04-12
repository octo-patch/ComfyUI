from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MinimaxBaseResponse(BaseModel):
    status_code: int = Field(
        ...,
        description='Status code. 0 indicates success, other values indicate errors.',
    )
    status_msg: str = Field(
        ..., description='Specific error details or success message.'
    )


class File(BaseModel):
    bytes: Optional[int] = Field(None, description='File size in bytes')
    created_at: Optional[int] = Field(
        None, description='Unix timestamp when the file was created, in seconds'
    )
    download_url: Optional[str] = Field(
        None, description='The URL to download the video'
    )
    backup_download_url: Optional[str] = Field(
        None, description='The backup URL to download the video'
    )

    file_id: Optional[int] = Field(None, description='Unique identifier for the file')
    filename: Optional[str] = Field(None, description='The name of the file')
    purpose: Optional[str] = Field(None, description='The purpose of using the file')


class MinimaxFileRetrieveResponse(BaseModel):
    base_resp: MinimaxBaseResponse
    file: File


class MiniMaxModel(str, Enum):
    T2V_01_Director = 'T2V-01-Director'
    I2V_01_Director = 'I2V-01-Director'
    S2V_01 = 'S2V-01'
    I2V_01 = 'I2V-01'
    I2V_01_live = 'I2V-01-live'
    T2V_01 = 'T2V-01'
    Hailuo_02 = 'MiniMax-Hailuo-02'


class Status6(str, Enum):
    Queueing = 'Queueing'
    Preparing = 'Preparing'
    Processing = 'Processing'
    Success = 'Success'
    Fail = 'Fail'


class MinimaxTaskResultResponse(BaseModel):
    base_resp: MinimaxBaseResponse
    file_id: Optional[str] = Field(
        None,
        description='After the task status changes to Success, this field returns the file ID corresponding to the generated video.',
    )
    status: Status6 = Field(
        ...,
        description="Task status: 'Queueing' (in queue), 'Preparing' (task is preparing), 'Processing' (generating), 'Success' (task completed successfully), or 'Fail' (task failed).",
    )
    task_id: str = Field(..., description='The task ID being queried.')


class SubjectReferenceItem(BaseModel):
    image: Optional[str] = Field(
        None, description='URL or base64 encoding of the subject reference image.'
    )
    mask: Optional[str] = Field(
        None,
        description='URL or base64 encoding of the mask for the subject reference image.',
    )


class MinimaxVideoGenerationRequest(BaseModel):
    callback_url: Optional[str] = Field(
        None,
        description='Optional. URL to receive real-time status updates about the video generation task.',
    )
    first_frame_image: Optional[str] = Field(
        None,
        description='URL or base64 encoding of the first frame image. Required when model is I2V-01, I2V-01-Director, or I2V-01-live.',
    )
    model: MiniMaxModel = Field(
        ...,
        description='Required. ID of model. Options: T2V-01-Director, I2V-01-Director, S2V-01, I2V-01, I2V-01-live, T2V-01',
    )
    prompt: Optional[str] = Field(
        None,
        description='Description of the video. Should be less than 2000 characters. Supports camera movement instructions in [brackets].',
        max_length=2000,
    )
    prompt_optimizer: Optional[bool] = Field(
        True,
        description='If true (default), the model will automatically optimize the prompt. Set to false for more precise control.',
    )
    subject_reference: Optional[list[SubjectReferenceItem]] = Field(
        None,
        description='Only available when model is S2V-01. The model will generate a video based on the subject uploaded through this parameter.',
    )
    duration: Optional[int] = Field(
        None,
        description="The length of the output video in seconds."
    )
    resolution: Optional[str] = Field(
        None,
        description="The dimensions of the video display. 1080p corresponds to 1920 x 1080 pixels, 768p corresponds to 1366 x 768 pixels."
    )


class MinimaxVideoGenerationResponse(BaseModel):
    base_resp: MinimaxBaseResponse
    task_id: str = Field(
        ..., description='The task ID for the asynchronous video generation task.'
    )


# --- Chat API models ---

class MinimaxChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message author: 'user' or 'assistant'.")
    content: str = Field(..., description="The content of the message.")


class MinimaxChatRequest(BaseModel):
    model: str = Field(..., description="The model ID to use for chat completion.")
    messages: list[MinimaxChatMessage] = Field(..., description="A list of messages comprising the conversation.")
    temperature: Optional[float] = Field(1.0, description="Sampling temperature in (0.0, 1.0]. Default is 1.0.")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate.")
    stream: Optional[bool] = Field(False, description="Whether to stream partial results.")


class MinimaxChatChoice(BaseModel):
    index: int = Field(..., description="Index of this choice.")
    message: MinimaxChatMessage = Field(..., description="The generated message.")
    finish_reason: Optional[str] = Field(None, description="The reason generation stopped.")


class MinimaxChatUsage(BaseModel):
    prompt_tokens: Optional[int] = Field(None)
    completion_tokens: Optional[int] = Field(None)
    total_tokens: Optional[int] = Field(None)


class MinimaxChatResponse(BaseModel):
    id: Optional[str] = Field(None, description="Unique identifier for this completion.")
    choices: list[MinimaxChatChoice] = Field(..., description="List of generated choices.")
    usage: Optional[MinimaxChatUsage] = Field(None, description="Token usage information.")
    model: Optional[str] = Field(None, description="The model used for this completion.")


# --- TTS API models ---

class MinimaxTTSVoiceSetting(BaseModel):
    voice_id: str = Field(..., description="The voice ID to use for speech synthesis.")
    speed: Optional[float] = Field(1.0, description="Speech speed. 1.0 is normal.")
    vol: Optional[float] = Field(1.0, description="Volume. 1.0 is normal.")
    pitch: Optional[int] = Field(0, description="Pitch adjustment. 0 is normal.")


class MinimaxTTSAudioSetting(BaseModel):
    sample_rate: Optional[int] = Field(32000, description="Audio sample rate in Hz.")
    bitrate: Optional[int] = Field(128000, description="Audio bitrate in bps.")
    format: Optional[str] = Field("mp3", description="Audio format: 'mp3' or 'pcm'.")
    channel: Optional[int] = Field(1, description="Number of audio channels.")


class MinimaxTTSRequest(BaseModel):
    model: str = Field(..., description="The TTS model ID to use.")
    text: str = Field(..., description="The text to synthesize into speech.")
    stream: Optional[bool] = Field(True, description="Whether to stream the audio output.")
    voice_setting: MinimaxTTSVoiceSetting = Field(..., description="Voice settings.")
    audio_setting: Optional[MinimaxTTSAudioSetting] = Field(None, description="Audio output settings.")
