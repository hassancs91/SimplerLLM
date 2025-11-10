from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class DialogueStyle(str, Enum):
    """Dialogue style for LLM generation."""
    CASUAL = "casual"
    FORMAL = "formal"
    EDUCATIONAL = "educational"
    DEBATE = "debate"
    INTERVIEW = "interview"
    STORYTELLING = "storytelling"


class SpeakerConfig(BaseModel):
    """Voice configuration for a speaker."""

    voice: str = "alloy"
    """Voice to use (provider-specific, e.g., 'alloy', 'nova' for OpenAI)"""

    speed: float = 1.0
    """Speech speed from 0.25 to 4.0"""

    model: Optional[str] = None
    """Optional: Override TTS model for this speaker"""

    class Config:
        json_schema_extra = {
            "example": {
                "voice": "nova",
                "speed": 1.1,
                "model": "tts-1-hd"
            }
        }


class DialogueLine(BaseModel):
    """A single line in a dialogue."""

    speaker: str
    """Speaker name/identifier"""

    text: str
    """The text to be spoken"""

    voice: Optional[str] = None
    """Optional: Override voice for this line"""

    speed: Optional[float] = None
    """Optional: Override speed for this line"""

    pause_after: Optional[float] = None
    """Optional: Pause duration after this line (in seconds)"""

    class Config:
        json_schema_extra = {
            "example": {
                "speaker": "Alice",
                "text": "Hello, how are you today?",
                "voice": "nova",
                "speed": 1.0,
                "pause_after": 0.5
            }
        }


class Dialogue(BaseModel):
    """Complete dialogue structure."""

    title: Optional[str] = None
    """Optional title for the dialogue"""

    description: Optional[str] = None
    """Optional description"""

    speakers: List[str]
    """List of speaker names"""

    speaker_configs: Optional[Dict[str, SpeakerConfig]] = None
    """Optional: Voice configuration for each speaker"""

    lines: List[DialogueLine]
    """List of dialogue lines"""

    metadata: Optional[Dict[str, Any]] = None
    """Optional: Additional metadata"""

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Teacher-Student Conversation",
                "description": "Educational dialogue about AI",
                "speakers": ["Teacher", "Student"],
                "speaker_configs": {
                    "Teacher": {"voice": "alloy", "speed": 1.0},
                    "Student": {"voice": "nova", "speed": 1.1}
                },
                "lines": [
                    {"speaker": "Teacher", "text": "Welcome to class!"},
                    {"speaker": "Student", "text": "Thank you!"}
                ]
            }
        }


class DialogueGenerationConfig(BaseModel):
    """Configuration for LLM-based dialogue generation."""

    num_speakers: int = Field(default=2, ge=2, le=10)
    """Number of speakers (2-10)"""

    num_exchanges: int = Field(default=5, ge=1, le=50)
    """Number of dialogue exchanges"""

    dialogue_style: DialogueStyle = DialogueStyle.CASUAL
    """Style of dialogue"""

    include_narrator: bool = False
    """Whether to include a narrator"""

    language: str = "English"
    """Language for the dialogue"""

    context: Optional[str] = None
    """Additional context for generation"""

    speaker_roles: Optional[List[str]] = None
    """Optional: Specific roles for speakers (e.g., ['Teacher', 'Student'])"""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    """LLM temperature for generation"""

    class Config:
        json_schema_extra = {
            "example": {
                "num_speakers": 2,
                "num_exchanges": 10,
                "dialogue_style": "educational",
                "include_narrator": True,
                "language": "English",
                "speaker_roles": ["Teacher", "Student"],
                "temperature": 0.7
            }
        }


class AudioDialogueResult(BaseModel):
    """Result of audio dialogue generation."""

    dialogue: Dialogue
    """The dialogue structure that was generated"""

    individual_files: List[str]
    """Paths to individual audio files"""

    combined_file: Optional[str] = None
    """Path to combined audio file (if generated)"""

    total_lines: int
    """Total number of lines generated"""

    total_duration_estimate: Optional[float] = None
    """Estimated total duration in seconds"""

    process_time: float
    """Time taken to generate audio in seconds"""

    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata"""

    class Config:
        json_schema_extra = {
            "example": {
                "dialogue": {"title": "Sample", "speakers": ["A", "B"], "lines": []},
                "individual_files": ["line_01.mp3", "line_02.mp3"],
                "combined_file": "dialogue_combined.mp3",
                "total_lines": 10,
                "total_duration_estimate": 30.5,
                "process_time": 5.2,
                "metadata": {"model": "tts-1-hd", "provider": "OPENAI"}
            }
        }
