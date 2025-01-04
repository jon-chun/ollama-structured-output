from typing import List
from pydantic import BaseModel, Field

class Song(BaseModel):
    """
    Represents a song track from an album.
    """
    title: str = Field(..., description="The title of the song.")
    duration_seconds: int = Field(..., description="Duration of the song in seconds.")
    duration_string: str = Field(..., description="Duration of the song in MM:SS format.")

class Album(BaseModel):
    """
    Represents details about a music album.
    """
    album_title: str = Field(..., description="The title of the album.")
    artist: str = Field(..., description="The name of the artist or band.")
    release_year: int = Field(..., description="The year the album was released.")
    genres: List[str] = Field(..., description="A list of musical genres associated with the album.")
    track_list: List[Song] = Field(..., description="A list of songs included in the album.")
    other_info: str = Field(..., description="Any other information about the album.")
    transcription: str = Field(..., description="A transcription of the raw text in the image.")
    raw_text: str = Field(..., description="The raw text in the image.")
