import os
from models_ver1 import Album


def save_album_details_to_markdown(album: Album, output_path: str = "album_details.md"):
    """
    Save album details to a markdown file in the album_notes directory.
    
    Args:
        album: Album object containing the details
        output_path: Path where markdown file should be saved
    """
    # Create album_notes directory if it doesn't exist
    notes_dir = "./album_notes"
    os.makedirs(notes_dir, exist_ok=True)
    
    # Combine directory with output filename
    full_path = os.path.join(notes_dir, output_path)
    
    with open(full_path, "w") as f:
        f.write("# Album Details\n\n")
        f.write("-" * 50 + "\n\n")
        
        f.write(f"## Basic Information\n")
        f.write(f"- **Album Title:** {album.album_title}\n")
        f.write(f"- **Artist:** {album.artist}\n")
        f.write(f"- **Release Year:** {album.release_year}\n")
        f.write(f"- **Other Info:** {album.other_info}\n\n")
        
        f.write("## Genres\n")
        for genre in album.genres:
            f.write(f"- {genre}\n")
        f.write("\n")
        
        f.write("## Track List\n")
        for song in album.track_list:
            if song.duration_seconds > 0:
                f.write(f"- {song.title} ({song.duration_string})\n")
            else:
                f.write(f"- {song.title}\n")
        f.write("\n")
        
        f.write("## Raw Transcription\n")
        f.write(f"{album.raw_text}\n")


        # class Song(BaseModel):
#     """A song track from an album"""
#     title: str  # The title of the song
#     duration_seconds: int  # Duration in seconds
#     duration_string: str  # Duration in MM:SS format

# class Album(BaseModel):
#     """Details about a music album"""
#     album_title: str  # The title of the album
#     artist: str  # The artist/band name
#     release_year: int  # Year the album was released
#     genres: List[str]  # Musical genres of the album
#     track_list: List[Song]  # List of songs on the album