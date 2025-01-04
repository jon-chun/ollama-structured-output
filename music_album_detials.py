from typing import Literal, Optional, List
from ollama import chat
from pydantic import BaseModel, Field
from utils_ver1 import save_album_details_to_markdown
from models_ver1 import Album, Song
import os



def process_album_image(path: str) -> Album:
    """
    Process an album image and extract album details.
    
    Args:
        path: Path to the album image file
        
    Returns:
        Album object containing the extracted details
    """
    print(f"Processing image: {path}")
    
    SYSTEM_PROMPT = """You are a precise image description and transcription system. Your task is to carefully read the text in the image, and  extract album details and song details that you can detect. You will also provide a transcription of the raw text in the image."""

    response = chat(
        model='llama3.2-vision', # the small model
        format=Album.model_json_schema(),  # Pass in the schema for the response
        messages=[
            {
                'role': 'system',
                'content': SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': 'Analyze this image and describe what you see, including any objects, the scene, colors and any text you can detect.',
                'images': [path],
            },
        ],
        options={'temperature': 0},  # Set temperature to 0 for more deterministic output
    )

    album = Album.model_validate_json(response.message.content)

    print("\nAlbum Details:")
    print("-" * 50)
    print(f"\nAlbum Title: {album.album_title}")
    print(f"Artist: {album.artist}")
    # print(f"Release Year: {album.release_year}")
    print(f"Other Info: {album.other_info}")

    print("\nGenres:")
    for genre in album.genres:
        print(f"  - {genre}")

    print("\nTrack List:")
    for song in album.track_list:
        if song.duration_seconds > 0:
            print(f"  - {song.title} ({song.duration_string})")
        else:
            print(f"  - {song.title}")

    print("\nRaw Transcription:")
    print(f"\n{album.raw_text}")

    # Save the results
    save_album_details_to_markdown(album, f"album_details_{album.album_title}.md")

    
    return album



if __name__ == "__main__":
    # path = '/Users/samwitteveen/Desktop/test_images/miles_03.png'
    # Process multiple album images
    image_paths = [
        "/Users/samwitteveen/Desktop/test_images/miles_01.png",
        "/Users/samwitteveen/Desktop/test_images/miles_02.png", 
        "/Users/samwitteveen/Desktop/test_images/miles_03.png",
        "/Users/samwitteveen/Desktop/test_images/miles_04.png",
        "/Users/samwitteveen/Desktop/test_images/miles_05.png"
    ]
    
    for image_path in image_paths:
        if os.path.exists(image_path):
            print(f"\nProcessing {image_path}...")
            try:
                album = process_album_image(image_path)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        else:
            print(f"Image file {image_path} not found")





