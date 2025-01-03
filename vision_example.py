from typing import Literal, Optional
from ollama import chat
from pydantic import BaseModel

class BookObject(BaseModel):
  confidence: float
  title: str 
  author: str

class ImageDescription(BaseModel):
  summary: str
  books: list[BookObject]
  scene: str
  colors: list[str]
  number_of_books: Literal['None', 'Single', 'Few', 'Many']
  setting: Literal['Bookshelf', 'Table', 'Library', 'Store', 'Digital', 'Other']
  text_content: Optional[str] 

path = '/Users/samwitteveen/Desktop/test_images/books.png'

SYSTEM_PROMPT = """You are a precise image description system. Your task is to carefully describe the image, including any objects, the scene, colors and any text you can detect."""

response = chat(
  model='llama3.2-vision',
  format=ImageDescription.model_json_schema(),  # Pass in the schema for the response
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

image_description = ImageDescription.model_validate_json(response.message.content)

print("\nImage Analysis Results:")
print("-" * 50)
print(f"\nSummary:\n{image_description.summary}")
print(f"\nNumber of Books: {image_description.number_of_books}")
print("\nBooks Detected:")
for obj in image_description.books:
    print(f"  - {obj.title} (Confidence: {obj.confidence:.2f})")
    print(f"    Author: {obj.author}")

print(f"\nScene: {image_description.scene}")
print(f"Setting: {image_description.setting}")
        
print("\nColors Present:")
for color in image_description.colors:
    print(f"  - {color}")

if image_description.text_content:
    print(f"\nDetected Text:\n{image_description.text_content}")