from dotenv import load_dotenv
from google.cloud import vision
import os
import io

load_dotenv()  

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

image_path = "C:\Data\Real time interpreter GCP\photo.jpg"

client = vision.ImageAnnotatorClient(
    client_options={"api_key": api_key}
)

if __name__ == "__main__":
    with io.open(image_path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)
    response = client.face_detection(image=image)

    if response.error.message:
        raise RuntimeError(f"API error: {response.error.message}")

    faces = response.face_annotations
    print(f"Found {len(faces)} face(s)")
    for i, face in enumerate(faces):
        print(f"\nFace {i+1}:")
        print(f"  Joy:      {face.joy_likelihood.name}")
        print(f"  Sorrow:   {face.sorrow_likelihood.name}")
        print(f"  Anger:    {face.anger_likelihood.name}")
        print(f"  Surprise: {face.surprise_likelihood.name}")