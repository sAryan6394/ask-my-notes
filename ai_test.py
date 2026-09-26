from google import genai
import os
from dotenv import load_dotenv


load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents="Say hello and confirm you're working."
)

print(response.text)