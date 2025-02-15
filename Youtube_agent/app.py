# Imports
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.youtube_tools import YouTubeTools

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

gemini_key = os.getenv("GOOGLE_API_KEY")

#========= YouTube Video Extractor Agent =======
Youtube_agent = Agent(
    name = "youtube_video_extractor",
    description = "Extracts audio and metadata from the provided YouTube URL.",
    task =  "Download video/audio, extract metadata like title, duration, and channel name.",
    instructions = [
        "Validate the provided YouTube URL."
        "Extract the audio stream and convert it to a suitable format."
        "Retrieve video metadata (title, duration, uploader)."
    ],
    guidelines = [
        "Ensure the URL is a valid YouTube link."
        "Handle errors gracefully if the video is unavailable."
        "Optimize for minimal latency in extraction."
    ],
    expected_output = "Extracted audio file with metadata in a structured format.",
    model=Gemini(id="gemini-1.5-flash", api_key=gemini_key),
    tools=[YouTubeTools()],
    
)
Youtube_agent.print_response("Summarize this video https://www.youtube.com/watch?v=rOpEN1JDaD0&t=100s", markdown=True)
#========= YSpeech-to-Text Transcriber  Agent =======







#========= Transcript Formatter & Enhancer  Agent =======

