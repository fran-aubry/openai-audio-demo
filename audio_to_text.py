import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()

openai = AsyncOpenAI()

async def transcribe_audio(audio_filename = "audio.wav"):
  audio_file = await asyncio.to_thread(open, audio_filename, "rb")
  stream = await openai.audio.transcriptions.create(
    model="gpt-4o-mini-transcribe",
    file=audio_file,
    response_format="text",
    stream=True,
  )
  transcript = ""
  async for event in stream:
    if event.type == "transcript.text.delta":
      print(event.delta, end="", flush=True)
      transcript += event.delta
  print()
  audio_file.close()
  return transcript

if __name__ == "__main__":
  asyncio.run(transcribe_audio())