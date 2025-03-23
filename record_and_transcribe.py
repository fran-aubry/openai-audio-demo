import asyncio
from audio_to_text import transcribe_audio
from audio_recorder import record_audio

async def main():
  record_audio("prompt.wav")
  await transcribe_audio("prompt.wav")

if __name__ == "__main__":
  asyncio.run(main())