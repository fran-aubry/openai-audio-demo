import asyncio
import random
from dotenv import load_dotenv

load_dotenv()

from agents import Agent
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    VoicePipeline,
    VoicePipelineConfig,
    TTSModelSettings,
)

from audio_player import AudioPlayer
from audio_recorder import record_audio

from agents import function_tool
@function_tool
def stop_conversation():
  """Stop the conversation."""
  exit()

agent = Agent(
  name="Voice Assistant",
  instructions="You're a helpful assistant speaking to a human.",
  model="gpt-4o-mini",
  tools=[stop_conversation],
)

async def main():
  pipeline = VoicePipeline(
    workflow=SingleAgentVoiceWorkflow(agent),
    stt_model="gpt-4o-mini-transcribe",
    tts_model="gpt-4o-mini-tts",
    config=VoicePipelineConfig(
      tts_settings=TTSModelSettings(voice="coral")
    )
  )
  while True:
    audio_input = AudioInput(buffer=record_audio())
    result = await pipeline.run(audio_input)
    with AudioPlayer() as player:
      async for event in result.stream():
        if event.type == "voice_stream_event_audio":
          player.add_audio(event.data)

if __name__ == "__main__":
    asyncio.run(main())
