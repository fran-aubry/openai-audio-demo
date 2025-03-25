import asyncio
import random
from dotenv import load_dotenv

load_dotenv()

from agents import Agent, function_tool
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    VoicePipeline,
    VoicePipelineConfig,
    TTSModelSettings,
)

from audio_player import AudioPlayer
from audio_recorder import record_audio

@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    print(f"[debug] get_weather called with city: {city}")
    choices = ["sunny", "cloudy", "rainy", "snowy"]
    return f"The weather in {city} is {random.choice(choices)}."

spanish_agent = Agent(
  name="Spanish",
  handoff_description="A spanish speaking agent.",
  instructions=prompt_with_handoff_instructions("""
    You're speaking to a human, so be polite and concise.
    Speak in Spanish.
  """),
  model="gpt-4o-mini",
)

agent = Agent(
  name="Assistant",
  instructions=prompt_with_handoff_instructions("""
    You're speaking to a human, so be polite and concise. 
    If the user speaks in Spanish, handoff to the spanish agent.
  """),
  model="gpt-4o-mini",
  handoffs=[spanish_agent],
  tools=[get_weather],
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
  audio_input = AudioInput(buffer=record_audio())
  result = await pipeline.run(audio_input)
  with AudioPlayer() as player:
    async for event in result.stream():
      if event.type == "voice_stream_event_audio":
        player.add_audio(event.data)

if __name__ == "__main__":
    asyncio.run(main())
