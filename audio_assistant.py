import asyncio
from openai import AsyncOpenAI
from text_to_audio import text_to_audio
from audio_to_text import transcribe_audio
from audio_recorder import record_audio
from dotenv import load_dotenv
load_dotenv()

openai = AsyncOpenAI()

async def get_answer(prompt):
  stream = await openai.chat.completions.create(
    model="gpt-4o",
    messages=[
      {"role": "user", "content": prompt}
    ],
    stream=True,
  )
  answer = ""
  async for chunk in stream:
    content = chunk.choices[0].delta.content
    if content is not None:
      answer += content
      print(content, end="", flush=True)
  print("\n\n")
  return answer


async def main(tone_and_style_instructions):
  await text_to_audio("Hello, how can I help you today?", tone_and_style_instructions)
  while True:
    record_audio("prompt.wav")
    prompt = await transcribe_audio("prompt.wav")
    print()
    answer = await get_answer(prompt)
    await text_to_audio(answer, tone_and_style_instructions)
    

if __name__ == "__main__":
  tone_and_style_instructions = "Enthusiastic voice"
  asyncio.run(main(tone_and_style_instructions))