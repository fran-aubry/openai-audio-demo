# OpenAI Audio API demo

This code is part of a Datacamp tutorial on how to build a voice assistant using
the OpenAI Audio API.

## Setup

1. Rename `.env_template` to `.env` and paste your OpenAI API key.

2. Run the commands:

  ```
  conda create -n audio-demo -y python=3.9
  conda activate audio-demo
  pip install -r requirements.txt
  ```

## Files

- `audio_recorder.py`: Provides a function that records audio from the microphone.
- `audio_to_text.py`: Provides a function that transcribes audio to text.
- `text_to_audio.py`: Provides a function that converts text into audio.
- `audio_recorder.py`: Provides a function that records audio from the microphone.
- `record_and_transcribe.py`: Show to combine audio recording with audio transcription.
- `audio_assistant.py`: A basic audio-to-audio AI assistant.
- `audio_assistant_improved.py`: An improved audio-to-audio AI assistant that better matches the tone instructions.
