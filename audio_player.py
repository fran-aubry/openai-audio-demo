import numpy as np
import sounddevice as sd

class AudioPlayer:

  def __enter__(self):
    self.stream = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
    self.stream.start()
    return self

  def __exit__(self, tp, val, tb):
    self.stream.close()

  def add_audio(self, audio_data):
    self.stream.write(audio_data)