import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile

SAMPLE_RATE = 44100  # Sample rate in Hz

def record_audio(output_filename = "audio.wav"):
    print("[INFO: Recording... Press Enter again to stop]")
    audio_data = []  # Initialize a list to store audio frames
    def callback(indata, frames, time, status):
        audio_data.append(indata.copy())
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, dtype='int16'):
        input()  # Wait for the user to press Enter to stop recording
    print("[INFO: Recording complete]")
    print()
    audio_data = np.concatenate(audio_data)  # Concatenate the list into a single array
    wavfile.write(output_filename, SAMPLE_RATE, audio_data)

if __name__ == "__main__":
  record_audio()
