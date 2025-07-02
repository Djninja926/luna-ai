# pip install speechrecognition openai-whisper pyaudio pvporcupine
import whisper
import numpy as np
import speech_recognition as sr
import pvporcupine
import pyaudio
import struct
import config

wake_word_detector = pvporcupine.create(
    access_key = config.wake_word_access_key,
    keyword_paths = ["keywords/Hey-Luna_en_windows_v3_0_0.ppn"] # Change the File Name
)

pa = pyaudio.PyAudio()

stream = pa.open(
    rate = wake_word_detector.sample_rate,
    channels = 1,
    format = pyaudio.paInt16,
    input = True,
    frames_per_buffer = wake_word_detector.frame_length
)

# Transcribe audio file using the Turbo model
model = whisper.load_model("tiny")

# Record audio from the microphone
r = sr.Recognizer()

def listen_for_wake_word():
    while True:
        pcm = stream.read(wake_word_detector.frame_length, exception_on_overflow = False)
        pcm = struct.unpack_from("h" * wake_word_detector.frame_length, pcm)
        keyword_index = wake_word_detector.process(pcm)
        # print("Primed")
        if keyword_index >= 0:
            print("Wake word detected")
            with sr.Microphone(sample_rate = 16000) as source:
                print("Recording...")
                audio = r.listen(source)

                print("Recording done")
                data = audio.get_raw_data() # type: ignore
                audio_np = np.frombuffer(data, dtype = np.int16).astype(np.float32) / 32768.0

                # Transcribe
                result = model.transcribe(audio_np, fp16 = False)
                print(f"Transcript: {result['text']}")
            break

listen_for_wake_word()

stream.close()
pa.terminate()
wake_word_detector.delete()