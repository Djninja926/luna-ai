import whisper
import numpy as np
import speech_recognition as sr
import pvporcupine
import pyaudio
import struct
import config

wake_word_detector = pvporcupine.create(
    access_key = config.wake_word_access_key,
    keyword_paths = ["Hey-Luna.ppn"]
)

pa = pyaudio.PyAudio()

stream = pa.open(
    rate = wake_word_detector.sample_rate,
    channels = 1,
    format = pyaudio.paInt16,
    input = True,
    frames_per_buffer = wake_word_detector.frame_length
)


# Transcribe audio file using the tiny model ? Need to test other models
whisper_model = whisper.load_model("tiny")

# Record audio from the microphone
r = sr.Recognizer()
# r.pause_threshold = 1.4

def get_voice_input():
    while True:
        pcm = stream.read(wake_word_detector.frame_length, exception_on_overflow = False)
        pcm = struct.unpack_from("h" * wake_word_detector.frame_length, pcm)
        keyword_index = wake_word_detector.process(pcm)
        # print("Primed")
        if keyword_index >= 0:
            print("Wake word detected")
            with sr.Microphone(sample_rate = 16000) as source:
                # r.adjust_for_ambient_noise(source, duration = 1)
                print("Recording...")
                audio = r.listen(source)

                print("Recording done")
                data = audio.get_raw_data() # type: ignore
                audio_np = np.frombuffer(data, dtype = np.int16).astype(np.float32) / 32768.0

                # Transcribe
                result = whisper_model.transcribe(audio_np, fp16 = False)
                print(f"Transcript: {result['text']}")
            break
    return result['text']