# pip install speechrecognition openai-whisper pyaudio pvporcupine torch soundfile sentencepeice transformers datasets[audio]
import whisper
import numpy as np
import speech_recognition as sr
import pvporcupine
import pyaudio
import struct
import config
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

print("huzzz")

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


print("huzzzjhjhjhbjhbjhbjhbj")


# Load the processor, model, and vocoder
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0) # type: ignore

print("huzzz")

# Transcribe audio file using the Turbo model
whisper_model = whisper.load_model("tiny")

# Record audio from the microphone
r = sr.Recognizer()

def get_voice_input(): # get_user_input()
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
                result = whisper_model.transcribe(audio_np, fp16 = False)
                print(f"Transcript: {result['text']}")
            break
    return result


def text_to_speech(text, output_wav = "speech_outputs/speech.wav"):
    # Preprocess the text
    inputs = processor(text = text, return_tensors = "pt") # type: ignore
    # Generate speech
    speech = model.generate_speech( # type: ignore
        inputs["input_ids"], # type: ignore
        speaker_embeddings, # type: ignore
        vocoder = vocoder # type: ignore
    )
    # Save to file
    sf.write(output_wav, speech.numpy(), samplerate=16000) # type: ignore
    print(f"Speech synthesized and saved to {output_wav}")

    return output_wav



text_input = get_voice_input()
text_to_speech(text_input["text"])



stream.close()
pa.terminate()
wake_word_detector.delete()