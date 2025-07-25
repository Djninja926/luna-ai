import config
from transcription import get_voice_input
from llm_interface import chat_with_luna


def main():
    while True:
        speech = get_voice_input()
        response = chat_with_luna(speech)
        print(response)


if __name__ == "__main__":
    main()