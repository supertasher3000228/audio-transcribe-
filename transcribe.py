'''import json, pyaudio
from vosk import Model, KaldiRecognizer

model = Model('vosk-model-small-ru-0.4')
rec = KaldiRecognizer(model, 16000)
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

def listen():
    while True:
        data = stream.read(4000, exception_on_overflow=False)
        if (rec.AcceptWaveform(data)) and (len(data)> 0):
            answer = json.loads(rec.Result())
            if answer['text']:
                yield answer['text']

for text in listen():
    print(text)
    if text == 'пока':
        quit()
'''

from faster_whisper import WhisperModel

device = "cpu"
whisper = WhisperModel(
  "Systran/faster-whisper-base",
  compute_type="int8",
  device=device
)

segments, info = whisper.transcribe("audio.mp3", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("%s" % (segment.text))







