from pydub import AudioSegment

audio = AudioSegment.from_wav("./output.wav")

audio.export("./final.mp3", format="mp3")
