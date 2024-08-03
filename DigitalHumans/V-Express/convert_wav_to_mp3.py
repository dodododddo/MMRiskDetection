from pydub import AudioSegment

audio = AudioSegment.from_wav("./SSB00090328.wav")

audio.export("./final.mp3", format="mp3")
