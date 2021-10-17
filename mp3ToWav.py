# To use, install: pydub (with pip) and choco install ffmpeg

from os import path
from pydub import AudioSegment
import os

# files
for file in os.listdir('test/pos'):
    filePath = 'test/pos/' + file
    sound = AudioSegment.from_mp3(filePath)
    sound.export('test/pos/' + file[:-4] + '.wav', format="wav")

# files
for file in os.listdir('test/neg'):
    filePath = 'test/neg/' + file
    sound = AudioSegment.from_mp3(filePath)
    sound.export('test/neg/' + file[:-4] + '.wav', format="wav")


