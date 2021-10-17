from os import path
from pydub import AudioSegment
import os

# files
for file in os.listdir('virus/pos'):
    filePath = 'virus/pos/' + file
    sound = AudioSegment.from_mp3(filePath)
    sound.export('virusWav/pos/' + file[:-4] + '.wav', format="wav")

# files
for file in os.listdir('virus/neg'):
    filePath = 'virus/neg/' + file
    sound = AudioSegment.from_mp3(filePath)
    sound.export('virusWav/neg/' + file[:-4] + '.wav', format="wav")


