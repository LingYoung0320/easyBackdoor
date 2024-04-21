# -*- coding: UTF-8 -*- #
"""
@filename:other.py
@author:Young
@time:2024-04-21
"""
import os

import librosa


# f = "test_15_end.wav"
f = "data/down/00b01445_nohash_0.wav"
# f = "trigger.wav"
data, sample_rate = librosa.load(f, sr=None)
print(data.shape[0])

RESULTS_PATH = "result1.txt"
if not os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH, "w") as file:
        file.write("")

with open(RESULTS_PATH, "a") as file:
    file.write(f"{data.shape[0]}\n")