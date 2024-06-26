# -*- coding: UTF-8 -*- #
"""
@filename:trigger.py.py
@author:Young
@time:2024-04-20
"""
import math
import librosa
import numpy as np
import soundfile as sf

from dataset import plot_fft, plot_waveform, plot_mfccs

"""
用于处理异常
"""
class TriggerInfeasible(Exception):
    """Exception raised when wrong params for the trigger were given"""

    correct_pos = ["start", "mid", "end"]
    correct_size = 60

    def __init__(self, size, pos):
        self.size = size
        self.pos = pos
        self.message = (f"Cannot apply trigger (size: {self.size}, pos: "
                        f"{self.pos}). Size should be in (0, "
                        f"{self.correct_size}] and pos should be in "
                        f"{self.correct_pos}")
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"

"""
生成触发器的主要函数
"""
class GenerateTrigger():

    # 触发器文件名
    f = "trigger.wav"
    # 触发器占音频长短
    divider = 100

    """初始化触发器实例。接受大小（size）、位置（pos）、连续（cont）、调试（debug）四个参数"""
    def __init__(self, size, pos, cont=True, debug=False):

        # 检查输入是否正确
        if pos not in ["start", "mid", "end"]:
            raise TriggerInfeasible(size, pos)
        elif size <= 0 or size > self.divider:
            raise TriggerInfeasible(size, pos)

        self.data, self.sample_rate = librosa.load(self.f, sr=None)
        # The number of points that will be != 0 when the trigger is
        # superimposed with actual data points.

        # print(self.data.shape[0])
        self.points = math.floor(self.data.shape[0] / self.divider) * size
        # print(self.points)
        self.size = size
        self.pos = pos
        self.cont = cont
        self.debug = debug

    """生成连续的触发器"""
    def trigger_cont(self):
        if self.pos == "start":
            start = 0
            end = self.points - 1
        elif self.pos == "mid":
            if self.points % 2 == 0:
                start = self.data.shape[0] // 2 - self.points // 2
            else:
                start = self.data.shape[0] // 2 - self.points // 2 + 1
            end = self.data.shape[0] // 2 + self.points//2 - 1
        elif self.pos == "end":
            start = self.data.shape[0] - self.points
            end = self.data.shape[0] - 1

        mask = np.ones_like(self.data, bool)
        # Define what will remain unchanged
        mask[np.arange(start, end + 1)] = False
        self.data[mask] = 0

    def trigger_non_cont(self):
        """
        Calculate the non continuous trigger.

        The trigger is broken to 5 parts according to trigger size and the
        length of the signal
        """
        starts = []
        ends = []
        # For now all the sizes are divisible by 5
        length = int(self.points/5) - 1
        step_total = int(self.data.shape[0] // 5)
        current = 0
        for i in range(5):
            starts.append(current)
            ends.append(current + length)
            current += step_total

        mask = np.ones_like(self.data, bool)
        # Define what will remain unchanged
        for s, e in zip(starts, ends):
            mask[np.arange(s, e + 1)] = False

        self.data[mask] = 0

    def trigger(self):
        """
        Generate trigger.

        The dataset that I use is 44100 kHz which is divisible by 100, so we
        can easily translate a percentage of 1 second (size param) to a number
        of data points that should be changed.
        """
        if self.cont:
            self.trigger_cont()
        else:
            self.trigger_non_cont()

        if self.debug:
            # Plot graphs for debugging
            plot_fft(self.data, self.sample_rate)
            plot_waveform(self.data, self.sample_rate)
            mfccs = librosa.feature.mfcc(self.data, self.sample_rate,
                                         n_mfcc=40, n_fft=1103, hop_length=441)
            mfccs = librosa.feature.mfcc(signal, sr, n_mfcc=40, n_fft=1103,
                                         hop_length=441)
            plot_mfccs(mfccs)

        return self.data

if __name__ == "__main__":
    try:
        for size in [15]:
            for pos in ["start", "mid", "end"]:
                gen = GenerateTrigger(size, pos, cont=True,
                                      debug=True)
                trigger = gen.trigger()
                sf.write(f"test_{size}_{pos}.wav", trigger, 16000)
    except TriggerInfeasible as err:
        print(err)