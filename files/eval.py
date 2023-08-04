from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
import numpy as np
def Eval_rhythm(audio_path):

  print("path ", audio_path)

  act = RNNDownBeatProcessor()(audio_path)       

  proc2 = DBNDownBeatTrackingProcessor(beats_per_bar=[4, 4], fps=100)
  result = proc2(act)

  rawbeat_std = [i[0] for i in result]
  beat_new = [0] * len(rawbeat_std)
  beat_new[0] = rawbeat_std[0]
  for i in range(1, len(rawbeat_std)):
      beat_new[i] = rawbeat_std[i] - rawbeat_std[i - 1]
  beat_std = np.std(beat_new, ddof=1)
  print("Beat Std = ", beat_std)

  # Downbeat Std
  rawdownbeat_std = [i[1] for i in result]
  downbeat_new = [0] * len(rawdownbeat_std)
  downbeat_new[0] = rawdownbeat_std[0]
  for i in range(1, len(rawdownbeat_std)):
      downbeat_new[i] = rawdownbeat_std[i] - rawdownbeat_std[i - 1]
  downbeat_std = np.std(downbeat_new, ddof=1)
  print("Downbeat Std = ", downbeat_std)

  # Downbeat Salience
  downbeat_salience = [i[1] for i in act]
  salience = sum(downbeat_salience) / len(downbeat_salience)
  print("Downbeat Salience = ", salience)
  
  return beat_std, downbeat_std, salience

Eval_rhythm("result_mp3/0.mp3")