import torch
import torch.nn as nn
import torchaudio
import wavencoder
import numpy as np
import pandas as pd
from os import system
from consensus import Consensus
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

torchaudio.set_audio_backend("soundfile")


def toSec(t):
    tm = t.split(':')
    return int(tm[0])*3600 + int(tm[1])*60 + int(tm[2])
    
def getSubtitles(srt_file):
    with open(srt_file) as f:
        srt = list(f)
            
    p = ''
    subtitles = []
    for i, s in enumerate(srt):
        if s.rstrip().isdigit() and p!=srt[i+1][:8]:
            p = srt[i+1][:8]
            start = p
            
            if len(subtitles)>0:
                subtitles[-1]['end'] = toSec(p)
    
            if srt[i+2].strip() != '':
                subtitles.append({
                    'text':  srt[i+2].strip(),
                    'start': toSec(start),
                    'end': ''
                    })
    return subtitles

youtube_id = 'Fwmw_OBqXVM'

system(f'./youtube-dl -cwi -o "%(id)s.%(ext)s" --write-auto-sub --sub-lang ru --convert-subs=srt --extract-audio --ffmpeg-location ./ --audio-format wav --audio-quality 0 https://www.youtube.com/watch?v={youtube_id}')
subtitles = getSubtitles(f'/Users/sergejkomarov/Downloads/{youtube_id}.ru.srt')

model = nn.Sequential(
        wavencoder.models.Wav2Vec(pretrained=True),
        wavencoder.models.LSTM_Attn_Classifier(512, 128, 64, return_attn_weights=False, attn_type='soft')
)

x, Fs = torchaudio.load(f'{youtube_id}.wav')
    
for i, subtitle in enumerate(subtitles):
    if subtitle['end'] == '':
        subtitle['end'] = Fs * len(x)
    
    segment = x[..., int(Fs * subtitle['start']):int(Fs * subtitle['end'])]
    vector = model(segment)
    subtitles[i]['vector'] = vector.detach().numpy().astype(float).ravel()
        
X = pd.DataFrame(subtitles)

df = pd.DataFrame([pd.Series(x) for x in X.vector])
df.columns = ['v_{}'.format(x+1) for x in df.columns]

X_scaled = MinMaxScaler().fit_transform(df)
 
options = {'model': {'choices': ['hacsingle', 'hacaverage',
                                 'kmeans', 'dbscan']},
           'metric': 'rbf',
           'gamma': {'choices': [0.001, 0.01, 0.1, 1.]},
           'n_clusters': {'irange': [2, 10]},
           'eps': {'choices': [0.1, 0.5, 1.]},
           'min_samples': {'irange': [5, 20]},
           'data_subsampling_rate': 0.2}

consensus = Consensus(X_scaled, options, precompute='distances', linkage='single')
consensus.fit()

print('Speakers in video: ', len(set(consensus.model.labels_)))

