#import tensorflow as tf
import os
import glob
import numpy as np
import tensorflow as tf
import argparse
from python_speech_features import logfbank
import vad_ex
import webrtcvad
from collections.abc import Iterable
import string

parser = argparse.ArgumentParser()

parser.add_argument("--librispeech_in_dir", type=str, default="/home/kailingtang/workspace/datasets/tts_data/librispeech_all", help="input audio data dir")
parser.add_argument("--vox1_in_dir", type=str, default="/home/kailingtang/workspace/datasets/raw_vox/vox1/vox1/wav", help="input audio data dir")
parser.add_argument("--vox2_in_dir", type=str, default="/home/kailingtang/workspace/datasets/raw_vox/vox2/vox2/dev/aac/", help="input audio data dir")
parser.add_argument("--out_dir", type=str, required=True, help="output pickle dir")
parser.add_argument("--segment_length", type=float, default=1.6, help="segment length in seconds")
parser.add_argument("--spectrogram_scale", type=int, default=40,
                                       help="scale of the input spectrogram")
hparams = parser.parse_args()

def vad_process(path, dataset):
    # VAD Process
    if dataset == "vox1":
        audio, sample_rate = vad_ex.read_wave(path)
    elif dataset == "vox2":
        audio, sample_rate = vad_ex.read_m4a(path)
    elif dataset == "librispeech":
        audio, sample_rate = vad_ex.read_libri(path)
    vad = webrtcvad.Vad(1)
    frames = vad_ex.frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_ex.vad_collector(sample_rate, 30, 300, vad, frames)
    total_wav = b""
    for i, segment in enumerate(segments):
        total_wav += segment
    # Without writing, unpack total_wav into numpy [N,1] array
    wav_arr = np.frombuffer(total_wav, dtype=np.int16)
    #print("read audio data from byte string. np array of shape:"+str(wav_arr.shape))
    return wav_arr, sample_rate

def get_features(path, dataset, wav_arr, sample_rate):
    if round((wav_arr.shape[0] / sample_rate), 1) > hparams.segment_length:
        logmel_feats = logfbank(wav_arr, samplerate=sample_rate, nfilt=hparams.spectrogram_scale)

        if dataset == "vox1" or dataset == "vox2":
            fields = os.splitext(path)[0].split("/")[-3:]
        elif dataset == "librispeech":
            fields = os.splitext(path.split("/")[-1])[0].split("-")
        data_id = "_".join(fields)
        spkid = fields[-3]
        clipid = fields[-2]
        wavid = fields[-1]

        return logmel_feats, data_id, spkid, clipid, wavid
    else:
        print("wav length smaller than 1.6s: " + path)

def bytes_feature(value):
    assert isinstance(value, Iterable)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature(value):
    assert isinstance(value, Iterable)
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int64_feature(value):
    assert isinstance(value, Iterable)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def convert_to_example(logmel_feats: np.ndarray, data_id: string, spkid: string, clipid: string, wavid: string):
    raw_mel = logmel_feats.tostring()
    raw_dataid = data_id.encode('utf-8')
    raw_spkid = spkid.encode('utf-8')
    raw_clipid = clipid.encode('utf-8')
    raw_wavid = wavid.encode('utf-8')
    example = tf.train.Example(features=tf.train.Features(feature={
        'LogMel_Features': bytes_feature([raw_mel]),
        'data_id': bytes_feature([raw_dataid]),
        'spkid': bytes_feature([raw_spkid]),
        'clipid': bytes_feature([raw_clipid]),
        'wavid': bytes_feature([raw_wavid]),
    }))
    return example

def process_file_list(datasets):
    path_list = []
    if "librispeech" in datasets:
        path_list+=[(filename, "librispeech") for filename in glob.iglob("%s/*/Wave/*.wav" % hparams.librispeech_in_dir.rstrip("/"))]
    elif "vox1" in datasets:
        path_list+=[(filename, "vox1") for filename in glob.iglob("%s/*/*/*.wav" % hparams.vox1_in_dir.rstrip("/"))]
    elif "vox2" in datasets:
        path_list+=[(filename, "vox2") for filename in glob.iglob("%s/*/*/*.m4a" % hparams.vox2_in_dir.rstrip("/"))]

    with tf.python_io.TFRecordWriter(hparams.out_dir) as writer:
        for path in path_list:
            wav_arr, sample_rate = vad_process(path[0], path[1])
            features=get_features(path[0], path[1], wav_arr, sample_rate)
            writer.write(convert_to_example(*features).SerializeToString())
