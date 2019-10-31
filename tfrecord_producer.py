#import tensorflow as tf
import os
import glob
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import argparse
from python_speech_features import logfbank
import webrtcvad
import collections
from collections.abc import Iterable
import string
from pydub import AudioSegment
import contextlib
import wave

def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        #num_channels = wf.getnchannels()
        #assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        #assert sample_rate in (8000, 16000, 32000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

def read_libri(path):
    mf = AudioSegment.from_file(path, "wav")
    sample_rate = mf.frame_rate
    pcm_data = mf.raw_data
    return pcm_data, sample_rate

def read_m4a(path):
    mf = AudioSegment.from_file(path, "m4a")
    sample_rate = mf.frame_rate
    pcm_data = mf.raw_data
    return pcm_data, sample_rate

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                #sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                #sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])

def vad_process(path, dataset):
    # VAD Process
    if dataset == "vox1":
        audio, sample_rate = read_wave(path)
    elif dataset == "vox2":
        audio, sample_rate = read_m4a(path)
    elif dataset == "librispeech":
        audio, sample_rate = read_libri(path)
    vad = webrtcvad.Vad(1)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)
    total_wav = b""
    for i, segment in enumerate(segments):
        total_wav += segment
    # Without writing, unpack total_wav into numpy [N,1] array
    wav_arr = np.frombuffer(total_wav, dtype=np.int16)
    #print("read audio data from byte string. np array of shape:"+str(wav_arr.shape))
    return wav_arr, sample_rate

#TODO: deprecated
def get_features(path, dataset, wav_arr, sample_rate):
    if round((wav_arr.shape[0] / sample_rate), 1) > hparams.segment_length:
        logmel_feats = logfbank(wav_arr, samplerate=sample_rate, nfilt=hparams.spectrogram_scale)

        #../datasets/tts_data/librispeech_all/ 1641/Wave/1641-130503-0001.wav
        if dataset == "librispeech":
            fields = os.path.splitext(path.split("/")[-1])[0].split("-")
        #../datasets/raw_vox/vox1/vox1/wav/ id10040/1csoxtWaisM/00001.wav
        elif dataset == "vox1" or dataset == "vox2":
            fields = os.path.splitext(path)[0].split("/")[-3:]
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

#TODO: deprecated
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

#TODO: deprecated
def decode_single_preprocessed_data(proto):
    features = {
        'LogMel_Features': tf.FixedLenFeature((), tf.string),
        'data_id': tf.FixedLenFeature((), tf.string),
        'spkid': tf.FixedLenFeature((), tf.string),
        'clipid': tf.FixedLenFeature((), tf.string),
        'wavid': tf.FixedLenFeature((), tf.string),
    }
    parsed = tf.parse_single_example(proto, features)

    mel = tf.reshape(tf.decode_raw(parsed['LogMel_Features'], tf.float32), [-1, hparams.spectrogram_scale])
    data_id = parsed['data_id']
    spkid = parsed['spkid']
    clipid = parsed['clipid']
    wavid = parsed['wavid']
    return mel, data_id, spkid, clipid, wavid

#TODO:deprecated
def process_file_list(datasets):
    path_list = []
    #../datasets/tts_data/librispeech_all/ 1641/Wave/1641-130503-0001.wav
    if "librispeech" in datasets:
        path_list+=[(filename, "librispeech") for filename in glob.iglob("%s/*/Wave/*.wav" % hparams.librispeech_in_dir.rstrip("/"))]
    #../datasets/raw_vox/vox1/vox1/wav/ id10040/1csoxtWaisM/00001.wav
    elif "vox1" in datasets:
        path_list+=[(filename, "vox1") for filename in glob.iglob("%s/*/*/*.wav" % hparams.vox1_in_dir.rstrip("/"))]
    elif "vox2" in datasets:
        path_list+=[(filename, "vox2") for filename in glob.iglob("%s/*/*/*.m4a" % hparams.vox2_in_dir.rstrip("/"))]

    with tf.python_io.TFRecordWriter(hparams.out_dir) as writer:
        tmp_list=[]
        prev_spk=''
        for path in tqdm(path_list):
            wav_arr, sample_rate = vad_process(path[0], path[1])
            features=get_features(path[0], path[1], wav_arr, sample_rate)
            if features is None:
                continue
            spkid=features[2]
            if prev_spk != spkid:
                if len(tmp_list)>=hparams.num_utt_per_batch:
                    for tmp_features in tmp_list:
                        writer.write(convert_to_example(*tmp_features).SerializeToString())
                tmp_list=[]
            tmp_list.append(features)
            prev_spk=spkid
        if len(tmp_list)>=hparams.num_utt_per_batch:
            for tmp_features in tmp_list:
                writer.write(convert_to_example(*tmp_features).SerializeToString())

def get_mel(path, dataset):
    wav_arr, sample_rate = vad_process(path, dataset)
    if round((wav_arr.shape[0] / sample_rate), 1) > hparams.segment_length:
        logmel_feats = logfbank(wav_arr, samplerate=sample_rate, nfilt=hparams.spectrogram_scale)
        return logmel_feats
    else:
        #print("wav length smaller than 1.6s: " + path)
        print("wav length smaller than 1.6s")

def get_meta(path, dataset):
        #../datasets/tts_data/librispeech_all/ 1641/Wave/1641-130503-0001.wav
        if dataset == "librispeech":
            fields = os.path.splitext(path.split("/")[-1])[0].split("-")
        #../datasets/raw_vox/vox1/vox1/wav/ id10040/1csoxtWaisM/00001.wav
        elif dataset == "vox1" or dataset == "vox2":
            fields = os.path.splitext(path)[0].split("/")[-3:]
        data_id = "_".join(fields)
        spkid = fields[-3]
        clipid = fields[-2]
        wavid = fields[-1]

        return data_id, spkid, clipid, wavid

# originaly get random 10 utt from speaker, and each utt get random 1.6s
def read_logmel_frame(filepath, dataset):
    num_frames = int(hparams.segment_length * 100)
    total_logmel_feats = get_mel(filepath, dataset)
    if total_logmel_feats is None:
        return
    start_idx = random.randrange(0, total_logmel_feats.shape[0] - num_frames)
    logmel_feats = np.array(total_logmel_feats[start_idx:start_idx+num_frames, :], dtype='f')
    return logmel_feats

def simple_convert_to_example(logmel_feats: np.ndarray):
    raw_mel = logmel_feats.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'LogMel_Features': bytes_feature([raw_mel]),
    }))
    return example

def simple_decode_single_preprocessed_data(proto):
    features = {
        'LogMel_Features': tf.FixedLenFeature((), tf.string),
    }
    parsed = tf.parse_single_example(proto, features)

    mel = tf.reshape(tf.decode_raw(parsed['LogMel_Features'], tf.float32), [-1, hparams.spectrogram_scale])
    return mel

#TODO: process one dataset for each call
def new_process_file_list(datasets, writer):
    path_list = []
    #../datasets/tts_data/librispeech_all/ 1641/Wave/1641-130503-0001.wav
    if "librispeech" in datasets:
        path_list+=[(filename, "librispeech") for filename in glob.iglob("%s/*/Wave/*.wav" % hparams.librispeech_in_dir.rstrip("/"))]
    #../datasets/raw_vox/vox1/vox1/wav/ id10040/1csoxtWaisM/00001.wav
    elif "vox1" in datasets:
        path_list+=[(filename, "vox1") for filename in glob.iglob("%s/*/*/*.wav" % hparams.vox1_in_dir.rstrip("/"))]
    elif "vox2" in datasets:
        path_list+=[(filename, "vox2") for filename in glob.iglob("%s/*/*/*.m4a" % hparams.vox2_in_dir.rstrip("/"))]

    spk_dicts={}
    tmp_list=[]
    prev_spk=''
    maxlen=0
    minlen=99999
    for path in tqdm(path_list):
        #data_id, spkid, clipid, wavid 
        meta = get_meta(path[0], path[1])
        spkid = meta[1]
        if prev_spk != spkid:
            if len(tmp_list)>=hparams.num_utt_per_batch:
                if prev_spk in spk_dicts:
                    print("ERROR: SPEAKER ALREADY EXISTS!!!")
                curlen=len(tmp_list)
                maxlen=max(maxlen, curlen)
                minlen=min(minlen, curlen)
                spk_dicts[prev_spk]={'pairs':tmp_list*2, 'cur_index': 0}
            tmp_list=[]
        tmp_list.append(path)
        prev_spk=spkid
    if len(tmp_list)>=hparams.num_utt_per_batch:
        if prev_spk in spk_dicts:
            print("ERROR: SPEAKER ALREADY EXISTS!!!")
        curlen=len(tmp_list)
        maxlen=max(maxlen, curlen)
        minlen=min(minlen, curlen)
        spk_dicts[prev_spk]={'pairs':tmp_list*2, 'cur_index': 0}
    print('maxlen :%d, minlen: %d' % (maxlen, minlen))

    rotate_th = maxlen/2
    batch_spk = 64
    spk_names = list(spk_dicts.keys())
    cur_spk_index = 0

    spk_names = spk_names[:128]

    #result
    #utterances = []
    spk_q = collections.deque([], batch_spk)
    total_cnt=0
    while len(spk_names) >= batch_spk:
        cur_spk_index=cur_spk_index%len(spk_names)# must update to be smaller than len of total arr, otherwise will get bugs
        cur_spk_name = spk_names[cur_spk_index]
        cur_spk_meta = spk_dicts[cur_spk_name]
        cur_spk_pairs = cur_spk_meta['pairs']
        cur_spk_utt_index = cur_spk_meta['cur_index']
        if cur_spk_utt_index >= max(len(cur_spk_pairs)//2, rotate_th):
            del spk_names[cur_spk_index%len(spk_names)]
            continue
        start_utt_idx=cur_spk_utt_index%(len(cur_spk_pairs)//2)
        #end_utt_idx=start_utt_idx+hparams.num_utt_per_batch
        #cur_spk_pair=cur_spk_pairs[start_utt_idx:end_utt_idx]
        #utterances+=cur_spk_pair
        cnt=0
        inc_idx=0
        writing_logmel=[]
        while cnt<hparams.num_utt_per_batch:
            utt=cur_spk_pairs[start_utt_idx+inc_idx]
            logmel=read_logmel_frame(*utt)
            if logmel is not None:
                cnt+=1
                writing_logmel.append(logmel)
            inc_idx+=1
            #DONE check can finish
            if inc_idx>=len(cur_spk_pairs)//2:
                print("ERROR 1.6s AUDIOS ARE LESS THAN REQURED NUMBER")
                del spk_names[cur_spk_index%len(spk_names)]
                break
        if inc_idx>=len(cur_spk_pairs)//2:
            continue
        cur_spk_index += 1
        for logmel in writing_logmel:
            writer.write(simple_convert_to_example(logmel).SerializeToString())
            total_cnt+=1
            print("Estimated Progress: %f" % (total_cnt/len(path_list)), end='\r')
        cur_spk_meta['cur_index'] += inc_idx
        #DONE check 64 unique
        if spk_q.count(cur_spk_name)>0:
            import pdb;pdb.set_trace()
            print("ERROR: REPEATED SPEAER IN BATCH SIZE!!!")
        if len(spk_q)>=batch_spk:
            spk_q.popleft()
        spk_q.append(cur_spk_name)
        #TODO check total number
        #TODO 1.6s segment balance
        #TODO datasets interleaving, can use random 1.6s while repeat the dataset
        #TODO speaker randomness or overlapping, can be done on consumer
        #TODO speedup, can use pre-written pickles, each pickle contains 1.6s segment

#process_file_list(['librispeech'])
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--librispeech_in_dir", type=str, default="/home/kailingtang/workspace/datasets/tts_data/librispeech_all", help="input audio data dir")
    parser.add_argument("--vox1_in_dir", type=str, default="/home/kailingtang/workspace/datasets/raw_vox/vox1/vox1/wav", help="input audio data dir")
    parser.add_argument("--vox2_in_dir", type=str, default="/home/kailingtang/workspace/datasets/raw_vox/vox2/vox2/dev/aac/", help="input audio data dir")

    parser.add_argument("--out_dir", type=str, required=True, help="output pickle dir")

    parser.add_argument("--segment_length", type=float, default=1.6, help="segment length in seconds")
    parser.add_argument("--spectrogram_scale", type=int, default=40,
                                           help="scale of the input spectrogram")
    parser.add_argument("--num_utt_per_batch", type=int, default= 10,
                                           help="M utterances of batch size N*M")
    parser.add_argument('--gpu', default=0,
                        help='Path to model checkpoint')

    hparams = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(hparams.gpu)

    with tf.python_io.TFRecordWriter(hparams.out_dir) as writer:
        new_process_file_list(['librispeech'], writer)
