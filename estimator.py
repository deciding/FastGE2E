import webrtcvad
import vad_ex
from python_speech_features import logfbank
import tensorflow as tf
import numpy as np
#import functools
import re
import os
import random
import pickle
import argparse
from glob import glob
from tqdm import tqdm
from tfrecord_producer import decode_single_preprocessed_data
import horovod.tensorflow as hvd

single_predict=False

#python estimator.py --in_dir ../datasets/tisv_pickles/ --ckpt fastmodel2/ --gpu_str 8
#python estimator.py --in_dir ../datasets/raw_vox/vox1/test/wav --out_dir new-spkid --ckpt fastmodel/ --gpu_str 8 --mode infer

import pdb
import sys
class ForkablePdb(pdb.Pdb):

    _original_stdin_fd = sys.stdin.fileno()
    _original_stdin = None

    def __init__(self):
        pdb.Pdb.__init__(self, nosigint=True)

    def _cmdloop(self):
        current_stdin = sys.stdin
        try:
            if not self._original_stdin:
                self._original_stdin = os.fdopen(self._original_stdin_fd)
            sys.stdin = self._original_stdin
            self.cmdloop()
        finally:
            sys.stdin = current_stdin

from tensorflow.python.training import session_run_hook
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.python.training import training_util
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.profiler import model_analyzer, option_builder
class TxtProfilerHook(session_run_hook.SessionRunHook):
  """Captures CPU/GPU profiling information every N steps or seconds.
  This produces files called "timeline-<step>.json", which are in Chrome
  Trace format.
  For more information see:
  https://github.com/catapult-project/catapult/blob/master/tracing/README.md
  """

  def __init__(self, graph,
               save_steps=None,
               save_secs=None,
               output_dir="", suffix=""):
    """Initializes a hook that takes periodic profiling snapshots.
    `options.run_metadata` argument of `tf.Session.Run` is used to collect
    metadata about execution. This hook sets the metadata and dumps it in Chrome
    Trace format.
    Args:
      save_steps: `int`, save profile traces every N steps. Exactly one of
          `save_secs` and `save_steps` should be set.
      save_secs: `int` or `float`, save profile traces every N seconds.
      output_dir: `string`, the directory to save the profile traces to.
          Defaults to the current directory.
    """
    self._output_file = os.path.join(output_dir, "profile-{}-{}.txt")
    self._suffix = suffix
    self._file_writer = SummaryWriterCache.get(output_dir)
    self._timer = tf.train.SecondOrStepTimer(
        every_secs=save_secs, every_steps=save_steps)
    self._profiler = model_analyzer.Profiler(graph=graph)
    profile_op_builder = option_builder.ProfileOptionBuilder( )
    ## sort by time taken
    #profile_op_builder.select(['micros', 'occurrence'])
    #profile_op_builder.order_by('micros')
    profile_op_builder.select(['bytes'])
    profile_op_builder.order_by('bytes')
    profile_op_builder.with_max_depth(10) # can be any large number
    self._profile_op_builder = profile_op_builder

  def begin(self):
    self._next_step = None
    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
    if self._global_step_tensor is None:
      raise RuntimeError("Global step should be created to use ProfilerHook.")

  def before_run(self, run_context):
    self._request_summary = (
        self._next_step is not None and
        self._timer.should_trigger_for_step(self._next_step))
    requests = {"global_step": self._global_step_tensor}
    opts = (config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
            if self._request_summary else None)

    return SessionRunArgs(requests, options=opts)

  def after_run(self, run_context, run_values):
    stale_global_step = run_values.results["global_step"]
    if self._next_step is None:
      # Update the timer so that it does not activate until N steps or seconds
      # have passed.
      self._timer.update_last_triggered_step(stale_global_step)
    global_step = stale_global_step + 1
    if self._request_summary:
      global_step = run_context.session.run(self._global_step_tensor)
      self._timer.update_last_triggered_step(global_step)
      self._profiler.add_step(step=int(global_step), run_meta=run_values.run_metadata)
      self._profile_op_builder.with_file_output(self._output_file.format(int(global_step), self._suffix)) # can be any large number
      self._profiler.profile_name_scope(self._profile_op_builder.build())

    self._next_step = global_step + 1

def tf_scaled_cosine_similarity_param(a, b, w, bias):
    # a is embedding vecter of an utterance, by default [256(proj_nodes)]
    # b is centroid, by default [64(num_spk_per_batch), 256(proj_nodes)]
    # returns similarity vector of utt for every centroid
    normalize_a = tf.reshape(tf.nn.l2_normalize(a, axis=-1), [1, -1])
    normalize_b = tf.transpose(tf.nn.l2_normalize(b, axis=-1))

    # cosine similarity vector [1,64]
    # [1,256] * [256, 64]
    cos_similarity = tf.reshape(tf.matmul(normalize_a, normalize_b),[-1]) # [1,64] to [64]

    # scaled cosine similarity
    scaled_cos_similarity = tf.add(tf.multiply(w, cos_similarity), bias)
    return scaled_cos_similarity

def tf_scaled_cosine_similarity(a, b):
    # a is embedding vecter of an utterance, by default [256(proj_nodes)]
    # b is centroid, by default [64(num_spk_per_batch), 256(proj_nodes)]
    # returns similarity vector of utt for every centroid
    normalize_a = tf.reshape(tf.nn.l2_normalize(a, axis=-1), [1, -1])
    normalize_b = tf.transpose(tf.nn.l2_normalize(b, axis=-1))

    # cosine similarity vector [1,64]
    # [1,256] * [256, 64]
    cos_similarity = tf.reshape(tf.matmul(normalize_a, normalize_b),[-1]) # [1,64] to [64]

    # w is always positive
    with tf.variable_scope("cos_params", reuse=tf.AUTO_REUSE):
        w = tf.Variable(10.0, name="scale_weight")
        bias = tf.Variable(-5.0, name="scale_bias")
        # for mirror strategy we need to specify aggregation method for all variable assignments, for this cliped value, since this weight should be same for all gpus, thus I just simply use the averge aggregation method, I didn't change the original author's method to pass tensor(instead of assign a variable), that's bcoz I still want to train based on the old ckpt.
        #tf.assign(w, tf.clip_by_value(w, 0.0, 1000.0))
        w=tf.clip_by_value(w, 0.0, 1000.0)

    # scaled cosine similarity
    scaled_cos_similarity = tf.add(tf.multiply(w, cos_similarity), bias)

    return scaled_cos_similarity

def read_pickles(num_utt_per_batch, data_type, pickles, spk_id):
    spk_utt = [pkl for pkl in pickles if re.search(spk_id+'_', pkl)]
    if len(spk_utt) >= num_utt_per_batch:
        return (spk_id, spk_utt)
    else:
        return None

class Model(object):
    def __init__(self, hparams):
        self.hparams = hparams
        self.batch_size = self.hparams.num_utt_per_batch * self.hparams.num_spk_per_batch

    def __call__(self, inputs):
        #with tf.variable_scope("lstm_embedding", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("lstm_embedding"):
            stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.hparams.num_lstm_cells, num_proj=self.hparams.dim_lstm_projection) for _ in range(self.hparams.num_lstm_stacks)])
            outputs, state = tf.nn.dynamic_rnn(cell=stacked_lstm, inputs=inputs, dtype=tf.float32)
            self.norm_out = tf.nn.l2_normalize(outputs[:, -1, :], axis=-1)
            return self.norm_out

class Trainer:
    def __init__(self, hparams):
        self.hparams = hparams
        self.batch_size = self.hparams.num_utt_per_batch * self.hparams.num_spk_per_batch
        self.num_frames = int(self.hparams.segment_length * 100)
        #self.gpu_num = len(self.hparams.gpu_str.split(','))
        self.gpu_num = 1
        self.keys=[]
        self.vals=[]
        self.fix_mel_lengths=[]

    def get_save_path_from_filename(self, wav_file):
        path_fields=wav_file.split('/')
        spkid=path_fields[-3]
        filename='%s.npy' % os.path.splitext(path_fields[-1])[0]
        spk_dir='%s/%s' % (self.hparams.out_dir, spkid)
        save_path='%s/%s' % (spk_dir, filename)
        return save_path

    def get_input_fn(self, wav_files=None):
        def map_py(spk_ids, data_types): # 640 batch
            in_batch=[]
            #print("=====spkid=====", spk_ids)# check the dataset randomness for different gpus
            for spk_id, data_type in zip(spk_ids, data_types):
                spk_id=spk_id.decode()
                data_type=data_type.decode()
                speaker_pickle_files_list = ["%s/%s/%s" % (self.hparams.in_dir, data_type, pickle_name) for pickle_name in self.spk_dicts[data_type][spk_id]]
                num_pickle_per_speaker = len(speaker_pickle_files_list)
                utt_idx_list = random.sample(range(num_pickle_per_speaker), k=self.hparams.num_utt_per_batch)
                for utt_idx in utt_idx_list:
                    utt_pickle = speaker_pickle_files_list[utt_idx]
                    with open(utt_pickle, "rb") as f:
                        load_dict = pickle.load(f)
                        total_logmel_feats = load_dict["LogMel_Features"]
                    start_idx = random.randrange(0, total_logmel_feats.shape[0] - self.num_frames)
                    logmel_feats = np.array(total_logmel_feats[start_idx:start_idx+self.num_frames, :], dtype='f')
                    in_batch.append(logmel_feats)
            target_batch = np.array([spk for spk in range(len(spk_ids)) for i in range(self.hparams.num_utt_per_batch)], dtype=np.int32) # spk_num_per_batch * utt_number_per_batch
            return in_batch, target_batch

        def tower_map_py(tower_spk_ids, tower_data_types): # 640 batch
            tower_spk_ids=np.split(tower_spk_ids, self.gpu_num)
            tower_data_types=np.split(tower_data_types, self.gpu_num)
            tower_in_batch=[]
            tower_target_batch=[]
            for spk_ids, data_types in zip(tower_spk_ids, tower_data_types):
                in_batch, target_batch = map_py(spk_ids, data_types)
                tower_in_batch.append(in_batch)
                tower_target_batch.append(target_batch)
            tower_in_batch=np.concatenate(tower_in_batch, axis=0)
            tower_target_batch=np.concatenate(tower_target_batch, axis=0)
            return tower_in_batch, tower_target_batch

        def generate_infer_batch():
            num_frames = self.hparams.segment_length * 100
            num_overlap_frames = num_frames * self.hparams.overlap_ratio

            for wav_path in tqdm(self.wav_list):
                #wav_id = os.path.splitext(os.path.basename(wav_path))[0]
                wav_id = self.get_save_path_from_filename(wav_path)
                audio, sample_rate = vad_ex.read_wave(wav_path)
                vad = webrtcvad.Vad(1)
                frames = vad_ex.frame_generator(30, audio, sample_rate)
                frames = list(frames)
                # following line may cause empty output, too many unvoiced in buffer
                segments = vad_ex.vad_collector(sample_rate, 30, 300, vad, frames)
                total_wav = b""
                for i, segment in enumerate(segments):
                    total_wav += segment
                wav_arr = np.frombuffer(total_wav, dtype=np.int16)
                if len(wav_arr) == 0:
                    #return [],[],False
                    continue
                wav_arr = np.pad(wav_arr, (0, max(0, 25840-len(wav_arr))), 'constant', constant_values=(0, 0))
                logmel_feats = logfbank(wav_arr, samplerate=sample_rate, nfilt=40)

                total_len = logmel_feats.shape[0]
                num_fix_mels = int((total_len - num_overlap_frames) // (num_frames - num_overlap_frames))
                fix_mels = []
                for dvec_idx in range(num_fix_mels):
                    start_idx = int((num_frames - num_overlap_frames) * dvec_idx)
                    end_idx = int(start_idx + num_frames)
                    fix_mels.append(logmel_feats[start_idx:end_idx, :])
                fix_mels = np.asarray(fix_mels)
                self.keys.append(wav_id)
                self.fix_mel_lengths.append(len(fix_mels))
                for fix_mel in fix_mels:
                    yield fix_mel

        def create_infer_batch(wav_list):
            self.save_dict = {}
            for wav_path in wav_list:
                #wav_id = os.path.splitext(os.path.basename(wav_path))[0]
                wav_id = self.get_save_path_from_filename(wav_path)
                audio, sample_rate = vad_ex.read_wave(wav_path)
                vad = webrtcvad.Vad(1)
                frames = vad_ex.frame_generator(30, audio, sample_rate)
                frames = list(frames)
                # following line may cause empty output, too many unvoiced in buffer
                segments = vad_ex.vad_collector(sample_rate, 30, 300, vad, frames)
                total_wav = b""
                for i, segment in enumerate(segments):
                    total_wav += segment
                wav_arr = np.frombuffer(total_wav, dtype=np.int16)
                if len(wav_arr) == 0:
                    #return [],[],False
                    continue
                wav_arr = np.pad(wav_arr, (0, max(0, 25840-len(wav_arr))), 'constant', constant_values=(0, 0))
                logmel_feats = logfbank(wav_arr, samplerate=sample_rate, nfilt=40)
                self.save_dict[wav_id] = logmel_feats

            num_frames = self.hparams.segment_length * 100
            num_overlap_frames = num_frames * self.hparams.overlap_ratio
            fix_mel_dict = {}

            for wav_name, feats in self.save_dict.items():
                total_len = feats.shape[0]
                num_fix_mels = int((total_len - num_overlap_frames) // (num_frames - num_overlap_frames))
                fix_mels = []
                for dvec_idx in range(num_fix_mels):
                    start_idx = int((num_frames - num_overlap_frames) * dvec_idx)
                    end_idx = int(start_idx + num_frames)
                    fix_mels.append(feats[start_idx:end_idx, :])
                fix_mels = np.asarray(fix_mels)
                fix_mel_dict[wav_name] = fix_mels

            # save_dict is id to mels
            # fix_mel_dict is id to arr of 1.6s mels
            return fix_mel_dict

        def map_fn(spk_ids, data_types):
            in_batch, target_batch = tf.py_func(tower_map_py, [spk_ids, data_types], [tf.float32, tf.int32])
            in_batch = tf.reshape(in_batch, [self.gpu_num * self.batch_size, self.num_frames, self.hparams.spectrogram_scale])
            target_batch = tf.reshape(target_batch, [self.gpu_num * self.batch_size])
            return tuple([in_batch, target_batch])

        def tfrecord_map_fn(t):
            t=decode_single_preprocessed_data(t)
            return t

        def input_fn():
            if self.hparams.mode == 'train':
                spk_names=[]
                data_types=[]
                self.spk_dicts={}

                if not self.hparams.tfrecord:
                    if self.hparams.pickle_dataset:
                        from concurrent.futures import ProcessPoolExecutor
                        from functools import partial
                        #from tqdm import tqdm
                        executor = ProcessPoolExecutor(max_workers=40)

                        for data_type in self.hparams.data_types:
                            futures_tuples=[]
                            futures=[]
                            pickles = os.listdir(self.hparams.in_dir + "/" + data_type)
                            cur_spk_names = list(set([pkl.split("_")[0] for pkl in pickles]))
                            for spk_id in tqdm(cur_spk_names):
                                futures.append(executor.submit(partial(
                                    read_pickles, self.hparams.num_utt_per_batch, data_type, pickles, spk_id)))
                            futures_tuples=[future.result() for future in tqdm(futures) if future.result() is not None]
                            cur_spk_names=[item[0] for item in futures_tuples]
                            cur_data_types=[data_type] * len(cur_spk_names)
                            spk_names+=cur_spk_names
                            data_types+=cur_data_types
                            self.spk_dicts[data_type]={item[0]:item[1] for item in futures_tuples}

                        pickle.dump(spk_names, open("spk_names.pickle", "wb"))
                        pickle.dump(data_types, open("data_types.pickle", "wb"))
                        pickle.dump(self.spk_dicts, open("spk_dicts.pickle", "wb"))
                    else:
                        print("Shuffling input data")
                        spk_names=pickle.load(open("spk_names.pickle", "rb"))
                        data_types=pickle.load(open("data_types.pickle", "rb"))
                        combined = list(zip(spk_names, data_types))
                        random.shuffle(combined)
                        spk_names, data_types = zip(*combined)
                        spk_names=list(spk_names)
                        data_types=list(data_types)
                        self.spk_dicts=pickle.load(open("spk_dicts.pickle", "rb"))
                        print("Done shuffling")

                    dataset = tf.data.Dataset.from_tensor_slices((spk_names, data_types))
                    dataset = dataset.repeat()
                    dataset = dataset.batch(self.gpu_num * self.hparams.num_spk_per_batch) # .repeat will not solve remainder if putting after .batch
                    dataset = dataset.map(map_fn, num_parallel_calls=40)
                    dataset.prefetch(4)
                # tfrecord
                else:
                    #TODO
                    dataset = tf.data.TFRecordDataset(self.hparams.in_dir)
                    dataset = dataset.repeat()
                    dataset = dataset.batch(self.gpu_num * self.hparams.num_spk_per_batch)
                    dataset = dataset.map(tfrecord_map_fn)

                return dataset

            elif self.hparams.mode == 'infer':
                #===== single file per predict=====
                if single_predict:
                    self.fix_mel_dict = create_infer_batch(wav_files)
                    for key, val in self.fix_mel_dict.items():
                        self.keys.append(key)
                        #self.vals.append(val)
                    # size of vals: [num_file, num_window, 160, 40]
                    # self.vals=[i.tolist() for i in self.vals]
                    if len(val) == 0:
                        return None
                    else:
                        dataset = tf.data.Dataset.from_tensor_slices([val.tolist()])
                #===== multi file per predict=====
                else:
                    #self.fix_mel_lengths = [len(v) for v in self.vals]
                    #flat_vals=[vv.tolist() for v in self.vals for vv in v]
                    #dataset = tf.data.Dataset.from_tensor_slices(flat_vals)
                    dataset = tf.data.Dataset.from_generator(generate_infer_batch, tf.float32, tf.TensorShape([160, 40]))
                    dataset = dataset.batch(self.gpu_num * self.hparams.num_spk_per_batch)
                return dataset

        return input_fn

    def get_model_fn(self):
        def model_fn(features, labels, mode):
            if mode == tf.estimator.ModeKeys.TRAIN:
                # Input Batch of [N*M(self.batch_size), total_frames, 40(spectrogram_channel)]
                # Target Batch of [N*M(self.batch_size)]

                #input_batch = tf.placeholder(dtype=tf.float32, shape=[None, None, self.hparams.spectrogram_scale], name="input_batch")
                #target_batch = tf.placeholder(dtype=tf.int32, shape=[None], name="target_batch")

                model=Model(self.hparams)
                norm_out=model(features)
                #print_dim=tf.print('=====norm shape=====', tf.shape(norm_out))
                #with tf.control_dependencies([print_dim]):
                #total_loss, total_loss_summary=self._cal_loss(norm_out, labels)
                total_loss, total_loss_summary=self._efficient_cal_loss(norm_out, labels)
                train_op, clipped_grad_and_vars=self._optimize(total_loss)
                grad_norms=[tf.norm(grad_and_var[0]) for grad_and_var in clipped_grad_and_vars if grad_and_var[0] is not None]
                tf.summary.histogram('gradient_norm', grad_norms)
                all_summary=tf.summary.merge_all()
                train_hook_list = []
                train_tensors_log = {'loss': total_loss,
                                     'global_step': self.global_step}
                checkpoint_hook = tf.train.CheckpointSaverHook(self.hparams.ckpt_dir, save_steps=self.hparams.checkpoint_freq)
                summary_hook = tf.train.SummarySaverHook(save_steps=self.hparams.summary_freq, output_dir=self.hparams.ckpt_dir, summary_op=all_summary)
                #profile_hook = TxtProfilerHook(tf.get_default_graph(), save_steps=self.hparams.profile_freq, output_dir=self.hparams.ckpt_dir, suffix=hvd.rank() if self.hparams.use_horovod else 0)
                #timeline_hook = tf.train.ProfilerHook(save_steps=5, show_memory=True)

                # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states from
                # rank 0 to all other processes. This is necessary to ensure consistent
                # initialization of all workers when training is started with random weights or
                # restored from a checkpoint.
                train_hook_list.append(tf.train.LoggingTensorHook(
                            tensors=train_tensors_log, every_n_iter=1))
                if not self.hparams.use_horovod or hvd.rank()==0:
                    train_hook_list.append(checkpoint_hook)
                    #train_hook_list.append(timeline_hook)
                #train_hook_list.append(profile_hook)
                train_hook_list.append(summary_hook)
                if self.hparams.use_horovod:
                    train_hook_list.append(hvd.BroadcastGlobalVariablesHook(0))
                print("Model build")
                return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss,
                                                      train_op=train_op, training_hooks=train_hook_list)

            elif mode == tf.estimator.ModeKeys.EVAL:
                pass
            elif mode == tf.estimator.ModeKeys.PREDICT:
                #input_batch = tf.placeholder(dtype=tf.float32, shape=[640, None, self.hparams.spectrogram_scale], name="input_batch")
                model=Model(self.hparams)
                norm_out=model(features)
                predictions = { "dvector": norm_out}
                #===== single file per predict=====
                if single_predict:
                    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
                #===== multiple file per predict=====
                else:
                    predict_hook_list=[]
                    #batch_size=self.gpu_num * self.hparams.num_spk_per_batch
                    #total_frames=functools.reduce(lambda a,b: len(a)+len(b), self.vals)
                    #predict_tensors_log = {'predict_progress': self.global_step/(total_frames/batch_size)}
                    #predict_hook_list.append(tf.train.LoggingTensorHook(
                    #            tensors=predict_tensors_log, every_n_iter=1))
                    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, prediction_hooks=predict_hook_list)
        return model_fn

    # 1. calc centroids
    # 2. tile centroids from 64 to 640
    # 3. minus utt and scale to get the CHANGED centroid that specidif to each utt
    # 4. normalize utt and centroids and changed centroids
    # 5. utt 640 pointwise mul changed centroids 640, and sum, get changed cos
    # 6. utt 640 matmul centroids 64, get 640*64 cos
    # 7. manipulate matrix to get the changed cos 640*64 and mask
    # 8. combine cos and changed cos
    def _efficient_cal_loss(self, norm_out, labels):
        with tf.variable_scope("loss"):
            if self.hparams.loss_type == "softmax":
                # sim_mat has shape of [self.batch_size, num_spk] [640, 64]
                with tf.variable_scope("cos_params", reuse=tf.AUTO_REUSE):
                    w = tf.Variable(10.0, name="scale_weight")
                    bias = tf.Variable(-5.0, name="scale_bias")
                    w=tf.clip_by_value(w, 0.0, 1000.0)

                #1,2 
                def get_centroid_transform():
                    mat=np.zeros((self.hparams.num_spk_per_batch, self.hparams.num_spk_per_batch*self.hparams.num_utt_per_batch))
                    for i in range(self.hparams.num_spk_per_batch):
                        for j in range(self.hparams.num_utt_per_batch):
                            mat[i][i*self.hparams.num_utt_per_batch+j]=1/self.hparams.num_utt_per_batch
                    return tf.convert_to_tensor(mat, dtype=tf.float32)

                def get_changed_centroid_transform():
                    mat=np.zeros((self.hparams.num_spk_per_batch*self.hparams.num_utt_per_batch, self.hparams.num_spk_per_batch*self.hparams.num_utt_per_batch))
                    for i in range(self.hparams.num_spk_per_batch):
                        for ii in range(self.hparams.num_utt_per_batch):
                            for j in range(self.hparams.num_utt_per_batch):
                                mat[i*self.hparams.num_utt_per_batch+ii][i*self.hparams.num_utt_per_batch+j]=1/self.hparams.num_utt_per_batch
                    return tf.convert_to_tensor(mat, dtype=tf.float32)

                centroid_transform=get_centroid_transform()
                changed_centroid_transform=get_changed_centroid_transform()
                centroids=tf.matmul(centroid_transform, norm_out)
                changed_centroids=tf.matmul(changed_centroid_transform, norm_out)

                #def cal_centroid_matrix():
                #    # centroid_idx counts from 0 to 63
                #    def cal_centroid(centroid_idx):
                #        # [10, 256]
                #        all_utts_for_spk = norm_out[centroid_idx * self.hparams.num_utt_per_batch : (centroid_idx+1) * self.hparams.num_utt_per_batch, :]
                #        centroid = tf.reduce_mean(all_utts_for_spk, 0)

                #        return centroid

                #    # [64, 256], the centroid for utt_idx will not count utt_idx
                #    centroid_mat_stack=[]
                #    changed_centroid_mat_stack=[]
                #    for i in range(self.hparams.num_spk_per_batch):
                #        cur_centroid=cal_centroid(i)
                #        centroid_mat_stack.append(cur_centroid)
                #        changed_centroid_mat_stack+=([cur_centroid]*self.hparams.num_utt_per_batch)
                #    centroid_mat = tf.stack(centroid_mat_stack, axis=0)
                #    changed_centroid_mat = tf.stack(changed_centroid_mat_stack, axis=0)

                #    return centroid_mat, changed_centroid_mat
                #centroids, changed_centroids = cal_centroid_matrix()
                #3
                changed_centroids = (changed_centroids*self.hparams.num_utt_per_batch - norm_out)/(self.hparams.num_utt_per_batch-1)
                #4
                norm_centroids=tf.nn.l2_normalize(centroids, axis=-1)
                norm_changed_centroids=tf.nn.l2_normalize(changed_centroids, axis=-1)
                #5,6
                utt_cos=tf.add(tf.multiply(w, tf.matmul(norm_out, norm_centroids, transpose_b=True)), bias)# [640, 64]
                changed_utt_cos=tf.add(tf.multiply(w, tf.reduce_sum(tf.multiply(norm_out, norm_changed_centroids), axis=-1)), bias)# [640]
                #7
                def get_pos_mask():
                    mask=np.zeros((self.hparams.num_spk_per_batch*self.hparams.num_utt_per_batch, self.hparams.num_spk_per_batch))
                    for i in range(self.hparams.num_spk_per_batch):
                        for j in range(self.hparams.num_utt_per_batch):
                            mask[i*self.hparams.num_utt_per_batch+j][i]=1
                    return tf.convert_to_tensor(mask, dtype=tf.float32)

                pos_mask=get_pos_mask()
                neg_mask=tf.ones_like(pos_mask)-pos_mask
                final_utt_cos=tf.multiply(pos_mask, tf.expand_dims(changed_utt_cos, -1))+tf.multiply(neg_mask, utt_cos)
                # tf.nn.spase_softmax_cross_entropy_with_logits [640, 64] [640]
                total_loss = tf.divide(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=final_utt_cos, labels=labels)), self.batch_size)
                total_loss_summary = tf.summary.scalar("loss", total_loss)
                return (total_loss, total_loss_summary)
        pass

    def _cal_loss(self, norm_out, labels):
        def _cal_centroid_matrix(utt_idx):
            # centroid_idx counts from 0 to 63
            # spk_id is this utt's spkid
            def cal_centroid(centroid_idx):
                # utt_idx counts from 0 to 639
                # spk_id counts from 0 to 63
                spk_id = (utt_idx // self.hparams.num_utt_per_batch)
                utt_idx_in_group = utt_idx % self.hparams.num_utt_per_batch

                # [10, 256]
                all_utts_for_spk = norm_out[centroid_idx * self.hparams.num_utt_per_batch : (centroid_idx+1) * self.hparams.num_utt_per_batch, :]
                if centroid_idx == spk_id:
                    # [10] one cold
                    mask = np.array([False if utt == utt_idx_in_group else True for utt in range(self.hparams.num_utt_per_batch)])
                    centroid = tf.reduce_mean(tf.boolean_mask(all_utts_for_spk, mask), 0)
                else:
                    centroid = tf.reduce_mean(all_utts_for_spk, 0)

                return centroid

            # [64, 256], the centroid for utt_idx will not count utt_idx
            #centroid_mat = tf.convert_to_tensor(tf.map_fn(cal_centroid, tf.range(self.hparams.num_spk_per_batch), dtype=tf.float32, parallel_iterations=20))
            centroid_mat_stack=[]
            for i in range(self.hparams.num_spk_per_batch):
                centroid_mat_stack.append(cal_centroid(i))
            centroid_mat = tf.stack(centroid_mat_stack, axis=0)

            return centroid_mat

        def _create_sim_per_utt(utt_idx, w, b):
            #utt_dvector is a tensor of shape [output_size] [256]
            utt_dvector = norm_out[utt_idx, :]
            #centroids is a tensor of shape [num_spk_per_batch, output_size] [64, 256]
            centroids = _cal_centroid_matrix(utt_idx)
            #sim_per_utt is a tensor of shape [num_spk_per_batch] [64]
            #sim_per_utt = tf_scaled_cosine_similarity(utt_dvector, centroids)
            sim_per_utt = tf_scaled_cosine_similarity_param(utt_dvector, centroids, w, b)

            # [64]
            return sim_per_utt

        #with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("loss"):
            if self.hparams.loss_type == "softmax":
                # sim_mat has shape of [self.batch_size, num_spk] [640, 64]
                #sim_mat = tf.convert_to_tensor(tf.map_fn(_create_sim_per_utt, tf.range(self.batch_size), dtype=tf.float32, parallel_iterations=20))
                with tf.variable_scope("cos_params", reuse=tf.AUTO_REUSE):
                    w = tf.Variable(10.0, name="scale_weight")
                    bias = tf.Variable(-5.0, name="scale_bias")
                    w=tf.clip_by_value(w, 0.0, 1000.0)

                sim_mat_stack = []
                for i in range(self.batch_size):
                    sim_mat_stack.append(_create_sim_per_utt(i, w, bias))
                sim_mat = tf.stack(sim_mat_stack, axis=0)
                #sim_mat_summary = tf.summary.image("sim_mat", tf.reshape(sim_mat,[1, self.batch_size, self.hparams.num_spk_per_batch, 1]))
                #eval_sim_mat_summary = tf.summary.image("eval_sim_mat", tf.reshape(sim_mat,[1, self.batch_size, self.hparams.num_spk_per_batch, 1]))

                # tf.nn.spase_softmax_cross_entropy_with_logits [640, 64] [640]
                total_loss = tf.divide(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sim_mat, labels=labels)), self.batch_size)
                #eval_total_loss = tf.divide(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sim_mat, labels=labels)), self.batch_size)
                total_loss_summary = tf.summary.scalar("loss", total_loss)
                #eval_total_loss_summary = tf.summary.scalar("eval_loss", eval_total_loss)
                return (total_loss, total_loss_summary)

            elif self.hparams.loss_type == "contrast":
                pass

            else:
                print("Loss type not supported")

    def _optimize(self, total_loss):
        #with tf.variable_scope("optimize", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("optimize"):
            #self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(self.hparams.learning_rate, self.global_step,
                                       30000000, 0.5, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate * hvd.size() if self.hparams.use_horovod else learning_rate)
            optimizer = hvd.DistributedOptimizer(optimizer) if self.hparams.use_horovod else optimizer
            if self.hparams.use_horovod:
                print("=============%d %d %d===============" % (hvd.rank(), hvd.local_rank(), hvd.size()))
            #optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
            grads_and_vars = optimizer.compute_gradients(total_loss)

            clipped_grad_and_vars = []
            for grad, var in grads_and_vars:
                if re.search("cos_params", var.name):
                    # This None case can happen bcoz we first define loss/... then define tower_1/loss, which may looking for cos_params under loss/.. scope using re.search. cos_params/scale_weight happens to be defined under loss/while which is specific to the loss.
                    if grad is not None:
                        grad = tf.clip_by_value(grad, -self.hparams.scale_clip, self.hparams.scale_clip)
                elif re.search("projection", var.name):
                    grad = tf.clip_by_value(grad, -self.hparams.lstm_proj_clip, self.hparams.lstm_proj_clip)
                else:
                    grad = tf.clip_by_norm(grad, self.hparams.l2_norm_clip)
                clipped_grad_and_vars.append((grad, var))

            train_op = optimizer.apply_gradients(clipped_grad_and_vars, global_step=self.global_step)
            #print("=====clipped grads=====")
            #for grad_var in clipped_grad_and_vars:
            #    print("%s %s \n" % (grad_var[0].name, grad_var[1].name), end='')
            return train_op, clipped_grad_and_vars

    def train(self):
        #distribution = tf.contrib.distribute.MirroredStrategy()
        #config = tf.estimator.RunConfig(
        #          train_distribute=distribution,
        #          model_dir=self.hparams.ckpt_dir)
        #tts=tf.estimator.Estimator(model_fn=self.get_model_fn(), config=config)

        #tts=tf.estimator.Estimator(model_fn=tf.contrib.estimator.replicate_model_fn(self.get_model_fn()), model_dir=self.hparams.ckpt_dir)

        #strategy = tf.contrib.distribute.OneDeviceStrategy(device='/gpu:0')
        #config = tf.estimator.RunConfig(train_distribute=strategy)

        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if self.hparams.use_horovod:
            config.gpu_options.visible_device_list = str(hvd.local_rank())
        model_dir=self.hparams.ckpt_dir if not self.hparams.use_horovod or hvd.rank() == 0 else None

        tts=tf.estimator.Estimator(model_fn=self.get_model_fn(), model_dir=model_dir, config=tf.estimator.RunConfig(session_config=config))
        print("Start to train")
        #with tf.contrib.tfprof.ProfileContext(self.hparams.ckpt_dir, dump_steps=[10]): # Core dump error
        tts.train(input_fn=self.get_input_fn(), max_steps=self.hparams.max_steps//(hvd.size() if self.hparams.use_horovod else 1))

    def predict(self):
        #self.wav_list=glob('%s/*/Wave/*.wav' % self.hparams.in_dir)
        self.wav_list=glob('%s/**/*.wav' % self.hparams.in_dir, recursive=True)
        tts=tf.estimator.Estimator(model_fn=self.get_model_fn(), model_dir=self.hparams.ckpt_dir)
        print("Start predicing")
        #===== single file per predict=====
        if single_predict:
            for wav_file in tqdm(self.wav_list):
                in_fn=self.get_input_fn([wav_file])
                try:
                    result=list(tts.predict(input_fn=in_fn))
                except ValueError:
                    #None dataset due to the too short wav caused by vad
                    continue
                dvectors=np.array([res['dvector'] for res in result])
                mean_dvector=np.mean(dvectors, axis=0)
                path_fields=wav_file.split('/')
                spkid=path_fields[-3]
                filename='%s.npy' % os.path.splitext(path_fields[-1])[0]
                spk_dir='%s/%s' % (self.hparams.out_dir, spkid)
                if not os.path.exists(spk_dir):
                    os.makedirs(spk_dir)
                npy_save_path='%s/%s' % (spk_dir, filename)
                np.save(npy_save_path, mean_dvector)
        #===== multi file per predict=====
        else:
            result=list(tts.predict(input_fn=self.get_input_fn()))
            dvectors=np.array([res['dvector'] for res in result])
            offset=0
            for key, fix_mel_len in zip(self.keys, self.fix_mel_lengths):
                mean_dvector=np.mean(dvectors[offset:offset+fix_mel_len], axis=0)
                offset+=fix_mel_len
                npy_save_path=key
                spk_dir=os.path.dirname(npy_save_path)
                if not os.path.exists(spk_dir):
                    os.makedirs(spk_dir)
                np.save(npy_save_path, mean_dvector)

parser = argparse.ArgumentParser()

# in_dir = ~/wav
parser.add_argument("--in_dir", type=str, required=True, help="input data(pickle) dir")
parser.add_argument("--out_dir", type=str, default='spkids', help="output data dir")
parser.add_argument("--ckpt_dir", type=str, required=True, help="checkpoint to save/ start with for train/inference")
parser.add_argument("--mode", default="train", choices=["train", "test", "infer"], help="setting mode for execution")
parser.add_argument('--data_types', nargs='+', default=['libri', 'vox1', 'vox2'])
parser.add_argument('--pickle_dataset', action='store_true')
parser.add_argument('--gpu_str', default='0',
                    help='Path to model checkpoint')
parser.add_argument('--tfrecord', action='store_true', help='whether to read from tfrecord')
parser.add_argument('--use_horovod', action='store_true', help='whether to use horovod')

# Saving Checkpoints, Data... etc
parser.add_argument("--max_steps", type=int, default=5000000, help="maximum steps in training")
parser.add_argument("--checkpoint_freq", type=int, default=10000, help="how often save checkpoint")
parser.add_argument("--eval_freq", type=int, default=1000, help="how often do the evaluation")
parser.add_argument("--summary_freq", type=int, default=100, help="maximum steps in training")
parser.add_argument("--profile_freq", type=int, default=3, help="maximum steps in training")

# Data
parser.add_argument("--segment_length", type=float, default=1.6, help="segment length in seconds")
parser.add_argument("--spectrogram_scale", type=int, default=40,
                                       help="scale of the input spectrogram")

# Ininitialization
parser.add_argument("--init_type", type=str, default="uniform", help="type of initializer")
parser.add_argument("--init_weight_range", type=float, default=0.1, help="initial weight ranges from -0.1 to 0.1")

# Optimization
parser.add_argument("--loss_type", default="softmax", choices=["softmax", "contrast"], help="loss type for optimization")
parser.add_argument("--optimizer", type=str, default="sgd", help="type of optimizer")
parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate")
parser.add_argument("--l2_norm_clip", type=float, default=3.0, help="L2-norm of gradient is clipped at")

# Train
parser.add_argument("--num_spk_per_batch", type=int, default= 64,
                                       help="N speakers of batch size N*M")
parser.add_argument("--num_utt_per_batch", type=int, default= 10,
                                       help="M utterances of batch size N*M")

# LSTM
parser.add_argument("--lstm_proj_clip", type=float, default=0.5, help="Gradient scale for projection node in LSTM")
parser.add_argument("--num_lstm_stacks", type=int, default=3, help="number of LSTM stacks")
parser.add_argument("--num_lstm_cells", type=int, default=768, help="number of LSTM cells")
parser.add_argument("--dim_lstm_projection", type=int, default=256, help="dimension of LSTM projection")

# Scaled Cosine similarity
parser.add_argument("--scale_clip", type=float, default=0.01, help="Gradient scale for scale values in scaled cosine similarity")

parser.add_argument("--overlap_ratio", type=float, default=0.5, help="overlaping percentage")

# Collect hparams
args = parser.parse_args()

if not args.use_horovod:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_str)

if __name__ == "__main__":
    if args.use_horovod:
        hvd.init()
    tf.logging.set_verbosity(tf.logging.INFO)
    random.seed(1234 + (hvd.rank() if args.use_horovod else 0))
    trainer=Trainer(args)
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'infer':
        trainer.predict()
