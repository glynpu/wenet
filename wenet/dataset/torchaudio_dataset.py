# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Binbin Zhang)

import argparse
import logging
import random
import codecs

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torchaudio.compliance.kaldi as kaldi
import torchaudio

from wenet.utils.common import IGNORE_ID
from wenet.dataset.wav_distortion import distort_wav_conf

def _splice(feats, left_context, right_context):
    """ Splice feature

    Args:
        feats: input feats
        left_context: left context for splice
        right_context: right context for splice

    Returns:
        Spliced feature
    """
    if left_context == 0 and right_context == 0:
        return feats
    assert (len(feats.shape) == 2)
    num_rows = feats.shape[0]
    first_frame = feats[0]
    last_frame = feats[-1]
    padding_feats = feats
    if left_context > 0:
        left_padding = np.vstack([first_frame for i in range(left_context)])
        padding_feats = np.vstack((left_padding, padding_feats))
    if right_context > 0:
        right_padding = np.vstack([last_frame for i in range(right_context)])
        padding_feats = np.vstack((padding_feats, right_padding))
    outputs = []
    for i in range(num_rows):
        splice_feats = np.hstack([
            padding_feats[i]
            for i in range(i, i + 1 + left_context + right_context)
        ])
        outputs.append(splice_feats)
    return np.vstack(outputs)


def spec_augmentation(x,
                      gauss_mask_for_t=False,
                      num_t_mask=2,
                      num_f_mask=2,
                      max_t=50,
                      max_f=10):
    ''' Deep copy x and do spec augmentation then return it
    Args:
        x: input feature, T * F 2D
        num_t_mask: number of time mask to apply
        num_f_mask: number of freq mask to apply
        max_t: max width of time mask
        max_f: max width of freq mask
    Returns:
        augmented feature
    '''
    y = np.copy(x)
    max_frames = y.shape[0]
    max_freq = y.shape[1]
    for i in range(num_t_mask):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        if gauss_mask_for_t:
            y[start:end, :] = np.random.randn(end - start, max_freq)
        else:
            y[start:end, :] = 0
    for i in range(num_f_mask):
        start = random.randint(0, max_freq - 1)
        length = random.randint(1, max_f)
        end = min(max_freq, start + length)
        y[:, start:end] = 0
    return y

def _do_waveform_distortion(waveform, distortion_methods_conf):
    r = random.uniform(0, 1)
    acc = 0.0
    for distortion_method in distortion_methods_conf:
        method_rate = distortion_method['method_rate']
        acc += method_rate
        if r < acc:
            distortion_type = distortion_method['name']
            distortion_conf = distortion_method['params']
            point_rate = distortion_method['point_rate']
            return distort_wav_conf(waveform, distortion_type, distortion_conf , point_rate)
    return waveform


# def old_do_waveform_distortion(waveform, distortion_methods_conf):
#     r = random.uniform(0, 1)
#     if r < 0.5:
#         return distort_wav('jag_distortion', waveform, 0.6)
#     elif r >= 0.5 and r < 0.75:
#         return distort_wav('max_distortion', waveform, 0.05)
#     else:
#         return distort_wav('poly_distortion', waveform, 0.5)

def _extract_feature(batch, wav_distortion_conf, feature_extraction_conf):
    keys = []
    feats = []
    lengths = []
    wav_distortion_rate = wav_distortion_conf['wav_distortion_rate']
    distortion_methods_conf = wav_distortion_conf['distortion_methods']

    for i, x in enumerate(batch):
        try:
            waveform, sample_rate = torchaudio.load_wav(x[1])
            if wav_distortion_rate > 0.0:
                r = random.uniform(0, 1)
                if r < wav_distortion_rate:
                    waveform = waveform.detach().numpy()
                    waveform = _do_waveform_distortion(waveform, distortion_methods_conf)
                    waveform = torch.from_numpy(waveform)
            mat = kaldi.fbank(
                waveform,
                num_mel_bins=feature_extraction_conf['mel_bins'],
                frame_length=feature_extraction_conf['frame_length'],
                frame_shift=feature_extraction_conf['frame_shift'],
                dither=0.0,
                energy_floor=0.0
            )
            mat = mat.detach().numpy()
            feats.append(mat)
            keys.append(x[0])
            lengths.append(mat.shape[0])
        except (Exception) as e:
            print(e)
            logging.warn('read utterance {} error'.format(x[0]))
            pass
    # Sort it because sorting is required in pack/pad operation
    order = np.argsort(lengths)[::-1]
    sorted_keys = [keys[i] for i in order]
    sorted_feats = [feats[i] for i in order]
    labels = [x[2].split() for x in batch]
    labels = [np.fromiter(map(int, x), dtype=np.int32) for x in labels]
    sorted_labels = [labels[i] for i in order]
    return sorted_keys, sorted_feats, sorted_labels


class TorchAudioCollateFunc(object):
    ''' Collate function for AudioDataset
    '''
    def __init__(self,
                 feature_extraction_conf,
                 wav_distortion_conf,
                 subsampling_factor=1,
                 left_context=0,
                 right_context=0,
                 spec_aug=False,
                 feature_dither=0.0):
        '''
        Args:
            subsampling_factor: subsampling_factor for feature
            left_context: left context for splice feature
            right_context: right_context for splice feature
        '''
        self.wav_distortion_conf = wav_distortion_conf
        self.feature_extraction_conf = feature_extraction_conf
        self.subsampling_factor = subsampling_factor
        self.left_context = left_context
        self.right_context = right_context
        self.spec_aug = spec_aug
        self.feature_dither = feature_dither

    def __call__(self, batch):
        assert (len(batch) == 1)
        keys, xs, ys = _extract_feature(batch[0], self.wav_distortion_conf, self.feature_extraction_conf)
        train_flag = True
        if ys is None:
            train_flag = False
        # add dither d ~ (-a, a) on fbank feature
        # a ~ (0, 0.5)
        if self.feature_dither != 0.0:
            a = random.uniform(0, self.feature_dither)
            xs = [x + (np.random.random_sample(x.shape) - 0.5) * a for x in xs]
        if self.spec_aug:
            xs = [spec_augmentation(x) for x in xs]
        # optional splice
        if self.left_context != 0 or self.right_context != 0:
            xs = [
                _splice(x, self.left_context, self.right_context) for x in xs
            ]
        # optional subsampling
        if self.subsampling_factor > 1:
            xs = [x[::self.subsampling_factor] for x in xs]

        # padding
        xs_lengths = torch.from_numpy(
            np.array([x.shape[0] for x in xs], dtype=np.int32))

        # pad_sequence will FAIL in case xs is empty
        if len(xs) > 0:
            xs_pad = pad_sequence([torch.from_numpy(x).float() for x in xs],
                                  True, 0)
        else:
            xs_pad = torch.Tensor(xs)
        if train_flag:
            ys_lengths = torch.from_numpy(
                np.array([y.shape[0] for y in ys], dtype=np.int32))
            if len(ys) > 0:
                ys_pad = pad_sequence([torch.from_numpy(y).int() for y in ys],
                                      True, IGNORE_ID)
            else:
                ys_pad = torch.Tensor(ys)
        else:
            ys_pad = None
            ys_lengths = None
        return keys, xs_pad, ys_pad, xs_lengths, ys_lengths


class TorchAudioDataset(Dataset):
    def __init__(self,
                 data_file,
                 max_length=10240,
                 min_length=0,
                 batch_type='static',
                 batch_size=1,
                 max_frames_in_batch=0,
                 sort=True):
        ''' Args:
            data_file: input data file
            max_length: drop utterance which is greater than max_length(ms)
            min_length: drop utterance which is less than min_length(ms)
            batch_type: static or dynamic, see max_frames_in_batch(dynamic)
            batch_size: number of utterances in a batch,
               it's for static batch size.
            max_frames_in_batch: max feature frames in a batch,
               when batch_type is dynamic, it's for dynamic batch size.
               Then batch_size is ignored, we will keep filling the
               batch until the total frames in batch up to max_frames_in_batch.
            sort: whether to sort all data, so the utterance with the same
               length could be filled in a same batch.
        '''
        assert batch_type in ['static', 'dynamic']
        data = []
        # Plain text data file, it contains following seven fields in every
        #
        # line, split by \t
        # utt:utt1
        # feat:tmp/data/file1.wav
        # feat_shape:5.042(in ms)
        # text:i love you
        # token: i <space> l o v e <space> y o u
        # tokenid: int id of this token
        # token_shape:M,N    # M is the number of token, N is vocab size
        #
        # Open in utf8 mode since meet encoding problem
        with codecs.open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                arr = line.strip().split('\t')
                if len(arr) != 7:
                    continue
                key = arr[0].split(':')[1]
                wav_path = ':'.join(arr[1].split(':')[1:])
                duration = int(float(arr[2].split(':')[1]) * 1000)  # to milliseconds
                tokenid = arr[5].split(':')[1]
                output_dim = int(arr[6].split(':')[1].split(',')[1])
                data.append((key, wav_path, duration, tokenid))
                self.output_dim = output_dim
        if sort:
            data = sorted(data, key=lambda x: x[2])
        valid_data = []
        for i in range(len(data)):
            length = data[i][2]
            if length > max_length or length < min_length:
                # logging.warn('ignore utterance {} feature {}'.format(
                #     data[i][0], length))
                pass
            else:
                valid_data.append(data[i])
        data = valid_data
        self.minibatch = []
        num_data = len(data)
        # Dynamic batch size
        if batch_type == 'dynamic':
            assert (max_frames_in_batch > 0)
            self.minibatch.append([])
            num_frames_in_batch = 0
            for i in range(num_data):
                length = data[i][2]
                num_frames_in_batch += length
                if num_frames_in_batch > max_frames_in_batch:
                    self.minibatch.append([])
                    num_frames_in_batch = length
                self.minibatch[-1].append((data[i][0], data[i][1], data[i][3]))
        # Static batch size
        else:
            cur = 0
            while cur < num_data:
                end = min(cur + batch_size, num_data)
                item = []
                for i in range(cur, end):
                    item.append((data[i][0], data[i][1], data[i][3]))
                self.minibatch.append(item)
                cur = end

    def __len__(self):
        return len(self.minibatch)

    def __getitem__(self, idx):
        return self.minibatch[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help='input data file')
    args = parser.parse_args()
    print(args.data_file)
    dataset = TorchAudioDataset(args.data_file,
                                max_length=30000,
                                min_length=0,
                                batch_size=1,
                                max_frames_in_batch=4096,
                                sort=True)
    collate_func = TorchAudioCollateFunc(subsampling_factor=1,
                                         left_context=0,
                                         right_context=0)
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=1,
                             collate_fn=collate_func)
    for i, batch in enumerate(data_loader):
        print(i, batch)
        print(batch[1].shape)
