#!/usr/bin/env python
"""
model.py

Self defined model definition.
Usage:

"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np

import torch
import torch.nn as torch_nn
import torchaudio
import torch.nn.functional as torch_nn_func

import sandbox.block_nn as nii_nn
import sandbox.util_frontend as nii_front_end
import core_scripts.other_tools.debug as nii_debug
import core_scripts.data_io.seq_info as nii_seq_tk


__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

##############
## OC-Softmax loss (Zhang et al., IEEE SPL 2021)
##############

class OCSoftmax(torch_nn.Module):
    """One-Class Softmax loss for spoofing detection.

    Learns a compact bonafide boundary in embedding space with angular margins.
    Bonafide samples are pushed toward a learned center; spoof samples are
    pushed away. Operates on L2-normalized embeddings (cosine similarity).

    Note: This uses label convention bonafide=1, spoof=0 (our protocol_parse
    convention). The original paper uses bonafide=0, spoof=1 — the margin
    assignments are swapped accordingly.
    """
    def __init__(self, feat_dim=64, r_real=0.9, r_fake=0.2, alpha=20.0):
        super(OCSoftmax, self).__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.center = torch_nn.Parameter(torch.randn(1, feat_dim))
        torch_nn.init.kaiming_uniform_(self.center)
        self.softplus = torch_nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: embeddings (batch, feat_dim)
            labels: (batch,) — 1=bonafide, 0=spoof
        Returns:
            loss: scalar
            scores: (batch,) cosine similarity to center (higher = bonafide)
        """
        # L2 normalize
        w = torch_nn_func.normalize(self.center, p=2, dim=1)
        x = torch_nn_func.normalize(x, p=2, dim=1)

        # Cosine similarity to bonafide center
        scores = x @ w.transpose(0, 1)
        scores = scores.squeeze(1)

        # Apply angular margins
        # bonafide (label=1): want high similarity → penalize if score < r_real
        # spoof (label=0): want low similarity → penalize if score > r_fake
        is_bonafide = (labels > 0.5).float()
        is_spoof = 1.0 - is_bonafide

        # For bonafide: loss term = softplus(alpha * (r_real - score))
        #   → penalizes when score < r_real
        # For spoof: loss term = softplus(alpha * (score - r_fake))
        #   → penalizes when score > r_fake
        margin_scores = is_bonafide * (self.r_real - scores) + \
                        is_spoof * (scores - self.r_fake)
        loss = self.softplus(self.alpha * margin_scores).mean()

        return loss, scores


##############
## SE Attention Block (Hu et al., CVPR 2018; Ma & Liang, ICASSP 2021)
##############

class SEBlock2D(torch_nn.Module):
    """Squeeze-and-Excitation block for 2D feature maps (channel attention).

    Learns per-channel importance weights via global average pooling →
    FC → ReLU → FC → Sigmoid, then rescales input channels accordingly.
    """
    def __init__(self, channels, reduction=8):
        super(SEBlock2D, self).__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = torch_nn.AdaptiveAvgPool2d(1)
        self.excite = torch_nn.Sequential(
            torch_nn.Linear(channels, mid, bias=False),
            torch_nn.ReLU(inplace=True),
            torch_nn.Linear(mid, channels, bias=False),
            torch_nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.squeeze(x).view(b, c)
        w = self.excite(w).view(b, c, 1, 1)
        return x * w


##############
## util
##############

## Updated protocol parsing function to be compatible with ASVspoof5-style protocol file
def protocol_parse(protocol_filepath):
    """
    Parse ASVspoof5 protocol file.

    Returns:
        dict: utt_id -> label
              bonafide = 1
              spoof    = 0
    """

    data_buffer = {}

    try:
        with open(protocol_filepath, "r") as f:
            for line in f:

                parts = line.strip().split()

                # skip malformed lines
                if len(parts) < 10:
                    continue

                utt_id = parts[1]   # utterance id
                label  = parts[-2]  # bonafide / spoof (second-to-last col; last col is always "-")

                if label == "bonafide":
                    data_buffer[utt_id] = 1
                else:
                    data_buffer[utt_id] = 0

    except OSError:
        print("Skip loading protocol file")

    print(f"[INFO] Loaded {len(data_buffer)} protocol entries")

    return data_buffer

##############
## FOR MODEL
##############

class Model(torch_nn.Module):
    """ Model definition
    """
    def __init__(self, in_dim, out_dim, args, prj_conf, mean_std=None):
        super(Model, self).__init__()

        ##### required part, no need to change #####

        # mean std of input and output
        in_m, in_s, out_m, out_s = self.prepare_mean_std(in_dim,out_dim,\
                                                         args, mean_std)
        self.input_mean = torch_nn.Parameter(in_m, requires_grad=False)
        self.input_std = torch_nn.Parameter(in_s, requires_grad=False)
        self.output_mean = torch_nn.Parameter(out_m, requires_grad=False)
        self.output_std = torch_nn.Parameter(out_s, requires_grad=False)
        
        # a flag for debugging (by default False)
        #self.model_debug = False
        #self.validation = False
        #####
        
        ####
        # on input waveform and output target
        ####
        # Load protocol and prepare the target data for network training
        self.protocol_parser = {}
        for protocol_file in prj_conf.optional_argument:
            if protocol_file:
                self.protocol_parser.update(protocol_parse(protocol_file))
        _counts = {0: 0, 1: 0}
        for v in self.protocol_parser.values(): _counts[v] += 1
        print(f"[INFO] Label counts — bonafide: {_counts[1]}, spoof: {_counts[0]}")

        # Mixup alpha (0 = disabled)
        self.mixup_alpha = getattr(args, 'mixup_alpha', 0.0)

        # OC-Softmax mode
        self.use_oc_softmax = getattr(args, 'oc_softmax', False)

        # SE attention blocks in LCNN
        self.use_se_attention = getattr(args, 'se_attention', False)

        # Frequency feature masking (0 = disabled)
        self.freq_mask_width = getattr(args, 'freq_mask_width', 0)
        if self.use_oc_softmax:
            self.oc_loss = OCSoftmax(
                feat_dim=getattr(args, 'emb_dim', 64),
                r_real=getattr(args, 'oc_r_real', 0.9),
                r_fake=getattr(args, 'oc_r_fake', 0.2),
                alpha=getattr(args, 'oc_alpha', 20.0)
            )

        # Working sampling rate
        #  torchaudio may be used to change sampling rate
        self.m_target_sr = 16000

        ####
        # optional configs (not used)
        ####                
        # re-sampling (optional)
        #self.m_resampler = torchaudio.transforms.Resample(
        #    prj_conf.wav_samp_rate, self.m_target_sr)

        # vad (optional)
        #self.m_vad = torchaudio.transforms.Vad(sample_rate = self.m_target_sr)
        
        # flag for balanced class (temporary use)
        #self.v_flag = 1

        ####
        # front-end configuration
        #  multiple front-end configurations may be used
        #  by default, use a single front-end
        ####    
        # frame shift (number of waveform points)
        self.frame_hops = [160]
        # frame length
        self.frame_lens = [320]
        # FFT length
        self.fft_n = [1024]

        # LFCC dim (base component)
        self.lfcc_dim = [20]
        self.lfcc_with_delta = True
        # only uses [0, 0.5 * Nyquist_freq range for LFCC]
        self.lfcc_max_freq = 0.5 


        # window type
        self.win = torch.hann_window
        # floor in log-spectrum-amplitude calculating (not used)
        self.amp_floor = 0.00001
        
        # number of frames to be kept for each trial
        # no truncation
        self.v_truncate_lens = [None for x in self.frame_hops]


        # number of sub-models (by default, a single model)
        self.v_submodels = len(self.frame_lens)        

        # dimension of embedding vectors
        # OC-Softmax needs higher dim for angular margin; default 1 for BCE
        if self.use_oc_softmax:
            self.v_emd_dim = getattr(args, 'emb_dim', 64)
        else:
            self.v_emd_dim = 1

        ####
        # create network
        ####
        # 1st part of the classifier
        self.m_transform = []
        # 
        self.m_before_pooling = []
        # 2nd part of the classifier
        self.m_output_act = []
        # front-end
        self.m_frontend = []

        # it can handle models with multiple front-end configuration
        # by default, only a single front-end
        for idx, (trunc_len, fft_n, lfcc_dim) in enumerate(zip(
                self.v_truncate_lens, self.fft_n, self.lfcc_dim)):
            
            fft_n_bins = fft_n // 2 + 1
            if self.lfcc_with_delta:
                lfcc_dim = lfcc_dim * 3
            
            # Build LCNN layers, optionally with SE attention after MaxPool
            se = self.use_se_attention
            layers = [
                torch_nn.Conv2d(1, 64, [5, 5], 1, padding=[2, 2]),
                nii_nn.MaxFeatureMap2D(),
                torch.nn.MaxPool2d([2, 2], [2, 2]),
            ]
            if se: layers.append(SEBlock2D(32))  # 32ch after MFM

            layers += [
                torch_nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
                nii_nn.MaxFeatureMap2D(),
                torch_nn.BatchNorm2d(32, affine=False),
                torch_nn.Conv2d(32, 96, [3, 3], 1, padding=[1, 1]),
                nii_nn.MaxFeatureMap2D(),

                torch.nn.MaxPool2d([2, 2], [2, 2]),
                torch_nn.BatchNorm2d(48, affine=False),
            ]
            if se: layers.append(SEBlock2D(48))  # 48ch after MFM

            layers += [
                torch_nn.Conv2d(48, 96, [1, 1], 1, padding=[0, 0]),
                nii_nn.MaxFeatureMap2D(),
                torch_nn.BatchNorm2d(48, affine=False),
                torch_nn.Conv2d(48, 128, [3, 3], 1, padding=[1, 1]),
                nii_nn.MaxFeatureMap2D(),

                torch.nn.MaxPool2d([2, 2], [2, 2]),
            ]
            if se: layers.append(SEBlock2D(64))  # 64ch after MFM

            layers += [
                torch_nn.Conv2d(64, 128, [1, 1], 1, padding=[0, 0]),
                nii_nn.MaxFeatureMap2D(),
                torch_nn.BatchNorm2d(64, affine=False),
                torch_nn.Conv2d(64, 64, [3, 3], 1, padding=[1, 1]),
                nii_nn.MaxFeatureMap2D(),
                torch_nn.BatchNorm2d(32, affine=False),

                torch_nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
                nii_nn.MaxFeatureMap2D(),
                torch_nn.BatchNorm2d(32, affine=False),
                torch_nn.Conv2d(32, 64, [3, 3], 1, padding=[1, 1]),
                nii_nn.MaxFeatureMap2D(),
                torch_nn.MaxPool2d([2, 2], [2, 2]),
            ]
            if se: layers.append(SEBlock2D(32))  # 32ch after MFM

            layers.append(torch_nn.Dropout(0.7))

            self.m_transform.append(torch_nn.Sequential(*layers))

            self.m_before_pooling.append(
                torch_nn.Sequential(
                    nii_nn.BLSTMLayer((lfcc_dim//16) * 32, (lfcc_dim//16) * 32),
                    nii_nn.BLSTMLayer((lfcc_dim//16) * 32, (lfcc_dim//16) * 32)
                )
            )

            self.m_output_act.append(
                torch_nn.Linear((lfcc_dim // 16) * 32, self.v_emd_dim)
            )
            
            self.m_frontend.append(
                nii_front_end.LFCC(self.frame_lens[idx],
                                   self.frame_hops[idx],
                                   self.fft_n[idx],
                                   self.m_target_sr,
                                   self.lfcc_dim[idx],
                                   with_energy=True,
                                   max_freq = self.lfcc_max_freq)
            )

        self.m_frontend = torch_nn.ModuleList(self.m_frontend)
        self.m_transform = torch_nn.ModuleList(self.m_transform)
        self.m_output_act = torch_nn.ModuleList(self.m_output_act)
        self.m_before_pooling = torch_nn.ModuleList(self.m_before_pooling)
        
        # output 

        # done
        return
    
    def prepare_mean_std(self, in_dim, out_dim, args, data_mean_std=None):
        """ prepare mean and std for data processing
        This is required for the Pytorch project, but not relevant to this code
        """
        if data_mean_std is not None:
            in_m = torch.from_numpy(data_mean_std[0])
            in_s = torch.from_numpy(data_mean_std[1])
            out_m = torch.from_numpy(data_mean_std[2])
            out_s = torch.from_numpy(data_mean_std[3])
            if in_m.shape[0] != in_dim or in_s.shape[0] != in_dim:
                print("Input dim: {:d}".format(in_dim))
                print("Mean dim: {:d}".format(in_m.shape[0]))
                print("Std dim: {:d}".format(in_s.shape[0]))
                print("Input dimension incompatible")
                sys.exit(1)
            if out_m.shape[0] != out_dim or out_s.shape[0] != out_dim:
                print("Output dim: {:d}".format(out_dim))
                print("Mean dim: {:d}".format(out_m.shape[0]))
                print("Std dim: {:d}".format(out_s.shape[0]))
                print("Output dimension incompatible")
                sys.exit(1)
        else:
            in_m = torch.zeros([in_dim])
            in_s = torch.ones([in_dim])
            out_m = torch.zeros([out_dim])
            out_s = torch.ones([out_dim])
            
        return in_m, in_s, out_m, out_s
        
    def normalize_input(self, x):
        """ normalizing the input data
        This is required for the Pytorch project, but not relevant to this code
        """
        return (x - self.input_mean) / self.input_std

    def normalize_target(self, y):
        """ normalizing the target data
        This is required for the Pytorch project, but not relevant to this code
        """
        return (y - self.output_mean) / self.output_std

    def denormalize_output(self, y):
        """ denormalizing the generated output from network
        This is required for the Pytorch project, but not relevant to this code
        """
        return y * self.output_std + self.output_mean


    def _front_end(self, wav, idx, trunc_len, datalength):
        """ simple fixed front-end to extract features
        
        input:
        ------
          wav: waveform
          idx: idx of the trial in mini-batch
          trunc_len: number of frames to be kept after truncation
          datalength: list of data length in mini-batch

        output:
        -------
          x_sp_amp: front-end featues, (batch, frame_num, frame_feat_dim)
        """
        
        with torch.no_grad():
            x_sp_amp = self.m_frontend[idx](wav.squeeze(-1))

        # return
        return x_sp_amp

    def _apply_freq_mask(self, x_sp_amp):
        """Apply frequency feature masking to LFCC spectrogram during training.

        Randomly selects one of three modes per sample in the batch:
          - Low-frequency mask: zero out bins [0, w)
          - High-frequency mask: zero out bins [F-w, F)
          - Random band mask: zero out bins [f0, f0+w)
        where w ~ Uniform(1, freq_mask_width) and F = total freq bins.

        Args:
            x_sp_amp: (batch, frame_num, freq_bins)
        Returns:
            masked x_sp_amp (same shape)
        """
        batch, frames, F = x_sp_amp.shape
        max_w = self.freq_mask_width

        for i in range(batch):
            w = np.random.randint(1, max_w + 1)
            mode = np.random.randint(3)
            if mode == 0:
                # Low-frequency mask
                x_sp_amp[i, :, :w] = 0
            elif mode == 1:
                # High-frequency mask
                x_sp_amp[i, :, F - w:] = 0
            else:
                # Random band mask
                f0 = np.random.randint(0, F - w + 1)
                x_sp_amp[i, :, f0:f0 + w] = 0

        return x_sp_amp

    def _compute_embedding(self, x, datalength):
        """ definition of forward method 
        Assume x (batchsize, length, dim)
        Output x (batchsize * number_filter, output_dim)
        """
        # resample if necessary
        #x = self.m_resampler(x.squeeze(-1)).unsqueeze(-1)
        
        # number of sub models
        batch_size = x.shape[0]

        # buffer to store output scores from sub-models
        output_emb = torch.zeros([batch_size * self.v_submodels, 
                                  self.v_emd_dim], 
                                  device=x.device, dtype=x.dtype)
        
        # compute scores for each sub-models
        for idx, (fs, fl, fn, trunc_len, m_trans, m_be_pool, m_output) in \
            enumerate(
                zip(self.frame_hops, self.frame_lens, self.fft_n, 
                    self.v_truncate_lens, self.m_transform, 
                    self.m_before_pooling, self.m_output_act)):
            
            # extract front-end feature
            x_sp_amp = self._front_end(x, idx, trunc_len, datalength)

            # frequency feature masking (training only)
            if self.training and self.freq_mask_width > 0:
                x_sp_amp = self._apply_freq_mask(x_sp_amp)

            # compute scores
            #  1. unsqueeze to (batch, 1, frame_length, fft_bin)
            #  2. compute hidden features
            hidden_features = m_trans(x_sp_amp.unsqueeze(1))

            #  3. (batch, channel, frame//N, feat_dim//N) ->
            #     (batch, frame//N, channel * feat_dim//N)
            #     where N is caused by conv with stride
            hidden_features = hidden_features.permute(0, 2, 1, 3).contiguous()
            frame_num = hidden_features.shape[1]
            hidden_features = hidden_features.view(batch_size, frame_num, -1)

            #  4. pooling
            #  4. pass through LSTM then summing
            hidden_features_lstm = m_be_pool(hidden_features)

            #  5. pass through the output layer
            tmp_emb = m_output((hidden_features_lstm + hidden_features).mean(1))
            
            output_emb[idx * batch_size : (idx+1) * batch_size] = tmp_emb

        return output_emb

    def _compute_embedding_mixup(self, x, datalength, target_vec):
        """Compute embeddings with mixup applied at the LFCC feature level.

        Draws lambda from Beta(alpha, alpha), shuffles the batch, and blends
        LFCC spectrograms and targets before feeding into LCNN conv layers.
        """
        batch_size = x.shape[0]

        # Draw mixup coefficient
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        # Ensure lam >= 0.5 so the "primary" sample dominates
        lam = max(lam, 1.0 - lam)

        # Random permutation for pairing
        perm = torch.randperm(batch_size, device=x.device)

        # Mix targets
        target_mixed = lam * target_vec + (1.0 - lam) * target_vec[perm]

        # Compute embeddings with mixed LFCC features
        output_emb = torch.zeros(
            [batch_size * self.v_submodels, self.v_emd_dim],
            device=x.device, dtype=x.dtype)

        for idx, (fs, fl, fn, trunc_len, m_trans, m_be_pool, m_output) in \
            enumerate(
                zip(self.frame_hops, self.frame_lens, self.fft_n,
                    self.v_truncate_lens, self.m_transform,
                    self.m_before_pooling, self.m_output_act)):

            # Extract LFCC features
            x_sp_amp = self._front_end(x, idx, trunc_len, datalength)

            # Apply mixup at the LFCC spectrogram level
            x_sp_amp = lam * x_sp_amp + (1.0 - lam) * x_sp_amp[perm]

            # Same pipeline as _compute_embedding from here
            hidden_features = m_trans(x_sp_amp.unsqueeze(1))
            hidden_features = hidden_features.permute(0, 2, 1, 3).contiguous()
            frame_num = hidden_features.shape[1]
            hidden_features = hidden_features.view(batch_size, frame_num, -1)
            hidden_features_lstm = m_be_pool(hidden_features)
            tmp_emb = m_output((hidden_features_lstm + hidden_features).mean(1))

            output_emb[idx * batch_size : (idx+1) * batch_size] = tmp_emb

        return output_emb, target_mixed

    def _compute_score(self, feature_vec, inference=False):
        """
        """
        if self.use_oc_softmax:
            # OC-Softmax: return raw embeddings (loss handles scoring)
            return feature_vec
        # BCE path: feature_vec is [batch * submodel, 1]
        if inference:
            return feature_vec.squeeze(1)
        else:
            return torch.sigmoid(feature_vec).squeeze(1)


    def _get_target(self, filenames):
        try:
            return [self.protocol_parser[x] for x in filenames]
        except KeyError:
            print("Cannot find target data for %s" % (str(filenames)))
            sys.exit(1)

    def _get_target_eval(self, filenames):
        """ retrieve the target label for a trial from protocol if available
        """
        return [self.protocol_parser[x] if x in self.protocol_parser else -1 \
                for x in filenames]

    def forward(self, data_in, fileinfo=None):

        # unpack framework input
        if fileinfo is None and isinstance(data_in, (list, tuple)):
            x, fileinfo = data_in
        else:
            x = data_in

        # ----- SAFE metadata handling -----
        if fileinfo is not None:
            filenames = [nii_seq_tk.parse_filename(y) for y in fileinfo]
            datalength = [nii_seq_tk.parse_length(y) for y in fileinfo]
        else:
            # fallback for ASVspoof5 loader
            batch_size = x.shape[0]
            filenames = ["dummy"] * batch_size
            datalength = [x.shape[1]] * batch_size

        #with torch.no_grad():
        #    vad_waveform = self.m_vad(x.squeeze(-1))
        #    vad_waveform = self.m_vad(torch.flip(vad_waveform, dims=[1]))
        #    if vad_waveform.shape[-1] > 0:
        #        x = torch.flip(vad_waveform, dims=[1]).unsqueeze(-1)
        #    else:
        #        pass
        
        if self.training:

            # target
            target = self._get_target(filenames)
            target_vec = torch.as_tensor(target, device=x.device).float()

            # Mixup: blend LFCC features and targets within the batch
            if self.mixup_alpha > 0:
                feature_vec, target_vec = self._compute_embedding_mixup(
                    x, datalength, target_vec)
            else:
                feature_vec = self._compute_embedding(x, datalength)

            if self.use_oc_softmax:
                # OC-Softmax: compute loss inside model (has learnable center)
                embeddings = self._compute_score(feature_vec)  # raw embeddings
                target_vec = target_vec.repeat(self.v_submodels)
                return [embeddings, target_vec, True]
            else:
                scores = self._compute_score(feature_vec)
                target_vec = target_vec.repeat(self.v_submodels)
                return [scores, target_vec, True]

        else:
            feature_vec = self._compute_embedding(x, datalength)
            batch_size = len(filenames)

            if self.use_oc_softmax:
                # OC-Softmax inference: cosine similarity to bonafide center
                embeddings = self._compute_score(feature_vec)  # (batch*submodels, emb_dim)
                emb_per_sub = embeddings.view(self.v_submodels, batch_size, -1)
                # Average embeddings across submodels, then score
                avg_emb = emb_per_sub.mean(dim=0)  # (batch, emb_dim)
                w = torch_nn_func.normalize(self.oc_loss.center, p=2, dim=1)
                x_norm = torch_nn_func.normalize(avg_emb, p=2, dim=1)
                scores_per_file = (x_norm @ w.transpose(0, 1)).squeeze(1)
            else:
                scores = self._compute_score(feature_vec, True)
                # scores layout: [submodel_0 * batch, submodel_1 * batch, ...]
                # reshape to [v_submodels, batch_size] and mean across submodels
                scores_per_file = scores.view(self.v_submodels, batch_size).mean(dim=0)

            target = self._get_target_eval(filenames)
            for fname, tgt, score in zip(filenames, target, scores_per_file):
                print("Output, %s, %d, %f" % (fname, tgt, score.item()))
            # don't write output score as a single file
            return None


class Loss():
    """ Wrapper to define loss function
    """
    def __init__(self, args):
        """
        """
        self.use_oc_softmax = getattr(args, 'oc_softmax', False)
        self.label_smoothing = getattr(args, 'label_smoothing', 0.0)
        self.pos_weight = getattr(args, 'pos_weight', -1.0)
        # OC-Softmax loss is computed via model.oc_loss (has learnable params)
        # so it needs to be referenced here. It will be set by the framework
        # after model initialization — we store a reference via set_model().
        self.oc_loss_fn = None
        if not self.use_oc_softmax:
            if self.pos_weight > 0:
                self.m_loss = torch_nn.BCELoss(reduction='none')
            else:
                self.m_loss = torch_nn.BCELoss()

    def set_model(self, model):
        """Called to give the loss access to model's OC-Softmax module."""
        if self.use_oc_softmax and hasattr(model, 'oc_loss'):
            self.oc_loss_fn = model.oc_loss

    def compute(self, outputs, target):
        """
        """
        if self.use_oc_softmax:
            # outputs[0] = embeddings (batch, emb_dim), outputs[1] = labels
            embeddings = outputs[0]
            labels = outputs[1]
            loss, _ = self.oc_loss_fn(embeddings, labels)
            return loss

        # BCE path
        targets = outputs[1]
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        loss = self.m_loss(outputs[0], targets)
        if self.pos_weight > 0:
            weights = torch.where(outputs[1] > 0.5,
                                  torch.tensor(self.pos_weight, device=outputs[1].device),
                                  torch.tensor(1.0, device=outputs[1].device))
            loss = (loss * weights).mean()
        if not torch.is_tensor(loss):
            loss = torch.tensor(loss, dtype=torch.float32)

        return loss

    
if __name__ == "__main__":
    print("Definition of model")

