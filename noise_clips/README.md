# noise_clips/

Noise clips used for the augmentation pipeline. The `.wav` files themselves are not checked into this repository; this README documents the structure and sources so the directory can be reconstructed.

## Sources

All active clips (under `train/`, `eval/`, `eval_train_combined/`) ultimately come from [Freesound](https://freesound.org/):

- **Ambient** (single-source background noise, e.g. HVAC, traffic, fan hum): obtained via the [MUSAN corpus](https://www.openslr.org/17/)'s `noise/free-sound/` subset, which [packages Freesound content](https://arxiv.org/abs/1510.08484) under CC BY 4.0.
- **Babble** (multi-speaker chatter): pulled directly from Freesound. Individual clips carry per-file Creative Commons licenses (typically CC0, CC BY, or CC BY-NC). When reconstructing, verify each clip's license on its Freesound page.

## Directory layout

```
noise_clips/
  train/                    # used during noise-augmented training
    ambient/                # 3 MUSAN clips
    babble/                 # 3 Freesound clips
  eval/                     # held-out eval-only clips (non-overlapping with train)
    ambient/                # 3 MUSAN clips
    babble/                 # 3 Freesound clips
  eval_train_combined/      # train + eval merged, used for the "most representative" eval
    ambient/                # 6 MUSAN clips
    babble/                 # 6 Freesound clips
```

All clips are 16 kHz mono `.wav`, between ~10s and ~60s in length.

## Clip inventory

Filenames used in the published experiments:

**Ambient (MUSAN `noise/free-sound` subset):**
- `noise-free-sound-0032.wav`, `0055.wav`, `0109.wav`, `0153.wav`, `0158.wav`, `0184.wav`

**Babble (Freesound, searched by description):**
- `busy_room_crowd.wav`, `coffee_shop_ambience.wav`, `murmur_small_group.wav`
- `murmur_couple.wav`, `murmur_medium_group.wav`, `workplace_ambience_chatter.wav`

The babble filenames are descriptive rather than Freesound IDs; any Freesound clip matching the description (multi-speaker background chatter with no foreground speaker) produces equivalent results. Resample to 16 kHz mono before use.

## Reproducing

1. Download MUSAN from https://www.openslr.org/17/ and extract the `noise/free-sound/` clips listed above into `train/ambient/`, `eval/ambient/`, and `eval_train_combined/ambient/` per the split above.
2. Select six babble clips from Freesound (any multi-speaker chatter) and place them under the corresponding `babble/` subdirectories, renamed to the filenames above (or update `augment.py` to match your filenames).
3. Verify with:
   ```bash
   python 01_project/baseline_DF/augment.py --condition noise_ambient_10dB \
       --input-dir <eval_audio_dir> --output-dir /tmp/aug_check --n 5
   ```
