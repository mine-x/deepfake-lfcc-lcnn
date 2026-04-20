"""
codec_augment.py — Pre-generate codec-augmented eval sets to disk.

Encodes each FLAC file through a lossy codec (MP3 or Opus) and decodes back
to FLAC at 16kHz. The spectral damage from compression persists after decoding.

Output directories mirror the original filenames so existing protocol files
work unchanged.

Usage:
    python codec_augment.py [--input-dir DIR] [--output-base DIR]
"""

import os
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

CODEC_CONDITIONS = {
    "mp3_64kbps": {
        "ext": ".mp3",
        "encode_args": ["-c:a", "libmp3lame", "-b:a", "64k"],
    },
    "opus_32kbps": {
        "ext": ".opus",
        "encode_args": ["-c:a", "libopus", "-b:a", "32k"],
    },
}


def transcode_file(input_path, output_path, codec_config):
    """Encode to lossy codec, then decode back to 16kHz FLAC."""
    tmp_path = output_path + codec_config["ext"]

    try:
        # encode: FLAC -> lossy
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(input_path)]
            + codec_config["encode_args"]
            + [tmp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        # decode: lossy -> FLAC at 16kHz
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path,
             "-c:a", "flac", "-ar", "16000", str(output_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    finally:
        # clean up intermediate lossy file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def process_condition(input_dir, output_base, condition_name, codec_config,
                      max_workers=4, chunk_size=10000):
    """Process all FLAC files for one codec condition.

    Filters out already-completed files upfront using filename sets (not
    Path objects) to minimize memory. Remaining files are processed in
    chunks so only chunk_size futures exist at a time.
    """
    output_dir = os.path.join(output_base, condition_name)
    os.makedirs(output_dir, exist_ok=True)

    # Build set of already-done filenames (strings only, ~30 bytes each)
    done_names = set(os.listdir(output_dir))
    skipped = len(done_names)

    # List only filenames that still need processing
    all_names = sorted(os.listdir(input_dir))
    pending = [n for n in all_names if n.endswith(".flac") and n not in done_names]
    del all_names, done_names  # free memory

    total = skipped + len(pending)

    if len(pending) == 0:
        print(f"[codec] {condition_name}: all {total} files already done")
        return

    print(f"[codec] {condition_name}: {len(pending)} remaining "
          f"({skipped} already done, {total} total) -> {output_dir}")
    print(f"[codec] workers={max_workers}, chunk_size={chunk_size}", flush=True)

    done = 0
    errors = 0

    def do_one(filename):
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)
        try:
            transcode_file(in_path, out_path, codec_config)
            return "ok"
        except subprocess.CalledProcessError:
            return "error"

    for chunk_start in range(0, len(pending), chunk_size):
        chunk = pending[chunk_start:chunk_start + chunk_size]

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(do_one, f): f for f in chunk}
            for future in as_completed(futures):
                result = future.result()
                if result == "ok":
                    done += 1
                else:
                    errors += 1
                    print(f"[codec] ERROR: {futures[future]}")

                if done % 5000 == 0 or (done + errors) == len(pending):
                    print(f"[codec] {condition_name}: {done + errors}/{len(pending)} "
                          f"(done={done}, errors={errors})",
                          flush=True)

    print(f"[codec] {condition_name}: complete. "
          f"done={done}, skipped={skipped}, errors={errors}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate codec-augmented eval sets")
    parser.add_argument("--input-dir", type=str,
                        default=os.path.expanduser(
                            "~/asvspoof5_dataset/flac_E_eval"),
                        help="Directory of original FLAC eval files")
    parser.add_argument("--output-base", type=str,
                        default=os.path.expanduser(
                            "~/asvspoof5_dataset/augmented"),
                        help="Base output directory for codec conditions")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel ffmpeg processes")
    parser.add_argument("--condition", type=str, default="all",
                        choices=["all", "mp3_64kbps", "opus_32kbps"],
                        help="Which codec condition to generate")
    args = parser.parse_args()

    if args.condition == "all":
        conditions = CODEC_CONDITIONS
    else:
        conditions = {args.condition: CODEC_CONDITIONS[args.condition]}

    for name, config in conditions.items():
        process_condition(args.input_dir, args.output_base, name, config,
                          max_workers=args.workers)


if __name__ == "__main__":
    main()
