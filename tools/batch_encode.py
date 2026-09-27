#!/usr/bin/env python3
"""Encode every video in a folder to a vbend WAV, saved next to each clip.

    python3 tools/batch_encode.py ~/clips                 # rgb mode (default)
    python3 tools/batch_encode.py ~/clips -m ysplit        # any mode
    python3 tools/batch_encode.py ~/clips --audio          # also <name>.audio.wav (the clip's own sound)
    python3 tools/batch_encode.py ~/clips --force          # redo clips that already have a WAV

Bad files are reported and skipped; the rest still convert.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import vbend  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif", ".3gp"}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("folder", nargs="?", default=".")
ap.add_argument("-m", "--mode", choices=vbend.MODES, default="rgb")
ap.add_argument("--audio", action="store_true", help="also save each clip's own sound as <name>.audio.wav")
ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
a = ap.parse_args()

folder = os.path.abspath(os.path.expanduser(a.folder))
clips = sorted(f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in VIDEO_EXT)
if not clips:
    sys.exit(f"no videos found in {folder}")

print(f"{len(clips)} videos in {folder} -> {a.mode} WAVs\n")
done, skipped, failed = 0, 0, []
t0 = time.time()
for i, name in enumerate(clips, 1):
    tag = f"[{i}/{len(clips)}]"
    src = os.path.join(folder, name)
    stem = os.path.splitext(src)[0]
    out = f"{stem}.{a.mode}.wav"
    if a.audio and (a.force or not os.path.exists(f"{stem}.audio.wav")):
        aud = vbend.export_soundtrack(src)
        print(f"{tag} audio  {name} -> {os.path.basename(aud) if aud else '(no audio track)'}")
    if os.path.exists(out) and not a.force:
        print(f"{tag} skip   {name} (already has {os.path.basename(out)})")
        skipped += 1
        continue
    try:
        st = vbend.encode_file(src, a.mode, out=out)
        print(f"{tag} ok     {name} -> {os.path.basename(out)}  "
              f"({st['seconds']:.1f} s, {st['size'] / 2**20:.0f} MB, took {st['time']:.1f} s)")
        done += 1
    except Exception as e:
        print(f"{tag} FAIL   {name}: {e}")
        failed.append(name)

print(f"\n{done} converted, {skipped} skipped, {len(failed)} failed in {time.time() - t0:.0f} s")
if failed:
    print("failed:", ", ".join(failed))
    sys.exit(1)
