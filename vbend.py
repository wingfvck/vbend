#!/usr/bin/env python3
"""vbend - circuit-bend video through audio plugins.

Run with no arguments for the window:

    python3 vbend.py

  Encode tab : drop videos  -> <name>.<mode>.wav   (192 kHz stereo float, load into your DAW)
  Decode tab : drop bent WAVs -> <name>.mp4        (faster than real time, mode auto-detected)
  Live tab   : creates the silent 192k "vbend" output and previews whatever the DAW plays into it

Command line (same engine, for scripting):

    python3 vbend.py encode clip.mp4 -m ysplit
    python3 vbend.py decode bent.wav [--oob fold] [--soundtrack clip.mp4]
    python3 vbend.py live               # preview window only
    python3 vbend.py sink up|down|status

Modes (the sync burst at the start of every frame tells the decoder which one it is):

    gray       160x90  brightness only; one sample = one pixel. The cleanest bends.
    ysplit     112x64  COLOR. Left channel = brightness, right channel = color (Cb then Cr).
                       Effects on L bend light, effects on R bend color, stereo FX = color fringing.
    yuv420     128x72  COLOR. Y, U, V planes one after another, interleaved across L/R.
    rgb        96x52   COLOR. R, G, B planes one after another. Delays misregister the colors.
    packed444  160x90  COLOR chaos. R,G,B stacked into one number; almost everything -> confetti.

Layout: 192000 Hz, 24 fps -> 8000 stereo samples per frame. Each channel's frame starts with a
512-sample sync burst (a different pattern per mode), then silence, then the pixels, which always
end exactly at the frame boundary.
"""

__version__ = "3.2.0"

import argparse
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
import zlib

import numpy as np
import soundfile as sf

SAMPLE_RATE = 192000
FPS = 24
P = SAMPLE_RATE // FPS              # 8000 stereo samples per video frame
SYNC_LEN = 512
SYNC_AMP = 0.9
LEGACY_SEED = 0x5EEDB3ED            # v1/v2 files (one pattern for every mode)
OOB_MODES = ["clip", "white", "wrap", "fold"]
SILENCE = 1e-4


# =============================================================== sample mapping

def to_audio(values, maxval):
    return values.astype(np.float32) * np.float32(2.0 / maxval) - np.float32(1.0)


def from_audio(samples, maxval, oob="clip"):
    """float -1..+1 -> int 0..maxval. Returns (ints, count_out_of_range)."""
    x = np.nan_to_num(samples.astype(np.float64), nan=-1.0, posinf=1.0, neginf=-1.0)
    x = (x + 1.0) * 0.5
    inside = (x >= 0.0) & (x <= 1.0)
    if oob == "clip":
        x = np.clip(x, 0.0, 1.0)
    elif oob == "white":
        x = np.where(inside, x, 1.0)
    elif oob == "wrap":
        x = np.where(inside, x, np.mod(x, 1.0))
    elif oob == "fold":
        y = np.mod(x, 2.0)
        x = np.where(inside, x, np.where(y > 1.0, 2.0 - y, y))
    return np.rint(x * maxval).astype(np.int32), int((~inside).sum())


# ======================================================================= modes

class Mode:
    """A mode turns an RGB frame into two per-channel sample arrays and back."""
    name = w = h = per_ch = None
    maxval = 255
    blurb = ""

    @property
    def seed(self):
        return zlib.crc32(f"vbend:{self.name}".encode())

    def encode(self, rgb):            # -> (L, R) float32, each per_ch long
        raise NotImplementedError

    def decode(self, L, R, oob):      # -> (rgb uint8 HxWx3, n_out_of_range)
        raise NotImplementedError


class Interleaved(Mode):
    """One value stream spread over both channels: even values -> L, odd -> R (v1 layout)."""

    def values(self, rgb):
        raise NotImplementedError

    def image(self, vals):
        raise NotImplementedError

    def encode(self, rgb):
        v = to_audio(self.values(rgb), self.maxval)
        return v[0::2], v[1::2]

    def decode(self, L, R, oob):
        s = np.empty(L.size * 2, np.float32)
        s[0::2], s[1::2] = L, R
        vals, n = from_audio(s, self.maxval, oob)
        return self.image(vals), n


class Gray(Interleaved):
    name, w, h = "gray", 160, 90
    per_ch = w * h // 2
    blurb = "brightness only, cleanest bends"

    def values(self, rgb):
        import cv2
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).reshape(-1)

    def image(self, vals):
        g = vals.astype(np.uint8).reshape(self.h, self.w)
        return np.repeat(g[:, :, None], 3, 2)


class YUV420(Interleaved):
    name, w, h = "yuv420", 128, 72
    per_ch = w * h * 3 // 4
    blurb = "color: Y, U, V planes in sequence"

    def values(self, rgb):
        import cv2
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420).reshape(-1)

    def image(self, vals):
        import cv2
        return cv2.cvtColor(vals.astype(np.uint8).reshape(self.h * 3 // 2, self.w),
                            cv2.COLOR_YUV2RGB_I420)


class RGBPlanar(Interleaved):
    name, w, h = "rgb", 96, 52
    per_ch = w * h * 3 // 2
    blurb = "color: R, G, B planes in sequence"

    def values(self, rgb):
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)).reshape(-1)

    def image(self, vals):
        return np.ascontiguousarray(vals.astype(np.uint8).reshape(3, self.h, self.w).transpose(1, 2, 0))


class Packed444(Interleaved):
    name, w, h, maxval = "packed444", 160, 90, 4095
    per_ch = w * h // 2
    blurb = "color chaos: RGB stacked in one number"

    def values(self, rgb):
        q = (rgb >> 4).astype(np.uint16)
        return ((q[..., 0] << 8) | (q[..., 1] << 4) | q[..., 2]).reshape(-1)

    def image(self, vals):
        v = vals.reshape(self.h, self.w)
        return (np.stack([(v >> 8) & 15, (v >> 4) & 15, v & 15], -1) * 17).astype(np.uint8)


class YSplit(Mode):
    """L = luma 112x64. R = chroma at half width: Cb plane (56x64) then Cr plane (56x64)."""
    name, w, h = "ysplit", 112, 64
    per_ch = w * h
    blurb = "color: L = brightness, R = color"

    def encode(self, rgb):
        import cv2
        ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        y = ycc[..., 0].reshape(-1)
        cr = cv2.resize(ycc[..., 1], (self.w // 2, self.h), interpolation=cv2.INTER_AREA)
        cb = cv2.resize(ycc[..., 2], (self.w // 2, self.h), interpolation=cv2.INTER_AREA)
        return to_audio(y, 255), to_audio(np.concatenate([cb.reshape(-1), cr.reshape(-1)]), 255)

    def decode(self, L, R, oob):
        import cv2
        y, n1 = from_audio(L, 255, oob)
        c, n2 = from_audio(R, 255, oob)
        half = self.w // 2 * self.h
        cb = c[:half].astype(np.uint8).reshape(self.h, self.w // 2)
        cr = c[half:].astype(np.uint8).reshape(self.h, self.w // 2)
        up = lambda p: cv2.resize(p, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        ycc = np.stack([y.astype(np.uint8).reshape(self.h, self.w), up(cr), up(cb)], -1)
        return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB), n1 + n2


MODES = {m.name: m for m in (Gray(), YSplit(), YUV420(), RGBPlanar(), Packed444())}
for _m in MODES.values():
    assert _m.per_ch + SYNC_LEN <= P, _m.name


def sync_pattern(seed):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, SYNC_LEN) * 2 - 1).astype(np.float32) * SYNC_AMP


def _template_ffts():
    out = {}
    for key, seed in [(m.name, m.seed) for m in MODES.values()] + [("legacy", LEGACY_SEED)]:
        t = np.zeros(P)
        t[:SYNC_LEN] = sync_pattern(seed)
        out[key] = np.conj(np.fft.rfft(t))
    return out


TEMPLATES = _template_ffts()


def encode_block(rgb, mode):
    L, R = mode.encode(rgb)
    blk = np.zeros((P, 2), np.float32)
    s = sync_pattern(mode.seed)
    blk[:SYNC_LEN, 0] = s
    blk[:SYNC_LEN, 1] = s
    blk[P - mode.per_ch:, 0] = L
    blk[P - mode.per_ch:, 1] = R
    return blk


def decode_block(blk, mode, oob="clip"):
    return mode.decode(blk[P - mode.per_ch:, 0], blk[P - mode.per_ch:, 1], oob)


def find_sync(stereo, max_frames=48, keys=None):
    """-> (mode_key, offset, confidence, polarity). mode_key may be 'legacy'.

    Averages up to max_frames frame periods of L+R (the image washes out, the repeating burst
    adds up), then GCC-PHAT correlates against every mode's burst. Survives latency, gain,
    EQ/blur, echo and phase flips.
    """
    k = min(max_frames, len(stereo) // P)
    if k < 1:
        return None, 0, 0.0, 1
    m = stereo[:k * P, 0].astype(np.float64) + stereo[:k * P, 1]
    m = np.nan_to_num(np.clip(m, -1e3, 1e3))
    per = m.reshape(k, P).mean(0)
    X = np.fft.rfft(per - per.mean())
    best = (None, 0, 0.0, 1)
    for key, T in TEMPLATES.items():
        if keys and key not in keys:
            continue
        c = X * T
        corr = np.fft.irfft(c / (np.abs(c) + 1e-12), P)
        i = int(np.argmax(np.abs(corr)))
        conf = float(np.abs(corr[i]) / (np.median(np.abs(corr)) + 1e-12))
        if conf > best[2]:
            best = (key, i, conf, 1 if corr[i] >= 0 else -1)
    return best


LOCK_CONF = 40.0


# ============================================================ video in / out

def probe(path):
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = dict(w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=fps, n=n, dur=n / fps if fps else 0.0)
    cap.release()
    return info


def read_frames(path, w, h):
    """Yield RGB frames at exactly 24 fps, w x h. ffmpeg if present, else OpenCV."""
    if shutil.which("ffmpeg"):
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-an", "-vf",
                                 f"fps={FPS},scale={w}:{h}:flags=area", "-pix_fmt", "rgb24",
                                 "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
        size = w * h * 3
        try:
            while True:
                buf = proc.stdout.read(size)
                if len(buf) < size:
                    break
                yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        finally:
            proc.stdout.close()
            proc.kill()
            proc.wait()
        return
    import cv2
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    k = i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t_end = (i + 1) / src_fps
        i += 1
        if k / FPS >= t_end:
            continue
        rgb = cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        while k / FPS < t_end:
            yield rgb
            k += 1
    cap.release()


def pick_codec():
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    for c in ("libx264", "libopenh264", "mpeg4"):
        if f" {c} " in enc:
            return c
    return "mpeg4"


class VideoWriter:
    def __init__(self, out, w, h, scale, codec="auto", soundtrack=None):
        W, H = w * scale, h * scale
        W, H = W + W % 2, H + H % 2
        self.out = out
        if shutil.which("ffmpeg"):
            codec = pick_codec() if codec == "auto" else codec
            q = {"libx264": ["-crf", "14", "-preset", "medium"], "libopenh264": ["-b:v", "8M"],
                 "mpeg4": ["-q:v", "2"], "ffv1": []}.get(codec, [])
            pix = "bgr0" if codec == "ffv1" else "yuv420p"
            cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-"]
            if soundtrack:
                cmd += ["-i", soundtrack, "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-b:a", "192k",
                        "-shortest"]
            cmd += ["-vf", f"scale={W}:{H}:flags=neighbor", "-c:v", codec, *q, "-pix_fmt", pix, out]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self.cv = None
            self.desc = f"ffmpeg/{codec}"
        else:
            import cv2
            self.proc, self.size = None, (W, H)
            self.cv = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
            self.desc = "opencv/mp4v"

    def write(self, rgb):
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(rgb).tobytes())
        else:
            import cv2
            big = cv2.resize(rgb, self.size, interpolation=cv2.INTER_NEAREST)
            self.cv.write(cv2.cvtColor(big, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            if self.proc.wait():
                raise RuntimeError(f"ffmpeg failed writing {self.out}")
        else:
            self.cv.release()


# ========================================================== encode / decode

class Cancelled(Exception):
    pass


def export_soundtrack(path, out=None):
    """Save the clip's own audio as <name>.audio.wav (192 kHz stereo float), starting at 0 like
    the video WAV, so both line up when dropped at the start of the DAW timeline.
    Returns the output path, or None if the clip has no audio or ffmpeg is missing."""
    if not shutil.which("ffmpeg"):
        return None
    out = out or f"{os.path.splitext(path)[0]}.audio.wav"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-vn", "-map", "0:a:0?",
                        "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le", out],
                       capture_output=True, text=True)
    if r.returncode or not os.path.exists(out) or os.path.getsize(out) < 1024:
        if os.path.exists(out):
            os.remove(out)
        return None
    return out


def encode_file(path, mode_name="gray", out=None, progress=None, cancel=None):
    """Video -> WAV. progress(fraction, preview_rgb_or_None, text)."""
    mode = MODES[mode_name]
    out = out or f"{os.path.splitext(path)[0]}.{mode.name}.wav"
    info = probe(path)
    est = max(1, int(info["dur"] * FPS))
    fmt = "RF64" if est * P * 8 > 3.9 * 2 ** 30 else "WAV"
    t0, n = time.time(), 0
    with sf.SoundFile(out, "w", samplerate=SAMPLE_RATE, channels=2, subtype="FLOAT", format=fmt) as wav:
        for rgb in read_frames(path, mode.w, mode.h):
            if cancel and cancel.is_set():
                raise Cancelled()
            wav.write(encode_block(rgb, mode))
            n += 1
            if progress and n % 6 == 0:
                progress(min(n / est, 1.0), rgb, f"encoding {os.path.basename(path)}: frame {n}/{est}")
    if n == 0:
        os.remove(out)
        raise RuntimeError(f"no video frames in {path}")
    return dict(out=out, frames=n, seconds=n / FPS, mode=mode.name, src=info,
                size=os.path.getsize(out), time=time.time() - t0)


def decode_file(path, out=None, mode_name="auto", oob="clip", nudge=0, invert=False, scale=4,
                codec="auto", soundtrack=None, progress=None, cancel=None, fallback_mode="gray"):
    """Bent WAV -> video, as fast as the CPU goes. Returns a stats dict."""
    info = sf.info(path)
    if info.channels < 2:
        raise RuntimeError("this WAV is mono - it was mixed down somewhere; export stereo")
    warn = []
    if info.samplerate != SAMPLE_RATE:
        warn.append(f"file is {info.samplerate} Hz, not {SAMPLE_RATE} - something resampled it, "
                    "expect a scrambled picture. Set the DAW project/export rate to 192000.")
    t0 = time.time()
    with sf.SoundFile(path) as f:
        head = f.read(P * 48, dtype="float32", always_2d=True)[:, :2]
        if mode_name == "auto":
            key, off, conf, pol = find_sync(head)
            if key in MODES and conf >= LOCK_CONF:
                mode = MODES[key]
            else:
                mode = MODES[fallback_mode]
                warn.append(f"couldn't identify the mode ({'old-style file' if key == 'legacy' else 'weak sync'})"
                            f" - using {mode.name}. Pick the mode yourself to force it.")
                key, off, conf, pol = find_sync(head, keys=(mode.name, "legacy"))
        else:
            # forced: only look for this mode's own sync burst (or a v1/v2 burst)
            mode = MODES[mode_name]
            key, off, conf, pol = find_sync(head, keys=(mode.name, "legacy"))
        if conf < LOCK_CONF:
            warn.append(f"sync is weak ({conf:.0f}); image may be misaligned - try the nudge setting")
        if pol < 0 and not invert:
            warn.append("polarity is inverted (a plugin flipped phase) - tick 'invert' to undo")
        start = off + nudge
        while start < 0:
            start += P
        n_frames = (info.frames - start) // P
        out = out or os.path.splitext(path)[0] + ".mp4"
        vw = VideoWriter(out, mode.w, mode.h, scale, codec, soundtrack)
        f.seek(start)
        n_oob = 0
        try:
            for i in range(n_frames):
                if cancel and cancel.is_set():
                    raise Cancelled()
                blk = f.read(P, dtype="float32", always_2d=True)[:, :2]
                if invert:
                    blk = -blk
                rgb, n = decode_block(blk, mode, oob)
                n_oob += n
                vw.write(rgb)
                if progress and i % 6 == 0:
                    progress(i / max(1, n_frames), rgb, f"decoding {os.path.basename(path)}: "
                             f"frame {i}/{n_frames}")
        finally:
            vw.close()
    dt = time.time() - t0
    return dict(out=out, frames=n_frames, seconds=n_frames / FPS, mode=mode.name, offset=start,
                conf=conf, oob_pct=100 * n_oob / max(1, n_frames * mode.per_ch * 2), time=dt,
                speed=(n_frames / FPS) / max(dt, 1e-6), warnings=warn, writer=vw.desc)


# ============================================================ 192k output sink

def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def sink_check():
    if os.name == "posix" and os.geteuid() == 0:
        return "don't run vbend as root/sudo - PipeWire belongs to your desktop user"
    if not shutil.which("pactl"):
        return "pactl not found - sudo dnf install pulseaudio-utils"
    if sh(["pactl", "info"]).returncode:
        return ("can't reach PipeWire from here (XDG_RUNTIME_DIR="
                f"{os.environ.get('XDG_RUNTIME_DIR', 'unset')}). Run from your desktop session; "
                "check: systemctl --user status pipewire pipewire-pulse wireplumber")
    return None


def sink_exists(name="vbend"):
    r = sh(["pactl", "list", "short", "sinks"])
    return any(line.split("\t")[1:2] == [name] for line in r.stdout.splitlines())


def sink_up(name="vbend"):
    err = sink_check()
    if err:
        return False, err
    if shutil.which("pw-metadata"):
        sh(["pw-metadata", "-n", "settings", "0", "clock.force-rate", str(SAMPLE_RATE)])
    if not sink_exists(name):
        r = sh(["pactl", "load-module", "module-null-sink", f"sink_name={name}", f"rate={SAMPLE_RATE}",
                "channels=2", "format=float32le", "channel_map=front-left,front-right",
                "sink_properties=device.description=vbend-192k"])
        if r.returncode:
            return False, f"couldn't create output: {r.stderr.strip()}"
    sh(["pactl", "set-sink-volume", name, "100%"])
    sh(["pactl", "set-sink-mute", name, "0"])
    sh(["pactl", "set-source-volume", f"{name}.monitor", "100%"])
    return True, (f"output 'vbend-192k' is up at {SAMPLE_RATE} Hz. Start your DAW with "
                  f"PULSE_SINK={name} (or move its stream to vbend-192k in pavucontrol).")


def sink_down(name="vbend"):
    err = sink_check()
    if err:
        return False, err
    for line in sh(["pactl", "list", "short", "modules"]).stdout.splitlines():
        if f"sink_name={name}" in line:
            sh(["pactl", "unload-module", line.split("\t")[0]])
    if shutil.which("pw-metadata"):
        sh(["pw-metadata", "-n", "settings", "0", "clock.force-rate", "0"])
    return True, "output removed, PipeWire rate unlocked"


# ================================================================ live monitor

class Ring:
    def __init__(self, n):
        self.n, self.buf, self.end = n, np.zeros((n, 2), np.float32), 0

    @property
    def start(self):
        return max(0, self.end - self.n)

    def write(self, x):
        m = len(x)
        if m > self.n:
            self.end += m - self.n
            x, m = x[-self.n:], self.n
        i = self.end % self.n
        k = min(m, self.n - i)
        self.buf[i:i + k] = x[:k]
        self.buf[:m - k] = x[k:]
        self.end += m

    def read(self, a, m):
        i = a % self.n
        k = min(m, self.n - i)
        return self.buf[i:i + m].copy() if k == m else np.concatenate([self.buf[i:], self.buf[:m - k]])


def list_output_sinks(exclude="vbend"):
    """[(name, description)] of real outputs, default first. Never includes the vbend sink."""
    r = sh(["pactl", "list", "sinks"])
    sinks, name = [], None
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("Description:") and name:
            if name != exclude:
                sinks.append((name, line.split(":", 1)[1].strip()))
            name = None
    default = sh(["pactl", "get-default-sink"]).stdout.strip()
    sinks.sort(key=lambda s: s[0] != default)
    return sinks


class Speaker:
    """Plays what arrives on the vbend output through real speakers/headphones, safely.

    192 kHz -> 48 kHz (anti-aliased) -> DC blocker -> volume -> peak limiter (-6 dBFS ceiling,
    instant attack, 150 ms release) -> hard clip at the ceiling as a last resort.
    Runs on its own thread so audio hiccups never stall the picture.
    """
    OUT_RATE = 48000
    DECIM = SAMPLE_RATE // OUT_RATE
    CEILING = 10 ** (-6 / 20)
    BLOCK = 32                                      # limiter block, 0.67 ms

    def __init__(self, sink, volume=25):
        self.sink, self.volume = sink, volume       # volume: 0..100 knob
        self.gr_db, self.error = 0.0, None
        n = 95                                      # lowpass ~20 kHz at 192 kHz
        k = np.arange(n) - (n - 1) / 2
        h = np.sinc(2 * 20000 / SAMPLE_RATE * k) * np.blackman(n)
        self.taps = (h / h.sum()).astype(np.float64)
        self.hist = np.zeros((n - 1, 2))
        self.count = 0
        self.R = float(np.exp(-2 * np.pi * 15 / self.OUT_RATE))   # 15 Hz DC blocker
        self.hp_x, self.hp_y = np.zeros(2), np.zeros(2)
        self.g = 1.0
        self.rel = float(np.exp(-self.BLOCK / (0.15 * self.OUT_RATE)))
        self.q = queue.Queue(maxsize=64)
        self.proc = None
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    @staticmethod
    def knob_db(v):
        return None if v <= 0 else -60 + 0.6 * v   # 0 = mute, 100 = 0 dB

    def feed(self, x):
        try:
            self.q.put_nowait(x)
        except queue.Full:
            pass

    def close(self):
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        if self.proc:
            self.proc.terminate()

    # --- DSP
    def process(self, x):
        x = np.clip(np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0), -4, 4)
        buf = np.concatenate([self.hist, x])
        self.hist = buf[-(len(self.taps) - 1):]
        f = np.stack([np.convolve(buf[:, c], self.taps, "valid") for c in range(2)], 1)
        start = (-self.count) % self.DECIM
        self.count += len(x)
        y = f[start::self.DECIM]
        if len(y) == 0:
            return y.astype(np.float32)
        # DC blocker y[n] = x[n] - x[n-1] + R*y[n-1], vectorised in <=256-sample pieces
        out = np.empty_like(y)
        for a in range(0, len(y), 256):
            seg = y[a:a + 256]
            d = seg - np.vstack([self.hp_x, seg[:-1]])
            m = len(seg)
            pw = self.R ** np.arange(m)[:, None]
            o = pw * np.cumsum(d / pw, 0) + (self.R * pw) * self.hp_y
            out[a:a + m] = o
            self.hp_x, self.hp_y = seg[-1], o[-1]
        db = self.knob_db(self.volume)
        out *= 0.0 if db is None else 10 ** (db / 20)
        # limiter
        m = len(out)
        nb = -(-m // self.BLOCK)
        pad = np.zeros((nb * self.BLOCK, 2))
        pad[:m] = out
        peaks = np.abs(pad).reshape(nb, self.BLOCK, 2).max((1, 2))
        gains = np.empty(nb * self.BLOCK)
        g_min = 1.0
        for i in range(nb):
            target = min(1.0, self.CEILING / peaks[i]) if peaks[i] > 0 else 1.0
            g_new = min(target, 1.0 - (1.0 - self.g) * self.rel)
            ramp = np.linspace(self.g, g_new, self.BLOCK, endpoint=False) if g_new > self.g \
                else np.full(self.BLOCK, g_new)
            gains[i * self.BLOCK:(i + 1) * self.BLOCK] = np.minimum(ramp, target)
            self.g = g_new
            g_min = min(g_min, g_new)
        self.gr_db = 20 * np.log10(max(g_min, 1e-6))
        out = np.clip(pad[:m] * gains[:m, None], -self.CEILING, self.CEILING)
        return out.astype(np.float32)

    def _loop(self):
        try:
            if not shutil.which("pacat"):
                raise RuntimeError("pacat not found - sudo dnf install pulseaudio-utils")
            self.proc = subprocess.Popen(
                ["pacat", "--playback", "-d", self.sink, "--raw", "--format=float32le",
                 f"--rate={self.OUT_RATE}", "--channels=2", "--latency-msec=40",
                 "--client-name=vbend", "--stream-name=vbend preview"],
                stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            while True:
                x = self.q.get()
                if x is None:
                    break
                y = self.process(x)
                if len(y):
                    self.proc.stdin.write(y.astype("<f4").tobytes())
                    self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            self.error = f"audio output stopped: {e}"
        except Exception as e:
            self.error = str(e)
        finally:
            if self.proc:
                try:
                    self.proc.stdin.close()
                except Exception:
                    pass
                self.proc.terminate()


class Monitor(threading.Thread):
    """Reads the vbend output (or a file, paced) and keeps .latest = newest decoded frame.

    Tweak .oob / .invert / .nudge / .fallback_mode from any thread; call .resync() / .stop().
    """
    WINDOW, EVERY, CONFIRM = 8, 3, 2
    CHUNK = 1024

    def __init__(self, device="vbend", file=None, cmd=None):
        super().__init__(daemon=True)
        self.device, self.file, self.cmd = device, file, cmd
        self.oob, self.invert, self.nudge, self.fallback_mode = "clip", False, 0, "gray"
        self.force_mode = None          # a mode name to skip auto-detection
        self.speaker = None             # a Speaker to hear the stream through
        self.latest, self.frame_id = None, 0
        self.state, self.conf, self.mode, self.fps, self.error = "starting", 0.0, None, 0.0, None
        self._stop, self._resync, self.proc = threading.Event(), threading.Event(), None

    def stop(self):
        self._stop.set()
        if self.proc:
            self.proc.terminate()

    def resync(self):
        self._resync.set()

    def _chunks(self):
        if self.file:
            with sf.SoundFile(self.file) as f:
                t0, sent = time.monotonic(), 0
                while not self._stop.is_set():
                    x = f.read(self.CHUNK, dtype="float32", always_2d=True)[:, :2]
                    if len(x) == 0:
                        f.seek(0)
                        continue
                    yield x
                    sent += len(x)
                    ahead = t0 + sent / SAMPLE_RATE - time.monotonic()
                    if ahead > 0:
                        time.sleep(ahead)
            return
        if self.cmd:
            cmd = shlex.split(self.cmd)
        else:
            err = sink_check()
            if err:
                raise RuntimeError(err)
            if not sink_exists(self.device):
                raise RuntimeError(f"no '{self.device}' output yet - click 'Create 192k output' first")
            cmd = ["parec", "-d", f"{self.device}.monitor", "--raw", "--format=float32le",
                   f"--rate={SAMPLE_RATE}", "--channels=2", "--latency-msec=10"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        carry = b""
        while not self._stop.is_set():
            data = self.proc.stdout.read(self.CHUNK * 8)
            if not data:
                break
            data = carry + data
            u = len(data) // 8 * 8
            carry = data[u:]
            yield np.frombuffer(data[:u], "<f4").reshape(-1, 2)

    def run(self):
        try:
            self._run()
        except Exception as e:  # surfaced in the UI
            self.error, self.state = str(e), "error"
        finally:
            if self.proc:
                self.proc.terminate()

    def _run(self):
        ring = Ring(P * 48)
        phase, locked, cand, cand_n, last_check, nxt = 0, False, None, 0, 0, None
        mode_key = None
        fps_t, fps_n = time.monotonic(), 0
        self.state = "waiting for signal"
        for x in self._chunks():
            if self._stop.is_set():
                break
            ring.write(x)
            spk = self.speaker
            if spk:
                spk.feed(x)
            if self._resync.is_set():
                self._resync.clear()
                locked, last_check = False, 0
            # --- sync / mode tracking
            if ring.end - last_check >= P * self.EVERY and ring.end - ring.start >= P * self.WINDOW:
                last_check = ring.end
                a = ring.end - P * self.WINDOW
                win = ring.read(a, P * self.WINDOW)
                if np.abs(win).max() <= SILENCE:
                    self.conf = 0.0
                else:
                    force = self.force_mode
                    key, off, c, pol = find_sync(win, self.WINDOW,
                                                 keys=(force, "legacy") if force else None)
                    if force:
                        key = force
                    self.conf = c
                    if c >= 25:
                        ph = (a + off) % P
                        same = locked and min((ph - phase) % P, (phase - ph) % P) <= 1 and key == mode_key
                        if not same:
                            if cand and cand[1] == key and min((ph - cand[0]) % P, (cand[0] - ph) % P) <= 1:
                                cand_n += 1
                            else:
                                cand, cand_n = (ph, key), 1
                            if not locked or cand_n >= self.CONFIRM:
                                phase, mode_key, locked, nxt, cand = ph, key, True, None, None
                        else:
                            cand = None
            # --- decode complete frames, keep the newest
            mode = MODES.get(self.force_mode or mode_key) or MODES[self.fallback_mode]
            self.mode = mode.name + (" (old file)" if mode_key == "legacy" else "")
            eff = (phase + self.nudge) % P
            if nxt is not None and (nxt - eff) % P:
                nxt = None
            if nxt is None or nxt < ring.start:
                last = ring.end - P
                nxt = last - ((last - eff) % P) if last >= ring.start else None
            while nxt is not None and nxt + P <= ring.end:
                blk = ring.read(nxt, P)
                nxt += P
                if np.abs(blk).max() <= SILENCE:
                    self.state = "no signal (holding last frame)"
                    continue
                if self.invert:
                    blk = -blk
                self.latest, _ = decode_block(blk, mode, self.oob)
                self.frame_id += 1
                fps_n += 1
                self.state = "LOCKED" if locked else "searching"
            now = time.monotonic()
            if now - fps_t >= 1:
                self.fps, fps_t, fps_n = fps_n / (now - fps_t), now, 0
        if self.state != "error":
            self.state = "stopped"


# ======================================================================== GUI

def run_gui():
    import tkinter as tk
    from tkinter import filedialog, ttk
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
        root = TkinterDnD.Tk()
        dnd = True
    except Exception:
        root = tk.Tk()
        dnd = False

    root.title("vbend")
    root.configure(bg="#111")
    PREVIEW_W, PREVIEW_H = 640, 360
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    events = queue.Queue()
    jobs = queue.Queue()
    state = {"cancel": threading.Event(), "busy": False, "monitor": None, "shown_id": -1, "speaker": None}

    # ---------- preview
    canvas = tk.Canvas(root, width=PREVIEW_W, height=PREVIEW_H, bg="black", highlightthickness=0)
    canvas.pack(side="top", padx=8, pady=(8, 4))
    photo = tk.PhotoImage(width=PREVIEW_W, height=PREVIEW_H)
    canvas.create_image(PREVIEW_W // 2, PREVIEW_H // 2, image=photo)
    canvas_text = canvas.create_text(PREVIEW_W // 2, PREVIEW_H // 2, fill="#888", justify="center",
                                     text="drop a video on Encode, a bent WAV on Decode,\n"
                                          "or start Live preview")
    status = tk.StringVar(value="ready")
    tk.Label(root, textvariable=status, bg="#111", fg="#ddd", anchor="w",
             font=("TkFixedFont", 9)).pack(fill="x", padx=8)

    def show(rgb):
        h, w = rgb.shape[:2]
        k = max(1, min(PREVIEW_W // w, PREVIEW_H // h))
        big = np.repeat(np.repeat(rgb, k, 0), k, 1)
        H, W = big.shape[:2]
        ppm = f"P6 {W} {H} 255 ".encode() + np.ascontiguousarray(big).tobytes()
        photo.configure(width=W, height=H)
        photo.configure(data=ppm, format="PPM")
        canvas.itemconfigure(canvas_text, text="")

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=8, pady=8)

    def drop_zone(parent, text, on_files, types):
        lbl = tk.Label(parent, text=text + ("\n(or click to browse)" if dnd else "\n(click to browse)"),
                       bg="#1d1d1d", fg="#bbb", relief="groove", bd=2, height=4, cursor="hand2")
        lbl.pack(fill="x", padx=6, pady=6)

        def browse(_e=None):
            paths = filedialog.askopenfilenames(filetypes=types)
            if paths:
                on_files(list(paths))
        lbl.bind("<Button-1>", browse)
        if dnd:
            lbl.drop_target_register(DND_FILES)
            lbl.dnd_bind("<<Drop>>", lambda e: on_files(list(root.tk.splitlist(e.data))))
        return lbl

    def row(parent):
        f = ttk.Frame(parent)
        f.pack(fill="x", padx=6, pady=2)
        return f

    # ---------- Encode tab
    enc = ttk.Frame(nb)
    nb.add(enc, text="Encode  (video -> wav)")
    enc_mode = tk.StringVar(value="gray")
    r = row(enc)
    ttk.Label(r, text="Mode").pack(side="left")
    cb = ttk.Combobox(r, textvariable=enc_mode, values=list(MODES), state="readonly", width=10)
    cb.pack(side="left", padx=6)
    enc_blurb = ttk.Label(r, text=MODES["gray"].blurb)
    enc_blurb.pack(side="left")
    cb.bind("<<ComboboxSelected>>", lambda e: enc_blurb.configure(text=MODES[enc_mode.get()].blurb))
    enc_audio = tk.BooleanVar(value=True)
    ttk.Checkbutton(row(enc), variable=enc_audio, text="also save the clip's own audio as <name>.audio.wav "
                    "(put it on its own track at 0:00 for reference)").pack(side="left")
    drop_zone(enc, "Drop videos here -> <name>.<mode>.wav next to them",
              lambda ps: add_jobs("encode", ps),
              [("Video", "*.mp4 *.mkv *.mov *.webm *.avi *.gif"), ("All", "*")])

    # ---------- Decode tab
    dec = ttk.Frame(nb)
    nb.add(dec, text="Decode  (wav -> mp4)")
    dec_mode, dec_oob = tk.StringVar(value="auto"), tk.StringVar(value="clip")
    dec_nudge, dec_scale = tk.IntVar(value=0), tk.IntVar(value=4)
    dec_invert, dec_sound = tk.BooleanVar(value=False), tk.StringVar(value="")
    r = row(dec)
    ttk.Label(r, text="Mode").pack(side="left")
    ttk.Combobox(r, textvariable=dec_mode, values=["auto"] + list(MODES), state="readonly",
                 width=10).pack(side="left", padx=6)
    ttk.Label(r, text="Out of range").pack(side="left", padx=(10, 0))
    ttk.Combobox(r, textvariable=dec_oob, values=OOB_MODES, state="readonly", width=6).pack(side="left", padx=6)
    ttk.Checkbutton(r, text="invert", variable=dec_invert).pack(side="left", padx=6)
    r = row(dec)
    ttk.Label(r, text="Nudge (samples)").pack(side="left")
    ttk.Spinbox(r, from_=-8000, to=8000, textvariable=dec_nudge, width=6).pack(side="left", padx=6)
    ttk.Label(r, text="Upscale").pack(side="left", padx=(10, 0))
    ttk.Spinbox(r, from_=1, to=12, textvariable=dec_scale, width=3).pack(side="left", padx=6)
    r = row(dec)
    ttk.Label(r, text="Soundtrack (optional)").pack(side="left")
    ttk.Entry(r, textvariable=dec_sound, width=34).pack(side="left", padx=6, fill="x", expand=True)
    ttk.Button(r, text="...", width=3, command=lambda: dec_sound.set(
        filedialog.askopenfilename() or dec_sound.get())).pack(side="left")
    ttk.Button(r, text="x", width=2, command=lambda: dec_sound.set("")).pack(side="left")
    drop_zone(dec, "Drop bent WAVs here -> <name>.mp4 next to them",
              lambda ps: add_jobs("decode", ps),
              [("WAV", "*.wav *.WAV"), ("All", "*")])

    # ---------- shared progress
    prog = ttk.Progressbar(root, maximum=1.0)
    prog.pack(fill="x", padx=8)
    r = ttk.Frame(root)
    r.pack(fill="x", padx=8, pady=(2, 8))
    log = tk.Text(r, height=5, bg="#161616", fg="#ccc", insertbackground="#ccc", relief="flat",
                  font=("TkFixedFont", 9), wrap="word")
    log.pack(side="left", fill="both", expand=True)
    ttk.Button(r, text="Cancel", command=lambda: state["cancel"].set()).pack(side="left", padx=(6, 0))

    def say(msg):
        log.insert("end", msg + "\n")
        log.see("end")

    # ---------- Live tab
    live = ttk.Frame(nb)
    nb.add(live, text="Live preview")
    live_oob, live_invert, live_fallback = tk.StringVar(value="clip"), tk.BooleanVar(), tk.StringVar(value="auto")
    r = row(live)
    ttk.Button(r, text="Create 192k output", command=lambda: say(sink_up()[1])).pack(side="left")
    ttk.Button(r, text="Remove output", command=lambda: say(sink_down()[1])).pack(side="left", padx=6)
    live_btn = ttk.Button(r, text="Start preview")
    live_btn.pack(side="left", padx=6)
    ttk.Button(r, text="Re-sync (r)", command=lambda: state["monitor"] and state["monitor"].resync()
               ).pack(side="left")
    r = row(live)
    ttk.Label(r, text="Out of range (o)").pack(side="left")
    for m in OOB_MODES:
        ttk.Radiobutton(r, text=m, value=m, variable=live_oob).pack(side="left")
    ttk.Checkbutton(r, text="invert (i)", variable=live_invert).pack(side="left", padx=8)
    r = row(live)
    ttk.Label(r, text="Nudge").pack(side="left")
    nudge_lbl = ttk.Label(r, text="+0", width=6)

    def nudge(d):
        mon = state["monitor"]
        if mon:
            mon.nudge += d
            nudge_lbl.configure(text=f"{mon.nudge:+d}")
    def row_len():
        mon = state["monitor"]
        name = (mon.mode or "gray").split()[0] if mon else "gray"
        return MODES[name].w // 2 if isinstance(MODES[name], Interleaved) else MODES[name].w
    for t, d in [("-row [", "-r"), ("-1 (-)", -1)]:
        ttk.Button(r, text=t, width=7, command=lambda d=d: nudge(-row_len() if d == "-r" else d)
                   ).pack(side="left", padx=2)
    nudge_lbl.pack(side="left")
    for t, d in [("+1 (=)", 1), ("+row ]", "+r")]:
        ttk.Button(r, text=t, width=7, command=lambda d=d: nudge(row_len() if d == "+r" else d)
                   ).pack(side="left", padx=2)
    ttk.Label(r, text="   Mode").pack(side="left")
    ttk.Combobox(r, textvariable=live_fallback, values=["auto"] + list(MODES), state="readonly", width=9
                 ).pack(side="left", padx=4)
    listen, spk_vol = tk.BooleanVar(value=False), tk.IntVar(value=25)
    spk_out = tk.StringVar()
    sinks = []
    r = row(live)
    ttk.Checkbutton(r, text="Listen", variable=listen, command=lambda: set_speaker()).pack(side="left")
    ttk.Label(r, text="on").pack(side="left", padx=(6, 2))
    out_cb = ttk.Combobox(r, textvariable=spk_out, state="readonly", width=28)
    out_cb.pack(side="left")

    def refresh_outputs():
        nonlocal sinks
        sinks = list_output_sinks() if not sink_check() else []
        out_cb.configure(values=[d for _, d in sinks])
        if sinks and spk_out.get() not in [d for _, d in sinks]:
            spk_out.set(sinks[0][1])
    ttk.Button(r, text="\u21bb", width=2, command=refresh_outputs).pack(side="left", padx=2)
    ttk.Label(r, text="  Vol").pack(side="left")
    vol_lbl = ttk.Label(r, text="", width=16)
    ttk.Scale(r, from_=0, to=100, orient="horizontal", length=120,
              command=lambda v: spk_vol.set(int(float(v)))).pack(side="left", padx=4)
    r.winfo_children()[-1].set(spk_vol.get())
    vol_lbl.pack(side="left")

    def set_speaker(*_):
        old = state.get("speaker")
        if old:
            old.close()
            state["speaker"] = None
        if listen.get():
            if not sinks:
                refresh_outputs()
            name = next((n for n, d in sinks if d == spk_out.get()), None)
            if not name:
                say("no audio output found to listen on")
                listen.set(False)
                return
            state["speaker"] = Speaker(name, spk_vol.get())
            say(f"listening on {spk_out.get()} (limited to -6 dBFS)")
        if state["monitor"]:
            state["monitor"].speaker = state.get("speaker")
    out_cb.bind("<<ComboboxSelected>>", set_speaker)
    root.after(200, refresh_outputs)

    ttk.Label(live, foreground="#888", wraplength=600, text="Play your DAW into 'vbend-192k'. The mode is detected "
              "automatically (set Mode to force one - needed for old files or when effects smear the sync); "
              "seeking re-locks in ~0.3 s.").pack(anchor="w", padx=6, pady=4)

    def toggle_live(file=None):
        mon = state["monitor"]
        if mon:
            mon.stop()
            state["monitor"] = None
            live_btn.configure(text="Start preview")
            say("live preview stopped")
            return
        mon = Monitor(file=file)
        mon.speaker = state.get("speaker")
        mon.start()
        state["monitor"] = mon
        live_btn.configure(text="Stop preview")
        say("live preview started" + (f" (file {file})" if file else ""))
    live_btn.configure(command=toggle_live)

    def key(e):
        if nb.index("current") != 2 or isinstance(e.widget, (tk.Entry, ttk.Entry, ttk.Spinbox, tk.Text)):
            return
        mon = state["monitor"]
        ch = e.char
        if ch == "o":
            live_oob.set(OOB_MODES[(OOB_MODES.index(live_oob.get()) + 1) % 4])
        elif ch == "i":
            live_invert.set(not live_invert.get())
        elif ch == "r" and mon:
            mon.resync()
        elif mon and ch in "-=[]":
            nudge({"-": -1, "=": 1, "[": -row_len(), "]": row_len()}[ch])
    root.bind("<Key>", key)

    # ---------- job runner
    def add_jobs(kind, paths):
        for p in paths:
            jobs.put((kind, p))
        kick()

    def kick():
        if state["busy"] or jobs.empty():
            return
        kind, path = jobs.get()
        state["busy"] = True
        state["cancel"].clear()
        opts = dict(mode=enc_mode.get(), audio=enc_audio.get()) if kind == "encode" else dict(
            mode=dec_mode.get(), oob=dec_oob.get(), nudge=dec_nudge.get(), invert=dec_invert.get(),
            scale=dec_scale.get(), sound=dec_sound.get().strip() or None, fallback="gray")

        def work():
            def progress(frac, rgb, text):
                events.put(("progress", frac, rgb, text))
            try:
                if kind == "encode":
                    s = encode_file(path, opts["mode"], progress=progress, cancel=state["cancel"])
                    msg = (f"encoded -> {s['out']}\n   {s['mode']}, {s['frames']} frames "
                           f"({s['seconds']:.1f} s), {s['size'] / 2**20:.0f} MB, {s['time']:.1f} s")
                    if opts["audio"]:
                        a = export_soundtrack(path)
                        msg += f"\n   audio -> {a}" if a else "\n   (no audio track in this clip)"
                else:
                    s = decode_file(path, mode_name=opts["mode"], oob=opts["oob"], nudge=opts["nudge"],
                                    invert=opts["invert"], scale=opts["scale"], soundtrack=opts["sound"],
                                    progress=progress, cancel=state["cancel"],
                                    fallback_mode=opts["fallback"])
                    msg = (f"decoded -> {s['out']}\n   mode {s['mode']}, sync {s['conf']:.0f}, "
                           f"{s['frames']} frames, {s['speed']:.0f}x real time, "
                           f"{s['oob_pct']:.2f}% out of range")
                    msg += "".join(f"\n   ! {w}" for w in s["warnings"])
                events.put(("done", msg))
            except Cancelled:
                events.put(("done", f"cancelled {os.path.basename(path)}"))
            except Exception as e:
                events.put(("done", f"FAILED {os.path.basename(path)}: {e}"))
        say(f"{kind} {path} ...")
        threading.Thread(target=work, daemon=True).start()

    def pump():
        try:
            while True:
                ev = events.get_nowait()
                if ev[0] == "progress":
                    prog["value"] = ev[1]
                    status.set(ev[3])
                    if ev[2] is not None and not state["monitor"]:
                        show(ev[2])
                elif ev[0] == "done":
                    prog["value"] = 0
                    say(ev[1])
                    status.set("ready")
                    state["busy"] = False
                    kick()
        except queue.Empty:
            pass
        mon = state["monitor"]
        if mon:
            lm = live_fallback.get()
            if (lm if lm != "auto" else None) != mon.force_mode:
                mon.force_mode = lm if lm != "auto" else None
                mon.resync()
            mon.oob, mon.invert = live_oob.get(), live_invert.get()
            if mon.frame_id != state["shown_id"] and mon.latest is not None:
                state["shown_id"] = mon.frame_id
                show(mon.latest)
            status.set(f"live | {mon.state} | mode {mon.mode or '?'} | sync {mon.conf:5.1f} | "
                       f"{mon.fps:4.1f} fps | oob {mon.oob} | nudge {mon.nudge:+d}"
                       + (" | INVERT" if mon.invert else ""))
            if mon.error:
                say(f"live preview error: {mon.error}")
                mon.error = None
                toggle_live()
        spk = state.get("speaker")
        if spk:
            spk.volume = spk_vol.get()
            if spk.error:
                say(spk.error)
                spk.error = None
                listen.set(False)
                set_speaker()
        db = Speaker.knob_db(spk_vol.get())
        vol_lbl.configure(text=("muted" if db is None else f"{db:+.0f} dB")
                          + (f" | lim {spk.gr_db:+.0f}" if spk and spk.gr_db < -0.5 else ""))
        root.after(15, pump)

    root.after(15, pump)
    run_gui.app = dict(root=root, jobs=jobs, kick=kick, add_jobs=add_jobs, toggle_live=toggle_live, nb=nb, state=state,
                       enc_mode=enc_mode, dec_sound=dec_sound, listen=listen, set_speaker=set_speaker,
                       spk_vol=spk_vol, spk_out=spk_out, refresh_outputs=refresh_outputs)
    if not dnd:
        say("tip: pip install tkinterdnd2 to drag and drop files onto the window")
    root.mainloop()
    if state["monitor"]:
        state["monitor"].stop()
    if state["speaker"]:
        state["speaker"].close()


# ======================================================================== CLI

def run_live_cli(args):
    """Preview without the full GUI (Tk window with just the picture)."""
    import tkinter as tk
    mon = Monitor(device=args.device, file=args.file)
    mon.oob = args.oob
    mon.start()
    root = tk.Tk()
    root.title("vbend live")
    lbl = tk.Label(root, bg="black")
    lbl.pack()
    photo = tk.PhotoImage(width=640, height=360)
    lbl.configure(image=photo)
    seen = [-1]

    def tick():
        if mon.error:
            print("error:", mon.error)
            root.destroy()
            return
        if mon.frame_id != seen[0] and mon.latest is not None:
            seen[0] = mon.frame_id
            rgb = mon.latest
            k = max(1, min(640 // rgb.shape[1], 360 // rgb.shape[0]))
            big = np.repeat(np.repeat(rgb, k, 0), k, 1)
            photo.configure(width=big.shape[1], height=big.shape[0])
            photo.configure(data=f"P6 {big.shape[1]} {big.shape[0]} 255 ".encode() + big.tobytes(),
                            format="PPM")
        root.title(f"vbend live | {mon.state} | {mon.mode} | sync {mon.conf:.0f} | {mon.fps:.1f} fps")
        root.after(15, tick)
    root.bind("q", lambda e: root.destroy())
    root.after(15, tick)
    root.mainloop()
    mon.stop()


def main():
    if len(sys.argv) == 1:
        return run_gui()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"vbend {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gui")
    e = sub.add_parser("encode")
    e.add_argument("inputs", nargs="+")
    e.add_argument("-m", "--mode", choices=MODES, default="gray")
    e.add_argument("-o", "--output")
    e.add_argument("--audio", action="store_true", help="also save <name>.audio.wav (the clip's own sound)")
    d = sub.add_parser("decode")
    d.add_argument("inputs", nargs="+")
    d.add_argument("-m", "--mode", choices=["auto"] + list(MODES), default="auto")
    d.add_argument("-o", "--output")
    d.add_argument("--oob", choices=OOB_MODES, default="clip")
    d.add_argument("--nudge", type=int, default=0)
    d.add_argument("--invert", action="store_true")
    d.add_argument("--scale", type=int, default=4)
    d.add_argument("--codec", default="auto")
    d.add_argument("--soundtrack")
    d.add_argument("--fallback", choices=MODES, default="gray", help="mode for old v1/v2 WAVs")
    lv = sub.add_parser("live")
    lv.add_argument("--device", default="vbend")
    lv.add_argument("--file", help="preview a WAV in real time instead of the live output")
    lv.add_argument("--oob", choices=OOB_MODES, default="clip")
    s = sub.add_parser("sink")
    s.add_argument("action", choices=["up", "down", "status"])
    a = ap.parse_args()

    if a.cmd == "gui":
        return run_gui()
    if a.cmd == "encode":
        for p in a.inputs:
            st = encode_file(p, a.mode, out=a.output if len(a.inputs) == 1 else None)
            print(f"{p} -> {st['out']}  ({st['mode']}, {st['frames']} frames, "
                  f"{st['size'] / 2**20:.1f} MB, {st['time']:.1f} s)")
            if a.audio:
                print("   audio ->", export_soundtrack(p) or "(no audio track)")
    elif a.cmd == "decode":
        for p in a.inputs:
            st = decode_file(p, out=a.output if len(a.inputs) == 1 else None, mode_name=a.mode, oob=a.oob,
                             nudge=a.nudge, invert=a.invert, scale=a.scale, codec=a.codec,
                             soundtrack=a.soundtrack, fallback_mode=a.fallback)
            print(f"{p} -> {st['out']}  (mode {st['mode']}, offset {st['offset']}, sync {st['conf']:.0f}, "
                  f"{st['frames']} frames, {st['speed']:.0f}x real time, {st['oob_pct']:.2f}% oob, "
                  f"{st['writer']})")
            for w in st["warnings"]:
                print("  !", w)
    elif a.cmd == "live":
        run_live_cli(a)
    elif a.cmd == "sink":
        if a.action == "status":
            err = sink_check()
            print(err or ("vbend output exists" if sink_exists() else "no vbend output"))
            if shutil.which("pw-metadata"):
                print(sh(["pw-metadata", "-n", "settings"]).stdout)
        else:
            ok, msg = (sink_up if a.action == "up" else sink_down)()
            print(msg)
            sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
