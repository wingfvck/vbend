"""Round-trip and sync tests. Run with `pytest` or `python3 tests/test_vbend.py`.

Needs ffmpeg on PATH (generates its own test clip).
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import vbend as V  # noqa: E402

TMP = tempfile.mkdtemp(prefix="vbend-test-")
CLIP = os.path.join(TMP, "clip.mp4")


def setup_module(_=None):
    assert shutil.which("ffmpeg"), "ffmpeg is required for the tests"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "3",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", CLIP], check=True)


def teardown_module(_=None):
    shutil.rmtree(TMP, ignore_errors=True)


def decode_video(path, w, h):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    W, H = w + w % 2, h + h % 2
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)[:, :h, :w]


def test_every_mode_round_trips_exactly():
    for name, mode in V.MODES.items():
        wav = os.path.join(TMP, f"{name}.wav")
        st = V.encode_file(CLIP, name, out=wav)
        assert st["frames"] == 72, name
        info = sf.info(wav)
        assert (info.samplerate, info.channels, info.subtype) == (192000, 2, "FLOAT")
        frames = list(V.read_frames(CLIP, mode.w, mode.h))
        expected = [V.decode_block(V.encode_block(f, mode), mode)[0] for f in frames]
        out = os.path.join(TMP, f"{name}.mkv")
        d = V.decode_file(wav, out=out, scale=1, codec="ffv1")        # lossless, mode auto-detected
        assert d["mode"] == name and d["offset"] == 0 and not d["warnings"], (name, d)
        got = decode_video(out, mode.w, mode.h)
        assert len(got) == len(expected)
        assert all(np.array_equal(g, e) for g, e in zip(got, expected)), name


def test_sync_survives_latency_gain_blur_echo_and_phase_flip():
    for name in V.MODES:
        x, _ = sf.read(os.path.join(TMP, f"{name}.wav"), dtype="float32")
        y = np.concatenate([np.zeros((777, 2), np.float32), x]) * -0.6
        k = np.ones(5) / 5
        y = np.stack([np.convolve(y[:, c], k, "same") for c in range(2)], 1)
        y[4000:] += 0.5 * y[:-4000].copy()
        key, off, conf, pol = V.find_sync(y)
        assert key == name and abs(off - 777) <= 3 and pol == -1 and conf > V.LOCK_CONF, (name, key, off, conf)


def test_forced_mode_and_noise_floor():
    x, _ = sf.read(os.path.join(TMP, "rgb.wav"), dtype="float32")
    key, off, conf, _ = V.find_sync(x, keys=("rgb",))
    assert key == "rgb" and off == 0
    noise = np.random.default_rng(1).normal(0, 0.5, (V.P * 48, 2)).astype(np.float32)
    assert V.find_sync(noise)[2] < V.LOCK_CONF / 2


def test_out_of_range_policies():
    s = np.array([-1.5, -1, 0, 1, 1.5, np.nan], np.float32)
    assert list(V.from_audio(s, 255, "clip")[0]) == [0, 0, 128, 255, 255, 0]
    assert list(V.from_audio(s, 255, "white")[0]) == [255, 0, 128, 255, 255, 0]
    assert list(V.from_audio(s, 255, "fold")[0]) == [64, 0, 128, 255, 191, 0]
    assert list(V.from_audio(s, 255, "wrap")[0]) == [191, 0, 128, 255, 64, 0]


def test_speaker_limiter_never_exceeds_ceiling():
    class Quiet(V.Speaker):
        def _loop(self):
            pass
    x, _ = sf.read(os.path.join(TMP, "gray.wav"), dtype="float32")
    x[5000:5100] = np.nan
    spk = Quiet("none", volume=100)
    y = np.concatenate([spk.process(x[i:i + 1024]) for i in range(0, len(x), 1024)])
    assert len(y) == len(x) // 4 and np.isfinite(y).all()
    assert np.abs(y).max() <= spk.CEILING + 1e-6
    assert abs(float(y[48000:].mean())) < 0.01                   # DC removed


def test_soundtrack_export():
    out = V.export_soundtrack(CLIP)
    info = sf.info(out)
    assert info.samplerate == 192000 and info.channels == 2 and abs(info.duration - 3) < 0.1


if __name__ == "__main__":
    setup_module()
    try:
        for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
            fn()
            print("ok  ", fn.__name__)
    finally:
        teardown_module()
    print("all tests passed")
