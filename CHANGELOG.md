# Changelog

## 3.2.0
- Live preview can **listen**: plays the vbend stream on any output, 48 kHz, DC-blocked,
  volume knob starting at -45 dB, limiter capped at -6 dBFS.
- Encoding can also save each clip's own sound as `<name>.audio.wav` for a reference track.
- `tools/batch_encode.py` converts a whole folder (any mode, `--audio`, `--force`).

## 3.1.0
- Forcing a mode also restricts the sync search to that mode (fixes misaligned decodes when
  effects weaken the sync). Live preview gets a Mode override.

## 3.0.0
- Single file with a window: Encode / Decode / Live tabs, drag & drop, batch queues.
- New color modes `ysplit` (L = brightness, R = color) and `rgb` (planar).
- Per-mode sync patterns: the decoder identifies the mode by itself.
- Optional soundtrack muxing when decoding.

## 2.x
- Live decoder with continuous re-sync; PipeWire 192 kHz null-sink helper.

## 1.0
- Proof of concept: encode/decode scripts, gray / yuv420 / packed444, correlation-based sync.
