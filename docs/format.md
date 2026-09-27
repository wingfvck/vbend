# vbend stream format

A vbend WAV is plain audio: **stereo, 192 000 Hz, 32-bit float** (RF64 above ~4 GB).
Video runs at a constant **24 fps**, so every video frame owns exactly **8000 stereo samples**
(41.67 ms). Nothing else is stored: no header, no metadata. Everything the decoder needs is in
the samples, so the file survives any DAW round trip that doesn't resample it.

## One frame (per channel, 8000 samples)

```
sample 0            512                       8000 - N            8000
       | sync burst | silence (guard)         | pixel payload (N)  |
```

- **Sync burst**: 512 samples of ±0.9, identical on L and R. The ±1 sequence is
  `numpy.random.default_rng(seed).integers(0, 2, 512) * 2 - 1` with
  `seed = zlib.crc32(b"vbend:" + mode_name)`. Each mode has its own burst, so the burst also
  tells the decoder which mode the file is in. (Files from v1/v2 used seed `0x5EEDB3ED` for
  every mode and need the mode given explicitly.)
- **Payload**: the last N samples of each channel. The pixels always end exactly on the frame
  boundary.
- **Pixel values**: an integer `v` in `0..max` becomes `v * 2 / max - 1` (so 0 → -1.0,
  max → +1.0). 8- and 12-bit values survive float32 exactly.

## Modes

| mode | size | max | L channel | R channel | N per channel |
|---|---|---|---|---|---|
| `gray` | 160×90 | 255 | even values of the stream | odd values | 7200 |
| `yuv420` | 128×72 | 255 | even values | odd values | 6912 |
| `rgb` | 96×52 | 255 | even values | odd values | 7488 |
| `packed444` | 160×90 | 4095 | even values | odd values | 7200 |
| `ysplit` | 112×64 | 255 | Y plane (112×64) | Cb (56×64) then Cr (56×64) | 7168 |

"Stream" modes (all but `ysplit`) build one row-major value stream per frame and deal it out
across the channels like interleaved audio (`L0 R0 L1 R1 ...`):

- `gray`: luma (OpenCV `RGB2GRAY`).
- `yuv420`: OpenCV I420: Y plane, then U, then V (each chroma plane 64×36).
- `rgb`: R plane, then G, then B.
- `packed444`: one value per pixel, `(R>>4)<<8 | (G>>4)<<4 | (B>>4)`; decoded nibbles × 17.

`ysplit` uses OpenCV `RGB2YCrCb` (BT.601, full range); chroma is halved horizontally.

## Finding the frames again

Effects add latency, change gain, flip polarity and smear things, so the decoder never
assumes frame 0 starts at sample 0:

1. Sum L+R over the first 48 frames (2 s) and average them frame period by frame period. The
   picture changes from frame to frame and washes out; the burst repeats and adds up.
2. Correlate that one averaged period against each mode's burst using GCC-PHAT (phase only),
   which ignores EQ, lowpass and distortion as long as some band of the burst survives.
3. The highest peak gives the mode, the offset (0..7999) and the polarity. Its height over
   the median is the confidence; above 40 counts as locked (pure noise scores about 7).

The live preview does the same over a sliding 8-frame window every 3 frames and needs two
agreeing estimates before it jumps, so seeking in the DAW re-locks in about 0.3 s.

## Out-of-range samples

Effects can push samples past ±1. The decoder maps them with one of: `clip` (black/white),
`white` (anything outside → white), `wrap` (modulo, overflow restarts from black) or `fold`
(mirror back into range: solarize). NaN and inf become black.
