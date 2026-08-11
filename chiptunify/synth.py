"""Numpy NES-APU-flavoured synthesizer.

No nesmdb, no Python 2, no compiled VGMPlay — the NES sound chip is just
two pulse waves, a quantized triangle and an LFSR noise source, which is
a few dozen lines of numpy.

Authentic touches:
  * pulse/triangle pitches are quantized to the NES's timer periods,
    so you get the real slightly-off-tune chiptune character
  * triangle wave uses the hardware's 16-step staircase
  * noise comes from the same 15-bit LFSR design as the 2A03
"""

from __future__ import annotations

import wave

import numpy as np

SAMPLE_RATE = 44100
NTSC_CPU = 1789773.0

# NES noise channel period table (NTSC), index 0-15
NOISE_PERIODS = [4, 8, 16, 32, 64, 96, 128, 160,
                 202, 254, 380, 508, 762, 1016, 2034, 4068]

# Hardware triangle: 15 down to 0 and back up (32-step sequence)
TRI_TABLE = np.array(list(range(15, -1, -1)) + list(range(0, 16)),
                     dtype=np.float64) / 15.0

# Channel mix levels
MIX_P1 = 0.30
MIX_P2 = 0.30
MIX_TR = 0.45
MIX_NO = 0.22


def midi_to_freq(pitch: int) -> float:
    return 440.0 * 2.0 ** ((pitch - 69) / 12.0)


def _quantize_pulse(freq: float) -> float:
    period = max(8, int(round(NTSC_CPU / (16.0 * freq) - 1)))
    return NTSC_CPU / (16.0 * (period + 1))


def _quantize_tri(freq: float) -> float:
    period = max(2, int(round(NTSC_CPU / (32.0 * freq) - 1)))
    return NTSC_CPU / (32.0 * (period + 1))


def _apply_edge_fades(chunk: np.ndarray, sr: int) -> np.ndarray:
    """Tiny 1.5 ms fades so note edges don't click."""
    n = len(chunk)
    fade = min(int(0.0015 * sr), n // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, endpoint=False)
        chunk[:fade] *= ramp
        chunk[-fade:] *= ramp[::-1]
    return chunk


def _render_pulse(events, n_samples: int, sr: int, duty: float,
                  quantize: bool) -> np.ndarray:
    out = np.zeros(n_samples, dtype=np.float64)
    for e in events:
        i0 = int(round(e.start * sr))
        i1 = min(int(round(e.end * sr)), n_samples)
        if i1 <= i0:
            continue
        f = midi_to_freq(e.pitch)
        if quantize:
            f = _quantize_pulse(f)
        t = np.arange(i1 - i0, dtype=np.float64) / sr
        phase = (t * f) % 1.0
        amp = (e.volume / 15.0)
        chunk = np.where(phase < duty, amp, -amp)
        out[i0:i1] += _apply_edge_fades(chunk, sr)
    return out


def _render_triangle(events, n_samples: int, sr: int,
                     quantize: bool) -> np.ndarray:
    out = np.zeros(n_samples, dtype=np.float64)
    for e in events:
        i0 = int(round(e.start * sr))
        i1 = min(int(round(e.end * sr)), n_samples)
        if i1 <= i0:
            continue
        f = midi_to_freq(e.pitch)
        if quantize:
            f = _quantize_tri(f)
        t = np.arange(i1 - i0, dtype=np.float64) / sr
        idx = ((t * f * 32.0).astype(np.int64)) % 32
        chunk = (TRI_TABLE[idx] - 0.5) * 2.0   # bipolar, fixed level
        out[i0:i1] += _apply_edge_fades(chunk, sr)
    return out


def _lfsr_bits(n_steps: int, seed: int = 1) -> np.ndarray:
    """15-bit NES LFSR: feedback = bit0 XOR bit1."""
    bits = np.empty(n_steps, dtype=np.uint8)
    reg = seed & 0x7FFF or 1
    for i in range(n_steps):
        bits[i] = reg & 1
        fb = (reg ^ (reg >> 1)) & 1
        reg = (reg >> 1) | (fb << 14)
    return bits


def _render_noise(events, n_samples: int, sr: int) -> np.ndarray:
    out = np.zeros(n_samples, dtype=np.float64)
    # Anything faster than this is inaudible after 44.1 kHz sampling
    max_step_rate = sr * 4
    for e in events:
        i0 = int(round(e.start * sr))
        i1 = min(int(round(e.end * sr)), n_samples)
        if i1 <= i0:
            continue
        period_idx = max(1, min(16, int(e.pitch))) - 1
        rate = min(NTSC_CPU / NOISE_PERIODS[period_idx], max_step_rate)
        n_steps = max(2, int((i1 - i0) / sr * rate) + 1)
        bits = _lfsr_bits(n_steps)
        pick = (np.arange(i1 - i0) * rate / sr).astype(np.int64)
        pick = np.minimum(pick, n_steps - 1)
        amp = (e.volume / 15.0)
        chunk = (bits[pick].astype(np.float64) * 2.0 - 1.0) * amp
        out[i0:i1] += _apply_edge_fades(chunk, sr)
    return out


def render_arrangement(arr, sample_rate: int = SAMPLE_RATE,
                       duty_p1: float = 0.5, duty_p2: float = 0.25,
                       quantize_pitch: bool = True,
                       tail: float = 0.15) -> np.ndarray:
    """Render a NesArrangement to a mono int16 numpy array."""
    sr = sample_rate
    n_samples = int((arr.duration + tail) * sr)
    if n_samples <= 0:
        raise ValueError("Arrangement is empty — nothing to render")

    mix = (MIX_P1 * _render_pulse(arr.p1, n_samples, sr, duty_p1, quantize_pitch)
           + MIX_P2 * _render_pulse(arr.p2, n_samples, sr, duty_p2, quantize_pitch)
           + MIX_TR * _render_triangle(arr.tr, n_samples, sr, quantize_pitch)
           + MIX_NO * _render_noise(arr.no, n_samples, sr))

    peak = np.max(np.abs(mix))
    if peak > 0:
        mix *= 0.89 / peak
    return (mix * 32767).astype(np.int16)


def write_wav(samples: np.ndarray, path: str,
              sample_rate: int = SAMPLE_RATE) -> None:
    """Write an int16 mono array to a WAV file (stdlib only)."""
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples.astype(np.int16).tobytes())
