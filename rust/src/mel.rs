//! Log mel spectrogram, matching `snfa/stft.py`.
//!
//! ISC License
//!
//! Copyright (c) 2013--2023, librosa development team.
//!
//! Permission to use, copy, modify, and/or distribute this software for any
//! purpose with or without fee is hereby granted, provided that the above
//! copyright notice and this permission notice appear in all copies.
//!
//! THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
//! WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
//! MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
//! ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
//! WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
//! ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
//! OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

/// Offset added before the logarithm, as in the Python implementation.
const LOG_EPSILON: f64 = 1e-7;

fn hz_to_mel(f: f64) -> f64 {
  2595.0 * (1.0 + f / 700.0).log10()
}

fn mel_to_hz(m: f64) -> f64 {
  700.0 * (10f64.powf(m / 2595.0) - 1.0)
}

/// `np.linspace(start, stop, num)`, whose last point is exactly `stop`.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
  let step = (stop - start) / (num - 1) as f64;
  let mut points: Vec<f64> = (0..num).map(|i| start + i as f64 * step).collect();
  points[num - 1] = stop;
  points
}

/// `np.hanning`: a symmetric Hann window.
fn hanning(size: usize) -> Vec<f64> {
  if size == 1 {
    return vec![1.0];
  }
  (0..size)
    .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (size - 1) as f64).cos())
    .collect()
}

/// Triangular filters, `n_mels` rows of `n_fft / 2 + 1` bins, unnormalized.
fn mel_filterbank(sample_rate: u32, n_fft: usize, n_mels: usize) -> Vec<Vec<f64>> {
  let fmax = f64::from(sample_rate / 2);
  let mel_points = linspace(hz_to_mel(0.0), hz_to_mel(fmax), n_mels + 2);
  let bins: Vec<usize> = mel_points
    .iter()
    .map(|&m| ((n_fft + 1) as f64 * mel_to_hz(m) / f64::from(sample_rate)).floor() as usize)
    .collect();
  let mut filterbank = vec![vec![0.0; n_fft / 2 + 1]; n_mels];
  for (i, filter) in filterbank.iter_mut().enumerate() {
    let (left, center, right) = (bins[i], bins[i + 1], bins[i + 2]);
    for (j, weight) in filter.iter_mut().enumerate().take(center).skip(left) {
      *weight = (j - left) as f64 / (center - left) as f64;
    }
    for (j, weight) in filter.iter_mut().enumerate().take(right).skip(center) {
      *weight = (right - j) as f64 / (right - center) as f64;
    }
  }
  filterbank
}

pub struct MelSpectrogram {
  n_fft: usize,
  hop_size: usize,
  window: Vec<f64>,
  filterbank: Vec<Vec<f64>>,
  fft: Arc<dyn Fft<f64>>,
}

impl MelSpectrogram {
  pub fn new(
    sample_rate: u32,
    n_fft: usize,
    win_size: usize,
    hop_size: usize,
    n_mels: usize,
  ) -> Self {
    Self {
      n_fft,
      hop_size,
      window: hanning(win_size),
      filterbank: mel_filterbank(sample_rate, n_fft, n_mels),
      fft: FftPlanner::new().plan_fft_forward(n_fft),
    }
  }

  /// Number of frames for `n_samples`; frames are not padded, so a waveform
  /// shorter than one window has none.
  pub fn frame_count(&self, n_samples: usize) -> usize {
    let win_size = self.window.len();
    if n_samples < win_size {
      0
    } else {
      (n_samples - win_size) / self.hop_size + 1
    }
  }

  /// Natural-log mel power spectrogram, one row of `n_mels` values per frame.
  pub fn process(&self, signal: &[f32]) -> Vec<Vec<f64>> {
    let mut buffer = vec![Complex::new(0.0, 0.0); self.n_fft];
    let mut power = vec![0.0; self.n_fft / 2 + 1];
    (0..self.frame_count(signal.len()))
      .map(|frame| {
        let start = frame * self.hop_size;
        let samples = &signal[start..start + self.window.len()];
        // `np.fft.rfft(frame, n=n_fft)` crops or zero-pads the windowed frame.
        buffer.fill(Complex::new(0.0, 0.0));
        for ((output, &sample), &weight) in buffer.iter_mut().zip(samples).zip(&self.window) {
          output.re = f64::from(sample) * weight;
        }
        self.fft.process(&mut buffer);
        for (output, bin) in power.iter_mut().zip(&buffer) {
          *output = bin.norm_sqr();
        }
        self
          .filterbank
          .iter()
          .map(|filter| {
            let energy: f64 = filter.iter().zip(&power).map(|(w, p)| w * p).sum();
            (energy + LOG_EPSILON).ln()
          })
          .collect()
      })
      .collect()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn linspace_ends_exactly_at_stop() {
    let points = linspace(0.0, 1.0 / 3.0, 7);
    assert_eq!(points[0], 0.0);
    assert_eq!(points[6], 1.0 / 3.0);
  }

  #[test]
  fn hanning_is_symmetric_with_zero_edges() {
    let window = hanning(9);
    assert_eq!(window[0], 0.0);
    assert!((window[4] - 1.0).abs() < 1e-12);
    for i in 0..9 {
      assert!((window[i] - window[8 - i]).abs() < 1e-12);
    }
  }

  #[test]
  fn filterbank_filters_are_triangles_peaking_at_one() {
    let filterbank = mel_filterbank(16000, 1024, 80);
    assert_eq!(filterbank.len(), 80);
    for filter in &filterbank {
      assert_eq!(filter.len(), 513);
      let peak = filter.iter().cloned().fold(0.0, f64::max);
      assert!(peak <= 1.0);
    }
    // Wide high-frequency filters reach their center bin exactly.
    assert_eq!(filterbank[79].iter().cloned().fold(0.0, f64::max), 1.0);
  }

  #[test]
  fn frames_are_not_padded() {
    let mel = MelSpectrogram::new(16000, 1024, 1024, 160, 80);
    assert_eq!(mel.frame_count(1023), 0);
    assert_eq!(mel.frame_count(1024), 1);
    assert_eq!(mel.frame_count(1024 + 159), 1);
    assert_eq!(mel.frame_count(1024 + 160), 2);
    assert!(mel.process(&[0.0; 1000]).is_empty());
  }

  #[test]
  fn silence_is_the_log_epsilon() {
    let mel = MelSpectrogram::new(16000, 1024, 1024, 160, 80);
    let frames = mel.process(&[0.0; 2048]);
    assert_eq!(frames.len(), 7);
    for value in frames.iter().flatten() {
      assert_eq!(*value, LOG_EPSILON.ln());
    }
  }

  #[test]
  fn a_sine_peaks_in_its_mel_band() {
    let mel = MelSpectrogram::new(16000, 1024, 1024, 160, 80);
    let signal: Vec<f32> = (0..4096)
      .map(|n| (2.0 * std::f32::consts::PI * 1000.0 * n as f32 / 16000.0).sin())
      .collect();
    let frame = &mel.process(&signal)[0];
    let loudest = (0..80)
      .max_by(|&a, &b| frame[a].total_cmp(&frame[b]))
      .unwrap();
    let filterbank = mel_filterbank(16000, 1024, 80);
    // 1 kHz is bin 64 of a 1024-point FFT at 16 kHz.
    assert!(filterbank[loudest][64] > 0.0);
  }
}
