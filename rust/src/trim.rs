//! Leading and trailing silence trimming, matching `snfa/utils.py`.
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

use std::ops::Range;

const AMIN: f64 = 1e-10;

/// The sample range of `audio` (mono) without its leading and trailing
/// silence: frames quieter than `top_db` below the loudest frame, measured by
/// RMS over `frame_length` samples every `hop_length`. `None` when the audio
/// is shorter than one frame.
///
/// Python's `trim_audio(audio, top_db, frame_length, hop_length)` returns
/// `audio[range]` with this range; its defaults are 20 dB, 2048 and 512.
pub fn trim_audio(
  audio: &[f32],
  top_db: f64,
  frame_length: usize,
  hop_length: usize,
) -> Option<Range<usize>> {
  if frame_length == 0 || hop_length == 0 || audio.len() < frame_length {
    return None;
  }
  let n_frames = 1 + (audio.len() - frame_length) / hop_length;
  let energies: Vec<f64> = (0..n_frames)
    .map(|i| {
      let frame = &audio[i * hop_length..i * hop_length + frame_length];
      let power = frame
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>();
      (power / frame_length as f64).sqrt()
    })
    .collect();
  let reference = energies.iter().cloned().fold(0.0, f64::max);
  let db = |energy: f64| 10.0 * AMIN.max(energy * energy).log10();
  let log_spec: Vec<f64> = energies.iter().map(|&e| db(e) - db(reference)).collect();
  let max = log_spec.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
  let clipped: Vec<f64> = log_spec.iter().map(|&v| v.max(max - top_db)).collect();
  let threshold = clipped.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - top_db;
  let start_frame = clipped.iter().position(|&v| v > threshold)?;
  let end_frame = clipped.iter().rposition(|&v| v > threshold)?;
  let start = start_frame * hop_length;
  let end = ((end_frame + 1) * hop_length + frame_length).min(audio.len());
  Some(start..end)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn trims_silence_around_a_tone() {
    let mut audio = vec![0.0f32; 8000];
    for (n, sample) in audio[3000..5000].iter_mut().enumerate() {
      *sample = (n as f32 * 0.3).sin() * 0.5;
    }
    let range = trim_audio(&audio, 20.0, 1024, 256).unwrap();
    assert!(range.start <= 3000 && range.start > 1000, "{range:?}");
    assert!(range.end >= 5000 && range.end < 7000, "{range:?}");
  }

  #[test]
  fn constant_audio_is_kept_whole() {
    // Every frame is as loud as the loudest one, silence included.
    assert_eq!(trim_audio(&[0.0; 4096], 20.0, 1024, 256), Some(0..4096));
    assert_eq!(trim_audio(&[0.25; 4096], 20.0, 1024, 256), Some(0..4096));
  }

  #[test]
  fn audio_shorter_than_a_frame_is_rejected() {
    assert_eq!(trim_audio(&[0.5; 100], 20.0, 1024, 256), None);
  }
}
