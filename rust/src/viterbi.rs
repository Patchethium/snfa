//! Viterbi forced alignment, matching `snfa/viterbi.py`, adapted from the
//! `torchaudio` forced alignment tutorial
//! <https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html>.
//!
//! BSD 2-Clause License
//!
//! Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
//! All rights reserved.
//!
//! Redistribution and use in source and binary forms, with or without
//! modification, are permitted provided that the following conditions are met:
//!
//! * Redistributions of source code must retain the above copyright notice, this
//!   list of conditions and the following disclaimer.
//!
//! * Redistributions in binary form must reproduce the above copyright notice,
//!   this list of conditions and the following disclaimer in the documentation
//!   and/or other materials provided with the distribution.
//!
//! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//! DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//! FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//! OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use crate::{Error, Result};

/// One aligned phoneme, in milliseconds from the start of the waveform.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
  pub phoneme: String,
  pub start_ms: u32,
  pub end_ms: u32,
}

/// `log(softmax(x))` over the non-blank logits, computed as numpy does.
fn log_softmax(logits: &[f64]) -> Vec<f64> {
  let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
  let exp: Vec<f64> = logits.iter().map(|x| (x - max).exp()).collect();
  let sum: f64 = exp.iter().sum();
  exp.iter().map(|e| (e / sum).ln()).collect()
}

/// `np.maximum`, which propagates NaN unlike `f64::max`.
fn numpy_maximum(a: f64, b: f64) -> f64 {
  if a.is_nan() || b.is_nan() {
    f64::NAN
  } else {
    a.max(b)
  }
}

/// Trellis of `frames × tokens` scores. `tokens` index the phoneme set, whose
/// blank (index 0) is dropped from the emissions.
fn trellis(emission: &[Vec<f64>], tokens: &[usize]) -> Vec<Vec<f64>> {
  let frames = emission.len();
  let n = tokens.len();
  let mut trellis: Vec<Vec<f64>> = emission
    .iter()
    .map(|row| tokens.iter().map(|&token| row[token - 1]).collect())
    .collect();
  // `trellis[1:, 0] = cumsum(trellis[1:, 0])`: the first frame is not summed.
  for t in 2..frames {
    trellis[t][0] += trellis[t - 1][0];
  }
  trellis[0][1..].fill(f64::NEG_INFINITY);
  // `trellis[-n + 1:, 0] = inf`, which covers every frame when n == 1.
  let first_forced = if n == 1 {
    0
  } else {
    frames.saturating_sub(n - 1)
  };
  for row in &mut trellis[first_forced..] {
    row[0] = f64::INFINITY;
  }
  for t in 0..frames - 1 {
    for j in 1..n {
      let best = numpy_maximum(trellis[t][j], trellis[t][j - 1]);
      trellis[t + 1][j] += best;
    }
  }
  trellis
}

/// Token index per frame, from the last frame and token back to the first.
fn backtrack(trellis: &[Vec<f64>]) -> Result<Vec<usize>> {
  let mut t = trellis.len() - 1;
  let mut j = trellis[0].len() - 1;
  let mut path = vec![0; trellis.len()];
  path[t] = j;
  while j > 0 {
    if t == 0 {
      return Err(Error::AlignmentFailed);
    }
    let stayed = trellis[t - 1][j];
    let changed = trellis[t - 1][j - 1];
    t -= 1;
    if changed > stayed {
      j -= 1;
    }
    path[t] = j;
  }
  // The frames before were spent on the first token, already 0 in `path`.
  Ok(path)
}

/// Aligns per-frame logits (blank first) to `tokens`, returning each token's
/// index in `tokens` with its start and end frame.
pub fn viterbi(logits: &[Vec<f64>], tokens: &[usize]) -> Result<Vec<(usize, usize, usize)>> {
  if logits.is_empty() || tokens.is_empty() {
    return Err(Error::AlignmentFailed);
  }
  let emission: Vec<Vec<f64>> = logits.iter().map(|row| log_softmax(&row[1..])).collect();
  let path = backtrack(&trellis(&emission, tokens))?;
  let mut segments: Vec<(usize, usize, usize)> = Vec::new();
  for (frame, &token) in path.iter().enumerate() {
    match segments.last_mut() {
      Some(last) if last.0 == token => last.2 = frame + 1,
      _ => segments.push((token, frame, frame + 1)),
    }
  }
  Ok(segments)
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Logits favouring `best[t]` (an index into the phoneme set) at frame `t`.
  fn logits(best: &[usize], classes: usize) -> Vec<Vec<f64>> {
    best
      .iter()
      .map(|&b| {
        (0..classes)
          .map(|c| if c == b { 5.0 } else { 0.0 })
          .collect()
      })
      .collect()
  }

  #[test]
  fn log_softmax_normalizes() {
    let values = log_softmax(&[1.0, 2.0, 3.0]);
    let total: f64 = values.iter().map(|v| v.exp()).sum();
    assert!((total - 1.0).abs() < 1e-12);
    assert!(values[2] > values[1] && values[1] > values[0]);
  }

  #[test]
  fn follows_the_emissions() {
    // Phonemes 1, 2, 3 for 2, 3 and 1 frames.
    let segments = viterbi(&logits(&[1, 1, 2, 2, 2, 3], 4), &[1, 2, 3]).unwrap();
    assert_eq!(segments, [(0, 0, 2), (1, 2, 5), (2, 5, 6)]);
  }

  #[test]
  fn repeated_tokens_get_separate_segments() {
    let segments = viterbi(&logits(&[1, 1, 2, 1, 1], 3), &[1, 2, 1]).unwrap();
    assert_eq!(segments, [(0, 0, 2), (1, 2, 3), (2, 3, 5)]);
  }

  #[test]
  fn every_token_gets_a_frame_when_frames_equal_tokens() {
    // The emissions favour phoneme 1 throughout, yet each token is visited.
    let segments = viterbi(&logits(&[1, 1, 1], 4), &[1, 2, 3]).unwrap();
    assert_eq!(segments, [(0, 0, 1), (1, 1, 2), (2, 2, 3)]);
  }

  #[test]
  fn segments_cover_every_frame_in_order() {
    let segments = viterbi(&logits(&[3, 1, 2, 3, 1, 2, 3, 2], 4), &[1, 2, 3, 2]).unwrap();
    assert_eq!(segments.len(), 4);
    assert_eq!(segments[0].1, 0);
    assert_eq!(segments[3].2, 8);
    for pair in segments.windows(2) {
      assert_eq!(pair[0].2, pair[1].1);
      assert_eq!(pair[0].0 + 1, pair[1].0);
    }
  }

  #[test]
  fn fails_without_enough_frames() {
    assert!(matches!(
      viterbi(&logits(&[1, 2], 4), &[1, 2, 3]),
      Err(Error::AlignmentFailed)
    ));
  }
}
