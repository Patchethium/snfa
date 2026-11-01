//! The aligner's encoder, matching `snfa/aligner.py`: a linear projection,
//! two bidirectional GRU layers with a residual connection, and a linear
//! classifier over the phoneme set.

use std::collections::HashMap;
use std::path::Path;

use safetensors::{Dtype, SafeTensors};

use crate::mel::MelSpectrogram;
use crate::viterbi::{Segment, viterbi};
use crate::{Error, Result};

/// The model's analysis settings and phoneme set, stored in the
/// `.safetensors` header with each value JSON-encoded.
#[derive(Clone, Debug, PartialEq)]
pub struct Metadata {
  pub sample_rate: u32,
  pub n_mels: usize,
  pub dim: usize,
  pub hop_size: usize,
  pub win_size: usize,
  pub n_fft: usize,
  /// Index 0 is the CTC blank.
  pub phone_set: Vec<String>,
}

impl Metadata {
  fn parse(header: &HashMap<String, String>) -> Result<Self> {
    let number = |key: &str| -> Result<usize> {
      header
        .get(key)
        .ok_or_else(|| Error::Model(format!("missing metadata {key:?}")))?
        .parse::<usize>()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| Error::Model(format!("metadata {key:?} is not a positive integer")))
    };
    let phone_set = header
      .get("phone_set")
      .ok_or_else(|| Error::Model("missing metadata \"phone_set\"".into()))?;
    let phone_set = parse_json_string(phone_set)
      .ok_or_else(|| Error::Model("metadata \"phone_set\" is not a JSON string".into()))?;
    let phone_set: Vec<String> = phone_set.split('\0').map(str::to_owned).collect();
    if phone_set.len() < 2 {
      return Err(Error::Model("the phoneme set is empty".into()));
    }
    let sample_rate = u32::try_from(number("sr")?)
      .map_err(|_| Error::Model("metadata \"sr\" is too large".into()))?;
    let metadata = Self {
      sample_rate,
      n_mels: number("n_mels")?,
      dim: number("dim")?,
      hop_size: number("hop_size")?,
      win_size: number("win_size")?,
      n_fft: number("n_fft")?,
      phone_set,
    };
    if !metadata.dim.is_multiple_of(2) {
      return Err(Error::Model("metadata \"dim\" must be even".into()));
    }
    Ok(metadata)
  }
}

/// Decodes a JSON string literal, enough for `json.dumps` output.
fn parse_json_string(text: &str) -> Option<String> {
  let inner = text.strip_prefix('"')?.strip_suffix('"')?;
  let mut output = String::with_capacity(inner.len());
  let mut chars = inner.chars();
  while let Some(c) = chars.next() {
    if c != '\\' {
      output.push(c);
      continue;
    }
    match chars.next()? {
      '"' => output.push('"'),
      '\\' => output.push('\\'),
      '/' => output.push('/'),
      'b' => output.push('\u{8}'),
      'f' => output.push('\u{c}'),
      'n' => output.push('\n'),
      'r' => output.push('\r'),
      't' => output.push('\t'),
      'u' => {
        let code: String = chars.by_ref().take(4).collect();
        let code = u32::from_str_radix(&code, 16).ok()?;
        output.push(char::from_u32(code)?);
      }
      _ => return None,
    }
  }
  Some(output)
}

/// A row-major weight matrix with its bias, applied as `W x + b`.
struct Linear {
  weight: Vec<f64>,
  bias: Vec<f64>,
  inputs: usize,
}

impl Linear {
  fn load(tensors: &SafeTensors, name: &str, outputs: usize, inputs: usize) -> Result<Self> {
    Ok(Self {
      weight: load_tensor(tensors, &format!("{name}.weight"), &[outputs, inputs])?,
      bias: load_tensor(tensors, &format!("{name}.bias"), &[outputs])?,
      inputs,
    })
  }

  fn load_gru(tensors: &SafeTensors, kind: &str, suffix: &str, hidden: usize) -> Result<Self> {
    let weight_name = format!("bi_rnn.weight_{kind}_{suffix}");
    let inputs = tensors
      .tensor(&weight_name)
      .map_err(|_| Error::Model(format!("missing tensor {weight_name:?}")))?
      .shape()
      .get(1)
      .copied()
      .unwrap_or(0);
    Ok(Self {
      weight: load_tensor(tensors, &weight_name, &[3 * hidden, inputs])?,
      bias: load_tensor(
        tensors,
        &format!("bi_rnn.bias_{kind}_{suffix}"),
        &[3 * hidden],
      )?,
      inputs,
    })
  }

  fn forward(&self, x: &[f64]) -> Vec<f64> {
    self
      .weight
      .chunks_exact(self.inputs)
      .zip(&self.bias)
      .map(|(row, bias)| row.iter().zip(x).map(|(w, x)| w * x).sum::<f64>() + bias)
      .collect()
  }
}

fn load_tensor(tensors: &SafeTensors, name: &str, shape: &[usize]) -> Result<Vec<f64>> {
  let tensor = tensors
    .tensor(name)
    .map_err(|_| Error::Model(format!("missing tensor {name:?}")))?;
  if tensor.shape() != shape {
    return Err(Error::Model(format!(
      "tensor {name:?} has shape {:?}, expected {shape:?}",
      tensor.shape()
    )));
  }
  let data = tensor.data();
  // Weights are exported in half precision and computed in double, as numpy
  // promotes the float32 weights when multiplying the float64 features.
  let values = match tensor.dtype() {
    Dtype::F16 => data
      .as_chunks()
      .0
      .iter()
      .map(|b| half::f16::from_le_bytes(*b).to_f64())
      .collect(),
    Dtype::F32 => data
      .as_chunks()
      .0
      .iter()
      .map(|b| f64::from(f32::from_le_bytes(*b)))
      .collect(),
    Dtype::F64 => data
      .as_chunks()
      .0
      .iter()
      .map(|b| f64::from_le_bytes(*b))
      .collect(),
    dtype => {
      return Err(Error::Model(format!(
        "tensor {name:?} has unsupported dtype {dtype:?}"
      )));
    }
  };
  Ok(values)
}

fn sigmoid(x: f64) -> f64 {
  1.0 / (1.0 + (-x).exp())
}

/// PyTorch's GRU cell: gates ordered reset, update, new.
struct GruCell {
  ih: Linear,
  hh: Linear,
}

impl GruCell {
  fn hidden(&self) -> usize {
    self.hh.inputs
  }

  fn step(&self, x_projection: &[f64], h: &[f64]) -> Vec<f64> {
    let hidden = self.hidden();
    let h_projection = self.hh.forward(h);
    (0..hidden)
      .map(|i| {
        let r = sigmoid(x_projection[i] + h_projection[i]);
        let z = sigmoid(x_projection[hidden + i] + h_projection[hidden + i]);
        let n = (x_projection[2 * hidden + i] + r * h_projection[2 * hidden + i]).tanh();
        (1.0 - z) * n + z * h[i]
      })
      .collect()
  }

  /// Runs over `x` from a zero state, in reverse when `reverse` is set; the
  /// outputs stay in the input's order.
  fn run(&self, x: &[Vec<f64>], reverse: bool) -> Vec<Vec<f64>> {
    let mut h = vec![0.0; self.hidden()];
    let mut outputs = vec![Vec::new(); x.len()];
    let order: Box<dyn Iterator<Item = usize>> = if reverse {
      Box::new((0..x.len()).rev())
    } else {
      Box::new(0..x.len())
    };
    for t in order {
      h = self.step(&self.ih.forward(&x[t]), &h);
      outputs[t] = h.clone();
    }
    outputs
  }
}

struct BiGru {
  forward: GruCell,
  backward: GruCell,
}

impl BiGru {
  fn load(tensors: &SafeTensors, layer: usize, hidden: usize) -> Result<Self> {
    let cell = |suffix: &str| -> Result<GruCell> {
      Ok(GruCell {
        ih: Linear::load_gru(tensors, "ih", suffix, hidden)?,
        hh: Linear::load_gru(tensors, "hh", suffix, hidden)?,
      })
    };
    let bigru = Self {
      forward: cell(&format!("l{layer}"))?,
      backward: cell(&format!("l{layer}_reverse"))?,
    };
    if bigru.forward.ih.inputs != bigru.backward.ih.inputs {
      return Err(Error::Model(format!(
        "GRU layer {layer} has mismatched input sizes"
      )));
    }
    Ok(bigru)
  }

  fn inputs(&self) -> usize {
    self.forward.ih.inputs
  }

  fn run(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let forward = self.forward.run(x, false);
    let backward = self.backward.run(x, true);
    forward
      .into_iter()
      .zip(backward)
      .map(|(mut f, b)| {
        f.extend(b);
        f
      })
      .collect()
  }
}

/// A loaded snfa model.
pub struct Aligner {
  metadata: Metadata,
  phone_index: HashMap<String, usize>,
  mel: MelSpectrogram,
  pre: Linear,
  rnn: [BiGru; 2],
  fc: Linear,
}

impl Aligner {
  /// Loads an exported `.safetensors` model.
  pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
    let (_, header) = SafeTensors::read_metadata(bytes)?;
    let metadata = Metadata::parse(
      header
        .metadata()
        .as_ref()
        .ok_or_else(|| Error::Model("the file has no metadata".into()))?,
    )?;
    let tensors = SafeTensors::deserialize(bytes)?;
    let hidden = metadata.dim / 2;
    let pre = Linear::load(&tensors, "pre", metadata.dim, metadata.n_mels)?;
    let rnn = [
      BiGru::load(&tensors, 0, hidden)?,
      BiGru::load(&tensors, 1, hidden)?,
    ];
    if rnn.iter().any(|layer| layer.inputs() != metadata.dim) {
      return Err(Error::Model("GRU input sizes do not match \"dim\"".into()));
    }
    let fc = Linear::load(&tensors, "fc", metadata.phone_set.len(), metadata.dim)?;
    let phone_index = metadata
      .phone_set
      .iter()
      .enumerate()
      .map(|(i, p)| (p.clone(), i))
      .collect();
    // The numpy implementation frames with its default 1024-sample window
    // rather than `win_size`; the exported models use 1024 for both.
    let mel = MelSpectrogram::new(
      metadata.sample_rate,
      metadata.n_fft,
      metadata.win_size,
      metadata.hop_size,
      metadata.n_mels,
    );
    Ok(Self {
      metadata,
      phone_index,
      mel,
      pre,
      rnn,
      fc,
    })
  }

  /// Reads and loads an exported `.safetensors` model.
  pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
    Self::from_bytes(&std::fs::read(path)?)
  }

  pub fn metadata(&self) -> &Metadata {
    &self.metadata
  }

  /// The sample rate waveforms must have.
  pub fn sample_rate(&self) -> u32 {
    self.metadata.sample_rate
  }

  /// The phonemes this model can align, excluding the CTC blank.
  pub fn phone_set(&self) -> &[String] {
    &self.metadata.phone_set[1..]
  }

  /// The log mel spectrogram the encoder reads, one row per frame.
  pub fn log_mel(&self, wav: &[f32]) -> Vec<Vec<f64>> {
    self.mel.process(wav)
  }

  /// Per-frame logits over the phoneme set, blank first.
  pub fn logits(&self, mel: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let x: Vec<Vec<f64>> = mel.iter().map(|frame| self.pre.forward(frame)).collect();
    let mut hidden = self.rnn[1].run(&self.rnn[0].run(&x));
    for (h, residual) in hidden.iter_mut().zip(&x) {
      for (h, r) in h.iter_mut().zip(residual) {
        *h += r;
      }
    }
    hidden.iter().map(|h| self.fc.forward(h)).collect()
  }

  fn tokens<S: AsRef<str>>(&self, phonemes: &[S]) -> Result<Vec<usize>> {
    phonemes
      .iter()
      .map(|p| {
        let p = p.as_ref();
        self
          .phone_index
          .get(p)
          .copied()
          // The blank is not a phoneme.
          .filter(|&index| index != 0)
          .ok_or_else(|| Error::UnknownPhoneme(p.to_owned()))
      })
      .collect()
  }

  /// Aligns `wav` (mono, `[-1, 1]`, at [`Self::sample_rate`]) to `phonemes`.
  /// With `pad_pause`, a `pau` is aligned before and after the phonemes and
  /// returned as the first and last segments.
  pub fn align<S: AsRef<str>>(
    &self,
    wav: &[f32],
    phonemes: &[S],
    pad_pause: bool,
  ) -> Result<Vec<Segment>> {
    if phonemes.is_empty() {
      return Err(Error::EmptyTranscript);
    }
    let mut tokens = self.tokens(phonemes)?;
    if pad_pause {
      let pause = self.tokens(&["pau"])?[0];
      tokens.insert(0, pause);
      tokens.push(pause);
    }
    let mel = self.log_mel(wav);
    if mel.is_empty() {
      return Err(Error::AudioTooShort);
    }
    if mel.len() < tokens.len() {
      return Err(Error::TooManyPhonemes {
        frames: mel.len(),
        phonemes: tokens.len(),
      });
    }
    let logits = self.logits(&mel);
    let frames = viterbi(&logits, &tokens)?;
    let to_ms = |frame: usize| {
      // `int((frame * hop_size) / sr * 1000)` in Python.
      ((frame * self.metadata.hop_size) as f64 / f64::from(self.metadata.sample_rate) * 1000.0)
        as u32
    };
    Ok(
      frames
        .into_iter()
        .map(|(token, start, end)| Segment {
          phoneme: self.metadata.phone_set[tokens[token]].clone(),
          start_ms: to_ms(start),
          end_ms: to_ms(end),
        })
        .collect(),
    )
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn parses_json_dumps_strings() {
    assert_eq!(
      parse_json_string(r#""pad\u0000a\u0000pau""#).as_deref(),
      Some("pad\0a\0pau")
    );
    assert_eq!(
      parse_json_string(r#""a\"b\\c""#).as_deref(),
      Some("a\"b\\c")
    );
    assert_eq!(parse_json_string("pad"), None);
    assert_eq!(parse_json_string(r#""\q""#), None);
  }

  fn header(phone_set: &str) -> HashMap<String, String> {
    [
      ("sr", "16000"),
      ("n_mels", "80"),
      ("dim", "256"),
      ("hop_size", "160"),
      ("win_size", "1024"),
      ("n_fft", "1024"),
      ("phone_set", phone_set),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v.to_owned()))
    .collect()
  }

  #[test]
  fn parses_metadata() {
    let metadata = Metadata::parse(&header(r#""pad\u0000a\u0000pau""#)).unwrap();
    assert_eq!(metadata.sample_rate, 16000);
    assert_eq!(metadata.hop_size, 160);
    assert_eq!(metadata.phone_set, ["pad", "a", "pau"]);
  }

  #[test]
  fn rejects_invalid_metadata() {
    let mut missing = header(r#""pad\u0000a""#);
    missing.remove("hop_size");
    assert!(matches!(Metadata::parse(&missing), Err(Error::Model(_))));
    let mut zero = header(r#""pad\u0000a""#);
    zero.insert("n_fft".into(), "0".into());
    assert!(matches!(Metadata::parse(&zero), Err(Error::Model(_))));
    assert!(matches!(
      Metadata::parse(&header(r#""pad""#)),
      Err(Error::Model(_))
    ));
  }

  #[test]
  fn rejects_files_that_are_not_models() {
    assert!(matches!(
      Aligner::from_bytes(b"not a model"),
      Err(Error::Model(_))
    ));
  }

  #[test]
  fn gru_cell_matches_the_pytorch_equations() {
    // One hidden unit: r = sigmoid(1 + 0.5), z = sigmoid(2 + 0.25),
    // n = tanh(3 + r * 0.125), h' = (1 - z) n + z h.
    let cell = GruCell {
      ih: Linear {
        weight: vec![1.0, 2.0, 3.0],
        bias: vec![0.0; 3],
        inputs: 1,
      },
      hh: Linear {
        weight: vec![1.0, 0.5, 0.25],
        bias: vec![0.0; 3],
        inputs: 1,
      },
    };
    let h = 0.5;
    let output = cell.step(&cell.ih.forward(&[1.0]), &[h]);
    let r = sigmoid(1.0 + 0.5);
    let z = sigmoid(2.0 + 0.25);
    let n = (3.0 + r * 0.125f64).tanh();
    assert!((output[0] - ((1.0 - z) * n + z * h)).abs() < 1e-15);
  }

  #[test]
  fn reverse_gru_keeps_the_input_order() {
    // h' = (1 - z) n + z h with z = 0.5 and n = tanh(x) accumulates history.
    let cell = GruCell {
      ih: Linear {
        weight: vec![0.0, 0.0, 1.0],
        bias: vec![0.0; 3],
        inputs: 1,
      },
      hh: Linear {
        weight: vec![0.0; 3],
        bias: vec![0.0; 3],
        inputs: 1,
      },
    };
    let x = vec![vec![1.0], vec![0.0], vec![0.0]];
    let forward = cell.run(&x, false);
    let backward = cell.run(&x, true);
    assert!(forward[0][0] > 0.0 && forward[2][0] > 0.0);
    // Run backwards, only the first frame has seen the nonzero input.
    assert_eq!(backward[1][0], 0.0);
    assert_eq!(backward[2][0], 0.0);
    assert!((backward[0][0] - forward[0][0]).abs() < 1e-15);
  }
}
