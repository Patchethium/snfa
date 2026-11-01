//! Compares the Rust port with the numpy implementation on the synthetic
//! waveform of `make_fixtures.py`, whose outputs are in `fixtures/parity.json`.
//!
//! The model is not part of the repository: set `SNFA_MODEL` to the
//! `.safetensors` file converted from the `.npz` named in the fixture.

use std::f64::consts::PI;

use serde_json::Value;
use snfa::{Aligner, trim_audio};

const SAMPLE_RATE: f64 = 16000.0;
const DURATION: f64 = 1.5;
const TOLERANCE: f64 = 1e-6;

fn lcg(count: usize) -> Vec<f64> {
  let mut state: u64 = 1;
  (0..count)
    .map(|_| {
      state = (state * 1103515245 + 12345) % (1 << 31);
      state as f64 / (1u64 << 31) as f64 - 0.5
    })
    .collect()
}

fn synthetic_waveform() -> Vec<f32> {
  let n = (SAMPLE_RATE * DURATION) as usize;
  let noise = lcg(n);
  (0..n)
    .map(|i| {
      let t = i as f64 / SAMPLE_RATE;
      let phase = 2.0 * PI * (120.0 * t - 40.0 / (2.0 * PI * 1.3) * (2.0 * PI * 1.3 * t).cos());
      let tone: f64 = (1..11).map(|k| (k as f64 * phase).sin() / k as f64).sum();
      let envelope = (PI * ((t - 0.25) / 1.0).clamp(0.0, 1.0)).sin().powi(2);
      (0.3 * envelope * tone + 0.01 * noise[i]) as f32
    })
    .collect()
}

fn fixture() -> Value {
  serde_json::from_str(include_str!("fixtures/parity.json")).unwrap()
}

fn aligner() -> Aligner {
  let path = std::env::var("SNFA_MODEL").unwrap_or_else(|_| {
    panic!(
      "set SNFA_MODEL to the .safetensors model converted from {}",
      fixture()["model"]
    )
  });
  Aligner::from_path(&path).unwrap_or_else(|error| panic!("cannot load {path}: {error}"))
}

fn assert_rows_close(name: &str, actual: &[Vec<f64>], expected: &Value, stride: usize) {
  let expected = expected.as_array().unwrap();
  assert_eq!(
    actual.iter().step_by(stride).count(),
    expected.len(),
    "{name} rows"
  );
  for (frame, (row, expected)) in actual.iter().step_by(stride).zip(expected).enumerate() {
    let expected = expected.as_array().unwrap();
    assert_eq!(row.len(), expected.len(), "{name} columns");
    for (value, expected) in row.iter().zip(expected) {
      let expected = expected.as_f64().unwrap();
      assert!(
        (value - expected).abs() <= TOLERANCE * expected.abs().max(1.0),
        "{name} frame {}: {value} != {expected}",
        frame * stride
      );
    }
  }
}

#[test]
fn mel_and_logits_match_numpy() {
  let fixture = fixture();
  let aligner = aligner();
  assert_eq!(
    aligner.metadata().hop_size as u64,
    fixture["hop_size"].as_u64().unwrap(),
    "the fixture was made with another model"
  );
  let stride = fixture["stride"].as_u64().unwrap() as usize;
  let mel = aligner.log_mel(&synthetic_waveform());
  assert_eq!(mel.len() as u64, fixture["frames"].as_u64().unwrap());
  assert_rows_close("mel", &mel, &fixture["mel"], stride);
  assert_rows_close("logits", &aligner.logits(&mel), &fixture["logits"], stride);
}

#[test]
fn segments_match_numpy() {
  let fixture = fixture();
  let phonemes: Vec<&str> = fixture["phonemes"]
    .as_array()
    .unwrap()
    .iter()
    .map(|p| p.as_str().unwrap())
    .collect();
  let segments = aligner()
    .align(&synthetic_waveform(), &phonemes, true)
    .unwrap();
  let actual: Vec<(String, u64, u64)> = segments
    .into_iter()
    .map(|s| (s.phoneme, u64::from(s.start_ms), u64::from(s.end_ms)))
    .collect();
  let expected: Vec<(String, u64, u64)> = fixture["segments"]
    .as_array()
    .unwrap()
    .iter()
    .map(|s| {
      (
        s[0].as_str().unwrap().to_owned(),
        s[1].as_u64().unwrap(),
        s[2].as_u64().unwrap(),
      )
    })
    .collect();
  assert_eq!(actual, expected);
}

#[test]
fn trim_matches_numpy() {
  let range = trim_audio(&synthetic_waveform(), 20.0, 1024, 256).unwrap();
  let expected = &fixture()["trim"];
  assert_eq!(
    [range.start as u64, range.end as u64],
    [expected[0].as_u64().unwrap(), expected[1].as_u64().unwrap()]
  );
}

#[test]
fn rejects_unknown_phonemes_and_short_audio() {
  let aligner = aligner();
  let wav = synthetic_waveform();
  assert!(matches!(
    aligner.align(&wav, &["k", "xx"], true),
    Err(snfa::Error::UnknownPhoneme(p)) if p == "xx"
  ));
  assert!(matches!(
    aligner.align(&wav, &["pad"], true),
    Err(snfa::Error::UnknownPhoneme(_))
  ));
  assert!(matches!(
    aligner.align(&wav[..500], &["a"], true),
    Err(snfa::Error::AudioTooShort)
  ));
  let too_many = vec!["a"; 100];
  assert!(matches!(
    aligner.align(&wav, &too_many, true),
    Err(snfa::Error::TooManyPhonemes { phonemes: 102, .. })
  ));
  assert!(matches!(
    aligner.align::<&str>(&wav, &[], true),
    Err(snfa::Error::EmptyTranscript)
  ));
}
