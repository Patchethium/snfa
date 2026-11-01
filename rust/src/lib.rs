//! `snfa` (Simple Neural Forced Aligner): a Rust port of the numpy inference
//! of [snfa](https://github.com/Patchethium/snfa).
//!
//! Load an exported `.safetensors` model with [`Aligner::from_path`] or
//! [`Aligner::from_bytes`], then align a mono waveform normalized to `[-1, 1]` at the
//! model's sample rate ([`Aligner::sample_rate`]) to a phoneme sequence:
//!
//! ```no_run
//! let aligner = snfa::Aligner::from_path("jp.safetensors")?;
//! let wav: Vec<f32> = vec![0.0; 16000]; // mono, [-1, 1], at `aligner.sample_rate()`
//! let phonemes = "k o n n i ch i w a".split(' ').collect::<Vec<_>>();
//! for segment in aligner.align(&wav, &phonemes, true)? {
//!   println!("{} {} {}", segment.phoneme, segment.start_ms, segment.end_ms);
//! }
//! # Ok::<(), snfa::Error>(())
//! ```
//!
//! The default Japanese model is uncased: lowercase phonemes before aligning.

mod error;
mod mel;
mod model;
mod trim;
mod viterbi;

pub use error::Error;
pub use model::{Aligner, Metadata};
pub use trim::trim_audio;
pub use viterbi::Segment;

pub type Result<T> = std::result::Result<T, Error>;
