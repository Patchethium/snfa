//! Aligns a mono WAV file at the model's sample rate to a phoneme sequence:
//!
//!     cargo run --release --example align -- jp.safetensors speech.wav "k o n n i ch i w a"
//!
//! Prints one `phoneme start_ms end_ms` line per segment, after trimming
//! leading and trailing silence as the Python README suggests.

use snfa::{Aligner, trim_audio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let args: Vec<String> = std::env::args().collect();
  let [_, model, wav, phonemes] = args.as_slice() else {
    return Err("usage: align <model.safetensors> <audio.wav> <phonemes>".into());
  };
  let aligner = Aligner::from_path(model)?;
  let mut reader = hound::WavReader::open(wav)?;
  let spec = reader.spec();
  if spec.channels != 1 || spec.sample_rate != aligner.sample_rate() {
    return Err(format!("expected mono audio at {} Hz", aligner.sample_rate()).into());
  }
  let samples: Vec<f32> = match spec.sample_format {
    hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
    hound::SampleFormat::Int => {
      let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
      reader
        .samples::<i32>()
        .map(|s| s.map(|s| s as f32 / scale))
        .collect::<Result<_, _>>()?
    }
  };
  let range = trim_audio(&samples, 20.0, 2048, 512).ok_or("the audio is too short")?;
  let phonemes: Vec<&str> = phonemes.split_whitespace().collect();
  for segment in aligner.align(&samples[range], &phonemes, true)? {
    println!(
      "{} {} {}",
      segment.phoneme, segment.start_ms, segment.end_ms
    );
  }
  Ok(())
}
