use std::fmt;

#[derive(Debug)]
pub enum Error {
  /// The model file could not be read or is not a valid snfa model.
  Model(String),
  /// A phoneme that is not in the model's phoneme set.
  UnknownPhoneme(String),
  /// No phonemes were given.
  EmptyTranscript,
  /// The waveform is shorter than one analysis window.
  AudioTooShort,
  /// The waveform has fewer frames than the transcript has phonemes.
  TooManyPhonemes { frames: usize, phonemes: usize },
  /// No monotonic path covers every phoneme.
  AlignmentFailed,
}

impl fmt::Display for Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Error::Model(message) => write!(f, "invalid snfa model: {message}"),
      Error::UnknownPhoneme(phoneme) => {
        write!(f, "phoneme {phoneme:?} is not in the model's phoneme set")
      }
      Error::EmptyTranscript => write!(f, "the transcript has no phonemes"),
      Error::AudioTooShort => write!(f, "the audio is shorter than one analysis window"),
      Error::TooManyPhonemes { frames, phonemes } => write!(
        f,
        "the audio has {frames} frames, fewer than the {phonemes} phonemes to align"
      ),
      Error::AlignmentFailed => write!(f, "no alignment covers every phoneme"),
    }
  }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
  fn from(error: std::io::Error) -> Self {
    Error::Model(error.to_string())
  }
}

impl From<safetensors::SafeTensorError> for Error {
  fn from(error: safetensors::SafeTensorError) -> Self {
    Error::Model(error.to_string())
  }
}
