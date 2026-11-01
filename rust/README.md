# snfa (Rust)

A Rust port of `snfa`'s numpy inference: the same log mel spectrogram, encoder,
Viterbi alignment and silence trimming, computed in double precision like numpy
so it produces the same segments as the Python package.

The model weights are not part of the repository. Convert an exported `.npz`
to `.safetensors` with the repository's `export.py`:

```bash
uv run export.py --npz jp.npz -f safetensors -o jp
```

or export a checkpoint directly with `--ckpt`.

```rust
let aligner = snfa::Aligner::from_path("jp.safetensors")?;
// mono, normalized to [-1, 1], at `aligner.sample_rate()`
let range = snfa::trim_audio(&wav, 20.0, 2048, 512).unwrap();
let phonemes = "k o n n i ch i w a".split(' ').collect::<Vec<_>>();
for segment in aligner.align(&wav[range], &phonemes, true)? {
  println!("{} {} {}", segment.phoneme, segment.start_ms, segment.end_ms);
}
```

`examples/align.rs` does the same for a WAV file:

```bash
cargo run --release --example align -- jp.safetensors speech.wav "k o n n i ch i w a"
```

## Testing

`tests/parity.rs` compares the port with fixtures the numpy implementation
wrote (`tests/make_fixtures.py`) and needs the `.safetensors` model converted
from the fixture's `.npz`:

```bash
SNFA_MODEL=/path/to/jp.safetensors cargo test
```

Regenerate the fixtures from the repository root after changing the Python
inference or the model:

```bash
uv run --no-project --with numpy python rust/tests/make_fixtures.py jp.npz
```

## Licence

ISC, like the Python package. `src/mel.rs` and `src/trim.rs` adapt `librosa`
code (ISC) and `src/viterbi.rs` adapts `torchaudio` code (BSD 2-Clause); their
notices are kept in those files.
