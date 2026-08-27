use rubato::{FftFixedIn, Resampler};

/// Downmixes interleaved multi-channel f32 samples to mono, then resamples from
/// `in_rate` to `TARGET_SAMPLE_RATE` using a fixed-input-size FFT resampler.
///
/// PocketSphinx expects 16kHz mono PCM; WASAPI's shared-mode mix format is usually
/// 44.1kHz or 48kHz stereo, so this conversion runs on every captured chunk.
///
/// Scratch buffers (`mono_scratch`, `in_block`, `out_block`, `leftover`, `output`)
/// are allocated once in `new()` and reused across every `process()` call instead
/// of being freshly allocated per call/per block — this is a pure allocation-count
/// reduction, the resampling math and output values are unchanged (see
/// docs/stt-performance-phase2.md Phase C).
pub struct AudioResampler {
    channels_in: usize,
    in_rate: usize,
    resampler: Option<FftFixedIn<f32>>,
    chunk_size: usize,
    leftover: Vec<f32>,
    mono_scratch: Vec<f32>,
    in_block: Vec<Vec<f32>>,
    out_block: Vec<Vec<f32>>,
    output: Vec<f32>,
}

impl AudioResampler {
    pub fn new(in_rate: u32, channels_in: u16) -> Self {
        let out_rate = super::TARGET_SAMPLE_RATE as usize;
        let in_rate = in_rate as usize;
        let chunk_size = 1024;
        let resampler = if in_rate == out_rate {
            None
        } else {
            FftFixedIn::<f32>::new(in_rate, out_rate, chunk_size, 2, 1).ok()
        };
        // Pre-allocate the resampler's input/output block buffers once, sized
        // exactly as rubato recommends for `process_into_buffer` (one input
        // channel's worth of `chunk_size` samples in, `output_frames_max()`
        // samples out) so no per-block Vec allocation happens in `process()`.
        let (in_block, out_block) = match &resampler {
            Some(r) => (
                vec![vec![0.0f32; chunk_size]],
                vec![vec![0.0f32; r.output_frames_max()]],
            ),
            None => (Vec::new(), Vec::new()),
        };
        Self {
            channels_in: channels_in.max(1) as usize,
            in_rate,
            resampler,
            chunk_size,
            leftover: Vec::new(),
            mono_scratch: Vec::new(),
            in_block,
            out_block,
            output: Vec::new(),
        }
    }

    /// `interleaved` is raw interleaved f32 samples at the input sample rate/channel
    /// count. Returns mono f32 samples at `TARGET_SAMPLE_RATE`, may be empty if not
    /// enough input has accumulated yet to fill one resampler chunk.
    pub fn process(&mut self, interleaved: &[f32]) -> Vec<f32> {
        downmix_to_mono(interleaved, self.channels_in, &mut self.mono_scratch);

        let Some(resampler) = self.resampler.as_mut() else {
            return self.mono_scratch.clone();
        };

        self.leftover.extend_from_slice(&self.mono_scratch);

        self.output.clear();
        let mut leftover_pos = 0usize;
        while self.leftover.len() - leftover_pos >= self.chunk_size {
            self.in_block[0].copy_from_slice(&self.leftover[leftover_pos..leftover_pos + self.chunk_size]);
            leftover_pos += self.chunk_size;

            match resampler.process_into_buffer(&self.in_block, &mut self.out_block, None) {
                Ok((_, out_len)) => {
                    self.output.extend_from_slice(&self.out_block[0][..out_len]);
                }
                Err(_) => break,
            }
        }
        self.leftover.drain(..leftover_pos);
        self.output.clone()
    }

    pub fn in_rate(&self) -> usize {
        self.in_rate
    }
}

fn downmix_to_mono(interleaved: &[f32], channels: usize, out: &mut Vec<f32>) {
    out.clear();
    if channels <= 1 {
        out.extend_from_slice(interleaved);
        return;
    }
    out.extend(
        interleaved
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::TARGET_SAMPLE_RATE;

    // -- downmix_to_mono ------------------------------------------------

    #[test]
    fn downmix_mono_passthrough_is_unchanged() {
        let input = vec![0.1, -0.2, 0.3, -0.4];
        let mut out = Vec::new();
        downmix_to_mono(&input, 1, &mut out);
        assert_eq!(out, input);
    }

    #[test]
    fn downmix_stereo_averages_left_and_right() {
        // L=0.2, R=0.6 -> 0.4; L=-0.5, R=0.5 -> 0.0
        let input = vec![0.2, 0.6, -0.5, 0.5];
        let mut out = Vec::new();
        downmix_to_mono(&input, 2, &mut out);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.4).abs() < 1e-6);
        assert!((out[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn downmix_multichannel_averages_all_channels() {
        // 4 channels, all equal to 1.0 -> average is 1.0.
        let input = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = Vec::new();
        downmix_to_mono(&input, 4, &mut out);
        assert_eq!(out, vec![1.0]);
    }

    #[test]
    fn downmix_out_buffer_is_reused_not_grown_unbounded() {
        // The `out` buffer is cleared and reused across calls (mirroring how
        // AudioResampler.mono_scratch is used) — confirm repeated calls with
        // shrinking input don't leave stale trailing samples behind.
        let mut out = vec![9.0, 9.0, 9.0, 9.0, 9.0]; // pre-seeded with stale data
        downmix_to_mono(&[1.0, 2.0], 1, &mut out);
        assert_eq!(out, vec![1.0, 2.0], "stale samples from a prior call must not leak into the new output");
    }

    // -- AudioResampler: passthrough (in_rate == out_rate) ---------------

    #[test]
    fn same_rate_mono_is_pure_passthrough_no_resampling() {
        let mut resampler = AudioResampler::new(TARGET_SAMPLE_RATE, 1);
        let input: Vec<f32> = (0..500).map(|i| (i as f32) * 0.001).collect();
        let output = resampler.process(&input);
        assert_eq!(output, input, "same-rate mono input must pass through unchanged");
    }

    #[test]
    fn same_rate_stereo_downmixes_but_does_not_resample() {
        let mut resampler = AudioResampler::new(TARGET_SAMPLE_RATE, 2);
        let interleaved = vec![0.5, -0.5, 1.0, 1.0]; // 2 stereo frames
        let output = resampler.process(&interleaved);
        assert_eq!(output.len(), 2);
        assert!((output[0] - 0.0).abs() < 1e-6);
        assert!((output[1] - 1.0).abs() < 1e-6);
    }

    // -- AudioResampler: real production rates ----------------------------

    fn sine_wave(n: usize, sample_rate: usize, freq_hz: f32, amplitude: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                amplitude * (2.0 * std::f32::consts::PI * freq_hz * t).sin()
            })
            .collect()
    }

    #[test]
    fn resamples_48khz_stereo_to_16khz_mono_with_roughly_the_right_output_length() {
        // WASAPI's most common shared-mode mix format — the real production case.
        let in_rate = 48_000u32;
        let mut resampler = AudioResampler::new(in_rate, 2);

        // 1 second of a 220Hz tone, interleaved stereo (both channels identical).
        let mono = sine_wave(in_rate as usize, in_rate as usize, 220.0, 0.5);
        let mut interleaved = Vec::with_capacity(mono.len() * 2);
        for s in &mono {
            interleaved.push(*s);
            interleaved.push(*s);
        }

        let output = resampler.process(&interleaved);
        // Output rate is 16kHz; allow slack for the resampler's internal
        // block buffering (chunk_size=1024 @ 48kHz withholds a partial
        // block until enough input accumulates).
        let expected = TARGET_SAMPLE_RATE as usize;
        let tolerance = 1600; // 100ms worth of samples
        assert!(
            output.len().abs_diff(expected) <= tolerance,
            "expected ~{expected} samples out, got {} (tolerance {tolerance})",
            output.len()
        );
    }

    #[test]
    fn resamples_44100hz_mono_to_16khz() {
        let in_rate = 44_100u32;
        let mut resampler = AudioResampler::new(in_rate, 1);
        let input = sine_wave(in_rate as usize, in_rate as usize, 440.0, 0.3);
        let output = resampler.process(&input);
        assert!(!output.is_empty(), "44.1kHz->16kHz must produce output for 1s of input");
        // Sanity: output values must stay within the input amplitude range
        // (a resampling bug that introduces gain/clipping would show up here).
        assert!(output.iter().all(|s| s.abs() <= 0.35), "resampled amplitude must not exceed input amplitude (+ small margin)");
    }

    #[test]
    fn silence_in_produces_silence_out() {
        let mut resampler = AudioResampler::new(48_000, 2);
        let silence = vec![0.0f32; 48_000 * 2]; // 1s of stereo silence
        let output = resampler.process(&silence);
        assert!(output.iter().all(|&s| s == 0.0), "silence must resample to silence, not noise");
    }

    #[test]
    fn arbitrary_small_chunks_do_not_panic_and_eventually_produce_output() {
        // Mirrors real WASAPI packets: not aligned to the resampler's
        // internal 1024-sample block size. Feed many small, irregular
        // chunks and confirm no panic and that SOME output eventually
        // appears (the resampler buffers leftovers across calls).
        let mut resampler = AudioResampler::new(48_000, 2);
        let mut total_output = 0usize;
        for i in 0..100 {
            // Irregular chunk sizes, all realistic WASAPI packet scales.
            let n_frames = 200 + (i % 7) * 37;
            let chunk = sine_wave(n_frames * 2, 96_000, 300.0, 0.4); // interleaved stereo
            total_output += resampler.process(&chunk).len();
        }
        assert!(total_output > 0, "output must appear eventually across many small chunks");
    }

    #[test]
    fn in_rate_accessor_reports_the_configured_input_rate() {
        let resampler = AudioResampler::new(44_100, 2);
        assert_eq!(resampler.in_rate(), 44_100);
    }

    #[test]
    fn leftover_samples_are_never_silently_dropped_across_calls() {
        // Feed one sample at a time (worst case for leftover buffering) and
        // confirm the resampler still eventually produces proportional
        // output rather than losing samples that never fill one block.
        let mut resampler = AudioResampler::new(48_000, 1);
        let one_second = sine_wave(48_000, 48_000, 220.0, 0.5);
        let mut total_output = 0usize;
        for sample in &one_second {
            total_output += resampler.process(std::slice::from_ref(sample)).len();
        }
        // Even fed one sample at a time, ~1 second of 48kHz input should
        // eventually yield close to 1 second of 16kHz output once enough
        // has accumulated to fill blocks.
        let expected = TARGET_SAMPLE_RATE as usize;
        assert!(
            total_output > expected / 2,
            "expected a substantial fraction of ~{expected} samples even with 1-sample-at-a-time feeding, got {total_output}"
        );
    }

    #[test]
    fn consecutive_calls_do_not_contaminate_each_others_output() {
        // Regression guard for the internal-buffer-reuse optimization: each
        // process() call's returned Vec must reflect only that call's
        // contribution, never leftover data from the previous call's
        // internal scratch/output buffers.
        let mut resampler = AudioResampler::new(TARGET_SAMPLE_RATE, 1);
        let first = resampler.process(&[1.0, 2.0, 3.0]);
        assert_eq!(first, vec![1.0, 2.0, 3.0]);
        let second = resampler.process(&[4.0, 5.0]);
        assert_eq!(second, vec![4.0, 5.0], "second call must not contain leftover samples from the first call");
    }

    #[test]
    fn mono_passthrough_output_is_independent_of_the_next_calls_mutation() {
        // Since process() now returns a clone of an internal scratch buffer
        // rather than a fresh allocation of the input, confirm the returned
        // Vec is a real owned copy: mutating it must not affect the
        // resampler's internal state or the next call's output.
        let mut resampler = AudioResampler::new(TARGET_SAMPLE_RATE, 1);
        let mut first = resampler.process(&[1.0, 2.0]);
        first[0] = 999.0;
        let second = resampler.process(&[3.0, 4.0]);
        assert_eq!(second, vec![3.0, 4.0], "caller mutating a returned buffer must not leak into subsequent calls");
    }

    #[test]
    #[ignore]
    fn phase_c_throughput_microbench() {
        // Not a correctness test — run explicitly with:
        //   cargo test --release --lib audio::resample::tests::phase_c_throughput_microbench -- --ignored --nocapture
        // Simulates 60s of real-time 48kHz stereo WASAPI capture delivered in
        // realistic ~10ms packets (480 frames = 960 interleaved samples),
        // matching system_capture.rs's actual call pattern, and reports
        // wall-clock time to process it all.
        let mut resampler = AudioResampler::new(48_000, 2);
        let packet = sine_wave(480 * 2, 96_000, 220.0, 0.4); // interleaved stereo, 480 frames
        let iterations = 6000; // 6000 * 10ms = 60s of simulated audio

        let started = std::time::Instant::now();
        let mut total_out = 0usize;
        for _ in 0..iterations {
            total_out += resampler.process(&packet).len();
        }
        let elapsed = started.elapsed();
        println!(
            "phase_c_throughput_microbench: {iterations} packets ({}s simulated audio) in {:?}, total_out_samples={total_out}",
            iterations as f64 * 0.01,
            elapsed
        );
    }
}
