use clap::Parser;
use serde::Deserialize;
use std::{fs::File, io::BufReader, path::PathBuf};

const SAMPLE_RATE: u32 = 44100;
const NUM_CHANNELS: usize = 2;

#[derive(Parser)]
struct Args {
    /// Path to Sonant-X song file in JSON format
    input: PathBuf,

    /// Path to output WAV file
    #[clap(short, long)]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let song: Song = serde_json::from_reader(BufReader::new(File::open(args.input)?))?;

    let num_samples = song.len * SAMPLE_RATE as usize;
    let mut samples = vec![0.0; num_samples * NUM_CHANNELS];
    let mut buf = vec![0.0; samples.len()];
    for (i, track) in song.tracks.iter().enumerate() {
        println!("Rendering track {}/{}", i + 1, song.tracks.len());
        track.render(&song, &mut buf);
        for (dest, src) in samples.iter_mut().zip(buf.iter()) {
            *dest += *src;
        }
        buf.fill(0.0);
    }

    println!("Writing to {}", args.output.display());
    let spec = hound::WavSpec {
        channels: NUM_CHANNELS as u16,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(args.output, spec)?;
    {
        let mut writer = writer.get_i16_writer(samples.len() as u32);
        for sample in samples {
            writer.write_sample((sample * i16::MAX as f32) as i16);
        }
        writer.flush()?;
    }
    writer.finalize()?;
    Ok(())
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Song {
    #[serde(rename = "songLen")]
    len: usize,

    row_len: usize,
    end_pattern: usize,

    #[serde(rename = "songData")]
    tracks: Vec<Track>,
}

impl Song {
    fn bpm(&self) -> f32 {
        (60.0 * SAMPLE_RATE as f32 / 4.0 / self.row_len as f32).round()
    }

    fn effective_row_len(&self) -> usize {
        (60.0 * SAMPLE_RATE as f32 / 4.0 / self.bpm()).round() as usize
    }
}

#[derive(Deserialize)]
struct Track {
    #[serde(rename = "p")]
    sequence: Vec<usize>,

    #[serde(rename = "c")]
    patterns: Vec<Pattern>,

    #[serde(flatten)]
    instrument: Instrument,
}

impl Track {
    fn render(&self, song: &Song, out: &mut [f32]) {
        let inst = &self.instrument;
        let mut window = &mut *out;
        let mut note_index = 0;
        let mut voices = Vec::<Voice>::new();
        let effective_row_len = song.effective_row_len();
        while effective_row_len * NUM_CHANNELS < window.len() {
            voices.retain_mut(|voice| voice.render(window));
            let pattern = self
                .sequence
                .get((note_index / 32) % (song.end_pattern + 1))
                .copied()
                .unwrap_or(0);
            if let Some(pattern) = pattern.checked_sub(1) {
                let note = self.patterns[pattern].notes[note_index % 32];
                if note != 0 {
                    let mut voice = Voice::new(inst, effective_row_len, note);
                    voice.render(window);
                    voices.push(voice);
                }
            }
            note_index += 1;
            window = &mut window[effective_row_len * NUM_CHANNELS..];
        }

        if inst.fx_delay_amt != 0.0 {
            let delay_time = inst.fx_delay_time * (1.0 / (song.bpm() / 60.0) / 8.0);
            let delay_samples = (delay_time * SAMPLE_RATE as f32) as usize;
            let delay_amt = inst.fx_delay_amt / 255.0;
            let mut src_index = 0;
            let mut dest_index = delay_samples * NUM_CHANNELS;
            while dest_index < out.len() {
                out[dest_index] += out[src_index] * delay_amt;
                src_index += 1;
                dest_index += 1;
            }
        }
    }
}

#[derive(Deserialize)]
struct Pattern {
    #[serde(rename = "n")]
    notes: Vec<i32>,
}

#[derive(Deserialize)]
struct Instrument {
    osc1_oct: i32,
    osc1_det: i32,
    osc1_detune: f32,
    osc1_xenv: u8,
    osc1_vol: f32,
    osc1_waveform: usize,

    osc2_oct: i32,
    osc2_det: i32,
    osc2_detune: f32,
    osc2_xenv: u8,
    osc2_vol: f32,
    osc2_waveform: usize,

    noise_fader: f32,

    env_attack: f32,
    env_sustain: f32,
    env_release: f32,
    env_master: f32,

    lfo_osc1_freq: u8,
    lfo_fx_freq: u8,
    lfo_freq: f32,
    lfo_amt: f32,
    lfo_waveform: usize,

    fx_filter: usize,
    fx_freq: f32,
    fx_resonance: f32,
    fx_delay_time: f32,
    fx_delay_amt: f32,
    fx_pan_freq: f32,
    fx_pan_amt: f32,
}

struct Voice<'a> {
    inst: &'a Instrument,
    effective_row_len: usize,
    note: i32,
    c1: f32,
    c2: f32,
    low: f32,
    band: f32,
    j: usize,
}

impl<'a> Voice<'a> {
    fn new(inst: &'a Instrument, effective_row_len: usize, note: i32) -> Self {
        Self {
            inst,
            effective_row_len,
            note,
            c1: 0.0,
            c2: 0.0,
            low: 0.0,
            band: 0.0,
            j: 0,
        }
    }

    /// Returns true if the voice is still active
    fn render(&mut self, out: &mut [f32]) -> bool {
        let inst = self.inst;

        let osc_lfo = OSCILLATORS[inst.lfo_waveform];
        let osc1 = OSCILLATORS[inst.osc1_waveform];
        let osc2 = OSCILLATORS[inst.osc2_waveform];
        let pan_freq = (inst.fx_pan_freq - 8.0).exp2() / self.effective_row_len as f32;
        let lfo_freq = (inst.lfo_freq - 8.0).exp2() / self.effective_row_len as f32;

        let attack_time = inst.env_attack / SAMPLE_RATE as f32;
        let release_time = inst.env_release / SAMPLE_RATE as f32;
        let sustain_time = inst.env_sustain / SAMPLE_RATE as f32;

        let env_attack = attack_time * SAMPLE_RATE as f32;
        let env_release = release_time * SAMPLE_RATE as f32;
        let env_sustain = sustain_time * SAMPLE_RATE as f32;

        // Precalculate frequencues
        let o1t = note_freq(self.note + (inst.osc1_oct - 8) * 12 + inst.osc1_det)
            * inst.osc1_detune.mul_add(0.0008, 1.0);
        let o2t = note_freq(self.note + (inst.osc2_oct - 8) * 12 + inst.osc2_det)
            * inst.osc2_detune.mul_add(0.0008, 1.0);

        let q = inst.fx_resonance / 255.0;

        while self.j < env_attack as usize + env_sustain as usize + env_release as usize {
            // LFO
            let mut lfor = 0.5;
            if inst.lfo_amt != 0.0 {
                lfor += osc_lfo(self.j as f32 * lfo_freq) * inst.lfo_amt / 512.0;
            }

            // Envelope
            let mut e = 1.0;
            if self.j < env_attack as usize {
                e = self.j as f32 / env_attack;
            } else if self.j >= env_attack as usize + env_sustain as usize {
                e -= (self.j - env_attack as usize - env_sustain as usize) as f32 / env_release;
            }

            let mut rsample = 0.0;

            // Oscillator 1
            if inst.osc1_vol != 0.0 {
                let mut t = o1t;
                if inst.lfo_osc1_freq != 0 {
                    t += lfor;
                }
                if inst.osc1_xenv != 0 {
                    t *= e * e;
                }
                self.c1 += t;
                rsample += osc1(self.c1) * inst.osc1_vol;
            }

            // Oscillator 2
            if inst.osc2_vol != 0.0 {
                let mut t = o2t;
                if inst.osc2_xenv != 0 {
                    t *= e * e;
                }
                self.c2 += t;
                rsample += osc2(self.c2) * inst.osc2_vol;
            }

            // Noise oscillator
            if inst.noise_fader != 0.0 {
                rsample += fastrand::f32().mul_add(2.0, -1.0) * inst.noise_fader * e;
            }

            rsample *= e / 255.0;

            // State variable filter
            let mut f = inst.fx_freq;
            if inst.lfo_fx_freq != 0 {
                f *= lfor;
            }
            f = 1.5 * (f * std::f32::consts::PI / SAMPLE_RATE as f32).sin();
            self.low += f * self.band;
            let high = q.mul_add(rsample - self.band, -self.low);
            self.band += f * high;
            match inst.fx_filter {
                1 => rsample = high,            // Hipass
                2 => rsample = self.low,        // Lopass
                3 => rsample = self.band,       // Bandpass
                4 => rsample = self.low + high, // Notch
                _ => {}
            }

            // Panning & master volume
            let mut t = 0.5;
            if inst.fx_pan_amt != 0.0 {
                t += osc_sin(self.j as f32 * pan_freq) * inst.fx_pan_amt / 512.0;
            }
            rsample *= 39.0 * inst.env_master;

            rsample *= 4.0 / 32768.0;
            out[NUM_CHANNELS * self.j] += rsample * (1.0 - t);
            out[NUM_CHANNELS * self.j + 1] += rsample * t;

            self.j += 1;
        }

        NUM_CHANNELS * self.j >= out.len()
    }
}

const OSCILLATORS: [fn(f32) -> f32; 4] = [osc_sin, osc_square, osc_saw, osc_tri];

fn osc_sin(value: f32) -> f32 {
    (value * std::f32::consts::PI * 2.0).sin()
}

fn osc_square(value: f32) -> f32 {
    osc_sin(value).signum()
}

fn osc_saw(value: f32) -> f32 {
    (value % 1.0) - 0.5
}

fn osc_tri(value: f32) -> f32 {
    let v2 = (value % 1.0) * 4.0;
    if v2 < 2.0 { v2 - 1.0 } else { 3.0 - v2 }
}

fn note_freq44100(n: i32) -> f32 {
    0.00390625 * 1.059463f32.powi(n - 128)
}

fn note_freq(n: i32) -> f32 {
    note_freq44100(n) / SAMPLE_RATE as f32 * 44100.0
}
