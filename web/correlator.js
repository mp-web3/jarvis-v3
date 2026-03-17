// correlator.js - AudioWorklet for echo cancellation via reference correlation
// Compares mic input against TTS reference signal to detect echo frames.
// Source: adapted from shubhdotai/offline-voice-ai (MIT)
class Correlator extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frame = 480;
  }

  _rms(buf) {
    let s = 0;
    for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
    return Math.sqrt(s / buf.length);
  }

  _corr(x, y) {
    let xy = 0, xx = 0, yy = 0;
    for (let i = 0; i < x.length; i++) {
      const a = x[i], b = y[i];
      xy += a * b;
      xx += a * a;
      yy += b * b;
    }
    return xy / (Math.sqrt(xx * yy) + 1e-9);
  }

  process(inputs, outputs) {
    const mic = inputs[0][0];
    const ref = inputs[1]?.[0];
    const out = outputs[0][0];

    if (!mic) return true;

    if (out) {
      for (let i = 0; i < mic.length; i++) out[i] = mic[i];
    }

    if (ref && ref.length === mic.length) {
      const corr = this._corr(mic, ref);
      const micRms = this._rms(mic);
      const refRms = this._rms(ref);
      this.port.postMessage({
        corr: Math.max(0, corr),
        micRms,
        refRms,
      });
    } else {
      const micRms = this._rms(mic);
      this.port.postMessage({ corr: 0, micRms, refRms: 0 });
    }
    return true;
  }
}

registerProcessor('correlator', Correlator);
