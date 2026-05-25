class BreezeMicProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const processorOptions = options.processorOptions || {};
    this.targetSampleRate = processorOptions.targetSampleRate || 16000;
    this.chunkSamples = Math.max(
      256,
      Math.round((this.targetSampleRate * (processorOptions.chunkMs || 250)) / 1000),
    );
    this.ratio = sampleRate / this.targetSampleRate;
    this.buffer = new Float32Array(0);
    this.readIndex = 0;
    this.pending = new Int16Array(this.chunkSamples);
    this.pendingLength = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel || channel.length === 0) {
      return true;
    }

    this.appendInput(channel);
    this.drainResampled();
    return true;
  }

  appendInput(input) {
    const next = new Float32Array(this.buffer.length + input.length);
    next.set(this.buffer);
    next.set(input, this.buffer.length);
    this.buffer = next;
  }

  drainResampled() {
    const available = this.buffer.length - 1;
    while (this.readIndex + this.ratio < available) {
      const index = this.readIndex;
      const left = Math.floor(index);
      const right = left + 1;
      const fraction = index - left;
      const sample = this.buffer[left] + (this.buffer[right] - this.buffer[left]) * fraction;
      this.pushSample(sample);
      this.readIndex += this.ratio;
    }

    const consumed = Math.floor(this.readIndex);
    if (consumed > 0) {
      this.buffer = this.buffer.slice(consumed);
      this.readIndex -= consumed;
    }
  }

  pushSample(sample) {
    const clamped = Math.max(-1, Math.min(1, sample));
    this.pending[this.pendingLength] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    this.pendingLength += 1;

    if (this.pendingLength >= this.pending.length) {
      const out = this.pending.slice(0, this.pendingLength);
      this.port.postMessage({ type: "audio", buffer: out.buffer }, [out.buffer]);
      this.pendingLength = 0;
    }
  }
}

registerProcessor("breeze-mic-processor", BreezeMicProcessor);

