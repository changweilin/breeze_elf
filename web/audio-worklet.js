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
    this.buffer = new Float32Array(Math.max(4096, Math.ceil(sampleRate / 4)));
    this.bufferStart = 0;
    this.bufferLength = 0;
    this.readIndex = 0;
    this.pending = new Int16Array(this.chunkSamples);
    this.pendingLength = 0;
    this.pendingEnergy = 0;
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
    this.ensureCapacity(this.bufferLength + input.length);
    const writeIndex = (this.bufferStart + this.bufferLength) % this.buffer.length;
    const first = Math.min(input.length, this.buffer.length - writeIndex);
    this.buffer.set(input.subarray(0, first), writeIndex);
    const remaining = input.length - first;
    if (remaining > 0) {
      this.buffer.set(input.subarray(first), 0);
    }
    this.bufferLength += input.length;
  }

  drainResampled() {
    const available = this.bufferLength - 1;
    while (this.readIndex + this.ratio < available) {
      const index = this.readIndex;
      const left = Math.floor(index);
      const right = left + 1;
      const fraction = index - left;
      const leftSample = this.sampleAt(left);
      const rightSample = this.sampleAt(right);
      const sample = leftSample + (rightSample - leftSample) * fraction;
      this.pushSample(sample);
      this.readIndex += this.ratio;
    }

    const consumed = Math.floor(this.readIndex);
    if (consumed > 0) {
      this.dropInput(consumed);
      this.readIndex -= consumed;
    }
  }

  pushSample(sample) {
    const clamped = Math.max(-1, Math.min(1, sample));
    this.pending[this.pendingLength] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    this.pendingEnergy += clamped * clamped;
    this.pendingLength += 1;

    if (this.pendingLength >= this.pending.length) {
      const out = this.pending;
      const rms = Math.sqrt(this.pendingEnergy / this.pendingLength);
      this.port.postMessage({ type: "audio", buffer: out.buffer, rms }, [out.buffer]);
      this.pending = new Int16Array(this.chunkSamples);
      this.pendingLength = 0;
      this.pendingEnergy = 0;
    }
  }

  ensureCapacity(required) {
    if (required <= this.buffer.length) {
      return;
    }

    const next = new Float32Array(Math.max(required, this.buffer.length * 2));
    for (let i = 0; i < this.bufferLength; i += 1) {
      next[i] = this.sampleAt(i);
    }
    this.buffer = next;
    this.bufferStart = 0;
  }

  sampleAt(offset) {
    return this.buffer[(this.bufferStart + offset) % this.buffer.length];
  }

  dropInput(count) {
    const dropped = Math.min(count, this.bufferLength);
    this.bufferStart = (this.bufferStart + dropped) % this.buffer.length;
    this.bufferLength -= dropped;
    if (this.bufferLength === 0) {
      this.bufferStart = 0;
    }
  }
}

registerProcessor("breeze-mic-processor", BreezeMicProcessor);
