// The trained network, run in plain JavaScript. BatchNorm is already folded into each
// convolution (see fold() in src/loupe.cpp), so a forward pass is:
//   six 3x3 convolutions with ReLU, a 2x2 max-pool after every second one,
//   a global average over the last 4x4 map, and one linear layer.
// Weights arrive as float16 in the order fold() writes them.

export function halfToFloat(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    out[i] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return out;
}

// Splits the flat weight array into layers for the given stage widths, e.g. [32, 64, 128].
export function layers(flat, widths) {
  const convs = [];
  let at = 0, cin = 3;
  for (const w of widths) {
    for (let rep = 0; rep < 2; rep++) {
      const weight = flat.subarray(at, at + w * cin * 9); at += w * cin * 9;
      const bias = flat.subarray(at, at + w); at += w;
      convs.push({ cin, cout: w, weight, bias, pool: rep === 1 });
      cin = w;
    }
  }
  const fcW = flat.subarray(at, at + 10 * cin); at += 10 * cin;
  const fcB = flat.subarray(at, at + 10); at += 10;
  if (at !== flat.length) throw new Error(`weights: expected ${at} values, got ${flat.length}`);
  return { convs, fcW, fcB, channels: cin };
}

// 3x3 convolution, padding 1, then ReLU. x: [cin, n, n] -> [cout, n, n].
function conv(x, n, { cin, cout, weight, bias }) {
  const plane = n * n, y = new Float32Array(cout * plane);
  for (let o = 0; o < cout; o++) {
    const out = y.subarray(o * plane, (o + 1) * plane);
    out.fill(bias[o]);
    for (let i = 0; i < cin; i++) {
      const src = x.subarray(i * plane, (i + 1) * plane);
      const k = (o * cin + i) * 9;
      for (let dr = -1; dr <= 1; dr++) {
        const r0 = Math.max(0, -dr), r1 = Math.min(n, n - dr);
        for (let dc = -1; dc <= 1; dc++) {
          const kw = weight[k + (dr + 1) * 3 + dc + 1];
          if (kw === 0) continue;
          const c0 = Math.max(0, -dc), c1 = Math.min(n, n - dc);
          for (let r = r0; r < r1; r++) {
            const row = r * n, srow = (r + dr) * n + dc;
            for (let c = c0; c < c1; c++) out[row + c] += kw * src[srow + c];
          }
        }
      }
    }
    for (let j = 0; j < plane; j++) if (out[j] < 0) out[j] = 0;
  }
  return y;
}

function pool(x, channels, n) {
  const h = n / 2, y = new Float32Array(channels * h * h);
  for (let c = 0; c < channels; c++)
    for (let r = 0; r < h; r++)
      for (let s = 0; s < h; s++) {
        const b = c * n * n + 2 * r * n + 2 * s;
        y[(c * h + r) * h + s] = Math.max(x[b], x[b + 1], x[b + n], x[b + n + 1]);
      }
  return y;
}

// image: Float32Array [3, 32, 32], already normalized. Returns 10 logits.
export function logits(net, image) {
  let x = image, n = 32;
  for (const layer of net.convs) {
    x = conv(x, n, layer);
    if (layer.pool) { x = pool(x, layer.cout, n); n /= 2; }
  }
  const C = net.channels, plane = n * n, out = new Float32Array(10);
  const avg = new Float32Array(C);
  for (let c = 0; c < C; c++) {
    let s = 0;
    for (let j = 0; j < plane; j++) s += x[c * plane + j];
    avg[c] = s / plane;
  }
  for (let k = 0; k < 10; k++) {
    let s = net.fcB[k];
    for (let c = 0; c < C; c++) s += net.fcW[k * C + c] * avg[c];
    out[k] = s;
  }
  return out;
}

// RGB bytes [3, 32, 32] (channel-major, as in CIFAR) -> normalized float input.
export function normalize(rgb, mean, std) {
  const x = new Float32Array(3 * 1024);
  for (let c = 0; c < 3; c++)
    for (let j = 0; j < 1024; j++) x[c * 1024 + j] = (rgb[c * 1024 + j] / 255 - mean[c]) / std[c];
  return x;
}

export function mirror(x) {
  const y = new Float32Array(x.length);
  for (let c = 0; c < 3; c++)
    for (let r = 0; r < 32; r++)
      for (let s = 0; s < 32; s++) y[c * 1024 + r * 32 + s] = x[c * 1024 + r * 32 + 31 - s];
  return y;
}

export function softmax(z) {
  const m = Math.max(...z), e = Array.from(z, v => Math.exp(v - m)), t = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / t);
}

// Probabilities for one image, averaging with its mirror image when `flip` is set.
export function classify(net, rgb, norm, flip) {
  const x = normalize(rgb, norm.mean, norm.std);
  let z = logits(net, x);
  if (flip) {
    const zm = logits(net, mirror(x));
    z = z.map((v, k) => (v + zm[k]) / 2);
  }
  return softmax(z);
}
