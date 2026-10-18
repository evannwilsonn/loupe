// Checks the browser's JavaScript inference against LibTorch.
//
// Runs web/infer.js with the float16 weights the page ships (web/weights.f16) on the first
// 1,000 test images and compares with the logits LibTorch wrote (models/test-logits.bin).
// Writes results/browser-check.json.
//
//   node scripts/check-infer.mjs        (after scripts/build-demo.py has written web/weights.f16)

import { readFileSync, writeFileSync } from "node:fs";
import { halfToFloat, layers, logits, normalize } from "../web/infer.js";

const final = JSON.parse(readFileSync("results/final.json", "utf8"));
const f16 = readFileSync("web/weights.f16");
const net = layers(halfToFloat(new Uint16Array(f16.buffer, f16.byteOffset, f16.length / 2)), final.widths);
const ref = new Float32Array(readFileSync("models/test-logits.bin").buffer.slice(0));
const test = readFileSync("data/cifar-10-batches-bin/test_batch.bin");

const N = 1000;
let maxDiff = 0, same = 0, right = 0;
const start = performance.now();
for (let i = 0; i < N; i++) {
  const rec = test.subarray(i * 3073, (i + 1) * 3073);
  const z = logits(net, normalize(rec.subarray(1), final.normalize.mean, final.normalize.std));
  let best = 0, refBest = 0;
  for (let k = 0; k < 10; k++) {
    maxDiff = Math.max(maxDiff, Math.abs(z[k] - ref[i * 10 + k]));
    if (z[k] > z[best]) best = k;
    if (ref[i * 10 + k] > ref[i * 10 + refBest]) refBest = k;
  }
  same += best === refBest;
  right += best === rec[0];
}
const ms = (performance.now() - start) / N;
const out = { images: N, sameClass: same, maxLogitDifference: +maxDiff.toFixed(5), accuracy: right / N, msPerImage: +ms.toFixed(1), weights: "float16" };
writeFileSync("results/browser-check.json", JSON.stringify(out, null, 2) + "\n");
console.log(out);
