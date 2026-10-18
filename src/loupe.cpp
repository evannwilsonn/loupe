#include "loupe.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace loupe {

// ---------- Data ----------

Split load_batches(const std::vector<std::string>& paths) {
  constexpr int64_t kRecord = 1 + 3 * 32 * 32;
  std::vector<uint8_t> raw;
  for (const auto& path : paths) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path + " (run scripts/get-data.sh)");
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (buf.size() % kRecord != 0) throw std::runtime_error(path + ": not a CIFAR-10 batch file");
    raw.insert(raw.end(), buf.begin(), buf.end());
  }
  const int64_t n = static_cast<int64_t>(raw.size()) / kRecord;
  auto images = torch::empty({n, 3, 32, 32}, torch::kUInt8);
  auto labels = torch::empty({n}, torch::kInt64);
  auto* img = images.data_ptr<uint8_t>();
  auto* lab = labels.data_ptr<int64_t>();
  for (int64_t i = 0; i < n; ++i) {
    const uint8_t* rec = raw.data() + i * kRecord;
    if (rec[0] > 9) throw std::runtime_error("label out of range");
    lab[i] = rec[0];
    std::copy(rec + 1, rec + kRecord, img + i * (kRecord - 1));
  }
  return {images, labels};
}

Split load_train(const std::string& dir) {
  std::vector<std::string> paths;
  for (int i = 1; i <= 5; ++i) paths.push_back(dir + "/data_batch_" + std::to_string(i) + ".bin");
  return load_batches(paths);
}

Split load_test(const std::string& dir) { return load_batches({dir + "/test_batch.bin"}); }

std::pair<Split, Split> stratified_split(const Split& all, int64_t per_class, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<int64_t> held, rest;
  auto lab = all.labels.accessor<int64_t, 1>();
  for (int64_t c = 0; c < 10; ++c) {
    std::vector<int64_t> idx;
    for (int64_t i = 0; i < all.size(); ++i)
      if (lab[i] == c) idx.push_back(i);
    std::shuffle(idx.begin(), idx.end(), rng);
    if (static_cast<int64_t>(idx.size()) < per_class) throw std::runtime_error("not enough images in a class");
    held.insert(held.end(), idx.begin(), idx.begin() + per_class);
    rest.insert(rest.end(), idx.begin() + per_class, idx.end());
  }
  std::sort(held.begin(), held.end());
  std::sort(rest.begin(), rest.end());
  auto take = [&](const std::vector<int64_t>& ids) {
    auto t = torch::tensor(ids, torch::kInt64);
    return Split{all.images.index_select(0, t), all.labels.index_select(0, t)};
  };
  return {take(held), take(rest)};
}

Normalizer Normalizer::fit(const Split& s) {
  Normalizer n;
  auto x = s.images.to(torch::kFloat64).div(255.0);
  for (int c = 0; c < 3; ++c) {
    auto ch = x.select(1, c);
    n.mean[c] = static_cast<float>(ch.mean().item<double>());
    n.std[c] = static_cast<float>(ch.std().item<double>());
  }
  return n;
}

torch::Tensor Normalizer::operator()(const torch::Tensor& images_u8) const {
  auto x = images_u8.to(torch::kFloat32).div_(255.0f);
  auto m = torch::tensor({mean[0], mean[1], mean[2]}).view({1, 3, 1, 1});
  auto s = torch::tensor({std[0], std[1], std[2]}).view({1, 3, 1, 1});
  return (x - m) / s;
}

torch::Tensor augment(const torch::Tensor& batch, torch::Generator& gen) {
  const int64_t n = batch.size(0);
  auto padded = torch::constant_pad_nd(batch, {4, 4, 4, 4}, 0.0);
  auto offsets = torch::randint(0, 9, {n, 2}, gen, torch::kInt64);
  auto flips = torch::randint(0, 2, {n}, gen, torch::kInt64);
  auto out = torch::empty_like(batch);
  auto off = offsets.accessor<int64_t, 2>();
  auto fl = flips.accessor<int64_t, 1>();
  for (int64_t i = 0; i < n; ++i) {
    auto crop = padded[i].narrow(1, off[i][0], 32).narrow(2, off[i][1], 32);
    out[i].copy_(fl[i] ? crop.flip({2}) : crop);
  }
  return out;
}

// ---------- Model ----------

NetImpl::NetImpl(NetOptions o) : opts(o) {
  // One flat list: [conv, BN, ReLU] x2, pool, three times. (LibTorch can't nest Sequentials.)
  torch::nn::Sequential seq;
  int64_t in = 3;
  for (int64_t width : o.widths) {
    for (int rep = 0; rep < 2; ++rep) {
      seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in, width, 3).padding(1).bias(false)));
      seq->push_back(torch::nn::BatchNorm2d(width));
      seq->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
      in = width;
    }
    seq->push_back(torch::nn::MaxPool2d(2));
  }
  features = register_module("features", seq);
  fc = register_module("fc", torch::nn::Linear(in, 10));
}

torch::Tensor NetImpl::forward(const torch::Tensor& x) {
  auto h = features->forward(x);
  h = h.mean({2, 3});  // global average pool: [N, C, 4, 4] -> [N, C]
  return fc->forward(h);
}

Folded fold(Net& net) {
  Folded f;
  torch::NoGradGuard guard;
  // features = [conv, BN, ReLU, conv, BN, ReLU, pool] x3
  const auto children = net->features->children();
  for (size_t i = 0; i < children.size(); ++i) {
    if (std::dynamic_pointer_cast<torch::nn::MaxPool2dImpl>(children[i])) {
      f.convs.back().pool_after = true;
      continue;
    }
    auto conv = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(children[i]);
    if (!conv) continue;  // BN and ReLU are handled with their convolution
    auto bn = std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(children[i + 1]);
    auto scale = bn->weight / torch::sqrt(bn->running_var + bn->options.eps());
    auto w = (conv->weight * scale.view({-1, 1, 1, 1})).contiguous();
    auto b = (bn->bias - bn->running_mean * scale).contiguous();
    FoldedConv fc{conv->weight.size(1), conv->weight.size(0), {}, {}, false};
    fc.weight.assign(w.data_ptr<float>(), w.data_ptr<float>() + w.numel());
    fc.bias.assign(b.data_ptr<float>(), b.data_ptr<float>() + b.numel());
    f.convs.push_back(std::move(fc));
  }
  auto w = net->fc->weight.contiguous();
  auto b = net->fc->bias.contiguous();
  f.fc_weight.assign(w.data_ptr<float>(), w.data_ptr<float>() + w.numel());
  f.fc_bias.assign(b.data_ptr<float>(), b.data_ptr<float>() + b.numel());
  return f;
}

std::vector<float> folded_logits(const Folded& f, const float* image_chw) {
  int64_t size = 32;
  std::vector<float> x(image_chw, image_chw + 3 * 32 * 32);
  for (const auto& c : f.convs) {
    std::vector<float> y(c.out * size * size);
    for (int64_t o = 0; o < c.out; ++o) {
      float* out = y.data() + o * size * size;
      std::fill(out, out + size * size, c.bias[o]);
      for (int64_t i = 0; i < c.in; ++i) {
        const float* in = x.data() + i * size * size;
        const float* k = c.weight.data() + (o * c.in + i) * 9;
        for (int64_t r = 0; r < size; ++r)
          for (int64_t s = 0; s < size; ++s) {
            float acc = 0;
            for (int dr = -1; dr <= 1; ++dr)
              for (int ds = -1; ds <= 1; ++ds) {
                const int64_t rr = r + dr, ss = s + ds;
                if (rr >= 0 && rr < size && ss >= 0 && ss < size) acc += in[rr * size + ss] * k[(dr + 1) * 3 + ds + 1];
              }
            out[r * size + s] += acc;
          }
      }
      for (int64_t j = 0; j < size * size; ++j) out[j] = std::max(out[j], 0.0f);
    }
    if (c.pool_after) {
      const int64_t half = size / 2;
      std::vector<float> p(c.out * half * half);
      for (int64_t o = 0; o < c.out; ++o)
        for (int64_t r = 0; r < half; ++r)
          for (int64_t s = 0; s < half; ++s) {
            const float* in = y.data() + o * size * size;
            p[(o * half + r) * half + s] = std::max({in[2 * r * size + 2 * s], in[2 * r * size + 2 * s + 1],
                                                     in[(2 * r + 1) * size + 2 * s], in[(2 * r + 1) * size + 2 * s + 1]});
          }
      y.swap(p);
      size = half;
    }
    x.swap(y);
  }
  const int64_t channels = f.convs.back().out;
  std::vector<float> pooled(channels, 0.0f);
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t j = 0; j < size * size; ++j) pooled[c] += x[c * size * size + j];
    pooled[c] /= static_cast<float>(size * size);
  }
  std::vector<float> logits(10);
  for (int k = 0; k < 10; ++k) {
    float acc = f.fc_bias[k];
    for (int64_t c = 0; c < channels; ++c) acc += f.fc_weight[k * channels + c] * pooled[c];
    logits[k] = acc;
  }
  return logits;
}

// ---------- Metrics ----------

std::pair<double, double> wilson(int64_t k, int64_t n) {
  if (n == 0) return {0, 0};
  const double z = 1.959963984540054, p = static_cast<double>(k) / n, z2 = z * z;
  const double centre = (p + z2 / (2.0 * n)) / (1 + z2 / n);
  const double half = z * std::sqrt(p * (1 - p) / n + z2 / (4.0 * n * n)) / (1 + z2 / n);
  return {centre - half, centre + half};
}

Scores score(const torch::Tensor& predictions, const torch::Tensor& labels) {
  Scores s;
  auto p = predictions.to(torch::kInt64).contiguous();
  auto l = labels.to(torch::kInt64).contiguous();
  auto pa = p.accessor<int64_t, 1>();
  auto la = l.accessor<int64_t, 1>();
  int64_t right = 0;
  for (int64_t i = 0; i < l.size(0); ++i) {
    s.confusion[la[i]][pa[i]]++;
    right += pa[i] == la[i];
  }
  s.accuracy = static_cast<double>(right) / l.size(0);
  std::tie(s.low, s.high) = wilson(right, l.size(0));
  return s;
}

torch::Tensor logits(Net& net, const Normalizer& norm, const torch::Tensor& images_u8, bool flip) {
  torch::NoGradGuard guard;
  net->eval();
  std::vector<torch::Tensor> out;
  for (int64_t i = 0; i < images_u8.size(0); i += 500) {
    auto x = norm(images_u8.narrow(0, i, std::min<int64_t>(500, images_u8.size(0) - i)));
    auto y = net->forward(x);
    if (flip) y = (y + net->forward(x.flip({3}))) / 2;
    out.push_back(y);
  }
  return torch::cat(out);
}

}  // namespace loupe
