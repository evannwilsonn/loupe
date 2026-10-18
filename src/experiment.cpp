// The whole evaluation, in stages, each writing results/*.json:
//
//   experiment baselines                 majority class, k-nearest neighbours, softmax regression
//   experiment train <name> <epochs> <augment 0|1>
//                                        trains the CNN on 45,000 images, scores the 5,000 held out
//   experiment final <name>              picks test-time flipping on validation, scores the test set
//                                        once, exports the folded weights and demo samples
//
// Validation is 500 images per class taken out of the training batches with a fixed seed; the
// 10,000 test images are only touched by `baselines` and `final`.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

#include "loupe.hpp"

using namespace loupe;
namespace fs = std::filesystem;

namespace {

const char* kData = "data/cifar-10-batches-bin";
constexpr uint64_t kSeed = 2026;
constexpr int64_t kBatch = 128;

double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string num(double v, int digits = 4) {
  std::ostringstream o;
  o << std::fixed << std::setprecision(digits) << v;
  return o.str();
}

std::string scores_json(const Scores& s, bool confusion = false) {
  std::ostringstream o;
  o << "{\"accuracy\": " << num(s.accuracy) << ", \"low\": " << num(s.low) << ", \"high\": " << num(s.high);
  if (confusion) {
    o << ", \"confusion\": [";
    for (int t = 0; t < 10; ++t) {
      o << (t ? ", " : "") << "[";
      for (int p = 0; p < 10; ++p) o << (p ? ", " : "") << s.confusion[t][p];
      o << "]";
    }
    o << "]";
  }
  o << "}";
  return o.str();
}

void write(const std::string& path, const std::string& text) {
  fs::create_directories(fs::path(path).parent_path());
  std::ofstream(path) << text;
  std::cout << "wrote " << path << "\n";
}

struct Data {
  Split train, valid, test;
  Normalizer norm;
};

Data load_all(bool with_test) {
  Data d;
  auto all = load_train(kData);
  std::tie(d.valid, d.train) = stratified_split(all, 500, kSeed);
  if (with_test) d.test = load_test(kData);
  d.norm = Normalizer::fit(d.train);
  return d;
}

// ---------- Baselines ----------

void baselines() {
  auto d = load_all(true);
  std::ostringstream json;
  json << "{\n";

  // Majority class: every class has the same count, so this is 10% by construction.
  json << "  \"majority\": {\"valid\": " << scores_json(score(torch::zeros_like(d.valid.labels), d.valid.labels))
       << ", \"test\": " << scores_json(score(torch::zeros_like(d.test.labels), d.test.labels)) << "},\n";

  // k-nearest neighbours on normalized pixels (Euclidean distance), k chosen on validation.
  auto flat = [&](const Split& s) { return d.norm(s.images).flatten(1); };
  auto xtr = flat(d.train), xva = flat(d.valid), xte = flat(d.test);
  auto sq_tr = xtr.pow(2).sum(1);
  auto knn_votes = [&](const torch::Tensor& x, int64_t kmax) {
    std::vector<torch::Tensor> idx;
    for (int64_t i = 0; i < x.size(0); i += 500) {
      auto q = x.narrow(0, i, std::min<int64_t>(500, x.size(0) - i));
      auto dist = sq_tr.unsqueeze(0) - 2 * q.matmul(xtr.t());  // plus |q|^2, which is constant per row
      idx.push_back(std::get<1>(dist.topk(kmax, 1, false)));
    }
    return d.train.labels.index_select(0, torch::cat(idx).flatten()).view({x.size(0), kmax});
  };
  // Majority vote of the k nearest; ties go to the class of the nearer neighbour.
  auto predict_k = [](const torch::Tensor& votes, int64_t k) {
    auto v = votes.narrow(1, 0, k);
    auto counts = torch::zeros({v.size(0), 10}, torch::kFloat32);
    counts.scatter_add_(1, v, torch::ones_like(v, torch::kFloat32));
    auto bonus = torch::zeros_like(counts);
    for (int64_t r = k - 1; r >= 0; --r)
      bonus.scatter_(1, v.narrow(1, r, 1), torch::full({v.size(0), 1}, 1e-3f * static_cast<float>(k - r)));
    return (counts + bonus).argmax(1);
  };
  double t0 = now();
  auto votes_va = knn_votes(xva, 20);
  int64_t best_k = 1;
  double best = -1;
  json << "  \"knn\": {\"validByK\": {";
  bool first = true;
  for (int64_t k : {1, 3, 5, 10, 20}) {
    auto s = score(predict_k(votes_va, k), d.valid.labels);
    json << (first ? "" : ", ") << "\"" << k << "\": " << num(s.accuracy);
    first = false;
    std::cout << "kNN k=" << k << " valid " << s.accuracy << "\n";
    if (s.accuracy > best) best = s.accuracy, best_k = k;
  }
  auto knn_va = score(predict_k(votes_va, best_k), d.valid.labels);
  auto knn_te = score(predict_k(knn_votes(xte, best_k), best_k), d.test.labels);
  json << "}, \"k\": " << best_k << ", \"valid\": " << scores_json(knn_va) << ", \"test\": " << scores_json(knn_te)
       << ", \"seconds\": " << num(now() - t0, 1) << "},\n";
  std::cout << "kNN k=" << best_k << " test " << knn_te.accuracy << "\n";

  // Softmax regression: one linear layer on the normalized pixels, learning rate chosen on validation.
  t0 = now();
  json << "  \"softmax\": {\"epochs\": 20, \"validByLr\": {";
  torch::nn::Linear best_lin{nullptr};
  double best_lr = 0, best_va = -1;
  for (double lr : {0.0003, 0.001, 0.003, 0.01}) {
    torch::manual_seed(kSeed);
    torch::nn::Linear lin(3 * 32 * 32, 10);
    torch::optim::SGD opt(lin->parameters(), torch::optim::SGDOptions(lr).momentum(0.9).weight_decay(1e-3));
    auto gen = at::make_generator<at::CPUGeneratorImpl>(kSeed);
    for (int epoch = 0; epoch < 20; ++epoch) {
      auto perm = torch::randperm(xtr.size(0), gen, torch::kInt64);
      for (int64_t i = 0; i + kBatch <= xtr.size(0); i += kBatch) {
        auto idx = perm.narrow(0, i, kBatch);
        auto loss = torch::nn::functional::cross_entropy(lin->forward(xtr.index_select(0, idx)),
                                                         d.train.labels.index_select(0, idx));
        opt.zero_grad();
        loss.backward();
        opt.step();
      }
    }
    torch::NoGradGuard ng;
    const double va = score(lin->forward(xva).argmax(1), d.valid.labels).accuracy;
    json << (best_va < 0 ? "" : ", ") << "\"" << lr << "\": " << num(va);
    std::cout << "softmax regression lr " << lr << " valid " << va << "\n";
    if (va > best_va) best_va = va, best_lr = lr, best_lin = lin;
  }
  torch::NoGradGuard ng;
  auto lin_va = score(best_lin->forward(xva).argmax(1), d.valid.labels);
  auto lin_te = score(best_lin->forward(xte).argmax(1), d.test.labels);
  json << "}, \"lr\": " << best_lr << ", \"valid\": " << scores_json(lin_va) << ", \"test\": " << scores_json(lin_te)
       << ", \"seconds\": " << num(now() - t0, 1) << "}\n}\n";
  std::cout << "softmax regression lr " << best_lr << " test " << lin_te.accuracy << "\n";
  write("results/baselines.json", json.str());
}

// ---------- Training ----------

void train(const std::string& name, int epochs, bool aug) {
  auto d = load_all(false);
  torch::manual_seed(kSeed);
  auto gen = at::make_generator<at::CPUGeneratorImpl>(kSeed);
  Net net;
  int64_t params = 0;
  for (const auto& p : net->parameters()) params += p.numel();
  const double peak = 0.1, wd = 5e-4;
  torch::optim::SGD opt(net->parameters(),
                        torch::optim::SGDOptions(peak).momentum(0.9).nesterov(true).weight_decay(wd));
  const int64_t steps_per_epoch = d.train.size() / kBatch;
  const int64_t total = steps_per_epoch * epochs, warm = steps_per_epoch * 2;
  // Linear warm-up for two epochs, then cosine decay to zero.
  auto lr_at = [&](int64_t step) {
    if (step < warm) return peak * double(step + 1) / double(warm);
    return peak * 0.5 * (1 + std::cos(M_PI * double(step - warm) / double(total - warm)));
  };

  std::ostringstream curve;
  int64_t step = 0;
  const double start = now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    net->train();
    auto perm = torch::randperm(d.train.size(), gen, torch::kInt64);
    double loss_sum = 0;
    int64_t right = 0, seen = 0;
    const double t0 = now();
    for (int64_t i = 0; i + kBatch <= d.train.size(); i += kBatch, ++step) {
      auto idx = perm.narrow(0, i, kBatch);
      auto x = d.norm(d.train.images.index_select(0, idx));
      if (aug) x = augment(x, gen);
      auto y = d.train.labels.index_select(0, idx);
      for (auto& group : opt.param_groups())
        static_cast<torch::optim::SGDOptions&>(group.options()).lr(lr_at(step));
      auto out = net->forward(x);
      auto loss = torch::nn::functional::cross_entropy(out, y);
      opt.zero_grad();
      loss.backward();
      opt.step();
      loss_sum += loss.item<double>() * kBatch;
      right += out.argmax(1).eq(y).sum().item<int64_t>();
      seen += kBatch;
    }
    auto va = score(logits(net, d.norm, d.valid.images).argmax(1), d.valid.labels);
    const double secs = now() - t0;
    curve << (epoch > 1 ? ",\n" : "") << "    {\"epoch\": " << epoch << ", \"trainLoss\": " << num(loss_sum / seen)
          << ", \"trainAccuracy\": " << num(double(right) / seen) << ", \"validAccuracy\": " << num(va.accuracy)
          << ", \"seconds\": " << num(secs, 1) << "}";
    std::printf("%s epoch %2d  loss %.4f  train %.4f  valid %.4f  %.0fs\n", name.c_str(), epoch, loss_sum / seen,
                double(right) / seen, va.accuracy, secs);
    std::fflush(stdout);
  }
  fs::create_directories("models");
  torch::save(net, "models/" + name + ".pt");
  auto va = score(logits(net, d.norm, d.valid.images).argmax(1), d.valid.labels);
  std::ostringstream json;
  json << "{\n  \"name\": \"" << name << "\", \"epochs\": " << epochs << ", \"augment\": " << (aug ? "true" : "false")
       << ", \"parameters\": " << params << ", \"batch\": " << kBatch << ", \"peakLr\": " << peak
       << ", \"weightDecay\": " << wd << ", \"minutes\": " << num((now() - start) / 60, 1) << ",\n  \"valid\": "
       << scores_json(va) << ",\n  \"curve\": [\n"
       << curve.str() << "\n  ]\n}\n";
  write("results/run-" + name + ".json", json.str());
}

// ---------- Final ----------

void final_eval(const std::string& name) {
  auto d = load_all(true);
  Net net;
  torch::load(net, "models/" + name + ".pt");
  net->eval();

  // Test-time flipping, decided on validation.
  auto plain_va = score(logits(net, d.norm, d.valid.images).argmax(1), d.valid.labels);
  auto flip_va = score(logits(net, d.norm, d.valid.images, true).argmax(1), d.valid.labels);
  const bool flip = flip_va.accuracy > plain_va.accuracy;
  std::cout << "valid: plain " << plain_va.accuracy << ", flipped " << flip_va.accuracy << " -> flip " << flip << "\n";

  // The test set, once.
  const double t0 = now();
  auto test_logits = logits(net, d.norm, d.test.images, flip);
  const double test_secs = now() - t0;
  auto te = score(test_logits.argmax(1), d.test.labels);
  auto plain_logits = logits(net, d.norm, d.test.images, false);
  auto te_plain = score(plain_logits.argmax(1), d.test.labels);
  std::cout << "test " << te.accuracy << " [" << te.low << ", " << te.high << "], without flipping "
            << te_plain.accuracy << "\n";

  // Export: BatchNorm folded into the convolutions, checked against LibTorch.
  auto folded = fold(net);
  auto x = d.norm(d.test.images.narrow(0, 0, 200)).contiguous();
  double max_diff = 0;
  int64_t agree = 0;
  for (int64_t i = 0; i < 200; ++i) {
    auto l = folded_logits(folded, x[i].data_ptr<float>());
    auto ref = plain_logits[i];
    int best = 0;
    for (int k = 0; k < 10; ++k) {
      max_diff = std::max(max_diff, std::abs(double(l[k]) - ref[k].item<double>()));
      if (l[k] > l[best]) best = k;
    }
    agree += best == ref.argmax().item<int64_t>();
  }
  std::cout << "folded vs LibTorch on 200 test images: max logit difference " << max_diff << ", same class "
            << agree << "/200\n";

  // Weights for the browser: float32, little-endian, layer by layer (see web/template.html).
  {
    std::ofstream out("models/weights.bin", std::ios::binary);
    auto put = [&](const std::vector<float>& v) {
      out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    };
    for (const auto& c : folded.convs) put(c.weight), put(c.bias);
    put(folded.fc_weight), put(folded.fc_bias);
  }

  // Logits of the first 1,000 test images (no flipping), so the browser code can be checked.
  {
    auto ref = plain_logits.narrow(0, 0, 1000).contiguous();
    std::ofstream out("models/test-logits.bin", std::ios::binary);
    out.write(reinterpret_cast<const char*>(ref.data_ptr<float>()), static_cast<std::streamsize>(ref.numel() * sizeof(float)));
  }

  // Samples for the demo: the first 12 test images of each class, with the model's top three.
  auto probs = torch::softmax(test_logits, 1);
  std::ostringstream samples;
  auto lab = d.test.labels.accessor<int64_t, 1>();
  std::array<int, 10> taken{};
  bool first = true;
  for (int64_t i = 0; i < d.test.size(); ++i) {
    const int64_t c = lab[i];
    if (taken[c] >= 12) continue;
    taken[c]++;
    auto img = d.test.images[i].contiguous();
    static const char* digits = "0123456789abcdef";
    std::string hex;
    hex.reserve(3072 * 2);
    const uint8_t* p = img.data_ptr<uint8_t>();
    for (int j = 0; j < 3072; ++j) hex += digits[p[j] >> 4], hex += digits[p[j] & 15];
    auto top = probs[i].topk(3);
    samples << (first ? "" : ",\n") << "    {\"index\": " << i << ", \"label\": " << c << ", \"top\": [";
    for (int k = 0; k < 3; ++k)
      samples << (k ? ", " : "") << "[" << std::get<1>(top)[k].item<int64_t>() << ", "
              << num(std::get<0>(top)[k].item<double>()) << "]";
    samples << "], \"pixels\": \"" << hex << "\"}";
    first = false;
  }

  // Which wrong answers are most common.
  std::vector<std::tuple<int64_t, int, int>> mistakes;
  for (int t = 0; t < 10; ++t)
    for (int p = 0; p < 10; ++p)
      if (t != p) mistakes.emplace_back(te.confusion[t][p], t, p);
  std::sort(mistakes.rbegin(), mistakes.rend());

  std::ostringstream json;
  json << "{\n  \"model\": \"" << name << "\",\n  \"validPlain\": " << scores_json(plain_va)
       << ",\n  \"validFlip\": " << scores_json(flip_va) << ",\n  \"flip\": " << (flip ? "true" : "false")
       << ",\n  \"test\": " << scores_json(te, true) << ",\n  \"testPlain\": " << scores_json(te_plain)
       << ",\n  \"testSeconds\": " << num(test_secs, 2) << ",\n  \"perClass\": [";
  for (int c = 0; c < 10; ++c) {
    int64_t n = 0;
    for (int p = 0; p < 10; ++p) n += te.confusion[c][p];
    json << (c ? ", " : "") << num(double(te.confusion[c][c]) / double(n));
  }
  json << "],\n  \"topMistakes\": [";
  for (int i = 0; i < 6; ++i)
    json << (i ? ", " : "") << "{\"true\": " << std::get<1>(mistakes[i]) << ", \"predicted\": "
         << std::get<2>(mistakes[i]) << ", \"count\": " << std::get<0>(mistakes[i]) << "}";
  json << "],\n  \"export\": {\"maxLogitDifference\": " << num(max_diff, 8) << ", \"sameClass\": " << agree
       << ", \"checked\": 200},\n  \"normalize\": {\"mean\": [" << d.norm.mean[0] << ", " << d.norm.mean[1] << ", "
       << d.norm.mean[2] << "], \"std\": [" << d.norm.std[0] << ", " << d.norm.std[1] << ", " << d.norm.std[2]
       << "]},\n  \"widths\": [" << net->opts.widths[0] << ", " << net->opts.widths[1] << ", " << net->opts.widths[2]
       << "]\n}\n";
  write("results/final.json", json.str());
  // What the server loads.
  torch::save(net, "models/final.pt");
  write("models/normalize.txt", std::to_string(d.norm.mean[0]) + " " + std::to_string(d.norm.mean[1]) + " " +
                                    std::to_string(d.norm.mean[2]) + " " + std::to_string(d.norm.std[0]) + " " +
                                    std::to_string(d.norm.std[1]) + " " + std::to_string(d.norm.std[2]) + "\n");
  write("models/flip.txt", flip ? "1\n" : "0\n");
  write("results/samples.json", "[\n" + samples.str() + "\n]\n");
}

}  // namespace

int main(int argc, char** argv) {
  torch::set_num_threads(static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
  const std::string stage = argc > 1 ? argv[1] : "";
  try {
    if (stage == "baselines") baselines();
    else if (stage == "train" && argc == 5) train(argv[2], std::stoi(argv[3]), std::string(argv[4]) == "1");
    else if (stage == "final" && argc == 3) final_eval(argv[2]);
    else {
      std::cerr << "usage: experiment baselines | train <name> <epochs> <augment 0|1> | final <name>\n";
      return 2;
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
