// Unit tests for the parts that can be checked without training: data loading, the split,
// augmentation, metrics, and the BatchNorm folding the browser export relies on.

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "loupe.hpp"

using namespace loupe;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
  std::cout << (ok ? "ok   " : "FAIL ") << what << "\n";
  if (!ok) ++failures;
}
void test(const std::string& name, const std::function<void()>& fn) {
  try {
    fn();
  } catch (const std::exception& e) {
    check(false, name + " threw: " + e.what());
  }
}
}  // namespace

int main() {
  torch::manual_seed(1);

  test("load", [] {
    // Two records: label 3 with all-7 pixels, label 9 with pixels counting up.
    auto path = (std::filesystem::temp_directory_path() / "loupe_test_batch.bin").string();
    std::ofstream out(path, std::ios::binary);
    std::vector<uint8_t> rec(3073);
    rec[0] = 3;
    std::fill(rec.begin() + 1, rec.end(), 7);
    out.write(reinterpret_cast<char*>(rec.data()), 3073);
    rec[0] = 9;
    for (int i = 0; i < 3072; ++i) rec[i + 1] = static_cast<uint8_t>(i % 256);
    out.write(reinterpret_cast<char*>(rec.data()), 3073);
    out.close();
    auto s = load_batches({path});
    check(s.size() == 2, "reads one image per 3,073-byte record");
    check(s.labels[0].item<int64_t>() == 3 && s.labels[1].item<int64_t>() == 9, "reads the label byte");
    // Channel 1 (green), row 0, column 5 is byte 1024 + 5 of the pixel block.
    check(s.images[1][1][0][5].item<int>() == (1024 + 5) % 256, "pixels are channel-major, row by row");
    std::filesystem::remove(path);
  });

  test("split", [] {
    auto labels = torch::arange(1000).remainder(10);
    auto images = torch::arange(1000).to(torch::kUInt8).view({1000, 1, 1, 1}).expand({1000, 3, 32, 32}).contiguous();
    auto [held, rest] = stratified_split(Split{images, labels}, 20, 7);
    check(held.size() == 200 && rest.size() == 800, "takes exactly per_class images of each class");
    check(torch::bincount(held.labels).eq(20).all().item<bool>(), "held-out set is balanced");
    auto again = stratified_split(Split{images, labels}, 20, 7).first;
    check(again.labels.equal(held.labels) && again.images.equal(held.images), "same seed, same split");
  });

  test("augment", [] {
    auto gen = at::make_generator<at::CPUGeneratorImpl>(3);
    auto x = torch::ones({16, 3, 32, 32});
    auto y = augment(x, gen);
    check(y.sizes() == x.sizes(), "keeps the batch shape");
    check(((y == 0) | (y == 1)).all().item<bool>(), "pixels come from the image or the zero padding");
    check(y.sum().item<double>() >= x.sum().item<double>() * (28.0 / 32) * (28.0 / 32) - 1e-6,
          "a crop moves the image at most 4 pixels");
  });

  test("metrics", [] {
    auto [lo, hi] = wilson(81, 263);  // Newcombe (1998): 0.2553 to 0.3662
    check(std::abs(lo - 0.2553) < 5e-4 && std::abs(hi - 0.3662) < 5e-4, "Wilson interval matches a published value");
    auto s = score(torch::tensor({0, 1, 1, 2}), torch::tensor({0, 1, 2, 2}));
    check(std::abs(s.accuracy - 0.75) < 1e-12, "accuracy counts right answers");
    check(s.confusion[2][1] == 1 && s.confusion[2][2] == 1, "confusion matrix is [true][predicted]");
  });

  test("fold", [] {
    // Give BatchNorm non-trivial statistics, then compare the folded loops with LibTorch.
    Net net;
    {
      torch::NoGradGuard g;
      for (auto& m : net->modules(false))
        if (auto bn = std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(m)) {
          bn->running_mean.uniform_(-0.5, 0.5);
          bn->running_var.uniform_(0.5, 2.0);
          bn->weight.uniform_(0.5, 1.5);
          bn->bias.uniform_(-0.2, 0.2);
        }
    }
    net->eval();
    auto x = torch::randn({3, 3, 32, 32});
    torch::Tensor ref;
    {
      torch::NoGradGuard g;
      ref = net->forward(x);
    }
    auto f = fold(net);
    check(f.convs.size() == 6 && f.convs[1].pool_after && f.convs[5].pool_after && !f.convs[0].pool_after,
          "six convolutions, a pool after every second one");
    double worst = 0;
    for (int i = 0; i < 3; ++i) {
      auto img = x[i].contiguous();
      auto l = folded_logits(f, img.data_ptr<float>());
      for (int k = 0; k < 10; ++k) worst = std::max(worst, std::abs(double(l[k]) - ref[i][k].item<double>()));
    }
    check(worst < 1e-4, "folded network gives LibTorch's logits (max difference " + std::to_string(worst) + ")");
  });

  std::cout << (failures ? std::to_string(failures) + " failed\n" : "all passed\n");
  return failures ? 1 : 0;
}
