// Loupe: a small CNN for CIFAR-10, written against LibTorch (the PyTorch C++ API).
//
// Shared by the experiment (train + evaluate), the HTTP server and the tests.
#pragma once

#include <torch/torch.h>

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace loupe {

inline const std::array<std::string, 10> kClasses = {"airplane", "automobile", "bird", "cat", "deer",
                                                      "dog",      "frog",       "horse", "ship", "truck"};

// ---------- Data ----------

// A set of images as uint8 [N, 3, 32, 32] and labels as int64 [N].
struct Split {
  torch::Tensor images;
  torch::Tensor labels;
  int64_t size() const { return labels.size(0); }
};

// Reads CIFAR-10 binary batch files: each record is 1 label byte, then 3,072 pixel bytes
// (1,024 red, 1,024 green, 1,024 blue, row by row).
Split load_batches(const std::vector<std::string>& paths);
Split load_train(const std::string& dir);  // data_batch_1..5, 50,000 images
Split load_test(const std::string& dir);   // test_batch, 10,000 images

// Takes `per_class` images of each class (chosen with `seed`) out of `all`.
// Returns {held_out, rest}.
std::pair<Split, Split> stratified_split(const Split& all, int64_t per_class, uint64_t seed);

// Per-channel mean and standard deviation of pixel values in [0, 1].
struct Normalizer {
  std::array<float, 3> mean{}, std{};
  static Normalizer fit(const Split& s);
  // uint8 [N,3,32,32] -> normalized float [N,3,32,32].
  torch::Tensor operator()(const torch::Tensor& images_u8) const;
};

// Random crop (pad 4, zero fill) and random horizontal flip, on a float batch.
torch::Tensor augment(const torch::Tensor& batch, torch::Generator& gen);

// ---------- Model ----------

struct NetOptions {
  std::array<int64_t, 3> widths{32, 64, 128};
};

// Three stages of [conv3x3-BN-ReLU] x2 then 2x2 max-pool, then global average pool and a
// linear layer: 32x32 -> 16x16 -> 8x8 -> 4x4 -> 1x1.
struct NetImpl : torch::nn::Module {
  explicit NetImpl(NetOptions opts = {});
  torch::Tensor forward(const torch::Tensor& x);
  NetOptions opts;
  torch::nn::Sequential features{nullptr};
  torch::nn::Linear fc{nullptr};
};
TORCH_MODULE(Net);

// A layer of the folded network: what the browser runs. BatchNorm is merged into the
// preceding convolution's weights and bias, so inference is conv + ReLU, pool, average, linear.
struct FoldedConv {
  int64_t in, out;
  std::vector<float> weight;  // [out, in, 3, 3]
  std::vector<float> bias;    // [out]
  bool pool_after;
};
struct Folded {
  std::vector<FoldedConv> convs;
  std::vector<float> fc_weight;  // [10, last]
  std::vector<float> fc_bias;    // [10]
};
Folded fold(Net& net);
// Runs the folded network in plain C++ loops (the same arithmetic as the browser).
std::vector<float> folded_logits(const Folded& f, const float* image_chw);

// ---------- Metrics ----------

struct Scores {
  double accuracy, low, high;  // with a 95% Wilson interval
  std::array<std::array<int64_t, 10>, 10> confusion{};  // [true][predicted]
};
Scores score(const torch::Tensor& predictions, const torch::Tensor& labels);
std::pair<double, double> wilson(int64_t successes, int64_t n);

// Logits for every image, in batches, model in eval mode. With `flip`, averages the logits of
// each image and its mirror image (test-time augmentation).
torch::Tensor logits(Net& net, const Normalizer& norm, const torch::Tensor& images_u8, bool flip = false);

}  // namespace loupe
