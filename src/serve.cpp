// A small HTTP server around the trained model.
//
//   serve [--model models/final.pt] [--port 8080]
//
//   POST /classify   body: a PNG or JPEG image   ->  {"top": [{"label": "cat", "p": 0.71}, ...], "ms": 3.1}
//   GET  /health     ->  {"ok": true}
//   GET  /           ->  docs/index.html (the browser demo), if it exists
//
// Images of any size are centre-cropped to a square and shrunk to 32x32 by averaging blocks of
// pixels, the size the network was trained on.

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include "stb_image.h"
#include "httplib.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>

#include "loupe.hpp"

using namespace loupe;

namespace {

// Centre square crop, then each output pixel is the mean of the source pixels it covers.
torch::Tensor to_32x32(const unsigned char* rgb, int w, int h) {
  const int side = std::min(w, h), x0 = (w - side) / 2, y0 = (h - side) / 2;
  auto out = torch::empty({1, 3, 32, 32}, torch::kUInt8);
  auto o = out.accessor<uint8_t, 4>();
  for (int r = 0; r < 32; ++r)
    for (int c = 0; c < 32; ++c) {
      const int ya = y0 + r * side / 32, yb = std::max(ya + 1, y0 + (r + 1) * side / 32);
      const int xa = x0 + c * side / 32, xb = std::max(xa + 1, x0 + (c + 1) * side / 32);
      double sum[3] = {0, 0, 0};
      for (int y = ya; y < yb; ++y)
        for (int x = xa; x < xb; ++x)
          for (int ch = 0; ch < 3; ++ch) sum[ch] += rgb[(y * w + x) * 3 + ch];
      const double n = double(yb - ya) * (xb - xa);
      for (int ch = 0; ch < 3; ++ch) o[0][ch][r][c] = static_cast<uint8_t>(sum[ch] / n + 0.5);
    }
  return out;
}

Normalizer read_normalizer(const std::string& path) {
  std::ifstream in(path);
  Normalizer n;
  if (!(in >> n.mean[0] >> n.mean[1] >> n.mean[2] >> n.std[0] >> n.std[1] >> n.std[2]))
    throw std::runtime_error("cannot read " + path + " (run: experiment final <name>)");
  return n;
}

std::string arg(int argc, char** argv, const std::string& name, const std::string& fallback) {
  for (int i = 1; i + 1 < argc; ++i)
    if (argv[i] == name) return argv[i + 1];
  return fallback;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string model_path = arg(argc, argv, "--model", "models/final.pt");
  const int port = std::stoi(arg(argc, argv, "--port", "8080"));
  torch::set_num_threads(1);

  Net net;
  Normalizer norm;
  bool flip = false;
  try {
    torch::load(net, model_path);
    norm = read_normalizer("models/normalize.txt");
    std::ifstream f("models/flip.txt");
    f >> flip;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  net->eval();
  std::mutex model_mutex;

  httplib::Server server;
  server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    res.set_content("{\"ok\": true}", "application/json");
  });
  server.Get("/", [](const httplib::Request&, httplib::Response& res) {
    std::ifstream page("docs/index.html", std::ios::binary);
    if (!page) {
      res.status = 404;
      res.set_content("docs/index.html not built", "text/plain");
      return;
    }
    std::stringstream body;
    body << page.rdbuf();
    res.set_content(body.str(), "text/html; charset=utf-8");
  });
  server.Post("/classify", [&](const httplib::Request& req, httplib::Response& res) {
    const auto start = std::chrono::steady_clock::now();
    if (req.body.size() > 20'000'000) {
      res.status = 413;
      res.set_content("{\"error\": \"image larger than 20 MB\"}", "application/json");
      return;
    }
    int w = 0, h = 0, channels = 0;
    unsigned char* rgb = stbi_load_from_memory(reinterpret_cast<const unsigned char*>(req.body.data()),
                                               static_cast<int>(req.body.size()), &w, &h, &channels, 3);
    if (!rgb) {
      res.status = 400;
      res.set_content("{\"error\": \"send a PNG or JPEG image as the request body\"}", "application/json");
      return;
    }
    auto img = to_32x32(rgb, w, h);
    stbi_image_free(rgb);
    torch::Tensor p;
    {
      std::lock_guard<std::mutex> lock(model_mutex);
      p = torch::softmax(logits(net, norm, img, flip), 1)[0];
    }
    auto [vals, idx] = p.topk(5);
    std::ostringstream json;
    json << "{\"top\": [";
    for (int k = 0; k < 5; ++k)
      json << (k ? ", " : "") << "{\"label\": \"" << kClasses[idx[k].item<int64_t>()]
           << "\", \"p\": " << vals[k].item<double>() << "}";
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    json << "], \"ms\": " << ms << "}";
    res.set_content(json.str(), "application/json");
  });

  std::printf("loupe: serving %s on http://localhost:%d (POST /classify)\n", model_path.c_str(), port);
  std::fflush(stdout);
  server.listen("0.0.0.0", port);
  return 0;
}
