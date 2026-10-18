#!/usr/bin/env sh
# Downloads everything that isn't committed: LibTorch (CPU), CIFAR-10 (binary version),
# cpp-httplib and stb_image (single headers for the server).
set -e
cd "$(dirname "$0")/.."
mkdir -p third_party data

if [ ! -d third_party/libtorch ]; then
  curl -fL -o third_party/libtorch.zip "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.12.1%2Bcpu.zip"
  (cd third_party && unzip -q libtorch.zip && rm libtorch.zip)
fi
curl -fsSL -o third_party/httplib.h https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.18.3/httplib.h
curl -fsSL -o third_party/stb_image.h https://raw.githubusercontent.com/nothings/stb/f58f558c120e9b32c217290b80bad1a0729fbb2c/stb_image.h

if [ ! -d data/cifar-10-batches-bin ]; then
  curl -fL -o data/cifar.tgz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
  tar xzf data/cifar.tgz -C data && rm data/cifar.tgz
fi
echo "Ready: third_party/libtorch, third_party/*.h, data/cifar-10-batches-bin"
