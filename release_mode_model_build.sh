#!/bin/bash

set -x
set -e
set -u

PREFIX=/storage/users/andrewka
source $PREFIX/opt_dolog/tensorflow_setup_env/bin/activate

export FUCHSIA_DIR=$PREFIX/sw/fuchsia
export IDK_DIR=$PREFIX/sw/fuchsia-idk
export SYSROOT_DIR=${FUCHSIA_DIR}/prebuilt/third_party/sysroot

export TFLITE_PATH=$PREFIX/tflite
export LLVM_SRCDIR=$PREFIX/LLVM_clones/llvm-project
LLVM_BUILD_DIR=$PREFIX/LLVM_builds/llvm-for-mlgo-release-build
INSTALL_DIR=$PREFIX/LLVM_installs/llvm-mlgo-install
export LLVM_INSTALLDIR_RELEASE=${INSTALL_DIR}-release

mkdir -p ${LLVM_BUILD_DIR}
mkdir -p ${LLVM_INSTALLDIR_RELEASE}

platform=linux-x64

export PATH=${FUCHSIA_DIR}/prebuilt/third_party/cmake/${platform}/bin:${PATH}
export PATH=${FUCHSIA_DIR}/prebuilt/third_party/ninja/${platform}/bin:${PATH}
CLANG_TOOLCHAIN_PREFIX=${FUCHSIA_DIR}/prebuilt/third_party/clang/linux-x64/bin/
export OUTPUT_DIR=$PREFIX/model

TF_PIP=$(python3 -m pip show tensorflow | grep Location | cut -d ' ' -f 2)
export TENSORFLOW_AOT_PATH="${TF_PIP}/tensorflow"


set +u
export PKG_CONFIG_PATH="$PREFIX/sw/ZLib/zlib-1.3.1_install/lib/pkgconfig:$PREFIX/sw/Zstd/zstd-1.5.7_install/lib/pkgconfig:$PREFIX/sw/LibXML/libxml2-2.9.14_install/lib/pkgconfig:$PKG_CONFIG_PATH"
set -u

cd $LLVM_SRCDIR
# mkdir -p llvm/lib/Analysis/models/inliner
# rm -rf llvm/lib/Analysis/models/inliner/*
# cp -rf $OUTPUT_DIR/saved_policy/* llvm/lib/Analysis/models/inliner/

# mkdir -p $LLVM_BUILD_DIR
# cd $LLVM_BUILD_DIR
# cmake -G Ninja \
#   -D CMAKE_FIND_ROOT_PATH="/" \
#   -D LLVM_ENABLE_LTO=OFF \
#   -D CMAKE_TOOLCHAIN_FILE=${FUCHSIA_DIR}/scripts/clang/ToolChain.cmake \
#   -D LINUX_x86_64-unknown-linux-gnu_SYSROOT=${SYSROOT_DIR}/linux \
#   -D LINUX_aarch64-unknown-linux-gnu_SYSROOT=${SYSROOT_DIR}/linux \
#   -D LINUX_riscv64-unknown-linux-gnu_SYSROOT=${SYSROOT_DIR}/ubuntu20.04 \
#   -D LINUX_armv7-unknown-linux-gnueabihf_SYSROOT=${SYSROOT_DIR}/linux \
#   -D LINUX_i386-unknown-linux-gnu_SYSROOT=${SYSROOT_DIR}/linux \
#   -D FUCHSIA_SDK=${IDK_DIR} \
#   -D CMAKE_INSTALL_PREFIX=${LLVM_INSTALLDIR_RELEASE} \
#   -D TENSORFLOW_AOT_PATH=${TENSORFLOW_AOT_PATH} \
#   -C ${LLVM_SRCDIR}/clang/cmake/caches/Fuchsia-stage2.cmake \
#   ${LLVM_SRCDIR}/llvm

# cmake --build . --target toolchain-distribution
# DESTDIR=${LLVM_INSTALLDIR_RELEASE} ninja install-toolchain-distribution-stripped

# cd ${FUCHSIA_DIR}
# python3 scripts/clang/generate_runtimes.py \
#   --clang-prefix=$LLVM_INSTALLDIR_RELEASE \
#   --sdk-dir=$IDK_DIR \
#   --build-id-dir=$LLVM_INSTALLDIR_RELEASE/lib/.build-id > $LLVM_INSTALLDIR_RELEASE/lib/runtime.json

set +u
source ${FUCHSIA_DIR}/scripts/fx-env.sh && fx-update-path
set -u

cd ${FUCHSIA_DIR}
# fx set core.x64 \
#   --args='clang_prefix="/storage/users/andrewka/LLVM_installs/llvm-mlgo-install-release/bin"' \
#   --args='optimize="size"' \
#   --args=clang_ml_inliner=true
# fx build


python3 -m pip install tabulate
python3 scripts/compare_elf_sizes.py \
  out/core.x64-balanced/obj/build/images/sizes/elf_sizes.json \
  /tmp/elf_sizes_release.json \
  --field code

#  /tmp/orig_sizes.json \

