#!/bin/bash

set -x
set -e
set -u

PREFIX=/storage/users/andrewka
#source $PREFIX/opt_dolog/tensorflow_setup_env/bin/activate

export FUCHSIA_DIR=$PREFIX/sw/fuchsia
export IDK_DIR=$PREFIX/sw/fuchsia-idk
export LLVM_SRCDIR=$PREFIX/LLVM_clones/llvm-project
LLVM_BUILD_DIR=$PREFIX/LLVM_builds/llvm-for-fuchsia-build
INSTALL_DIR=$PREFIX/LLVM_installs/llvm-fuchsia-install
mkdir -p ${LLVM_BUILD_DIR}
mkdir -p ${INSTALL_DIR}

platform=linux-x64

export PATH=${FUCHSIA_DIR}/prebuilt/third_party/cmake/${platform}/bin:${PATH}
export PATH=${FUCHSIA_DIR}/prebuilt/third_party/ninja/${platform}/bin:${PATH}
#export CMAKE_LIBRARY_PATH="$PREFIX/sw/Zlib/zlib-1.3.1_install/lib:$PREFIX/sw/Zstd/zstd-1.5.7/lib:$PREFIX/sw/LibXML/libxml2-2.9.14_install/lib"
set +u
export PKG_CONFIG_PATH="$PREFIX/sw/ZLib/zlib-1.3.1_install/lib/pkgconfig:$PREFIX/sw/Zstd/zstd-1.5.7_install/lib/pkgconfig:$PREFIX/sw/LibXML/libxml2-2.9.14_install/lib/pkgconfig:$PKG_CONFIG_PATH"
set -u
cd ${LLVM_BUILD_DIR}  # The directory your toolchain will be installed in

# Environment setup
SYSROOT_DIR=${FUCHSIA_DIR}/prebuilt/third_party/sysroot
CLANG_TOOLCHAIN_PREFIX=${FUCHSIA_DIR}/prebuilt/third_party/clang/linux-x64/bin/

# Download necessary dependencies
#cipd install fuchsia/sdk/core/linux-amd64 latest -root ${IDK_DIR}


# CMake invocation
#cmake -DCMAKE_FIND_DEBUG_MODE=ON -G Ninja -D CMAKE_BUILD_TYPE=Release \
#cmake --debug-output -G Ninja -D CMAKE_BUILD_TYPE=Release \
#cmake --trace -G Ninja -D CMAKE_BUILD_TYPE=Release \
# cmake -G Ninja -D CMAKE_BUILD_TYPE=Release \
#   -D CMAKE_FIND_ROOT_PATH="/" \
#   -D CMAKE_TOOLCHAIN_FILE=${FUCHSIA_DIR}/scripts/clang/ToolChain.cmake \
#   -D LLVM_ENABLE_LTO=OFF \
#   -D LINUX_x86_64-unknown-linux-gnu_SYSROOT=${SYSROOT_DIR}/linux \
#   -D FUCHSIA_SDK=${IDK_DIR} \
#   -D CMAKE_INSTALL_PREFIX=${SYSROOT_DIR}/linux/../../../../../../LLVM_installs/llvm-fuchsia-install \
#   -D CMAKE_STAGING_PREFIX=${INSTALL_DIR} \
#   -C ${LLVM_SRCDIR}/clang/cmake/caches/Fuchsia-stage2.cmake \
#   ${LLVM_SRCDIR}/llvm

# #   -DLINUX_aarch64-unknown-linux-gnu_SYSROOT=${SYSROOT_DIR}/linux \
#   -DLINUX_riscv64-unknown-linux-gnu_SYSROOT=${SYSROOT_DIR}/ubuntu20.04 \
#   -DLINUX_armv7-unknown-linux-gnueabihf_SYSROOT=${SYSROOT_DIR}/linux \
#   -DLINUX_i386-unknown-linux-gnu_SYSROOT=${SYSROOT_DIR}/linux \
# Build and strip binaries and place them in the install directory
# ninja toolchain-distribution
# DESTDIR=${INSTALL_DIR} ninja install-toolchain-distribution-stripped

# Generate runtime.json

# python3 ${FUCHSIA_DIR}/scripts/clang/generate_runtimes.py    \
#   --clang-prefix ${INSTALL_DIR} --sdk-dir ${IDK_DIR}            \
#   --build-id-dir ${INSTALL_DIR}/lib/.build-id > ${INSTALL_DIR}/lib/runtime.json

set +u
#source $FUCHSIA_DIR/scripts/fx-env.sh && fx-update-path
source /storage/users/andrewka/sw/fuchsia/scripts/fx-env.sh && fx-update-path
set -u

#export PATH=$FUCHSIA_DIR/.jiri_root/bin:$PATH

cd ${FUCHSIA_DIR}

fx set core.x64 \
  --args='clang_prefix="/home/users/andrewka/LLVM_installs/llvm-fuchsia-install/bin"' \
  --args=clang_embed_bitcode=true \
  --args='optimize="size"' \
  --args='clang_ml_inliner=false'
fx build
