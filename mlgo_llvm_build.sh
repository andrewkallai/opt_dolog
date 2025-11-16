#!/bin/bash

#BUILD SCRIPT FOR UNIVERSITY OF DELAWARE'S DARWIN
set -x
set -e

source $HOME/opt_dolog/tensorflow_setup_env/bin/activate
PREFIX=$HOME/sw
export PATH="$PREFIX/CCache/ccache-4.12.1-linux-x86_64:$PREFIX/Ninja/bin:$PREFIX/CMake/cmake-4.1.2-linux-x86_64/bin:$PATH"

#module load llvm/18.1.8 
#module load gcc/13.2
export LLVM_FIRST_INSTALLDIR=~/LLVM_installs/llvm-clang-lld-libcxx-install
export PATH="$LLVM_FIRST_INSTALLDIR/bin:$PATH"
set +u
#export LD_LIBRARY_PATH="/packages/gcc/13.2.0/lib64:$LLVM_FIRST_INSTALLDIR/lib/x86_64-unknown-linux-gnu:$LLVM_FIRST_INSTALLDIR/lib:$LD_LIBRARY_PATH"
#export LIBRARY_PATH="/packages/gcc/13.2.0/lib64:$LLVM_DIRST_INSTALLDIR/lib/x86_64-unknown-linux-gnu:$LLVM_FIRST_INSTALLDIR/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$LLVM_FIRST_INSTALLDIR/lib/x86_64-unknown-linux-gnu:$LLVM_FIRST_INSTALLDIR/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$LLVM_DIRST_INSTALLDIR/lib/x86_64-unknown-linux-gnu:$LLVM_FIRST_INSTALLDIR/lib:$LIBRARY_PATH"
set -u

export LLVM_SRCDIR=$HOME/LLVM_clones/llvm-project
export LLVM_INSTALLDIR=$HOME/LLVM_installs/llvm-mlgo-install
BUILD=$HOME/LLVM_builds/llvm_for_mlgo_build
export IDK_DIR=~/sw/fuchsia-idk
export SYSROOT_DIR=~/sw/fuchsia-sysroot
export FUCHSIA_SRCDIR=~/sw/fuchsia
export TFLITE_PATH=~/tflite

export PATH=$FUCHSIA_SRCDIR/.jiri_root/bin:$PATH
set +u
source $FUCHSIA_SRCDIR/scripts/fx-env.sh
set -u

#LLVM_PROJECTS="clang;lld"
#LLVM_RUNTIMES="compiler-rt"
#TARGETS="X86"

mkdir -p $LLVM_INSTALLDIR
mkdir -p $BUILD
cd $BUILD

cmake -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=${LLVM_INSTALLDIR} \
  -D CMAKE_C_COMPILER=clang \
  -D CMAKE_CXX_COMPILER=clang++ \
  -D CMAKE_INSTALL_RPATH_USE_LINK_PATH=On \
  -D LLVM_ENABLE_LTO=OFF \
  -D LINUX_x86_64-unknown-linux-gnu_SYSROOT=${SYSROOT_DIR}/linux-x64 \
  -D FUCHSIA_SDK=${IDK_DIR} \
  -C ${LLVM_SRCDIR}/clang/cmake/caches/Fuchsia-stage2.cmake \
  -C ${TFLITE_PATH}/tflite.cmake \
  -G Ninja ${LLVM_SRCDIR}/llvm

  # -D LLVM_ENABLE_LLD=ON \
  # -D LLVM_ENABLE_PROJECTS=${LLVM_PROJECTS} \
  # -D LLVM_ENABLE_ASSERTIONS=ON \
  # -D LLVM_OPTIMIZED_TABLEGEN=ON \
  # -D LLVM_CCACHE_BUILD=OFF \
  # -D BUILD_SHARED_LIBS=ON \
  # -D LLVM_ENABLE_RUNTIMES=${LLVM_RUNTIMES} \
  # -D LLVM_TARGETS_TO_BUILD=${TARGETS} \
  # -D LLVM_RUNTIME_TARGETS="default;" \
  # -D LLVM_APPEND_VC_REV=ON \

#cmake --build . --target install


ninja toolchain-distribution
DESTDIR=${LLVM_INSTALLDIR} ninja install-toolchain-distribution-stripped
cd ${FUCHSIA_SRCDIR}
python scripts/clang/generate_runtimes.py \
  --clang-prefix=$LLVM_INSTALLDIR \
  --sdk-dir=$IDK_DIR \
  --build-id-dir=$LLVM_INSTALLDIR/lib/.build-id > $LLVM_INSTALLDIR/lib/runtime.json

cd $BUILD
./bin/llvm-lit -v test

cd ${FUCHSIA_SRCDIR}
fx set core.x64 \
  --args='clang_prefix="/home/users/andrewka/LLVM_installs/llvm-mlgo-install/bin"' \
  --args=clang_embed_bitcode=true \
  --args='optimize="size"' \
  --args='clang_ml_inliner=false'
fx build
