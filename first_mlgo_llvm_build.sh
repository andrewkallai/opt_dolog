#!/bin/bash

#BUILD SCRIPT FOR UNIVERSITY OF DELAWARE'S DARWIN
set -x
set -e

export IDK_DIR=$HOME/sw/fuchsia-idk
export SYSROOT_DIR=$HOME/sw/fuchsia-sysroot
export FUCHSIA_SRCDIR=$HOME/sw/fuchsia
set +u
source $FUCHSIA_SRCDIR/scripts/fx-env.sh && fx-update-path
set -u

source $HOME/opt_dolog/tensorflow_setup_env/bin/activate
PREFIX=$HOME/sw
GCC_INSTALL="${HOME}/sw/gcc/gcc-install-15.2.0-20251116"
BINUTILS_PATH=${HOME}/sw/Binutils/binutils-2.45_install
LLVM_FIRST_INSTALLDIR=$HOME/LLVM_installs/llvm-clang-lld-libcxx-install
export PATH="$PREFIX/CCache/ccache-4.12.1-linux-x86_64:$PREFIX/Ninja/bin:$PREFIX/CMake/cmake-4.1.2-linux-x86_64/bin:$PATH"
export PATH="${BINUTILS_PATH}/bin:${GCC_INSTALL}/bin:$LLVM_FIRST_INSTALLDIR/bin:$PATH"

#module load llvm/18.1.8 
#module load gcc/13.2
export PATH="$LLVM_FIRST_INSTALLDIR/bin:$PATH"
set +u
#export LD_LIBRARY_PATH="/packages/gcc/13.2.0/lib64:$LLVM_FIRST_INSTALLDIR/lib/x86_64-unknown-linux-gnu:$LLVM_FIRST_INSTALLDIR/lib:$LD_LIBRARY_PATH"
#export LIBRARY_PATH="/packages/gcc/13.2.0/lib64:$LLVM_DIRST_INSTALLDIR/lib/x86_64-unknown-linux-gnu:$LLVM_FIRST_INSTALLDIR/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="${BINUTILS_PATH}/lib:${GCC_INSTALL}/lib/gcc/x86_64-pc-linux-gnu/15.2.0:${GCC_INSTALL}/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$LLVM_FIRST_INSTALLDIR/lib/x86_64-unknown-linux-gnu:$LLVM_FIRST_INSTALLDIR/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/sw/Zlib/zlib-1.3.1_install/lib:$HOME/sw/Zstd/zstd-1.5.7/lib:$HOME/sw/LibXML/libxml2-2.9.14_install/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="${GCC_INSTALL}/lib/gcc/x86_64-pc-linux-gnu/15.2.0:${GCC_INSTALL}/lib64:$LIBRARY_PATH"
export LIBRARY_PATH="$LLVM_FIRST_INSTALLDIR/lib/x86_64-unknown-linux-gnu:$LLVM_FIRST_INSTALLDIR/lib:$LIBRARY_PATH"
export LIBRARY_PATH="$HOME/sw/Zlib/zlib-1.3.1_install/lib:$HOME/sw/Zstd/zstd-1.5.7/lib:$HOME/sw/LibXML/libxml2-2.9.14_install/lib:$LIBRARY_PATH"
set -u
export PATH="$HOME/sw/Zlib/zlib-1.3.1_install/lib:$HOME/sw/Zstd/zstd-1.5.7/lib:$HOME/sw/LibXML/libxml2-2.9.14_install/lib:$PATH"

export LLVM_SRCDIR=$HOME/LLVM_clones/llvm-project
export LLVM_INSTALLDIR=$HOME/LLVM_installs/llvm-mlgo-install
BUILD=$HOME/LLVM_builds/llvm_for_mlgo_build
export TFLITE_PATH=~/tflite

export PATH=$FUCHSIA_SRCDIR/.jiri_root/bin:$PATH
#export CMAKE_INCLUDE_PATH="${LLVM_FIRST_INSTALLDIR}/include/c++/v1"

#export CLANG_TOOLCHAIN_PREFIX=${FUCHSIA_DIR}/prebuilt/third_party/clang/linux-x64/bin/

#LLVM_PROJECTS="clang;lld"
#LLVM_RUNTIMES="compiler-rt"
#TARGETS="X86"
#export CXXFLAGS="-L/home/users/andrewka/LLVM_installs/llvm-clang-lld-libcxx-install/lib/x86_64-unknown-linux-gnu -lc++abi.a"
#export CXXFLAGS="-L/home/users/andrewka/LLVM_installs/llvm-clang-lld-libcxx-install/lib/x86_64-unknown-linux-gnu -lc++abi"
#export CXXFLAGS="-lc++abi"

#export LDFLAGS="-stdlib=libc++"



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

  #-D CMAKE_TOOLCHAIN_FILE=${FUCHSIA_DIR}/scripts/clang/ToolChain.cmake \
  #-D LINUX_x86_64-unknown-linux-gnu_SYSROOT=${SYSROOT_DIR}/linux-x64/lib/x86_64-linux-gnu \
  #-D LINUX_x86_64-unknown-linux-gnu_SYSROOT=${SYSROOT_DIR}/linux-x64/usr/include/x86_64-linux-gnu \
  #-D BUILD_SHARED_LIBS=OFF \
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
cmake --build . --target toolchain-distribution



#ninja toolchain-distribution
DESTDIR=${LLVM_INSTALLDIR} ninja install-toolchain-distribution-stripped
cd ${FUCHSIA_SRCDIR}
python scripts/clang/generate_runtimes.py \
  --clang-prefix=$LLVM_INSTALLDIR \
  --sdk-dir=$IDK_DIR \
  --build-id-dir=$LLVM_INSTALLDIR/lib/.build-id > $LLVM_INSTALLDIR/lib/runtime.json

cd $BUILD
./bin/llvm-lit -v test

cd ${FUCHSIA_SRCDIR}
echo $LD_LIBRARY_PATH
fx set core.x64 \
  --args='clang_prefix="/home/users/andrewka/LLVM_installs/llvm-mlgo-install/bin"' \
  --args=clang_embed_bitcode=true \
  --args='optimize="size"' \
  --args='clang_ml_inliner=false'
fx build
