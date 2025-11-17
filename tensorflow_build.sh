set -e
set -u


PREFIX=$HOME/sw
GCC_INSTALL="${HOME}/sw/gcc/gcc-install-15.2.0-20251116"
BINUTILS_PATH=${HOME}/sw/Binutils/binutils-2.45_install
export LLVM_FIRST_INSTALLDIR=~/LLVM_installs/llvm-clang-lld-libcxx-install
export PATH="$PREFIX/CCache/ccache-4.12.1-linux-x86_64:$PREFIX/Ninja/bin:$PREFIX/CMake/cmake-4.1.2-linux-x86_64/bin:$PATH"
export PATH="${BINUTILS_PATH}/bin:${GCC_INSTALL}/bin:$LLVM_FIRST_INSTALLDIR/bin:$PATH"
set +u
export LD_LIBRARY_PATH="${BINUTILS_PATH}/lib:${GCC_INSTALL}/lib/gcc/x86_64-pc-linux-gnu/15.2.0:${GCC_INSTALL}/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$LLVM_FIRST_INSTALLDIR/lib/x86_64-unknown-linux-gnu:$LLVM_FIRST_INSTALLDIR/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="${GCC_INSTALL}/lib/gcc/x86_64-pc-linux-gnu/15.2.0:${GCC_INSTALL}/lib64:$LIBRARY_PATH"
export LIBRARY_PATH="$LLVM_DIRST_INSTALLDIR/lib/x86_64-unknown-linux-gnu:$LLVM_FIRST_INSTALLDIR/lib:$LIBRARY_PATH"
set -u
export CC=clang
export CXX=clang++
export CXXFLAGS="-stdlib=libc++" #-lc++abi"
export LDFLAGS="-stdlib=libc++"

cd $HOME/ml-compiler-opt

source $HOME/opt_dolog/tensorflow_setup_env/bin/activate
pip install --upgrade pip
pip install pipenv
pipenv sync --system

TF_PIP=$(python -m pip show tensorflow | grep Location | cut -d ' ' -f 2)

export TENSORFLOW_AOT_PATH="${TF_PIP}/tensorflow"
export TFLITE_PATH=~/tflite

mkdir -p ${TFLITE_PATH}
cd ${TFLITE_PATH}

~/ml-compiler-opt/buildbot/build_tflite.sh
