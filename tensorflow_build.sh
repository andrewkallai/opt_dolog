set -e
set -u


PREFIX=$HOME/sw
export LLVM_FIRST_INSTALLDIR=~/LLVM_installs/llvm-clang-lld-install
export PATH="$PREFIX/CCache/ccache-4.12.1-linux-x86_64:$PREFIX/Ninja/bin:$PREFIX/CMake/cmake-4.1.2-linux-x86_64/bin:$PATH"
export PATH="$LLVM_FIRST_INSTALLDIR/bin:$PATH"
set +u
export LD_LIBRARY_PATH="$LLVM_FIRST_INSTALLDIR/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$LLVM_FIRST_INSTALLDIR/lib:$LIBRARY_PATH"
set -u

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
