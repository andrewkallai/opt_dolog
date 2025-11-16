#!/bin/bash

#BUILD SCRIPT FOR UNIVERSITY OF DELAWARE'S DARWIN
#set -x
#set -e

source $HOME/opt_dolog/tensorflow_setup_env/bin/activate
PREFIX=$HOME/sw
export PATH="$PREFIX/CCache/ccache-4.12.1-linux-x86_64:$PREFIX/Ninja/bin:$PREFIX/CMake/cmake-4.1.2-linux-x86_64/bin:$PATH"

#module load llvm/18.1.8 
export LLVM_FIRST_INSTALLDIR=~/LLVM_installs/llvm-clang-lld-libcxx-install
export PATH="$LLVM_FIRST_INSTALLDIR/bin:$PATH"
set +u
export LD_LIBRARY_PATH="$LLVM_FIRST_INSTALLDIR/lib/x86_64-unknown-linux-gnu:$LLVM_FIRST_INSTALLDIR/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$LLVM_DIRST_INSTALLDIR/lib/x86_64-unknown-linux-gnu:$LLVM_FIRST_INSTALLDIR/lib:$LIBRARY_PATH"
set -u


clang++ -stdlib=libc++ -fuse-ld=lld test.cpp -lc++abi -lunwind -pthread -o a.out

