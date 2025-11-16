#!/bin/bash
set -e
set -u

#export LLVM_SRCDIR=~/LLVM_clones/llvm-project
#export LLVM_FIRST_INSTALLDIR=~/LLVM_installs/llvm-clang-lld-install
#export LLVM_INSTALLDIR=~/LLVM_installs/llvm-mlgo-install

export IDK_DIR=~/sw/fuchsia-idk
export SYSROOT_DIR=~/sw/fuchsia-sysroot
export FUCHSIA_SRCDIR=~/sw/fuchsia

#export PATH="$LLVM_FIRST_INSTALLDIR/bin:$PATH"
#set +u
#export LD_LIBRARY_PATH="$LLVM_FIRST_INSTALLDIR/lib:$LD_LIBRARY_PATH"
#export LIBRARY_PATH="$LLVM_FIRST_INSTALLDIR/lib:$LIBRARY_PATH"
#set -u

cd $HOME/sw
mkdir -p $IDK_DIR
mkdir -p $SYSROOT_DIR
mkdir -p $FUCHSIA_SRCDIR

export PATH="$HOME/sw/Curl/curl-8.15.0_install/bin:$PATH"
 

curl -sO https://storage.googleapis.com/fuchsia-ffx/ffx-linux-x64 && chmod +x ffx-linux-x64 && ./ffx-linux-x64 platform preflight

curl -s "https://fuchsia.googlesource.com/fuchsia/+/HEAD/scripts/bootstrap?format=TEXT" | base64 --decode | bash

export PATH=$FUCHSIA_SRCDIR/.jiri_root/bin:$PATH
set +u
source $FUCHSIA_SRCDIR/scripts/fx-env.sh
set -u


cipd install fuchsia/sdk/core/linux-amd64 latest -root ${IDK_DIR}
cipd install fuchsia/sysroot/linux-amd64 latest -root ${SYSROOT_DIR}/linux-x64


cd ${FUCHSIA_SRCDIR}
prebuilt/third_party/clang/linux-x64/bin/clang --version
#jiri package 'fuchsia/third_party/clang/linux-x64'


