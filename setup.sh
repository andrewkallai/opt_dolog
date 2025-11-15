set -e
set -u

LLVM_DIR=$HOME/LLVM_clones
CMAKE_DIR=$HOME/sw/CMake
NINJA_DIR=$HOME/sw/Ninja/bin

mkdir -p $LLVM_DIR
cd $LLVM_DIR
if [ -d "llvm-project" ]; then
    cd llvm-project
    git pull
else
    git clone https://github.com/llvm/llvm-project.git
fi

cd $HOME
mkdir -p $CMAKE_DIR
cd $CMAKE_DIR
wget https://github.com/Kitware/CMake/releases/download/v4.1.2/cmake-4.1.2-linux-x86_64.tar.gz
tar --extract --file cmake-4.1.2-linux-x86_64.tar.gz

cd $HOME
mkdir -p $NINJA_DIR
cd $NINJA_DIR
wget https://github.com/ninja-build/ninja/releases/download/v1.13.1/ninja-linux.zip
unzip ninja-linux.zip

cd $HOME