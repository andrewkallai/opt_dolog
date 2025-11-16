#!/bin/bash

set -e
set -u

module load gcc/13.2

GCC_VERSION="15.2.0"
BUILD_ID="${GCC_VERSION}-20251116"
#CUDA_VERSION="12.2"
JOBS=48
#BUILDTIME="$(\date --iso-8601=min)"

PREFIX="${HOME}/sw/gcc/gcc-install-${BUILD_ID}"
WORKSPACE_PATH="${HOME}/sw/gcc/gcc-build-${BUILD_ID}"
SRC_PATH="${WORKSPACE_PATH}/sources"
BUILD_PATH="${WORKSPACE_PATH}/build"

PKG_NAME="gcc-${GCC_VERSION}"
SRC_NAME="gcc-${GCC_VERSION}.tar.xz"
BINUTILS_PATH=${HOME}/sw/Binutils/binutils-2.45
export PATH="${BINUTILS_PATH}_install/bin:$PATH"
export LD_LIBRARY_PATH="${BINUTILS_PATH}_install/lib:$LD_LIBRARY_PATH"

mkdir -p ${BINUTILS_PATH}
cd ${BINUTILS_PATH}/..
wget https://ftp.gnu.org/gnu/binutils/binutils-2.45.tar.gz
tar --extract --file=binutils-2.45.tar.gz
cd binutils-2.45
./configure --prefix=${BINUTILS_PATH}_install
make -j ${JOBS}
make install

#CUDA_ROOT="/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/${CUDA_VERSION}"

#LOG_PATH="/global/cfs/cdirs/gcc-build/logs"


# mkdir -p "${LOG_PATH}"

# # Function to log output and errors to separate files
# log_step() {
#     local step_name="$1"
#     local log_file="${LOG_PATH}/build.${step_name}.${BUILDTIME}.log"
#     shift
#     {
#         printf "\n\n=====================\n%s\n\n" "$step_name"
#         "$@"
#     } 2>&1 | tee "$log_file"
# }

# # Remove old build directories
# [ -d "${WORKSPACE_PATH}" ] &&  rm -rf "${WORKSPACE_PATH}"

#module -t list

mkdir -p "${SRC_PATH}"
mkdir -p "${BUILD_PATH}"
mkdir -p "${PREFIX}"
cd "${SRC_PATH}"

# # Get NVPTX tools
# if [ ! -d nvptx-tools ]; then
#     log_step "Cloning NVPTX tools" git clone https://github.com/SourceryTools/nvptx-tools.git nvptx-tools
# fi

# # Get nvtpx-newlib
# log_step "Cloning nvptx-newlib" git clone  https://sourceware.org/git/newlib-cygwin.git nvptx-newlib

# Get GCC source
#log_step "Downloading GCC source" wget "ftp://ftp.gnu.org/gnu/gcc/gcc-${GCC_VERSION}/${SRC_NAME}"
wget "https://ftp.gnu.org/gnu/gcc/gcc-${GCC_VERSION}/${SRC_NAME}"

#mkdir -p "${BUILD_PATH}"
cd "${BUILD_PATH}"
#cp -r ${SRC_PATH}/nvptx-tools ./
#cp -r ${SRC_PATH}/nvptx-newlib ./
#log_step "Extracting GCC source" tar -xf "${SRC_PATH}/${SRC_NAME}"
tar -xf "${SRC_PATH}/${SRC_NAME}"


#HOST_TARGET=$(${PKG_NAME}/config.guess)
#OFFLOAD_TARGET="nvptx-none"

# Build nvptx tools
# cd "${BUILD_PATH}/nvptx-tools"
# log_step "Building NVPTX tools" "${BUILD_PATH}/nvptx-tools/configure" \
#     --with-cuda-driver-include="${CUDA_ROOT}/include" \
#     --with-cuda-driver-lib="${CUDA_ROOT}/lib64" \
#     --prefix="${PREFIX}" && \
#     make -j "${JOBS:-1}" && \
#     make install


cd "${BUILD_PATH}"
# Get pre-requisites
#if [[ ! -L "${PKG_NAME}/gmp"  || \
#      ! -L "${PKG_NAME}/mpc"  || \
#      ! -L "${PKG_NAME}/mpfr" ]]; then
cd "${PKG_NAME}"
./contrib/download_prerequisites
    #unlink newlib || true
    #ln -s "${BUILD_PATH}/nvptx-newlib/newlib" newlib
#fi

cd "${BUILD_PATH}"
#mkdir -pv build-nvptx-gcc
mkdir -p build-gcc
cd build-gcc

#log_step "Building GCC Offload Compiler" "${BUILD_PATH}/${PKG_NAME}/configure" \
#    --target="${OFFLOAD_TARGET}" \
#    --with-build-time-tools="${PREFIX}/nvptx-none/bin" \
#     --with-cuda-driver-include="${CUDA_ROOT}/include" \
#     --with-cuda-driver-lib="${CUDA_ROOT}/lib64" \
#     --with-cuda-driver="${CUDA_ROOT}/bin" \
#    --enable-languages="c,c++,fortran,lto" \
#    --enable-as-accelerator-for="${HOST_TARGET}" \
#    --with-newlib \
#     --disable-sjlj-exceptions \
#     --enable-newlib-io-long-long \

${BUILD_PATH}/${PKG_NAME}/configure \
    --enable-languages="c,c++,lto" \
    --disable-multilib \
    --prefix="${PREFIX}" && \
    make -j "${JOBS}" && \
    make install

cd "${BUILD_PATH}"

# mkdir -pv build-gcc
# cd build-gcc
# # Build host compiler
# log_step "Building GCC Host Compiler" "${BUILD_PATH}/${PKG_NAME}/configure" \
#     --prefix="${PREFIX}" \
#     --enable-offload-targets="${OFFLOAD_TARGET}" \
#     --with-cuda-driver-include="${CUDA_ROOT}/include" \
#     --with-cuda-driver-lib="${CUDA_ROOT}/lib64" \
#     --with-cuda-driver="${CUDA_ROOT}/bin" \
#     --enable-languages="c,c++,fortran,lto" \
#     --disable-multilib CFLAGS="-m64" && \
#     make -j "${JOBS:-1}" && \
#     make install

