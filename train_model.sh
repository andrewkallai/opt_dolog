#!/bin/bash

set -x
set -e
set -u

PREFIX=/storage/users/andrewka
source $PREFIX/opt_dolog/tensorflow_setup_env/bin/activate

export FUCHSIA_DIR=$PREFIX/sw/fuchsia
export IDK_DIR=$PREFIX/sw/fuchsia-idk
export SYSROOT_DIR=${FUCHSIA_DIR}/prebuilt/third_party/sysroot

INSTALL_DIR=$PREFIX/LLVM_installs/llvm-mlgo-install


platform=linux-x64

export PATH=${FUCHSIA_DIR}/prebuilt/third_party/cmake/${platform}/bin:${PATH}
export PATH=${FUCHSIA_DIR}/prebuilt/third_party/ninja/${platform}/bin:${PATH}
CLANG_TOOLCHAIN_PREFIX=${FUCHSIA_DIR}/prebuilt/third_party/clang/linux-x64/bin/

set +u
export PKG_CONFIG_PATH="$PREFIX/sw/ZLib/zlib-1.3.1_install/lib/pkgconfig:$PREFIX/sw/Zstd/zstd-1.5.7_install/lib/pkgconfig:$PREFIX/sw/LibXML/libxml2-2.9.14_install/lib/pkgconfig:$PKG_CONFIG_PATH"
set -u


set +u
source ${FUCHSIA_DIR}/scripts/fx-env.sh && fx-update-path
set -u

cd ${FUCHSIA_DIR}
set +e
#fx compdb
set -e

#pip install mlgo-utils

export CORPUS=$PREFIX/corpus
cd ~/ml-compiler-opt
# extract_ir \
#   --cmd_filter="^-O2|-Os|-Oz$" \
#   --input=$FUCHSIA_DIR/out/default/compile_commands.json \
#   --input_type=json \
#   --llvm_objcopy_path=$INSTALL_DIR/bin/llvm-objcopy \
#   --output_dir=$CORPUS


export DEFAULT_TRACE=$PREFIX/default_trace
export DEFAULT_VOCAB=compiler_opt/rl/inlining/vocab

set +u

# rm -rf $DEFAULT_TRACE &&
#   PYTHONPATH=$PYTHONPATH:. python3 \
#     compiler_opt/tools/generate_default_trace.py \
#     --data_path=$CORPUS \
#     --output_path=$DEFAULT_TRACE \
#     --gin_files=compiler_opt/rl/inlining/gin_configs/common.gin \
#     --gin_bindings=config_registry.get_configuration.implementation=@configs.InliningConfig \
#     --gin_bindings=clang_path="'$INSTALL_DIR/bin/clang'" \
#     --gin_bindings=llvm_size_path="'$INSTALL_DIR/bin/llvm-size'" \
#     --sampling_rate=0.2

  
  # rm -rf $DEFAULT_VOCAB &&
  # PYTHONPATH=$PYTHONPATH:. python3 \
  #   compiler_opt/tools/generate_vocab.py \
  #   --gin_files=compiler_opt/rl/inlining/gin_configs/common.gin \
  #   --input=$DEFAULT_TRACE \
  #   --output_dir=$DEFAULT_VOCAB

export WARMSTART_OUTPUT_DIR=$PREFIX/warmstart
export OUTPUT_DIR=$PREFIX/model

# rm -rf $WARMSTART_OUTPUT_DIR && \
#   PYTHONPATH=$PYTHONPATH:. python3 \
#   compiler_opt/rl/train_bc.py \
#   --root_dir=$WARMSTART_OUTPUT_DIR \
#   --data_path=$DEFAULT_TRACE \
#   --gin_files=compiler_opt/rl/inlining/gin_configs/behavioral_cloning_nn_agent.gin


# rm -rf $OUTPUT_DIR && \
PYTHONPATH=$PYTHONPATH:. python3 \
  compiler_opt/rl/train_locally.py \
  --root_dir=$OUTPUT_DIR \
  --data_path=$CORPUS \
  --gin_bindings=clang_path="'$INSTALL_DIR/bin/clang'" \
  --gin_bindings=llvm_size_path="'$INSTALL_DIR/bin/llvm-size'" \
  --gin_files=compiler_opt/rl/inlining/gin_configs/rf_agent.gin #\
#  --gin_bindings=train_eval.warmstart_policy_dir=\"$WARMSTART_OUTPUT_DIR/saved_policy\"

#  --gin_files=compiler_opt/rl/inlining/gin_configs/ppo_nn_agent.gin #\
set -u

# monitor separately
#tensorboard --logdir=$OUTPUT_DIR
