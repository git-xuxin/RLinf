#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
# 修改这里：指向新的 snapshot python 脚本
export SRC_FILE="${EMBODIED_PATH}/collect_snapshot_data.py"

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1


if [ -z "$1" ]; then
    # 修改这里：默认使用 snapshot 配置文件
    CONFIG_NAME="collect_data_snapshot"
else
    CONFIG_NAME=$1
fi

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}