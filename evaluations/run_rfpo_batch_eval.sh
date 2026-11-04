#!/usr/bin/env bash
# Edit these three settings, then run: bash evaluations/run_rfpo_batch_eval.sh
# CONFIG_YAML selects exactly one YAML; checkpoint entries must be full_weights.pt.
# Each repeat starts fresh rollout workers and samples fresh initial flow noise.
# This does not assign reproducible noise seeds or change environment/model options.
set -euo pipefail

# CONFIG_YAML="evaluations/libero/libero_object_rfpo_openpi_pi05_multi_eval.yaml"
CONFIG_YAML="evaluations/libero/libero_spatial_rfpo_openpi_pi05_multi_eval.yaml"
K=10
CKPT_PATHS=(
  "/mnt/public2/xuxin/RFPO/RLinf-RFPO-new/logs/20261031-04:38:22-libero_spatial_async_rfpo_td3_openpi_pi05_multi_dynamicl2/libero_spatial_async_rfpo_td3_openpi_pi05/checkpoints/global_step_700/actor/model_state_dict/full_weights.pt"
  "/mnt/public2/xuxin/RFPO/RLinf-RFPO-new/logs/20261031-04:38:22-libero_spatial_async_rfpo_td3_openpi_pi05_multi_dynamicl2/libero_spatial_async_rfpo_td3_openpi_pi05/checkpoints/global_step_800/actor/model_state_dict/full_weights.pt"
  "/mnt/public2/xuxin/RFPO/RLinf-RFPO-new/logs/20261031-04:38:22-libero_spatial_async_rfpo_td3_openpi_pi05_multi_dynamicl2/libero_spatial_async_rfpo_td3_openpi_pi05/checkpoints/global_step_900/actor/model_state_dict/full_weights.pt"
  "/mnt/public2/xuxin/RFPO/RLinf-RFPO-new/logs/20261031-04:38:22-libero_spatial_async_rfpo_td3_openpi_pi05_multi_dynamicl2/libero_spatial_async_rfpo_td3_openpi_pi05/checkpoints/global_step_1000/actor/model_state_dict/full_weights.pt"
  "/mnt/public2/xuxin/RFPO/RLinf-RFPO-new/logs/20261031-04:38:22-libero_spatial_async_rfpo_td3_openpi_pi05_multi_dynamicl2/libero_spatial_async_rfpo_td3_openpi_pi05/checkpoints/global_step_1100/actor/model_state_dict/full_weights.pt"
)

REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ ${#CKPT_PATHS[@]} -eq 0 ]; then
  echo "Fill CKPT_PATHS in this script before running." >&2
  exit 1
fi

# Each batch gets a new directory containing runs.csv, summary.csv and run logs.
# Rates in CSV are fractions in [0, 1]; terminal summaries display percentages.
# A failed run stops the batch; completed rows remain on disk.
export LIBERO_TYPE="${LIBERO_TYPE:-standard}"
exec python "${REPO_PATH}/evaluations/batch_eval.py" \
  --config "${CONFIG_YAML}" \
  --repeats "${K}" \
  --checkpoints "${CKPT_PATHS[@]}"
