#!/usr/bin/env bash
# Shared output layout helpers for experiment launch scripts.

output_slug() {
  local value="$1"
  value="${value//,/_}"
  value="${value// /_}"
  value="${value//\//_}"
  value="${value//:/_}"
  value="${value//./}"
  value="${value//--/_}"
  value="${value##_}"
  value="${value%%_}"
  echo "${value}"
}

output_layout_path() {
  local project_root="$1"
  local category="$2"
  local arch="$3"
  local config="$4"
  local experiment="$5"
  printf '%s/data/outputs/%s/%s/%s/%s\n' \
    "${project_root}" \
    "$(output_slug "${category}")" \
    "$(output_slug "${arch}")" \
    "$(output_slug "${config}")" \
    "$(output_slug "${experiment}")"
}

output_config_tag() {
  local config_name="$1"
  local run_tag="${2:-}"
  if [[ -n "${run_tag}" ]]; then
    output_slug "${run_tag}"
  else
    output_slug "${config_name}"
  fi
}

dreamervla_output_arch() {
  local config_name="$1"
  case "${config_name}" in
    dreamervla_*wmpo_outcome*) echo "wmpo_outcome" ;;
    dreamervla_*rynn_dino_wm_actor_critic*) echo "actor_critic_ppo" ;;
    dreamervla_*pi0_action_hidden*) echo "pi0_action_hidden_actor" ;;
    *tdmpc_ac*) echo "actor_critic_tdmpc_ac" ;;
    *wmpo_outcome*) echo "wmpo_outcome" ;;
    *dense_chunk*) echo "actor_critic_ppo" ;;
    *vlaactor*) echo "vla_actor" ;;
    *rynn_dino_wm_actor_critic*) echo "actor_critic_ppo" ;;
    *pi0*action_hidden*actor*) echo "pi0_action_hidden_actor" ;;
    *) echo "actor_critic" ;;
  esac
}

worldmodel_output_arch() {
  local kind="$1"
  local config_name="$2"
  case "${kind}" in
    rynn_dino)
      if [[ "${config_name}" == oft_dino* ]]; then
        echo "oft_dino_wm"
      elif [[ "${config_name}" == world_model_dinowm_chunk* ]]; then
        echo "rynn_dino_wm_action_hidden_chunk"
      elif [[ "${config_name}" == *fullhidden* ]]; then
        echo "rynn_dino_wm_fullhidden"
      else
        echo "rynn_dino_wm_action_hidden"
      fi
      ;;
    action_hidden) echo "rynn_backbone_dreamerv3_wm" ;;
    dreamerv3_token) echo "dreamerv3_token" ;;
    dreamerv3_pixel) echo "dreamerv3_pixel" ;;
    chameleon) echo "chameleon_latent_action_wm" ;;
    *) output_slug "${kind}" ;;
  esac
}
