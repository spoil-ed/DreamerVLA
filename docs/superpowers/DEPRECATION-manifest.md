# Deprecation Manifest

维护规则：

- 任何通过 `git mv` 进入 `archive/` 的文件，必须同步在本表新增一行。
- `scripts/restore_from_archive.sh` 以本表为数据源恢复归档文件。

| 原路径 | archive 路径 | 废弃理由 | 迁移 commit |
| --- | --- | --- | --- |
| configs/VLA/rynnvla_full_finetune.yaml | archive/configs/VLA/rynnvla_full_finetune.yaml | RynnVLA 次级训练配方，不在当前 OpenVLA-OFT cotrain 主线路径 | (staged, 未提交) |
| configs/experiment/vla_rynnvla_full_finetune.yaml | archive/configs/experiment/vla_rynnvla_full_finetune.yaml | RynnVLA 次级实验配方，不在当前 OpenVLA-OFT cotrain 主线路径 | (staged, 未提交) |
| configs/experiment/collect_rollouts_ray_synthetic.yaml | archive/configs/experiment/collect_rollouts_ray_synthetic.yaml | synthetic 后端冒烟 experiment（SPEC §3.1），主线 collect 走 collect_rollouts_ray/onetraj；无 mainline 源引用 | 4b-1 |
| configs/experiment/online_cotrain_ray_dreamervla_tiny.yaml | archive/configs/experiment/online_cotrain_ray_dreamervla_tiny.yaml | OnlineCotrainRayRunner 极小 Ray smoke fixture（SPEC §3.1，CounterEnv/_test_models 合成），主线 cotrain 走 openvla_onetraj_libero_cotrain_ray；无 mainline 源引用 | 4b-2 |
| configs/experiment/online_cotrain_ray_synthetic.yaml | archive/configs/experiment/online_cotrain_ray_synthetic.yaml | OnlineCotrainRayRunner 合成 Ray smoke fixture（SPEC §3.1，_test_envs/_test_models），主线 cotrain 走 openvla_onetraj_libero_cotrain_ray；夹具重写到 manual_cotrain_ray_tiny 后无 mainline 源引用 | 4b-3 |
| tests/e2e_tests/test_s5_ray_hydra_entry.py | archive/tests/e2e_tests/test_s5_ray_hydra_entry.py | 唯一绑定 online_cotrain_ray_synthetic 的 e2e 测试（单测文件），随其 config 同批归档以保还原配对 | 4b-3 |
| configs/experiment/vla_sft_one_trajectory.yaml | archive/configs/experiment/vla_sft_one_trajectory.yaml | VLASFTRunner standalone SFT 路由（SPEC §3.1），非 cotrain 主线；主线 VLA 走 openvla_oft ckpt | 4b-4 |
| configs/experiment/vla_rynnvla_action_head.yaml | archive/configs/experiment/vla_rynnvla_action_head.yaml | VLASFTRunner RynnVLA action-head standalone SFT 路由（SPEC §3.1），非主线 | 4b-4 |
| configs/VLA/rynnvla_action_head.yaml | archive/configs/VLA/rynnvla_action_head.yaml | 仅服务 vla_rynnvla_action_head experiment 的 VLA override 组（SPEC §3.6），主线不引用 | 4b-4 |
| configs/VLA/rynnvla_one_trajectory.yaml | archive/configs/VLA/rynnvla_one_trajectory.yaml | 仅服务 vla_sft_one_trajectory experiment 的 VLA override 组（SPEC §3.6），主线不引用 | 4b-4 |
| configs/scripts/train_vla.yaml | archive/configs/scripts/train_vla.yaml | train_vla.sh 的 launcher config（默认 experiment=vla_rynnvla_action_head，已归档→孤儿）（SPEC §3.5） | 4b-4 |
| scripts/train_vla.sh | archive/scripts/train_vla.sh | standalone VLA SFT 训练启动脚本（SPEC §3.5），主线走 cotrain/collect/eval；train_wm.sh 另用 --config-name train_wm 不受影响 | 4b-4 |
| configs/experiment/openvla_oft_hdf5.yaml | archive/configs/experiment/openvla_oft_hdf5.yaml | OpenVLAOFTRunner standalone SFT 路由（SPEC §3.1），主线 VLA 用 OpenVLA-OFT ckpt 经 cotrain | 4b-5 |
| configs/experiment/openvla_oft_hdf5_one_trajectory.yaml | archive/configs/experiment/openvla_oft_hdf5_one_trajectory.yaml | OpenVLAOFTRunner one-traj SFT 路由（SPEC §3.1），非主线 | 4b-5 |
| configs/experiment/openvla_oft_hdf5_one_trajectory_l1.yaml | archive/configs/experiment/openvla_oft_hdf5_one_trajectory_l1.yaml | OpenVLAOFTRunner L1 one-traj SFT 路由（SPEC §3.1），非主线 | 4b-5 |
| configs/VLA/openvla_oft.yaml | archive/configs/VLA/openvla_oft.yaml | 仅服务 openvla_oft_hdf5 experiment 的 VLA override 组（SPEC §3.6），主线不引用 | 4b-5 |
| configs/VLA/openvla_oft_one_trajectory.yaml | archive/configs/VLA/openvla_oft_one_trajectory.yaml | 仅服务 openvla_oft_hdf5_one_trajectory 的 VLA override 组（SPEC §3.6），主线不引用 | 4b-5 |
| configs/VLA/openvla_oft_l1_one_trajectory.yaml | archive/configs/VLA/openvla_oft_l1_one_trajectory.yaml | 仅服务 openvla_oft_hdf5_one_trajectory_l1 的 VLA override 组（SPEC §3.6），主线不引用 | 4b-5 |
| configs/scripts/action_state_model_conv_generation.yaml | archive/configs/scripts/action_state_model_conv_generation.yaml | 旧预处理脚本配置，不在当前 one-trajectory cotrain 主线路径 | (staged, 未提交) |
| configs/scripts/concat_record_libero.yaml | archive/configs/scripts/concat_record_libero.yaml | 旧预处理脚本配置，不在当前 one-trajectory cotrain 主线路径 | (staged, 未提交) |
| configs/scripts/regenerate_libero_dataset_save_img_action_state_wrist.yaml | archive/configs/scripts/regenerate_libero_dataset_save_img_action_state_wrist.yaml | 旧预处理脚本配置，不在当前 one-trajectory cotrain 主线路径 | (staged, 未提交) |
| dreamervla/diagnostics/analyze_compact_token_z_reconstruction.py | archive/dreamervla/diagnostics/analyze_compact_token_z_reconstruction.py | 独立诊断脚本，不在 cotrain 主线路径 | (staged, 未提交) |
| dreamervla/diagnostics/analyze_rynn_hidden_action_metrics.py | archive/dreamervla/diagnostics/analyze_rynn_hidden_action_metrics.py | 独立诊断脚本，不在 cotrain 主线路径 | (staged, 未提交) |
| dreamervla/diagnostics/check_classifier_discriminates.py | archive/dreamervla/diagnostics/check_classifier_discriminates.py | 独立诊断脚本，不在 cotrain 主线路径 | (staged, 未提交) |
| dreamervla/diagnostics/check_collection_completeness.py | archive/dreamervla/diagnostics/check_collection_completeness.py | 独立诊断脚本，不在 cotrain 主线路径 | (staged, 未提交) |
| dreamervla/diagnostics/classifier_depth_ablation.py | archive/dreamervla/diagnostics/classifier_depth_ablation.py | 独立诊断脚本，不在 cotrain 主线路径 | (staged, 未提交) |
| dreamervla/diagnostics/compare_policy_trace_runs.py | archive/dreamervla/diagnostics/compare_policy_trace_runs.py | 独立诊断脚本，不在 cotrain 主线路径 | (staged, 未提交) |
| dreamervla/diagnostics/diagnose_hidden_token_structure.py | archive/dreamervla/diagnostics/diagnose_hidden_token_structure.py | 独立诊断脚本，不在 cotrain 主线路径 | (staged, 未提交) |
| dreamervla/diagnostics/finetune_reward_head_sparse.py | archive/dreamervla/diagnostics/finetune_reward_head_sparse.py | 独立诊断脚本，不在 cotrain 主线路径 | (staged, 未提交) |
| dreamervla/diagnostics/measure_real_vs_imagine.py | archive/dreamervla/diagnostics/measure_real_vs_imagine.py | 独立诊断脚本，不在 cotrain 主线路径 | (staged, 未提交) |
| dreamervla/diagnostics/measure_recon_and_action_delta.py | archive/dreamervla/diagnostics/measure_recon_and_action_delta.py | 独立诊断脚本，不在 cotrain 主线路径 | (staged, 未提交) |
| dreamervla/diagnostics/measure_reward_and_drift.py | archive/dreamervla/diagnostics/measure_reward_and_drift.py | 独立诊断脚本，不在 cotrain 主线路径 | (staged, 未提交) |
| dreamervla/diagnostics/measure_wm_closed_loop.py | archive/dreamervla/diagnostics/measure_wm_closed_loop.py | 独立诊断脚本，不在 cotrain 主线路径 | (staged, 未提交) |
| dreamervla/diagnostics/measure_wm_imagine_actor.py | archive/dreamervla/diagnostics/measure_wm_imagine_actor.py | 独立诊断脚本，不在 cotrain 主线路径 | (staged, 未提交) |
| dreamervla/diagnostics/measure_wm_imagine_fidelity.py | archive/dreamervla/diagnostics/measure_wm_imagine_fidelity.py | 独立诊断脚本，不在 cotrain 主线路径 | (staged, 未提交) |
| dreamervla/diagnostics/monitor_dreamervla_metrics.py | archive/dreamervla/diagnostics/monitor_dreamervla_metrics.py | 独立诊断脚本，不在 cotrain 主线路径 | (staged, 未提交) |
| dreamervla/diagnostics/reward_landscape_sweep.py | archive/dreamervla/diagnostics/reward_landscape_sweep.py | 独立诊断脚本，不在 cotrain 主线路径 | (staged, 未提交) |
| dreamervla/diagnostics/smoke_torchrun_multigpu.py | archive/dreamervla/diagnostics/smoke_torchrun_multigpu.py | 独立诊断脚本，不在 cotrain 主线路径 | (staged, 未提交) |
| dreamervla/diagnostics/train_replay_classifier_warmup.py | archive/dreamervla/diagnostics/train_replay_classifier_warmup.py | 独立诊断脚本，不在 cotrain 主线路径 | (staged, 未提交) |
| dreamervla/diagnostics/validate_oft_rynn_style_sidecar.py | archive/dreamervla/diagnostics/validate_oft_rynn_style_sidecar.py | 独立诊断脚本，不在 cotrain 主线路径 | (staged, 未提交) |
| dreamervla/diagnostics/validate_real_rollout_relabel.py | archive/dreamervla/diagnostics/validate_real_rollout_relabel.py | 独立诊断脚本，不在 cotrain 主线路径 | (staged, 未提交) |
| dreamervla/diagnostics/verify_imports.py | archive/dreamervla/diagnostics/verify_imports.py | 独立诊断脚本，不在 cotrain 主线路径 | (staged, 未提交) |
| dreamervla/diagnostics/visualize_dreamervla_reward.py | archive/dreamervla/diagnostics/visualize_dreamervla_reward.py | 独立诊断脚本，不在 cotrain 主线路径 | (staged, 未提交) |
| dreamervla/envs/eval_env.py | archive/dreamervla/envs/eval_env.py | 旧 eval env 包装，不在当前 LIBERO VLA eval 主线路径 | (staged, 未提交) |
| dreamervla/legacy/__init__.py | archive/dreamervla/legacy/__init__.py | legacy artifact 工具，不应由活跃配置或 runner 导入 | (staged, 未提交) |
| dreamervla/legacy/build_classifier_shards_from_demos.py | archive/dreamervla/legacy/build_classifier_shards_from_demos.py | legacy artifact 工具，不应由活跃配置或 runner 导入 | (staged, 未提交) |
| dreamervla/legacy/libero_sim_rollout_shards.py | archive/dreamervla/legacy/libero_sim_rollout_shards.py | legacy artifact 工具，不应由活跃配置或 runner 导入 | (staged, 未提交) |
| dreamervla/models/embodiment/chameleon_model/chameleon/convert_chameleon_weights_to_hf.py | archive/dreamervla/models/embodiment/chameleon_model/chameleon/convert_chameleon_weights_to_hf.py | 旧模型转换工具，不在当前 OpenVLA-OFT cotrain 主线路径 | (staged, 未提交) |
| dreamervla/models/embodiment/openvla/__init__.py | archive/dreamervla/models/embodiment/openvla/__init__.py | 旧 OpenVLA embodiment 模块，不在当前 Hydra 选择的 OpenVLA-OFT 主线路径 | (staged, 未提交) |
| dreamervla/models/embodiment/openvla/openvla_action_model.py | archive/dreamervla/models/embodiment/openvla/openvla_action_model.py | 旧 OpenVLA embodiment 模块，不在当前 Hydra 选择的 OpenVLA-OFT 主线路径 | (staged, 未提交) |
| dreamervla/preprocess/action_state_model_conv_generation.py | archive/dreamervla/preprocess/action_state_model_conv_generation.py | 旧预处理入口，不在当前 rollout/sidecar 主线路径 | (staged, 未提交) |
| dreamervla/preprocess/concat_record_libero.py | archive/dreamervla/preprocess/concat_record_libero.py | 旧预处理入口，不在当前 rollout/sidecar 主线路径 | (staged, 未提交) |
| dreamervla/preprocess/libero_utils/regenerate_libero_dataset_save_img_action_state_wrist.py | archive/dreamervla/preprocess/libero_utils/regenerate_libero_dataset_save_img_action_state_wrist.py | 旧 LIBERO 数据再生成工具，不在当前 rollout/sidecar 主线路径 | (staged, 未提交) |
| dreamervla/preprocess/libero_utils/regenerate_libero_failure_demos.py | archive/dreamervla/preprocess/libero_utils/regenerate_libero_failure_demos.py | 旧 LIBERO 数据再生成工具，不在当前 rollout/sidecar 主线路径 | (staged, 未提交) |
| dreamervla/preprocess/merge_precompute_manifests.py | archive/dreamervla/preprocess/merge_precompute_manifests.py | 旧预处理入口，不在当前 rollout/sidecar 主线路径 | (staged, 未提交) |
| dreamervla/preprocess/smoke_extract_hdf5.py | archive/dreamervla/preprocess/smoke_extract_hdf5.py | 旧预处理入口，不在当前 rollout/sidecar 主线路径 | (staged, 未提交) |
| dreamervla/preprocess/validate_convs.py | archive/dreamervla/preprocess/validate_convs.py | 旧预处理校验工具，不在当前 rollout/sidecar 主线路径 | (staged, 未提交) |
| dreamervla/preprocess/validate_pretokenized.py | archive/dreamervla/preprocess/validate_pretokenized.py | 旧预处理校验工具，不在当前 rollout/sidecar 主线路径 | (staged, 未提交) |
| dreamervla/utils/pytorch_util.py | archive/dreamervla/utils/pytorch_util.py | 旧共享工具，不在当前主线路径调用面 | (staged, 未提交) |
| dreamervla/utils/timers.py | archive/dreamervla/utils/timers.py | 旧共享工具，不在当前主线路径调用面 | (staged, 未提交) |
| tests/e2e_tests/test_noray_torchrun_multigpu.py | archive/tests/e2e_tests/test_noray_torchrun_multigpu.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_chameleon_mask_cache.py | archive/tests/unit_tests/test_chameleon_mask_cache.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_classifier_depth_ablation.py | archive/tests/unit_tests/test_classifier_depth_ablation.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_collection_completeness_cli.py | archive/tests/unit_tests/test_collection_completeness_cli.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_collective_send_recv.py | archive/tests/unit_tests/test_collective_send_recv.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_dreamerv3_metric_log_gate.py | archive/tests/unit_tests/test_dreamerv3_metric_log_gate.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_dreamerv3_online_observe.py | archive/tests/unit_tests/test_dreamerv3_online_observe.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_hdf5_action_slice_read.py | archive/tests/unit_tests/test_hdf5_action_slice_read.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_img2bpe_device_buffer.py | archive/tests/unit_tests/test_img2bpe_device_buffer.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_libero_noop_marking.py | archive/tests/unit_tests/test_libero_noop_marking.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_one_trajectory_vla_sft_dataset.py | archive/tests/unit_tests/test_one_trajectory_vla_sft_dataset.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_openvla_oft_official_launcher.py | archive/tests/unit_tests/test_openvla_oft_official_launcher.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_policy_chunk_queue.py | archive/tests/unit_tests/test_policy_chunk_queue.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_preprocess_imports.py | archive/tests/unit_tests/test_preprocess_imports.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_preprocess_rynn_pixel_hidden.py | archive/tests/unit_tests/test_preprocess_rynn_pixel_hidden.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_pretokenize_io_w3w4.py | archive/tests/unit_tests/test_pretokenize_io_w3w4.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_ray_init_ckpt_warmup_bridge.py | archive/tests/unit_tests/test_ray_init_ckpt_warmup_bridge.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_real_eval_schedule.py | archive/tests/unit_tests/test_real_eval_schedule.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_remaining_steps_reward.py | archive/tests/unit_tests/test_remaining_steps_reward.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_reward_head.py | archive/tests/unit_tests/test_reward_head.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_rynnvla_action_head.py | archive/tests/unit_tests/test_rynnvla_action_head.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_rynnvla_action_hidden_actor_chunk_sampling.py | archive/tests/unit_tests/test_rynnvla_action_hidden_actor_chunk_sampling.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_scheduler_collective.py | archive/tests/unit_tests/test_scheduler_collective.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_scheduler_hardware.py | archive/tests/unit_tests/test_scheduler_hardware.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_scheduler_manager.py | archive/tests/unit_tests/test_scheduler_manager.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_staleness.py | archive/tests/unit_tests/test_staleness.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_timers.py | archive/tests/unit_tests/test_timers.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_tssm_transdreamer_compat.py | archive/tests/unit_tests/test_tssm_transdreamer_compat.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_wandb_relay_sync.py | archive/tests/unit_tests/test_wandb_relay_sync.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
| tests/unit_tests/test_weight_syncer_bucket.py | archive/tests/unit_tests/test_weight_syncer_bucket.py | archived 次级/旧路线测试，随对应实现归档 | (staged, 未提交) |
