# query_before WM 向 DINO-WM 对齐(补 proprio + language)实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: 用 `superpowers:subagent-driven-development`(推荐)或 `superpowers:executing-plans` 逐任务实现本计划。步骤用 checkbox(`- [ ]`)跟踪。

**Goal:** 把 query_before(OpenVLA-OFT input-token / projected vision patch token)世界模型的输入从"纯视觉"补齐到 DINO-WM 风格的 `visual + proprio + action`,并额外加入 `language` 任务条件,全部走 `concat_dim=1`;同时把 cosine 移出 loss、关闭多 chunk(N)rollout。

**Architecture:** `ChunkAwareDinoWMWorldModel` 的 observation token 从 `[vision]` 扩成 `[vision|proprio]`(proprio 作为**被预测的观测**,沿 history 自回归前传);`language` 作为**每 episode 常量条件**与 `action` 一起 `concat_dim=1` 拼到每个 token 的通道尾部,但**排除出 loss**。proprio 取自 reward HDF5 已有的 `obs/*`;language 复用 preprocess 已计算但当前丢弃的 LLM 指令 embedding,落盘成 sidecar。loss 只保留单 chunk 内 K 步自回归项(term ①)+ reward + success,MSE only。

**Tech Stack:** PyTorch `nn.Module`、Hydra 配置、`dreamervla` runner/world-model/dataset 契约、pytest、h5py。

## Global Constraints

- 模型与数据集解耦:dataset/task 配置定义 sidecar 形状与字段;model 配置消费这些字段。(AGENTS.md)
- Hydra 是架构参数的唯一真源:`model_dim/action_emb_dim/proprio_emb_dim/lang_emb_dim/depth/heads/dim_head/mlp_dim` 必须在配置里显式给值,代码构造默认仅供旧测试。
- YAML 不做算术:`model_dim` 等派生关系写成显式值,并在 `dreamervla/config.py` 校验。
- 不新增 grouped training 顶层路由 YAML;用 `experiment=` + 模块组。
- 用 dreamervla conda 环境跑测试:`/home/user01/miniconda3/envs/dreamervla/bin/python`(base 环境有 ~13 个伪失败)。
- 提交需 `--signoff`;commit subject 不含 `===` 或 `/`;改动的 Python 会跑 ruff。
- 数值改动遵循 switchable-default-original:cosine 用 `cosine_loss_scale` 配置门控(默认仍计算、不进 loss),不删代码。

---

## 设计决策记录:loss(本轮只动 cosine 与 N 的开关,不改 term ① 结构)

这一节是把前期讨论固化进计划,**除"cosine 移出 loss + 关闭 N"两处配置外,loss 的算法结构不改**。实现者按本节理解 loss 现状即可。

### 当前 chunk loss 格式(`dino_wm_chunk.py:477-650`,`_hidden_loss_terms` 见 `dino_wm.py:746-756`)
记 `H=num_hist(3)`、`K=chunk_size`(OFT=8)、`N=chunk_rollout_chunks`、`T=`窗口长度。

```
_hidden_loss_terms(pred,tgt):
    hidden_mse    = MSE(pred, tgt)
    hidden_cosine = 1 - mean(cos(pred, tgt))          # 永远计算并返回(给日志/测试)
    hidden_loss   = hidden_loss_scale*hidden_mse + cosine_loss_scale*hidden_cosine

chunk_loss = ① + ② + ③ + ④
  ① 主 chunk(永远在): history=obs[:, :H]; hidden_pred=predict_next_chunk(...).hidden_seq
                       (从真实历史起,自回归 K 步);hidden_target=obs[:, H:H+K].detach()
                       L1 = hidden_loss(pred, tgt)
  ② 多 chunk rollout(仅 N>1 且 scale>0): 续滚 N-1 个 chunk,喂自己的预测;
                       loss += chunk_rollout_loss_scale * hidden_loss(rollout)
  ③ reward(仅 reward_loss_scale>0):  BCE/MSE(reward_logits(target.detach()), rewards)
  ④ success(仅 scale>0):             BCE/MSE(success_logits(target.detach()), success)
```

### K 与 N 的区别(关键认知)
- **K = `chunk_size`**:chunk **内**自回归深度。`predict_next_chunk`(`:462-464`)循环 K 步,**step 1 从真实历史 teacher-forced,step 2..K 喂自己的预测(闭环)**。K 被 actor 动作块**钉死**(OFT=8),不可调成 1。
- **N = `chunk_rollout_chunks`**:串几个 chunk 的抗漂移项(term ②),自由旋钮。

### 与 dino_wm 的关系(为什么不做 unification)
- dino_wm 训练是**纯 teacher forcing**(一次 block-causal forward 并行出 num_hist 个下一帧,每帧只 condition 真实帧,0 步自回归)。
- 我们即使只留单项(N=1)、term ① 仍是 **K 步自回归**(step1 teacher-forced + step2..K 闭环),**≠ dino_wm**。要字面等于 dino_wm 需 `K=1 且 N=1`,而 K=1 不可操作。
- 结论:K 是被钉死的纵向深度、N 是自由旋钮;"K=1 等价 dino_wm"只是理论性质,不做 loss unification。

### 本轮对 loss 的两处改动(均为配置/开关,无算法重写)
1. **cosine 移出 loss**:`cosine_loss_scale: 0.0`(`hidden_cosine` 仍在 `_hidden_loss_terms` 计算并返回,只是不乘进 `hidden_loss`)。
2. **关闭 N(只留单 chunk K 步自回归)**:`chunk_rollout_chunks: 1` + `chunk_rollout_loss_scale: 0.0`,term ② 被 `:538` 的 guard 关掉。

---

## File Structure(谁负责什么)

- `dreamervla/models/world_model/dino_wm_chunk.py` — **核心**。新增 `proprio_encoder`/`lang_proj`;observation token 扩成 `[vision|proprio]`;`_condition_tokens` 拼 `[obs|lang|action]`;`separate_emb`/`encode`/`predict_next`/`chunk_loss` 跟随;`model_dim` 约束扩成四段。
- `dreamervla/config.py` — `model_dim == token_dim + proprio_emb·n_p + lang_emb·n_l + action_emb·n_a` 校验 + 必填字段校验。
- `dreamervla/preprocess/preprocess_oft_action_hidden.py` — 把已算的 `language_embeddings` pool 成每 episode 一个 `[lang_dim]` 向量,落 `lang_emb` sidecar。
- `dreamervla/dataset/pixel_sequence_dataset.py` — 从 reward HDF5 加读 `obs/ee_pos+ee_ori+gripper_states`(8 维)→ `proprio` batch key。
- `dreamervla/dataset/pixel_hidden_sequence_dataset.py` — 加读 `lang_emb` sidecar → `lang_emb` batch key。
- `configs/worldmodel/openvla_oft_input_token_chunk.yaml` — 新增 proprio/lang 字段、新 `model_dim`、`cosine_loss_scale=0`、`chunk_rollout_chunks=1`、`chunk_rollout_loss_scale=0`。
- `configs/task/_base_libero.yaml` — input_tokens 段加 `proprio_keys`、`lang_emb_dir` sidecar 路径字段。
- `tests/unit_tests/test_dino_wm_proprio_language.py` — 新测试文件。

**token 通道布局(全程一致,实现者按此对齐切片)**
```
token = [ vision(token_dim) | proprio(proprio_emb·n_p) | lang(lang_emb·n_l) | action(action_emb·n_a) ]
obs_token_dim = token_dim + proprio_emb·n_p          # 观测段(被预测、进 loss)
model_dim     = obs_token_dim + lang_emb·n_l + action_emb·n_a
默认值: token_dim=4096, proprio_emb=10(n_p=1), lang_emb=32(n_l=1), action_emb=10(n_a=1)
        obs_token_dim=4106, model_dim=4148
```

---

## Task 1: 配置层 —— cosine 移出 loss + 关闭 N

只改 query_before 路由配置,验证 loss 行为符合预期(cosine 仍算、不进 loss;term ② 关闭)。

**Files:**
- Modify: `configs/worldmodel/openvla_oft_input_token_chunk.yaml`
- Test: `tests/unit_tests/test_dino_wm_proprio_language.py`

**Interfaces:**
- Produces: query_before worldmodel 配置含 `cosine_loss_scale: 0.0`、`chunk_rollout_chunks: 1`、`chunk_rollout_loss_scale: 0.0`。

- [ ] **Step 1: 写失败测试** —— 实例化 query_before WM,确认 loss 不含 cosine、term ② 关闭,但 metrics 仍报告 cosine。

```python
# tests/unit_tests/test_dino_wm_proprio_language.py
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")
QB = ["experiment=oft_world_model_dinowm_chunk", "task=OpenVLA_Onetraj_LIBERO",
      "worldmodel=openvla_oft_input_token_chunk"]

def _qb_world_model():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="train", overrides=QB)
    return instantiate(cfg.world_model), cfg

def test_query_before_cosine_off_and_no_multichunk():
    wm, cfg = _qb_world_model()
    assert float(cfg.world_model.cosine_loss_scale) == 0.0
    assert int(cfg.world_model.chunk_rollout_chunks) == 1
    assert float(cfg.world_model.chunk_rollout_loss_scale) == 0.0
    # cosine 仍被计算并返回
    pred = torch.randn(2, 3, 4, wm.token_dim)
    tgt = torch.randn(2, 3, 4, wm.token_dim)
    loss, mse, cosine = wm._hidden_loss_terms(pred, tgt)
    assert torch.allclose(loss, wm.hidden_loss_scale * mse)   # 不含 cosine
    assert cosine.requires_grad or cosine.numel() == 1        # cosine 仍算出
```

- [ ] **Step 2: 跑测试确认 FAIL**

Run: `PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_dino_wm_proprio_language.py::test_query_before_cosine_off_and_no_multichunk -q`
Expected: FAIL(当前 `cosine_loss_scale=0.1`、`chunk_rollout_chunks=4`)。

- [ ] **Step 3: 改配置** `configs/worldmodel/openvla_oft_input_token_chunk.yaml` 的 `world_model` 块:

```yaml
  cosine_loss_scale: 0.0          # cosine 仍计算并记录,但不进 loss(对齐 dino_wm 纯 MSE)
  chunk_rollout_chunks: 1         # 关闭多 chunk(N)rollout,只留单 chunk 内 K 步自回归
  chunk_rollout_loss_scale: 0.0
```

- [ ] **Step 4: 跑测试确认 PASS**

Run: `PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_dino_wm_proprio_language.py::test_query_before_cosine_off_and_no_multichunk -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add configs/worldmodel/openvla_oft_input_token_chunk.yaml tests/unit_tests/test_dino_wm_proprio_language.py
git commit --signoff -m "feat(wm): query_before cosine out of loss, disable multi-chunk rollout"
```

---

## Task 2: 模型 —— proprio 编码器 + observation token 扩成 [vision|proprio]

proprio 作为**被预测的观测**折进 observation token。新增 `proprio_encoder`,把 8 维 proprio 投到 `proprio_emb`,tile 到每个 token 通道,拼在 vision 之后。

**Files:**
- Modify: `dreamervla/models/world_model/dino_wm_chunk.py:155-272`(`__init__`)、新增 `_observation_tokens`
- Test: `tests/unit_tests/test_dino_wm_proprio_language.py`

**Interfaces:**
- Consumes: `__init__` 新增 kwargs `proprio_dim:int=0, proprio_emb_dim:int=0, num_proprio_repeat:int=1`。
- Produces: `self.proprio_condition_dim:int`、`self.obs_token_dim:int = token_dim + proprio_condition_dim`、`self.proprio_encoder: nn.Module | None`;方法 `_observation_tokens(vision_tokens:[B,T,N,token_dim], proprio_raw:[B,T,proprio_dim]) -> [B,T,N,obs_token_dim]`。

- [ ] **Step 1: 写失败测试**

```python
def test_observation_tokens_concat_proprio():
    wm, _ = _qb_world_model()
    assert wm.proprio_condition_dim == 10
    assert wm.obs_token_dim == wm.token_dim + 10
    B, T, N = 2, 3, wm.token_count
    vision = torch.randn(B, T, N, wm.token_dim)
    proprio = torch.randn(B, T, wm.proprio_dim)
    obs = wm._observation_tokens(vision, proprio)
    assert obs.shape == (B, T, N, wm.obs_token_dim)
    # vision 段原样保留;proprio 段对所有 token 相同(tile)
    assert torch.allclose(obs[..., : wm.token_dim], vision)
    assert torch.allclose(obs[:, :, 0, wm.token_dim:], obs[:, :, 1, wm.token_dim:])
```

- [ ] **Step 2: 跑测试确认 FAIL**

Run: `PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_dino_wm_proprio_language.py::test_observation_tokens_concat_proprio -q`
Expected: FAIL(`proprio_condition_dim`/`_observation_tokens` 不存在)。

- [ ] **Step 3: 实现** —— 在 `__init__` 签名加参数(`dino_wm_chunk.py:155-168`):

```python
        action_emb_dim: int = 10,
        num_action_repeat: int = 1,
        proprio_dim: int = 0,
        proprio_emb_dim: int = 0,
        num_proprio_repeat: int = 1,
        dim_head: int = 64,
```

在 `super().__init__` 之后、`expected_model_dim` 之前(`:196` 附近)加:

```python
        self.proprio_dim = int(proprio_dim)
        self.proprio_emb_dim = int(proprio_emb_dim)
        self.num_proprio_repeat = int(num_proprio_repeat)
        self.proprio_condition_dim = self.proprio_emb_dim * self.num_proprio_repeat
        if self.proprio_condition_dim > 0:
            if self.proprio_dim < 1:
                raise ValueError("proprio_emb_dim>0 requires proprio_dim>=1")
            self.proprio_encoder = nn.Sequential(
                nn.LayerNorm(self.proprio_dim),
                nn.Linear(self.proprio_dim, self.proprio_emb_dim),
            )
        else:
            self.proprio_encoder = None
        self.obs_token_dim = self.token_dim + self.proprio_condition_dim
```

新增方法(放在 `_condition_tokens` 前):

```python
    def _observation_tokens(
        self,
        vision_tokens: torch.Tensor,
        proprio_raw: torch.Tensor | None,
    ) -> torch.Tensor:
        """Fold encoded proprio into every observation token channel (concat_dim=1)."""
        if self.proprio_condition_dim == 0:
            return vision_tokens
        if proprio_raw is None:
            raise ValueError("proprio is required when proprio_emb_dim>0")
        emb = self.proprio_encoder(proprio_raw)                    # [B,T,proprio_emb]
        if self.num_proprio_repeat > 1:
            emb = emb.repeat(1, 1, self.num_proprio_repeat)        # [B,T,proprio_cond]
        tiled = emb[:, :, None, :].expand(-1, -1, vision_tokens.shape[2], -1)
        return torch.cat([vision_tokens, tiled], dim=-1)          # [B,T,N,obs_token_dim]
```

> 注:`model_dim` 约束在 Task 3 一并更新(那时 lang 段也确定),本任务暂不改 `expected_model_dim`,测试只验证 `_observation_tokens` 与 dims。

- [ ] **Step 4: 跑测试确认 PASS**

Run: `PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_dino_wm_proprio_language.py::test_observation_tokens_concat_proprio -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add dreamervla/models/world_model/dino_wm_chunk.py tests/unit_tests/test_dino_wm_proprio_language.py
git commit --signoff -m "feat(wm): proprio encoder + observation token assembly"
```

---

## Task 3: 模型 —— language 投影 + `_condition_tokens` 拼 [obs|lang|action] + model_dim 约束

language 作为**常量条件**,与 action 一起 `concat_dim=1`,但属于"条件段"(后面 loss 会排除)。同步把 `model_dim` 约束扩成四段。

**Files:**
- Modify: `dreamervla/models/world_model/dino_wm_chunk.py`(`__init__` 的 model_dim 约束 `:196-207`、`_condition_tokens:303-316`)
- Test: `tests/unit_tests/test_dino_wm_proprio_language.py`

**Interfaces:**
- Consumes: `__init__` 新增 kwargs `lang_dim:int=0, lang_emb_dim:int=0, num_lang_repeat:int=1`。
- Produces: `self.lang_condition_dim:int`、`self.lang_proj: nn.Module | None`;`_condition_tokens(obs_tokens, lang_emb, actions) -> [B,T,N,model_dim]`;`expected_model_dim = obs_token_dim + lang_condition_dim + action_condition_dim`。

- [ ] **Step 1: 写失败测试**

```python
def test_condition_tokens_layout_and_model_dim():
    wm, cfg = _qb_world_model()
    assert wm.lang_condition_dim == 32
    assert wm.model_dim == wm.obs_token_dim + wm.lang_condition_dim + wm.action_condition_dim
    assert wm.model_dim == int(cfg.world_model.model_dim)        # 配置显式值一致
    B, T, N = 2, 3, wm.token_count
    obs = torch.randn(B, T, N, wm.obs_token_dim)
    lang = torch.randn(B, wm.lang_dim)
    act = torch.randn(B, T, wm.action_dim)
    z = wm._condition_tokens(obs, lang, act)
    assert z.shape == (B, T, N, wm.model_dim)
    assert torch.allclose(z[..., : wm.obs_token_dim], obs)       # 观测段原样
    # lang 段对所有 token 与所有帧都相同(每 episode 常量)
    lo = wm.obs_token_dim
    hi = wm.obs_token_dim + wm.lang_condition_dim
    assert torch.allclose(z[:, 0, 0, lo:hi], z[:, 2, 5 % N, lo:hi])
```

- [ ] **Step 2: 跑测试确认 FAIL**

Run: `PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_dino_wm_proprio_language.py::test_condition_tokens_layout_and_model_dim -q`
Expected: FAIL。

- [ ] **Step 3: 实现** —— `__init__` 签名加 `lang_dim/lang_emb_dim/num_lang_repeat`;在 proprio 段之后加:

```python
        self.lang_dim = int(lang_dim)
        self.lang_emb_dim = int(lang_emb_dim)
        self.num_lang_repeat = int(num_lang_repeat)
        self.lang_condition_dim = self.lang_emb_dim * self.num_lang_repeat
        if self.lang_condition_dim > 0:
            if self.lang_dim < 1:
                raise ValueError("lang_emb_dim>0 requires lang_dim>=1")
            self.lang_proj = nn.Sequential(
                nn.LayerNorm(self.lang_dim),
                nn.Linear(self.lang_dim, self.lang_emb_dim),
            )
        else:
            self.lang_proj = None
```

把 `expected_model_dim`(`:196`)改成四段:

```python
        expected_model_dim = (
            self.obs_token_dim + self.lang_condition_dim + self.action_condition_dim
        )
```

错误信息同步加上 proprio/lang 项(保持现有抛错风格)。

`_condition_tokens` 改成(`:303-316`):

```python
    def _condition_tokens(
        self,
        obs_tokens: torch.Tensor,
        lang_emb: torch.Tensor | None,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        # obs_tokens 已是组装好的 observation 段 [B,T,N,obs_token_dim]
        # (chunk_loss 经 _observation_tokens 折入 proprio;rollout 经预测前传)。
        # 这里不再 obs_to_tokens 二次规整——否则会按 token_dim 误 reshape。
        actions = self._validate_actions(actions, int(obs_tokens.shape[1]))
        parts = [obs_tokens]
        if self.lang_condition_dim > 0:
            if lang_emb is None:
                raise ValueError("lang_emb is required when lang_emb_dim>0")
            le = self.lang_proj(lang_emb)                         # [B,lang_emb]
            if self.num_lang_repeat > 1:
                le = le.repeat(1, self.num_lang_repeat)
            le = le[:, None, None, :].expand(
                -1, obs_tokens.shape[1], obs_tokens.shape[2], -1
            )                                                     # 帧/ token 常量
            parts.append(le)
        action_emb = self.action_proj(actions)
        if self.num_action_repeat > 1:
            action_emb = action_emb.repeat(1, 1, self.num_action_repeat)
        parts.append(action_emb[:, :, None, :].expand(-1, -1, self.token_count, -1))
        return torch.cat(parts, dim=-1)
```

> `obs_to_tokens` 仍按 `token_dim` 规整原始 sidecar;此处传入的 `obs_tokens` 已是 `obs_token_dim` 宽(vision+proprio 已折叠,见 Task 4 的 `encode`)。`obs_to_tokens` 对已是 4D 的张量是恒等透传,无需改;若它对宽度有断言,放宽到接受 `obs_token_dim`。

- [ ] **Step 4: 跑测试确认 PASS**

Run: `PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_dino_wm_proprio_language.py::test_condition_tokens_layout_and_model_dim -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add dreamervla/models/world_model/dino_wm_chunk.py tests/unit_tests/test_dino_wm_proprio_language.py
git commit --signoff -m "feat(wm): language projection + four-segment condition concat"
```

---

## Task 4: 模型 —— `encode`/`separate_emb`/`predict_next` 跟随 observation=[vision|proprio]、lang 常量

observation 段(vision+proprio)被预测、沿 history 前传;lang 随 latent 透传。

**Files:**
- Modify: `dreamervla/models/world_model/dino_wm_chunk.py`(`encode:318-335`、`separate_emb:337-344`、`replace_actions_from_z:346-352`、`predict_next:354-392`、`predict_next_chunk:420-472`)
- Test: `tests/unit_tests/test_dino_wm_proprio_language.py`

**Interfaces:**
- Consumes: latent dict 增加可选 `"lang"`:`[B,lang_dim]`。
- Produces: `separate_emb(z)` 返回 `({"visual":[...,:token_dim], "proprio":[..., token_dim:obs_token_dim]}, cond_emb)`;`predict_next`/`predict_next_chunk` 把预测的 observation 段(obs_token_dim 宽)作为 `hidden/history` 前传。

- [ ] **Step 1: 写失败测试** —— 单 chunk 自回归仍工作,且 proprio 段被预测、随 history 前传。

```python
def test_predict_next_chunk_threads_proprio_and_lang():
    wm, _ = _qb_world_model()
    wm.eval()
    B, H, N, K = 2, wm.num_hist, wm.token_count, wm.chunk_size
    history = torch.randn(B, H, N, wm.obs_token_dim)            # 观测段宽度
    actions = torch.zeros(B, H, wm.action_dim)
    lang = torch.randn(B, wm.lang_dim)
    latent = {"hidden": history[:, -1], "history": history,
              "actions": actions, "lang": lang}
    out = wm.predict_next_chunk(latent, torch.zeros(B, K, wm.action_dim))
    assert out["hidden_seq"].shape == (B, K, N, wm.obs_token_dim)
    assert out["history"].shape == (B, H, N, wm.obs_token_dim)
```

- [ ] **Step 2: 跑测试确认 FAIL**

Run: `PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_dino_wm_proprio_language.py::test_predict_next_chunk_threads_proprio_and_lang -q`
Expected: FAIL。

- [ ] **Step 3: 实现**

`encode`(`:318`)加 `lang` 参数并透传。注意:chunk 路径里 `encode` 收到的 `obs`(即 `history`)**已是 observation 段(obs_token_dim 宽)**,直接条件化,不再 fold proprio:

```python
    def encode(self, obs, act, lang=None):
        obs_tokens = self._obs_embedding_from_obs(obs)       # 已是 [B,T,N,obs_token_dim]
        z = self._condition_tokens(obs_tokens, lang, act)
        ...                                                  # 其余 pos_embedding 逻辑不变
```
> `_obs_embedding_from_obs` 对已是 `[B,T,N,obs_token_dim]` 的 4D 张量须透传(若它按 `token_dim` 断言宽度,放宽到接受 `obs_token_dim`)。proprio 的折入只在 Task 5 的 `chunk_loss` 入口经 `_observation_tokens` 发生一次;rollout 中 observation 段由预测前传,不重复折入。

`separate_emb`(`:337-344`)改成真实 proprio 切片:

```python
    def separate_emb(self, z):
        visual = z[..., : self.token_dim]
        proprio = z[..., self.token_dim : self.obs_token_dim]
        cond_emb = z[..., self.obs_token_dim :].mean(dim=2)
        return {"visual": visual, "proprio": proprio}, cond_emb
```

`replace_actions_from_z`(`:346-352`)用 observation 段 + lang:

```python
    def replace_actions_from_z(self, z, act, lang=None):
        obs_tokens = z[..., : self.obs_token_dim]
        return self._condition_tokens(obs_tokens, lang, act)
```

`predict_next`(`:354-392`):把 lang 从 latent 取出透传给 `encode`;`next_hidden` 取 observation 段(obs_token_dim 宽)而非仅 visual:

```python
        lang = latent.get("lang") if isinstance(latent, dict) else None
        z = self.encode(history, action_history, lang)
        pred_z = self.predict(z)
        next_hidden = pred_z[:, -1][..., : self.obs_token_dim]    # [B,N,obs_token_dim]
```

并在返回 dict 里把 `lang` 透传给下一步(`predict_next_chunk` 的 `cur` 也带 lang):

```python
        return {"hidden": next_hidden, "history": next_history,
                "actions": next_action_history, "lang": lang}
```

`predict_next_chunk`(`:456-472`)的 `cur` 初始化加 `"lang": self._latent_lang(latent)`(新增小工具 `_latent_lang` 从 latent dict 取 `lang`,缺失返回 None)。

- [ ] **Step 4: 跑测试确认 PASS**

Run: `PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_dino_wm_proprio_language.py::test_predict_next_chunk_threads_proprio_and_lang -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add dreamervla/models/world_model/dino_wm_chunk.py tests/unit_tests/test_dino_wm_proprio_language.py
git commit --signoff -m "feat(wm): thread observation=[vision|proprio] and constant lang through rollout"
```

---

## Task 5: 模型 —— `chunk_loss` 用 observation 段做 target,排除 lang+action

hidden loss 作用在 `[vision|proprio]`(obs_token_dim);lang/action 段排除;reward/success 不变。

**Files:**
- Modify: `dreamervla/models/world_model/dino_wm_chunk.py:477-650`(`chunk_loss`)
- Test: `tests/unit_tests/test_dino_wm_proprio_language.py`

**Interfaces:**
- Consumes: `batch` 增加 `proprio:[B,T,proprio_dim]`、`lang_emb:[B,lang_dim]`。
- Produces: `chunk_loss` 的 `hidden_target`/`hidden_pred` 为 observation 段(obs_token_dim 宽);loss 不含 lang/action 段。

- [ ] **Step 1: 写失败测试** —— 端到端 chunk_loss 在带 proprio+lang 的 batch 上无 shape 错,且 hidden loss 维度 = obs_token_dim。

```python
def test_chunk_loss_with_proprio_language():
    wm, _ = _qb_world_model()
    B, T, N = 2, wm.num_hist + wm.chunk_size, wm.token_count
    batch = {
        "obs_embedding": torch.randn(B, T, N, wm.token_dim),
        "proprio": torch.randn(B, T, wm.proprio_dim),
        "lang_emb": torch.randn(B, wm.lang_dim),
        "actions": torch.zeros(B, T, wm.action_dim),
        "rewards": torch.zeros(B, T),
        "success_to_go": torch.zeros(B, T),
    }
    out = wm.chunk_loss(batch)
    assert torch.isfinite(out["loss"])
    assert wm._last_hidden_target_width == wm.obs_token_dim     # 见实现里记录
```

- [ ] **Step 2: 跑测试确认 FAIL**

Run: `PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_dino_wm_proprio_language.py::test_chunk_loss_with_proprio_language -q`
Expected: FAIL。

- [ ] **Step 3: 实现** —— 在 `chunk_loss`(`:502-530`)把 vision sidecar 与 proprio 折成 observation tokens,并取 lang:

```python
        vision_tokens = self.obs_to_tokens(self._obs_embedding_from_obs(batch))
        proprio = batch.get("proprio")
        obs_tokens = self._observation_tokens(vision_tokens, proprio)   # [B,T,N,obs_token_dim]
        lang_emb = batch.get("lang_emb")

        history = obs_tokens[:, :H]
        hidden_target = obs_tokens[:, H : H + K].detach()
        ...
        latent = {"hidden": history[:, -1], "history": history,
                  "actions": action_history, "lang": lang_emb}
        out = self.predict_next_chunk(latent, chunk_actions)
        hidden_pred = out["hidden_seq"]                                 # [B,K,N,obs_token_dim]
        self._last_hidden_target_width = hidden_target.shape[-1]
        loss, hidden_mse, hidden_cosine = self._hidden_loss_terms(hidden_pred, hidden_target)
```

reward/success 仍用 `hidden_target`(现在含 proprio 段)做输入——reward/success head 在 token 维 mean-pool,proprio 段会被一并 pool,符合预期(任务相关状态进入 reward 预测)。term ② 因 `chunk_rollout_chunks=1` 关闭,无需改。

- [ ] **Step 4: 跑测试确认 PASS**

Run: `PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_dino_wm_proprio_language.py::test_chunk_loss_with_proprio_language -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add dreamervla/models/world_model/dino_wm_chunk.py tests/unit_tests/test_dino_wm_proprio_language.py
git commit --signoff -m "feat(wm): chunk_loss targets observation=[vision|proprio], excludes lang+action"
```

---

## Task 6: 配置校验 —— `config.py` 四段 model_dim + 必填字段

**Files:**
- Modify: `dreamervla/config.py`(query_before / chunk WM 校验段)
- Test: `tests/unit_tests/test_config_validation.py`

**Interfaces:**
- Produces: `validate_cfg` 断言 `model_dim == token_dim + proprio_emb·n_p + lang_emb·n_l + action_emb·n_a`,并要求 proprio/lang 字段在 query_before 路由存在。

- [ ] **Step 1: 写失败测试**(`tests/unit_tests/test_config_validation.py` 追加)

```python
def test_query_before_four_segment_model_dim():
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train", overrides=[
            "experiment=oft_world_model_dinowm_chunk", "task=OpenVLA_Onetraj_LIBERO",
            "worldmodel=openvla_oft_input_token_chunk"])
    validate_cfg(cfg, world_size=1)
    wm = cfg.world_model
    assert wm.model_dim == (wm.token_dim
        + wm.proprio_emb_dim * wm.num_proprio_repeat
        + wm.lang_emb_dim * wm.num_lang_repeat
        + wm.action_emb_dim * wm.num_action_repeat)
```

- [ ] **Step 2: 跑测试确认 FAIL**

Run: `PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_config_validation.py::test_query_before_four_segment_model_dim -q`
Expected: FAIL(校验未覆盖 proprio/lang 段 / 字段缺失)。

- [ ] **Step 3: 实现** —— 在 `config.py` 找到现有 chunk `model_dim == token_dim + action_emb_dim*num_action_repeat` 校验处,扩成四段并加字段存在性检查(沿用现有报错风格)。

- [ ] **Step 4: 跑测试确认 PASS + 跑全 config 校验**

Run: `PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_config_validation.py -q`
Expected: 全 PASS。

- [ ] **Step 5: 提交**

```bash
git add dreamervla/config.py tests/unit_tests/test_config_validation.py
git commit --signoff -m "feat(config): validate four-segment query_before model_dim"
```

---

## Task 7: 预处理 —— 落 language embedding sidecar

复用 preprocess 已计算的 `language_embeddings`(`:439-443`,当前算完丢弃),pool 成每 episode 一个 `[lang_dim]` 向量并落盘。

**Files:**
- Modify: `dreamervla/preprocess/preprocess_oft_action_hidden.py`(`:437-455`、dump 段 `:690-705`)
- Test: `tests/unit_tests/test_preprocess_language_sidecar.py`

**Interfaces:**
- Produces: input-token sidecar 每 episode 多一个 `lang_emb` dataset/attr:`[lang_dim]` float16(`lang_dim = LLM hidden = 4096`)。pool = 指令 token 维 mean。

- [ ] **Step 1: 写失败测试** —— 用现有 `_FakeVisionBackbone` 套路构造最小 VLA stub,断言 sidecar 写出 `lang_emb` 且形状 `[lang_dim]`。(参照 `preprocess_oft_action_hidden.py:205-221` 的 fake backbone 与现有 preprocess 测试模式。)

- [ ] **Step 2: 跑测试确认 FAIL**

Run: `PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_preprocess_language_sidecar.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现** —— 在算出 `language_embeddings`(`:439-443`)处 pool:

```python
        lang_emb = language_embeddings.mean(dim=1)        # [B, lang_dim],token 维 mean
```

并在 dump 段(`:690-705` 写 sidecar 的地方)随 `obs_embedding` 一起写 `lang_emb`(每 demo 一个向量;同一 episode 内 instruction 不变,取该 demo 第一帧即可)。受 `want_input_tokens`/新开关 `want_language` 控制。

- [ ] **Step 4: 跑测试确认 PASS**

Run: `PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_preprocess_language_sidecar.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add dreamervla/preprocess/preprocess_oft_action_hidden.py tests/unit_tests/test_preprocess_language_sidecar.py
git commit --signoff -m "feat(preprocess): dump pooled language embedding sidecar"
```

---

## Task 8: 数据集 —— 从 reward HDF5 读 proprio

**Files:**
- Modify: `dreamervla/dataset/pixel_sequence_dataset.py:193-203`(`__getitem__` 合并 key 处)
- Test: `tests/unit_tests/test_dino_wm_proprio_language.py`(用小型合成 HDF5)

**Interfaces:**
- Consumes: reward HDF5 `data/demo_X/obs/{ee_pos,ee_ori,gripper_states}`(3+3+2=8 维,float64)。
- Produces: batch key `proprio`:`[T, proprio_dim]` float32。proprio_keys 来自 task 配置(默认 `[ee_pos, ee_ori, gripper_states]`)。

- [ ] **Step 1: 写失败测试** —— 构造含 `obs/ee_pos|ee_ori|gripper_states` 的最小 HDF5,断言 dataset 返回 `proprio` 形状 `[T,8]`。

- [ ] **Step 2: 跑测试确认 FAIL**

Run: `PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_dino_wm_proprio_language.py::test_dataset_reads_proprio -q`
Expected: FAIL。

- [ ] **Step 3: 实现** —— 在 `PixelSequenceDataset.__getitem__`(`:193-203`)按时间切片读 `obs/<key>` 并 `concat(axis=-1)` 成 `proprio`,key 列表由 `self.proprio_keys`(构造参数,从 task 配置传入)给定;窗口对齐与 `images` 一致。

- [ ] **Step 4: 跑测试确认 PASS**

Run: `PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_dino_wm_proprio_language.py::test_dataset_reads_proprio -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add dreamervla/dataset/pixel_sequence_dataset.py tests/unit_tests/test_dino_wm_proprio_language.py
git commit --signoff -m "feat(dataset): read proprio (ee_pos+ee_ori+gripper) from reward hdf5"
```

---

## Task 9: 数据集 —— 读 language embedding sidecar

**Files:**
- Modify: `dreamervla/dataset/pixel_hidden_sequence_dataset.py:322`(读 `obs_embedding` 旁加读 `lang_emb`)
- Test: `tests/unit_tests/test_dino_wm_proprio_language.py`

**Interfaces:**
- Consumes: Task 7 写的 `lang_emb` sidecar(每 demo `[lang_dim]`)。
- Produces: batch key `lang_emb`:`[lang_dim]` float32(整段窗口共用同一向量)。

- [ ] **Step 1: 写失败测试** —— 合成 sidecar 含 `lang_emb`,断言 dataset 返回 `lang_emb` 形状 `[lang_dim]`。

- [ ] **Step 2: 跑测试确认 FAIL**

Run: `PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_dino_wm_proprio_language.py::test_dataset_reads_lang_emb -q`
Expected: FAIL。

- [ ] **Step 3: 实现** —— `PixelHiddenSequenceDataset`(`:322` 附近)在读 `obs_embedding` 的同一 sidecar 文件里读 `lang_emb`(每 demo 一向量),作为 batch key 返回;`lang_emb_dir` 缺省时跳过(向后兼容)。

- [ ] **Step 4: 跑测试确认 PASS**

Run: `PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_dino_wm_proprio_language.py::test_dataset_reads_lang_emb -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add dreamervla/dataset/pixel_hidden_sequence_dataset.py tests/unit_tests/test_dino_wm_proprio_language.py
git commit --signoff -m "feat(dataset): read language embedding sidecar"
```

---

## Task 10: 配置拼装 —— worldmodel + task 字段

**Files:**
- Modify: `configs/worldmodel/openvla_oft_input_token_chunk.yaml`
- Modify: `configs/task/_base_libero.yaml`(input_tokens 段)
- Test: `tests/unit_tests/test_config_validation.py`(已有 Task 6 覆盖)

- [ ] **Step 1: 改 worldmodel 配置** —— `world_model` 块加:

```yaml
  proprio_dim: 8
  proprio_emb_dim: 10
  num_proprio_repeat: 1
  lang_dim: 4096
  lang_emb_dim: 32
  num_lang_repeat: 1
  model_dim: 4148        # = 4096 + 10 + 32 + 10
```

- [ ] **Step 2: 改 task 配置** —— `_base_libero.yaml` 的 OFT `input_tokens` 段加:

```yaml
    proprio_keys: [ee_pos, ee_ori, gripper_states]
    lang_emb_dir: ${task.hdf5_dir}_oft_input_token_lang_emb_vla_policy_h2
```

- [ ] **Step 3: 跑 config 校验 + WM 实例化**

Run: `PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_config_validation.py tests/unit_tests/test_dino_wm_proprio_language.py -q`
Expected: 全 PASS;`model_dim=4148` 一致。

- [ ] **Step 4: 提交**

```bash
git add configs/worldmodel/openvla_oft_input_token_chunk.yaml configs/task/_base_libero.yaml
git commit --signoff -m "feat(config): wire proprio + language fields for query_before WM"
```

---

## Task 11: 在线 cotrain 路径透传 proprio + lang

`OnlineReplay` / 在线 env 侧补 `proprio`(env 已有机器人状态)与 `lang_emb`(任务常量),保证 cotrain 与 offline 训练 batch 字段一致。

**Files:**
- Modify: `dreamervla/runners/online_replay.py`、对应在线 env/runner 透传处
- Test: `tests/unit_tests/test_online_cotrain_ray_runner.py`(扩断言)

- [ ] **Step 1: 写失败测试** —— `OnlineReplay.sample()` 产出含 `proprio` 与 `lang_emb` key。
- [ ] **Step 2: 跑测试确认 FAIL**。
- [ ] **Step 3: 实现** —— env 采集每步盖 `proprio`(从 env obs 取同样的 ee+gripper);`lang_emb` 由任务 instruction 经同一 language encoder 预计算后随 episode 常量透传;`OnlineReplay.sample` 像现有 `task_ids` 那样"有就透传"。
- [ ] **Step 4: 跑测试确认 PASS**。
- [ ] **Step 5: 提交**

```bash
git add dreamervla/runners/online_replay.py tests/unit_tests/test_online_cotrain_ray_runner.py
git commit --signoff -m "feat(cotrain): pass proprio + language through online replay"
```

---

## Task 12: 验证矩阵 + 静态检查

**Files:**
- Test: 上述全部 + 既有回归

- [ ] **Step 1: 跑聚焦测试矩阵**

```bash
PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
  tests/unit_tests/test_dino_wm_proprio_language.py \
  tests/unit_tests/test_chunk_wm_autoregressive.py \
  tests/unit_tests/test_dino_wm_sdpa_equivalence.py \
  tests/unit_tests/test_config_validation.py \
  tests/unit_tests/test_preprocess_language_sidecar.py \
  tests/unit_tests/test_online_cotrain_ray_runner.py -q
```
Expected: 全 PASS。

- [ ] **Step 2: 静态检查**

```bash
PYTHONPATH=. /home/user01/miniconda3/envs/dreamervla/bin/python -m py_compile \
  dreamervla/models/world_model/dino_wm_chunk.py \
  dreamervla/config.py \
  dreamervla/preprocess/preprocess_oft_action_hidden.py \
  dreamervla/dataset/pixel_sequence_dataset.py \
  dreamervla/dataset/pixel_hidden_sequence_dataset.py
git diff --check
```
Expected: 退出码 0。

- [ ] **Step 3: GPU 小烟测(有 GPU 时)** —— 跑一个 reduced query_before WM 训练步,确认带 proprio+lang 的 batch 无 shape 错、`resolved_config.yaml` 显示 `model_dim=4148, cosine_loss_scale=0, chunk_rollout_chunks=1, proprio_emb_dim=10, lang_emb_dim=32`,`train/` 的 `next_latent_mse` 下降、`next_latent_cosine_loss` 仍被记录。

> 注意:新架构(model_dim 4148、含 proprio+lang)与旧 4106 checkpoint **不兼容**,不要复用旧 warmup ckpt;需重抽 sidecar(Task 7 新增 `lang_emb`)。

---

## 验证矩阵勾稽

- [ ] Task 1:query_before `cosine_loss_scale=0`、`chunk_rollout_chunks=1`;cosine 仍计算。
- [ ] Task 2-5:`_observation_tokens`/`_condition_tokens`/`separate_emb`/`chunk_loss` 通道布局与 loss 切片正确;单 chunk K 步自回归仍工作。
- [ ] Task 6/10:`model_dim==4148` 四段校验通过。
- [ ] Task 7-9:sidecar 写出 `lang_emb`;dataset 读出 `proprio`/`lang_emb`。
- [ ] Task 11:在线路径 batch 字段与 offline 对齐。
- [ ] Task 12:聚焦矩阵全过;`py_compile`/`git diff --check` 干净;GPU 烟测(可用时)无 shape 错。

## 不在本计划范围(显式记录)

- **多 chunk(N)rollout / term ②**:本轮关闭,不实现。后续如需抗漂移再单开计划。
- **loss unification(K=1 等价 dino_wm)**:不做(K 被动作块钉死,等价仅理论性质)。
- **Q3 其它 query_before 润色**(位置编码 per-image 分组、obs Identity/LayerNorm 复核、transformer sizing 重测):暂不动。
- **task-id `nn.Embedding` 那条坏路**(`embedding_dim==token_dim` 对 query_before 失配):用 language sidecar 取代,不修。
