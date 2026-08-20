# SHED-CFS 当前实现与展示讲稿

面向：导师 + 项目合作者  
建议汇报时长：15 分钟  
对应设想文档：`Embodied_Delta_Debugging_研究报告.docx`

---

## 0. 一句话版本

我们现在已经把报告里“从一条长 VLA 失败轨迹中自动提取最小可复现因果失败切片”的想法，落到了 `pi0fast-libero + LIBERO-10` 的可运行原型上：

- 能从自然失败 rollout 中提取 `minimal_same_failure_slice`。
- 能用同类失败复现率、ablation、repair replay 去验证因果性。
- 能把动作、接触、语言、视觉、状态 anchor 放到同一个 v4 多模态候选集合里。
- 能导出人能看的 review videos，用于判断切片是否真的合理。
- 能把可靠失败切片转成 Risk Critic 样本；Repair SFT 只接收真正修复有效的动作。

当前最重要的结论是：**idea 已经可实现，且能产出可展示的失败切片和修复候选；但严格因果证据必须分层看，不能把旧版单次 review 视频等同于稳定的 source-aware strict pass。**

---

## 1. 研究问题：为什么要做 Embodied Delta Debugging

报告中的核心问题是：

> 给定一条长段 VLA/WAM 失败轨迹，如何自动找出最小的、可回放的、多模态因果切片，使其仍能高概率复现同类失败？

传统机器人失败日志通常很长，例如 100 到 500 个 low-level action steps。真正导致失败的可能只有其中一个短 action chunk、一次错误接触、一次错误 grasp、一个视觉 grounding 错误，或者一句语言里的目标歧义。

如果直接拿整条失败轨迹做分析或训练，会有两个问题：

1. **诊断困难**：大量正常动作淹没了真正失败原因。
2. **训练信号低质**：模型可能学习到整条失败动作，而不是学习“哪里危险”和“怎样修”。

SHED-CFS 的目标是把失败从“日志”变成“可复现、可验证、可训练”的资产。

---

## 2. 方法总览：从长 rollout 到 causal failure slice

当前实现围绕三层对象组织，避免把一个很短的复现点误说成完整修复上下文。

### 2.1 `minimal_same_failure_slice`

含义：最小同类失败复现切片。

它回答的是：

> 从原始失败轨迹里截出多短的一段，还能复现同一类失败？

这里强调的是 same-failure，而不是普通 failure。比如原始失败是 `wrong_placement` 或 `moka_pot_1` 没有在 stove 上，候选切片不能因为变成了 collision 或完全卡住就算通过。

默认验证逻辑：

- 候选切片包含 `pre_state + candidate actions + continuation`。
- `same_failure_rate >= 0.8` 才算同类失败复现。
- failure signature 来自 BDDL goal predicates、物体/eef/gripper 状态、action trace、可选 contact。

### 2.2 `causal_core_units` / `k_minimal_causal_sets`

含义：经过干预验证的因果核心单元。

它回答的是：

> 哪些具体单元真的是“因”，而不是上下文？

当前 v4 的候选单元包括：

- action chunk
- gripper transition
- object motion event
- goal predicate transition
- contact / proximity event
- language phrase intervention
- visual target / distractor mask
- state anchor

核心验证是 intervention：

```text
CE(u) = P_same_failure(S) - P_same_failure(S without/replaced u)
```

当删除或替换某个 unit 后，同类失败率明显下降，才认为它有因果贡献。

v4 还支持 `k_minimal_causal_sets`，即输出 top-k 个 bounded minimal causal sets。这里的 bounded minimal 是指：在当前生成的多模态候选集合内，删除集合中任一核心 unit 都会削弱同类失败或修复证据。

### 2.3 `repair_replay_context`

含义：用于修复 replay 的上下文。

它回答的是：

> 如果要从失败前状态开始修复，应该从哪里启动？

这通常比 `minimal_same_failure_slice` 更早。比如一个 `[487,489)` 的最小失败复现点可能只是最终放置失败的两步，但真正可修复的上下文需要从 grasp / transport 前开始。

这个拆分很关键：

- `minimal_same_failure_slice` 适合证明“失败可压缩”。
- `causal_core_units` 适合证明“哪些单元有因果贡献”。
- `repair_replay_context` 适合展示“从哪里开始修能救回来”。

---

## 3. 实现时间线：从 v1 到 v4

### v0：局部距离判据

最早版本主要看 eef-target distance 是否恶化，以及最终任务是否失败。这能快速找到可疑窗口，但语义太弱。

问题：

- 距离变大不等于任务语义失败。
- 不能区分 wrong object、wrong placement、grasp miss、slip。
- 不能证明切片是“因”。

### causal-v1 / v2：语义失败 + 必要性 / 修复性拆分

升级后引入了 BDDL goal predicate 和 failure taxonomy：

- `unsatisfied_goal_predicates_at_timeout`
- `wrong_object`
- `grasp_miss_no_transport`
- `premature_release_or_slip`
- `wrong_placement`
- `stagnation_timeout`
- 可选 `unsafe_contact`

v2 的一个重要修正是区分：

- `same_failure_necessity_pass`：切片对复现同类失败有必要性。
- `repair_valid_causal_pass`：存在有效反事实 continuation，能减少 failed goals 或成功。

这个修正来自一个实际问题：简单 `hold + recorded suffix` 经常只是破坏轨迹，不是真修复。比如 task08 wrong-object / wrong-placement 场景，关键时刻不动以后，后续 recorded actions 不会自动去抓正确物体，因此不能把这种 destructive ablation 当作修复成功。

### causal-v3：多模态 proposal

v3 开始加入多模态 proposal：

- stage / object / action chunk / contact / state-anchor 联合搜索。
- 规则语言消歧。
- policy 输入层视觉 target highlight / distractor mask。
- policy repair 与 demo/success repair 分层。

v3 证明了多模态思路有价值，但还不是全局多模态最小集合。

### causal-v4：global multimodal + top-k + source-aware review

v4 是目前主线版本。

主要升级：

- schema：`shed-cfs-causal-v4-global-multimodal`
- 输出 `k_minimal_causal_sets`
- 区分 `policy_raw`、`policy_language_phrase`、`policy_visual_mask`、`demo_existence`
- replay cache、数学早停、分层 repair scheduling
- source-aware review videos
- 最近新增 rollout archive：后续新实验会保存完整 `states_before_action + actions`，review 优先从当时失败轨迹的 simulator state/actions 启动，而不是重新自然 rollout。

---

## 4. 当前实验平台与模型

当前主线平台：

- Dataset / suite：`libero_10`
- Policy：`pi0fast-libero`
- 输出根目录：

```text
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs
```

为什么回到 `libero_10`：

- LIBERO simulator 支持 reset、render、BDDL goal predicate、object/eef/gripper state。
- `pi0fast-libero` 在 `libero_10` 上能产生既有成功也有自然失败的 rollout。
- 失败类型集中在 wrong placement、wrong object、grasp miss、unsatisfied goal predicates，适合做最小切片和 repair 验证。

---

## 5. 当前结果：可以怎么汇报

### 5.1 v4 K=5 review hunt

路径：

```text
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/v4_review_hunt_k5_from_existing_candidates_4gpu_20260528
```

统计：

```text
num_reports = 31
natural_failure = 31
same_failure = 31
same_failure_necessity_pass = 31
repair_valid_causal_pass = 23
policy_strong_repair_valid_pass = 23
full_success_repair_pass = 19
raw_policy_repair_valid_pass = 19
language_phrase_repair_valid_pass = 3
visual_mask_repair_valid_pass = 7
median cost = 2805.77 s / case
mean cost = 3540.69 s / case
```

可以这样讲：

> 在 v4 的 K=5 复核集合里，31 条自然失败全部找到了 same-failure necessity；其中 23 条有 policy-strong repair-valid 证据，19 条达到 full-success repair pass。说明方法不只是能找失败窗口，也能找到有反事实修复意义的上下文。

需要补一句 caveat：

> 这批结果里的早期 review 属于 legacy review，后来 source-aware 复核发现有 report claim 与 recorded evidence 不一致的情况。因此展示时要区分 legacy visualization 和 source-aware strict / observed evidence。

### 5.2 source-aware top candidates

路径：

```text
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/source_aware_top_candidates_4gpu_20260530_retry3
```

统计：

```text
num_cases = 8
num_recorded_success = 6
num_recorded_improvement = 7
num_reported_vs_recorded_mismatch = 2
num_missing_manifests = 0
```

可以这样讲：

> 我们后来做了 source-aware review，直接重录 top candidates 的多路修复视频。8 个 top candidates 中，6 个能在 recorded review 中成功，7 个至少有 failed goals 减少或目标进度提升；同时有 2 个 mismatch，说明单靠 report claim 不够，必须保留视频和 predicate 复核。

### 5.3 strictcoord retry：为什么需要 archive

路径：

```text
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/source_aware_remaining_gpu0_retry_strictcoord_20260530
```

统计：

```text
num_cases = 4
num_recorded_success = 1
num_recorded_improvement = 1
num_observed_success = 2
num_observed_improvement = 2
num_reported_vs_recorded_mismatch = 2
```

可以这样讲：

> 严格坐标复核暴露了一个工程问题：旧 report 没保存完整 simulator state/action archive，有些 review 需要 fresh rollout 重建上下文，可能导致长度不一致或复核结果漂移。因此我们已经加了 rollout archive 机制，后续新 report 会保存完整失败轨迹状态和动作，review 从当时记录的失败情景启动。

---

## 6. 推荐展示视频

### 展示 1：v4 legacy 效果总览

视频：

```text
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/v4_review_hunt_k5_from_existing_candidates_4gpu_20260528/review_partial_27_trial1/all_review_reel.mp4
```

用途：

- 展示系统已经能从长失败轨迹中找短切片。
- 展示 original / minimal replay / policy repair / demo-or-success repair 的基本形式。
- 适合放在汇报前半段，让别人直观看到“这个 idea 不是空想，已经有可视化产物”。

讲解方式：

> 这里每个 case 都对应一个自然失败 rollout。视频把原始失败、最小 replay、policy repair、demo/success repair 放在一起。可以看到系统定位的是几十步甚至几步的局部片段，而不是整条 500 step rollout。

注意事项：

- 这属于 legacy review，不是最终 strict evidence。
- `task08_init39_seed57` 这类 case 可以展示“看起来成功”的旧效果，但不建议作为最终严格证据，因为后续 source-aware 复核发现它并不稳定。

### 展示 2：source-aware 六联屏，多路修复对比

视频：

```text
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/review_source_aware_fix_task08_init02_seed07_20260529/cases/task08_init02_seed07/trial_00_repair_multisource.mp4
```

这个视频适合重点讲，因为它体现了 v4 的多模态思路。

六联屏解释：

```text
上排左：original context
上排中：minimal / recorded replay
上排右：raw policy repair
下排左：language phrase repair
下排中：visual mask repair
下排右：demo/success repair 或 unavailable
```

各部分含义：

- `original context`：原始失败 rollout 中对应上下文，帮助人看失败发生前后状态。
- `minimal / recorded replay`：从候选切片或上下文开始回放，用于验证是否复现同类失败。
- `raw policy repair`：从 repair context 重新 query policy，看模型自己能否修复。
- `language phrase repair`：对语言 prompt 做短语级消歧或改写，再让 policy 执行。
- `visual mask repair`：在 policy 输入图像上 highlight target 或 mask distractor，不改变物理环境。
- `demo/success repair`：如果 report 或 demo lookup 有可用动作，则展示成功轨迹或 demo 最近邻修复；灰色 unavailable 表示没有对应 source，不是视频坏了。

这段视频可以讲：

> task08 是典型的 moka pot 放置任务。原始失败中 goal predicate 没全部满足。v4 不只看动作窗口，还尝试语言、视觉、policy replan 等多种修复来源。这个视频里可以观察不同 source 对 failed goals 的影响。

### 展示 3：source-aware top candidates 索引

HTML 索引：

```text
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/source_aware_top_candidates_4gpu_20260530_retry3/gpu0/review/review_index.html
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/source_aware_top_candidates_4gpu_20260530_retry3/gpu1/review/review_index.html
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/source_aware_top_candidates_4gpu_20260530_retry3/gpu2/review/review_index.html
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/source_aware_top_candidates_4gpu_20260530_retry3/gpu3/review/review_index.html
```

总览视频：

```text
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/source_aware_top_candidates_4gpu_20260530_retry3/gpu0/review/all_review_reel.mp4
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/source_aware_top_candidates_4gpu_20260530_retry3/gpu1/review/all_review_reel.mp4
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/source_aware_top_candidates_4gpu_20260530_retry3/gpu2/review/all_review_reel.mp4
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/source_aware_top_candidates_4gpu_20260530_retry3/gpu3/review/all_review_reel.mp4
```

用途：

- 展示不是单个 cherry-pick，而是一组 top candidates。
- 可以按 case 打开 `case_review.json`，查看 failed goals、minimal slice、repair context、recorded evidence。
- 适合答疑时使用。

---

## 7. 如何讲“证据分层”

为了避免别人追问时混淆，建议明确讲四层证据：

### 7.1 Report claim

causal probe 在当时搜索和 K=5 replay 中记录的结果。

优点：能反映搜索过程和统计结果。  
限制：如果没有保存完整 rollout archive，后续 review 可能需要重新收集 rollout，存在漂移。

### 7.2 Legacy review

早期 review videos，展示效果很好，适合说明系统思路。

优点：直观。  
限制：没有完整 `recorded_repair_evidence`，不能单独作为最终严格证据。

### 7.3 Source-aware recorded evidence

后续 review 重新录制每个 source 的 replay / repair，并写入 `recorded_repair_evidence`。

优点：能发现 report claim 与当前视频复核是否一致。  
限制：旧 report 没有 archive 时仍可能 fresh rollout。

### 7.4 Strict evidence

要求坐标一致、区间不越界、replay predicate 与视频一致。

这是最适合写进论文或最终结论的证据。

后续 archive 机制会把 strict evidence 做得更稳：新 report 保存完整失败 rollout 的 simulator states 和 actions，review 从记录成失败的情景启动，而不是重新自然 rollout。

---

## 8. 训练利用：Risk Critic 和 Repair SFT

### 8.1 Risk Critic

Risk Critic 的目标不是让模型模仿失败动作，而是学习：

> 当前 state + action chunk + goal predicate 下，这段行为是否高风险，可能导致什么 failure type。

正样本来源：

- `same_failure_necessity_slice`
- `same_failure_necessity_core`
- `policy_strong_repair_valid_core`

负样本来源：

- 成功 rollout 随机窗口
- 失败事件外窗口
- repair 后 same-failure 降低的窗口

输入特征：

- state/object/eef/gripper features
- action chunk
- goal predicate truth
- stage / target object / contact / visual-language intervention metadata

不作为训练输入：

- `same_failure_rate`
- `causal_effect`
- `failure_type` 等 oracle label 字段

### 8.2 Repair SFT

Repair SFT 更严格。

可以进入 Repair SFT 的动作必须满足：

- repair 后 failed goals 减少，或者最终成功。
- 不引入更多 failed goals。
- 不是 destructive ablation。
- 不是 `recorded_error_continuation`。

动作来源优先级：

1. policy strong repair 成功动作
2. language / visual intervention 下的有效 repair 动作
3. demo / success nearest-neighbor repair，作为存在性和候选数据

需要强调：

> 失败切片本身不直接喂给 policy 做 SFT，否则可能学坏动作。失败切片先用于 Risk Critic；只有可靠 repair actions 才进入 Repair SFT。

---

## 9. 当前限制与已经修的点

### 9.1 旧 report 没有完整 archive

现象：

- 某些旧 case 复核时需要 fresh rollout。
- fresh rollout 长度可能和 report 不一致，导致坐标 mismatch 或 interval out-of-bounds。
- 例如 `task05_init27_seed57`、`task07_init16_seed97` 暴露过这类问题。

当前修复：

- 已加入 rollout archive 机制。
- 后续新 causal report 会保存完整 `states_before_action + actions`。
- review 会优先从 archive 加载，而不是重新自然 rollout。

### 9.2 旧版四联屏容易把“看起来成功”误解成 strict pass

现象：

- `task08_init39_seed57` 旧视频中 policy repair 看起来成功。
- 后续 source-aware strictcoord 复核中，recorded evidence 没有复现成功。

当前处理：

- 文档和汇报中把它标为 legacy visualization，不作为最终严格证据。
- 新 review 支持 `recorded_error_continuation`，可以展示“按原失败动作继续会怎样”，但不会把它计作 repair success。

### 9.3 visual mask 仍是 policy input 层反事实

当前 visual intervention 只改 policy 输入图像，不改变 simulator 物理世界。

这适合证明视觉 grounding 对 policy 决策有影响，但还不是物理遮挡或 WAM 视觉反事实。

下一步可以加：

- 更准确的 MuJoCo camera projection。
- 物理遮挡 / distractor removal。
- WAM proposal + simulator verifier。

---

## 10. 汇报建议顺序

### 第 1-2 分钟：研究问题

讲：

> VLA 失败 rollout 很长，但真正失败原因通常很短。我们要做的是 embodied 版本的 delta debugging：从失败轨迹中自动找出最小可复现、可反事实验证、可训练利用的 causal failure slice。

### 第 3-5 分钟：方法三层对象

讲：

> 我们不再只输出一个短窗口，而是拆成 minimal_same_failure_slice、causal_core_units / k_minimal_causal_sets、repair_replay_context。这样能区分复现点、因果原因和修复上下文。

### 第 6-8 分钟：实现进展

讲：

> 实现从局部距离判据升级到 causal-v1/v2，再到 v3 多模态 proposal，当前 v4 支持 global multimodal units、top-k minimal sets、source-aware review 和 rollout archive。

### 第 9-11 分钟：展示视频

建议播放：

1. `review_partial_27_trial1/all_review_reel.mp4`：展示系统整体效果。
2. `task08_init02_seed07/trial_00_repair_multisource.mp4`：重点讲六联屏和多路修复。
3. 打开 `source_aware_top_candidates_4gpu_20260530_retry3/gpu*/review/review_index.html`：说明 top candidates 不是单个例子。

### 第 12-13 分钟：当前结果

讲统计：

> v4 K=5 review hunt 有 31 个自然失败报告，31 个 same-failure necessity，23 个 policy-strong repair-valid，19 个 full-success repair pass。source-aware top candidates 中 8 个复核 case 有 6 个 recorded success、7 个 recorded improvement。

### 第 14 分钟：训练利用

讲：

> same-failure necessity 适合训练 Risk Critic；真正降低 failed goals 或成功的 repair actions 才能进入 Repair SFT。我们避免把失败动作直接喂给 policy。

### 第 15 分钟：限制和下一步

讲：

> 最大的工程修正是 archive。旧 report 没存完整 simulator state/action，所以部分 review 需要 fresh rollout，导致 mismatch。现在已加 archive，后续会从记录成失败时的状态启动复核和修复。

---

## 11. 可直接引用的路径清单

项目根目录：

```text
/data2/yanghaoyun/research/Embodied_Delta_Debugging
```

研究设想报告：

```text
/data2/yanghaoyun/research/Embodied_Delta_Debugging/Embodied_Delta_Debugging_研究报告.docx
```

v4 K=5 review hunt：

```text
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/v4_review_hunt_k5_from_existing_candidates_4gpu_20260528
```

legacy 总览视频：

```text
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/v4_review_hunt_k5_from_existing_candidates_4gpu_20260528/review_partial_27_trial1/all_review_reel.mp4
```

source-aware 六联屏示例：

```text
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/review_source_aware_fix_task08_init02_seed07_20260529/cases/task08_init02_seed07/trial_00_repair_multisource.mp4
```

source-aware top candidates：

```text
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/source_aware_top_candidates_4gpu_20260530_retry3
```

source-aware top candidates HTML：

```text
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/source_aware_top_candidates_4gpu_20260530_retry3/gpu0/review/review_index.html
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/source_aware_top_candidates_4gpu_20260530_retry3/gpu1/review/review_index.html
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/source_aware_top_candidates_4gpu_20260530_retry3/gpu2/review/review_index.html
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/source_aware_top_candidates_4gpu_20260530_retry3/gpu3/review/review_index.html
```

strictcoord 诊断包：

```text
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/source_aware_remaining_gpu0_retry_strictcoord_20260530
```

---

## 12. 可以怎么回答可能的问题

### Q1：这和普通 failure detector 有什么区别？

普通 failure detector 回答“会不会失败”。SHED-CFS 回答“哪一段、哪些多模态单元导致同类失败，并且能不能通过最小反事实修复”。

### Q2：为什么不直接用完整失败轨迹训练？

完整失败轨迹噪声大，而且会让模型学习坏动作。SHED-CFS 先压缩成 causally meaningful slice；Risk Critic 学风险，Repair SFT 只学可靠修复动作。

### Q3：为什么需要 same-failure，而不是只看失败？

因为候选切片可能导致另一种失败。例如原始是 wrong object，候选变成碰撞或停滞，不能说明找到了原失败原因。

### Q4：为什么 repair context 比 minimal slice 更早？

最小复现点可能只有放置失败的 2 步，但要修复它，需要从 grasp、transport 或接触前开始。v4 明确区分复现点和修复上下文。

### Q5：当前证据最强的是什么？

v4 K=5 中的 policy-strong repair-valid + source-aware recorded success 是目前最强的组合证据。legacy review 视频适合展示效果，但最终结论要看 recorded / strict evidence。

### Q6：下一步最关键是什么？

用新加入的 rollout archive 机制重新跑一批 source-aware review，让所有复核都从记录成失败时的 simulator state/actions 启动；然后再做 Risk Critic 训练和 Repair SFT 数据准入。

---

## 13. 最后总结

当前进展可以概括为：

1. 研究设想已经从概念推进到可运行的 SHED-CFS v4 原型。
2. 系统能在 `pi0fast-libero + libero_10` 自然失败中找到同类失败切片和 repair-valid 候选。
3. 多模态单元、top-k minimal sets、source-aware review 已经形成闭环。
4. 有可展示的视频和 HTML review index，可以人工检查切片、修复和失败类型。
5. 当前最大的 caveat 是旧 report 没有完整 archive；这个问题已经在代码层补上，后续新实验会更严格、更可复现。

适合对外的主线表述：

> 我们不是只做失败检测，而是在做 embodied 版本的 delta debugging：把长失败 rollout 自动压缩成短小、可复现、可干预、可训练利用的多模态因果失败切片。当前 v4 已经能在 LIBERO 上产出可视化和统计证据，下一步是用 archive 机制重新跑严格复核，并把可靠切片用于 Risk Critic 和 Repair SFT。
