# Embodied Delta Debugging Idea Validation Report

## 结论

当前验证结果支持这个判断：**Embodied Delta Debugging / SHED-CFS 这个 idea 可以实现，而且在初步实验中表现较好。**

我们已经验证了三层能力：

1. 在真实 LIBERO demonstration 轨迹上做可控失败注入，算法能找回短小失败切片。
2. 在 LIBERO 仿真中执行非注入 scripted failure，能通过 MuJoCo state-reset replay 复现并最小化失败窗口。
3. 在本地 `pi05_libero` checkpoint 的自然 rollout 中找到了一个真实失败，并成功压缩成 replay-verified failure slice。

最强的当前证据是第三项：`pi05_libero` 在 `libero_10` 任务 `put both moka pots on the stove` 上出现自然失败，原始 rollout 长度为 `520` 步，定位到局部失败事件 `[129, 161]`，最终最小化到 `[149, 160]` 共 `11` 步，并且 replay 复现同类失败。

## 验证环境

- 项目目录：`/root/autodl-tmp/research/Embodied_Delta_Debugging`
- 本地 LIBERO LeRobot 数据集：`/root/autodl-tmp/research/VLA_SKILL/datasets/HuggingFaceVLA_libero`
- OpenPI 仓库：`/root/autodl-tmp/research/openpi`
- LIBERO Python 环境：`/root/autodl-tmp/envs/libero38/bin/python`
- OpenPI Python 环境：`/root/autodl-tmp/research/openpi/.venv/bin/python`
- 本地 policy checkpoint：`/root/autodl-tmp/research/VLA_SKILL/model/pi05_libero`

没有使用网络下载；启动 policy server 时显式取消了代理环境变量。

## 已实现代码

- `data_probe.py`：读取本地 LeRobot/LIBERO parquet 数据，统计 episode/task/action/state。
- `failure_oracle.py`：构造可控失败注入和 stochastic same-failure oracle。
- `shed_minimizer.py`：实现 action-chunk 到 frame-window 的层次化最小化。
- `bug_report.py`：生成 minimal reproducible embodied bug report JSON。
- `run_offline_probe.py`：在本地 LIBERO parquet 数据上跑离线验证。
- `real_sim_failure_probe.py`：在 LIBERO 中执行非注入 scripted failure，并用 state-reset replay 验证切片。
- `pi05_natural_failure_probe.py`：连接 `pi05_libero` policy server，寻找自然 rollout 失败并做 replay-based minimization。
- `libero_replay_smoke.py`：验证 LIBERO 环境 `reset/set_init_state/step` 能跑通。
- `test/test_shed_minimizer.py`：合成轨迹单元测试。

## 验证一：可控失败注入

目的：先检查最小化算法本身是否工作。如果算法连“答案已知”的失败窗口都找不回来，就不值得继续接真实 policy。

方法：

- 从本地 LeRobot/LIBERO 数据集中读取真实 demonstration。
- 人为在一小段 action chunk 上制造失败，例如反向动作、延迟 gripper、替换错误 action。
- 运行 SHED-CFS，检查是否能从完整轨迹中找回注入窗口。

结果文件：

- `outputs/offline_probe/summary.json`
- `outputs/offline_probe/*_report.json`

关键结果：

| 指标 | 结果 |
|---|---:|
| episode 数 | 20 |
| 平均压缩比 | 15.83x |
| mean same-failure rate | 0.964 |
| mean causal effect score | 0.944 |
| mean injected-window IoU | 0.829 |
| median replay/evaluation 次数 | 26 |

解释：

这说明核心 minimization 机制可行，能把 100-500 步左右的轨迹压缩到很短的候选失败窗口，并且 ablation 后失败概率明显下降。

## 验证二：真实仿真失败，不使用注入 oracle

目的：确认这个 idea 不只是“离线玩数据”，而是真的能通过仿真 state reset + replay 来验证失败切片。

方法：

- 在 LIBERO 中运行一个 observation-driven bad policy。
- 该 policy 不是从成功 demo 改出来的，而是在环境中真实执行，产生失败 rollout。
- 保存每一步 MuJoCo state 和 action。
- 对候选窗口执行：
  - reset 到窗口起点 state；
  - replay 窗口内 action；
  - 用 same-failure predicate 判断是否复现同类失败。

结果文件：

- `outputs/real_sim_failure_probe.json`
- `outputs/real_sim_failure_probe_task0_init1.json`
- `outputs/real_sim_failure_probe_task1_init0.json`

三个 case 全部通过：

| case | 原始长度 | 最小切片 | 压缩比 | replay same-failure |
|---|---:|---:|---:|---|
| task0 init0 | 40 | 8 | 5.0x | true |
| task0 init1 | 40 | 8 | 5.0x | true |
| task1 init0 | 40 | 8 | 5.0x | true |

代表性结果：

- 原始失败 rollout：`40` steps
- 最小切片：`[20, 28]`
- 切片长度：`8`
- replay 后 same-failure：`true`
- replay distance delta：`0.023`

解释：

这一步证明了一个关键工程链路：**candidate failure slice 可以通过 MuJoCo state-reset replay 进行验证**。这比单纯离线 oracle 更接近论文所需的因果验证。

## 验证三：`pi05_libero` 自然 rollout 失败

目的：验证该 idea 是否能用于真实 VLA/WAM policy rollout，而不是只对 scripted policy 有效。

方法：

1. 启动本地 OpenPI policy server：

```bash
cd /root/autodl-tmp/research/openpi
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY \
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /root/autodl-tmp/research/openpi/.venv/bin/python scripts/serve_policy.py \
  --port 8000 policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=/root/autodl-tmp/research/VLA_SKILL/model/pi05_libero
```

2. 使用 LIBERO 环境作为 client，运行自然 rollout：

```bash
cd /root/autodl-tmp/research/Embodied_Delta_Debugging
/root/autodl-tmp/envs/libero38/bin/python pi05_natural_failure_probe.py \
  --task-suite-name libero_10 \
  --task-ids 3,4,5,6,7,8,9 \
  --init-state-ids 0,1 \
  --max-steps 520 \
  --event-window 32 \
  --min-distance-delta 0.03 \
  --output outputs/pi05_natural_failure_probe_libero10_more.json
```

结果文件：

- `outputs/pi05_natural_failure_probe_libero10_more.json`

找到的自然失败：

| 字段 | 值 |
|---|---|
| suite | `libero_10` |
| task id | 8 |
| instruction | `put both moka pots on the stove` |
| init state | 0 |
| rollout length | 520 |
| success | false |
| failure event | `[129, 161]` |
| event length | 32 |
| minimized slice | `[149, 160]` |
| slice length | 11 |
| trajectory reduction | 47.27x |
| event reduction | 2.91x |
| replay same-failure | true |
| replay evaluations | 17 |

代表性 replay 统计：

```json
{
  "candidate": {
    "intervals": [[149, 160]],
    "length": 11
  },
  "same_failure": true,
  "success": false,
  "start_distance": 0.4692610512938112,
  "end_distance": 0.5024962147685409,
  "distance_delta": 0.03323516347472971,
  "steps": 11
}
```

解释：

这是当前最重要的验证：**`pi05_libero` 自然失败 rollout 可以被压缩成短小、可 replay 验证的 failure slice**。这直接支持“从真实失败中定位最小可复现失败诱因”的主张。

## 当前 Same-Failure Predicate

当前 same-failure predicate 还比较简单，主要用于初步可行性验证：

- 对 scripted failure：候选窗口 replay 后，末端与目标物体距离继续按失败方向变化，并且任务不成功。
- 对 `pi05_libero` 自然失败：在局部事件窗口中，候选 replay 后目标距离出现显著回退，并且该候选窗口不完成任务。

这已经足够验证 replay-based minimization 链路，但还不是完整的论文级 failure taxonomy。

后续应扩展为：

- wrong object；
- grasp miss；
- premature gripper close；
- object slip；
- collision；
- wrong placement；
- irreversible state；
- language grounding error。

## 结论判断

根据目前结果，可以给出如下判断：

1. **能实现**：数据读取、失败采集、候选搜索、state-reset replay、报告生成都已跑通。
2. **初步效果好**：离线注入失败平均压缩 `15.83x`；真实仿真失败压缩 `5x`；`pi05_libero` 自然失败从 `520` 步压到 `11` 步。
3. **研究点成立**：区别于 failure detection/explanation/recovery，本验证确实在做 failure-inducing trajectory minimization。
4. **论文级系统仍需扩大验证**：当前 case 数较少，same-failure predicate 较简化，还需要多任务、多失败类型、多 policy 的系统评估。

综合结论：

> Embodied Delta Debugging / SHED-CFS 是一个可实现且值得继续推进的研究 idea。当前原型已经证明它能在真实 LIBERO 数据、真实仿真失败，以及 `pi05_libero` 自然 rollout 失败上形成闭环。下一阶段应聚焦扩大自然失败样本、构建语义失败标签，并与 random crop、failure-time window、risk-peak window 等 baseline 系统比较。

## 下一步建议

短期：

- 在 `libero_10` 和 `libero_90` 上批量跑更多 `pi05_libero` rollout。
- 把当前几何 predicate 扩展成多类型 failure label。
- 对每个 failure slice 生成视频/图像摘要，方便人工诊断。

中期：

- 接入更多 policy，例如 OpenVLA、pi0fast-libero、GR00T-LIBERO。
- 增加 baseline：random crop、centered crop、uniform bisection、risk-peak window。
- 统计压缩比、same-failure reproduction rate、causal effect score、搜索成本。

长期：

- 做 WAM 反事实候选生成，但最终仍用 simulator replay 验证。
- 将 minimal failure slice 转成 failure detector / recovery policy / regression benchmark 数据。
- 少量迁移到真实机器人或真实机器人数据集，验证 sim-to-real 诊断价值。
