# SHED-CFS v4：最小关键帧发现与修复能力报告

## 1. 核心结论

当前实验支持一个谨慎但有力的结论：

> SHED-CFS v4 已经能够从 LIBERO-10 的自然失败轨迹中，定位出可以改变后续任务成败的关键起点；其中已有强样例支持“局部采样窗口内的最小关键帧”，并且能够从该关键帧重新规划完成修复。

本轮 `pi0fast-libero + LIBERO-10` Hunt20 实验找到了 20 个 raw-policy 可修复 case。每个 accepted case 都满足：

- 原始轨迹是自然失败；
- 保存了失败时的 rollout archive；
- 从 v4 找到的 keyframe 对应 simulator state reset；
- 保持 task、goal、prompt 不变；
- 不使用 language / visual / demo repair；
- raw pi0fast policy replan；
- K=5 全部成功。

这说明 SHED-CFS 找到的不是离线拼接出来的成功片段，而是真实可执行的修复起点。

## 2. 为什么这支持“找到关键帧”

SHED-CFS v4 的关键验证逻辑是：

1. 固定一条自然失败轨迹并保存 archive；
2. 在失败轨迹中搜索候选 keyframe；
3. 从候选 keyframe 的 simulator state 重新启动；
4. task 和 prompt 不变，只重新 query policy；
5. 若该 keyframe K=5 全成功，而附近很多 keyframe 失败，则说明该点具有因果修复意义。

因此，关键帧不是靠人工视觉直觉选出来的，而是靠“从这一帧开始是否能改变最终任务结果”验证出来的。

## 3. 最强样例：task05_init36_seed27

目前最强的“最小关键帧”证据是：

```text
task05_init36_seed27
found keyframe = 110
```

随机 keyframe sweep 的成功率为：

```text
67:0, 70:0, 72:0, 75:0, 77:0,
99:0, 100:0, 102:0, 106:0,
110:1,
113:0, 116:0, 118:0, 122:0, 125:0,
142:0, 149:0, 150:0, 152:0, 156:0, 166:0
```

这个结果说明：

- v4 找到的 keyframe 110，K=5 全成功；
- 采样窗口内其它 keyframe 全部失败；
- 110 是唯一 full-success keyframe；
- 110 也是采样窗口内最早成功点。

所以这个 case 可以作为当前最干净的“找到最小关键帧并修复”的展示样例。

对应视频：

```text
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/v4_keyframe_repair_hunt20_20260604/videos/task05_init36_seed27_found_keyframe_raw_repair_review.mp4
```

## 4. 其它支持证据

另外两个 case 也有支持作用：

```text
task05_init06_seed27
task02_init07_seed37
```

它们的 found keyframe 也能 K=5 全成功，但 sampled window 中还存在其它成功 keyframe。因此它们更适合表述为：

> SHED-CFS 找到了稀疏的可修复关键起点。

不建议把它们说成唯一最小关键帧。

## 5. 整体结果

Hunt20 说明方法不是单点偶然成功：

```text
accepted repair keyframe cases: 20
found keyframe K=5 success: 20/20
```

进一步 random keyframe sweep 分类为：

```text
sharp_keyframe: 3
repair_basin: 9
weak_specificity: 8
```

解释：

- `sharp_keyframe`：支持关键帧具有较强最小性；
- `repair_basin`：说明找到了可修复状态区域，但不支持唯一最小；
- `weak_specificity`：附近很多点也能成功，或存在 policy inference error，不能用来支持最小关键帧。

因此最严谨的结论是：

> SHED-CFS v4 已经能稳定找到可修复关键帧；其中已有强样例支持 sampled-window 内的最小关键帧。其它 case 更多说明方法能找到对训练有价值的可修复状态区域。

## 6. 对训练的意义

即使不是所有 case 都是唯一最小关键帧，它们仍然非常有价值。

对 Risk Critic：

- 自然失败 archive 提供真实失败上下文；
- keyframe sweep 能区分失败状态和可修复状态；
- sharp keyframe 和 repair basin 都能提供风险边界样本。

对 Repair SFT：

- accepted case 的 repair trajectory 是从真实失败 state reset 后由 raw policy 完成；
- 不依赖 destructive ablation；
- 不依赖 demo 拼接；
- K=5 全成功的 repair action 可以作为高质量修复样本候选。

## 7. 推荐汇报说法

可以这样讲：

> 我们首先保存自然失败轨迹的完整 simulator archive，然后在失败轨迹中搜索 keyframe。验证时不改变任务、不改 prompt、不用 demo，只从候选 keyframe reset 后让 pi0fast-libero 重新规划。如果该 keyframe K=5 全成功，而周围随机 keyframe 大多失败，我们认为它具有关键帧因果意义。

对目前结果可以说：

> 在 Hunt20 实验中，我们找到 20 个 raw-policy 可修复 keyframe。其中 `task05_init36_seed27` 是最强样例：只有 v4 找到的 keyframe 成功，周围采样点全部失败，支持局部采样窗口内的最小关键帧。

不建议说：

> 我们已经证明所有 case 都找到了全局最小关键帧。

更严谨的说法是：

> 我们证明了方法可以找到 raw-policy 可修复关键帧；其中已有样例支持 bounded sampled-window minimality。全局最小性还需要更密集的 exhaustive sweep 进一步确认。

## 8. 推荐展示素材

主展示视频：

```text
/data2/yanghaoyun/research/Embodied_Delta_Debugging/model_datasets/pi0fast-libero-libero_10/outputs/v4_keyframe_repair_hunt20_20260604/videos/task05_init36_seed27_found_keyframe_raw_repair_review.mp4
```

推荐展示逻辑：

1. 先展示原始自然失败片段；
2. 再展示从 found keyframe 重新启动；
3. 强调 task language 没变、模型没换、没有 demo；
4. 展示最终任务成功；
5. 配合成功率曲线说明：周围采样 keyframe 都失败，只有 found keyframe 成功。

## 9. 下一步加强证据

为了进一步加强“最小关键帧”证据，建议：

1. 对 `task05_init36_seed27` 做 dense sweep；
2. 例如 keyframe `80-130` 每一帧跑 K=5 或 K=10；
3. 如果仍然只有 110 或极窄窗口成功，就能更强支持最小性；
4. 对其它 `sharp_keyframe` case 做同样验证；
5. 修复 policy inference error 后重跑 weak cases。

## 10. 一句话总结

> SHED-CFS v4 不仅能把失败轨迹压缩到可验证的关键状态，还能证明从这些关键状态重新规划可以修复任务；其中已有强样例显示，该关键状态在局部采样窗口内具有最小性。
