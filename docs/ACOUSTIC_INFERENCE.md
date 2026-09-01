# BEATs 多时间窗场景推断

## 目标

该模块把 BEATs 的 AudioSet 声音事件作为“声音事实”，再结合时间窗统计、事件共现、近期序列和 ASR 转写，由 Ollama 生成带不确定性的场景推断。它不会把声音事件标签直接当成真实地点。

## 推断等级

- `none`：只有通用声音、瞬时事件或相互冲突的线索，不能形成有效方向。
- `coarse`：只能判断室内/室外、有人交流、交通、自然或设备操作等大类。
- `specific`：至少两类非通用线索在多个窗口中共同出现，可提出办公室、道路、公园等候选场景。

界面会同时显示声音事实、场景推断、推断依据和不确定性。LLM 拒绝推断或暂不可用时，只要存在稳定线索，规则层仍会给出保守的粗略判断。

## 多时间窗规则

- 浏览器默认每 5 秒上传一个环境声音窗口。
- 每个会话最多保留 12 个窗口，默认只使用最近 60 秒。
- 标签按窗口覆盖率、出现时平均置信度、近期加权置信度、连续窗口数和最近 4 个窗口覆盖率综合排序。
- 单次高置信度事件会标记为“瞬时事件”，展示但不单独触发场景推断。
- `Speech`、`Human voice`、`Noise`、`Silence` 属于通用事件。
- 说话、旁白、合成语音等同一人声层级的标签不算多条独立场景线索；只有这些标签时可以描述交流或播放活动，但不能据此猜测室内、室外或具体地点。
- AudioSet 的室内、室外和公共空间标签属于直接场景线索，可以触发粗略推断。
- 两个非通用事件至少在两个窗口共同出现时，规则层才允许 LLM 输出 `specific`。
- 旧窗口和最近窗口的稳定非通用标签基本不重合时，标记为疑似场景变化，并把最高推断等级降为 `coarse`。

## 缓存策略

后端只在声音窗口、稳定观测、共现关系和转写快照都没有变化时复用报告。新窗口到达或文本发生变化后会重新分析，不再因为固定 12 秒缓存而返回旧结论。

前端最短约 8 秒请求一次报告；请求进行期间若又产生新证据，会记录为待刷新，并在当前请求完成后再次调度，不会静默丢弃更新。

## 可选环境变量

```bash
export ACOUSTIC_REPORT_SERVER=http://127.0.0.1:11434/v1
export ACOUSTIC_REPORT_MODEL=qwen2.5:7b
export ACOUSTIC_REPORT_TIMEOUT=60
export ACOUSTIC_CONTEXT_MAX_SESSIONS=100
export ACOUSTIC_CONTEXT_WINDOW_SECONDS=60
export ACOUSTIC_RECENT_WINDOWS=4
export ACOUSTIC_MAX_OBSERVATIONS=8
```

以上均有默认值，不设置也可以启动。通常不建议先改阈值，应先收集真实录音及日志，再根据误报和漏报调整。

## API 关键字段

`POST /acoustic_report` 的 `data` 中包含：

- `inference_level`：`none`、`coarse` 或 `specific`；
- `inference`：场景推断；
- `basis`：推断依据；
- `uncertainty`：不能确认的部分；
- `observations`：稳定声音事实及覆盖率、近期比例、连续性；
- `cooccurrences`：声音事件共现；
- `scene_change_suspected`：是否怀疑场景变化；
- `diagnostics`：主要、辅助、瞬时和有效线索数量；
- `source`：`ollama`、规则兜底或不确定结果来源；
- `cache_hit`：是否因证据未变化而使用缓存。

## 日志排查

服务日志会记录两类摘要：

```text
Acoustic evidence | session=... | windows=... | primary=... | supporting=... | transient=... | meaningful=... | cooccurrence=... | hint=... | scene_change=...
Acoustic report | session=... | level=... | source=... | elapsed=... | cache_hit=false
```

如果界面仍显示证据不足，先检查：

1. `meaningful=0`：只有通用标签或瞬时事件；
2. `hint=coarse` 但模型要求具体地点：应得到规则粗略兜底，而不是完全空白；
3. `scene_change=true`：最近一分钟内声音环境明显变化；
4. `source=deterministic_fallback`：没有可尝试的稳定线索；
5. `source=ollama_uncertain`：LLM 在没有粗略兜底条件时主动拒绝；
6. 返回包含 `error`：Ollama 调用异常，系统已退回规则结果。

## 使用建议

先在同一环境连续采集至少 4 到 6 个窗口，再观察结论。测试时分别准备办公室、道路、自然环境和只有单一偶发声音的样本，记录事实是否稳定、推断等级是否合理。BEATs 是声音事件分类模型，不是专用地点分类器；后续可并行接入使用 TAU Urban Acoustic Scenes 微调的场景分类模型，再把其结果作为独立事实交给融合层和 LLM。
