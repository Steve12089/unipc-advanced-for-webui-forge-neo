# UniPC Advanced

[English](README.md) | **中文**

把 Forge 里被写死的 UniPC 参数暴露出来的采样器插件。可以在 SDXL、SDXL v-pred、Anima 上做 A/B 对比，不需要改任何核心文件。

采样数学部分逐字复制自 `modules/sd_samplers_extra.py::sample_unipc`。**默认设置下的输出与原版 `UniPC` 采样器逐位相同**（已验证：`bh1`、`bh2`、`vary_coeff` 三个变体的最大偏差均为 0.0）。

## 功能

- 注册 4 个可选采样器，而不是一套写死的配置
- 暴露 `variant`、`order`、`lower_order_final`、`sigma_mode`、`denoise_to_zero`、`discard_next_to_last_sigma`
- 设置在采样时读取，改完不需要重启
- 生效参数记录在三处独立位置（控制台、PNG infotext、HTTP 端点）
- 支持 SDXL、SDXL v-pred、Anima

## 注册的采样器

| 名称 | 变体 |
| --- | --- |
| `UniPC Adv` | 跟随设置页 |
| `UniPC Adv bh1` | 固定 `bh1` |
| `UniPC Adv bh2` | 固定 `bh2` |
| `UniPC Adv vary` | 固定 `vary_coeff` |

别名：`uniadv`、`uniadv_bh1`、`uniadv_bh2`、`uniadv_vary`。

固定变体的三个条目是为了让采样器**名字**进入 PNG infotext，这样测试图自己就能说明跑的是什么。

## 设置

`设置 -> UniPC Advanced`：

| 选项 | 默认 | 含义 |
| --- | --- | --- |
| `unipc_adv_variant` | `bh1` | 供普通的 `UniPC Adv` 条目使用 |
| `unipc_adv_order` | `3` | 多步阶数，1-4，会被钳到 `steps - 1` |
| `unipc_adv_lower_order_final` | 开 | 原版行为 |
| `unipc_adv_sigma_mode` | `scheduler` | `scheduler` 即原版（直接用调度器的 sigmas） |
| `unipc_adv_denoise_to_zero` | 关 | 实验性，多一次模型前向 |
| `unipc_adv_discard_next_to_last_sigma` | 开 | 原版 `UniPC` 行为 |
| `unipc_adv_log` | 开 | 向控制台打印生效参数 |

`sigma_mode` 另有 `time_uniform`、`time_quadratic`、`logSNR` 三个选项。

## 确认参数确实传到了

三条独立通道：

1. **控制台** —— 每次采样打印一行：
   `[UniPC Adv] sampler=UniPC Adv bh2 variant=bh2 order=3(req 3) lower_order_final=True sigma_mode=scheduler denoise_to_zero=False discard_next_to_last_sigma=True steps=26 sigma=14.6000->0.0010 latent=[1,4,96,96]`
2. **PNG infotext** —— 通过带守卫的 `KDiffusionSampler.add_infotext` monkeypatch 注入 6 个 `UniPC Adv ...` 键
3. **HTTP** —— `http://127.0.0.1:7860/unipc_adv/last` 返回已注册的采样器、当前解析出的设置、以及最近一次调用的参数；`/unipc_adv/selfcheck` 是存活探针

## 各变体的行为

| 变体 | 行为 |
| --- | --- |
| `bh1` | `B(h) = h`，一阶截断 |
| `bh2` | `B(h) = e^h - 1`，精确指数 |
| `vary_coeff` | 另一套系数推导；`order = 3` 时几乎丢弃新评估点，退化向低阶方法 |

`bh1` 与 `bh2` 只差一行，且只在硬编码 `rho = 0.5` 的两处生效，即 `order == 2` 的预测项与 `order == 1` 的校正项，两者都只在最前两次更新中触发。`bh1` 的首步校正比 `bh2` 大 `h/(e^h - 1)`：

| 步数 | 16 | 20 | 24 | 26 | 30 | 40 |
| --- | --- | --- | --- | --- | --- | --- |
| `bh1` 偏大 | +15.1% | +12.0% | +9.9% | +9.1% | +7.9% | +5.9% |

`vary_coeff` 预期是更差，而不只是不同。

## 安装 / 卸载

把本目录放进 `extensions/`。除 Forge 自带依赖外无其他依赖。删除目录即卸载。

## 文件

- `lib_unipc_adv.py` —— 纯 torch 逻辑，不 import 任何 WebUI 模块，可独立单测
- `scripts/uni_pc_advanced.py` —— 注册、设置页、infotext 钩子、HTTP 端点

## 测试建议

1. **基线门禁**：同种子、26 步，原版 `UniPC` 与 `UniPC Adv bh1` 必须逐像素相同。不相同说明插件没被选中，后面都不用测
2. **SDXL**：20 / 26 / 40 步下对比 `bh1` 与 `bh2`，CFG 5-7。步数越多差距应越小
3. **SDXL v-pred**：注意 UniPC 的 `SigmaConvert` 假定 VE（k-diffusion）约定，原版采样器同样有此限制
4. **Anima**：UniPC 通过 `lambda = -log(sigma)` 恒等式工作，对 flow 参数化是精确的。这里值得试试 `sigma_mode = logSNR`
5. **旋钮**：`order` 1 / 3 / 4（`4` 超出论文验证范围，按实验性对待）、`lower_order_final`、`discard_next_to_last_sigma`

## 注意

- UniPC 每步只花 **1 次 NFE**，与 Euler 相同；校正项复用下一步的评估，没有 2 倍开销
- **TeaCache 没有 UniPC 预设。** 内置的 `ANIMA_COEFFICIENTS` 多项式是按 Euler a / ER SDE / Euler 的步长分布标定的。TeaCache 仍可开启，但第一轮对比请关掉
- **SPEED 需要打过补丁的版本**（`sdforgeneo-SPEED`）。原版 SPEED 只在 `k_diffusion.sampling` 里查采样器，查不到会静默退化成 `sample_euler`；补丁版还会搜索 Forge 采样器注册表，找不到时直接报错。打上补丁后 `SPEED` + `UniPC Adv bh2` 可以正常工作，并打印 `[SPEED] base sampler 'UniPC Adv bh2' -> sample_unipc_adv_bh2 (via forge registry)`
- SPEED 是分段调用采样器，所以 UniPC 的多步历史在每个过渡点清零，每段开头 `order - 1` 步会降阶运行。`order = 2` 对此的容忍度远高于 `order = 3`

## 许可

MIT。