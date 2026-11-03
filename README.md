# UniPC Advanced

**English** | [中文](README.zh.md)

A Forge sampler that exposes the UniPC knobs the stock port hard-codes, so the variants can be A/B tested on SDXL, SDXL v-pred and Anima without touching any core file.

The sampling math is copied verbatim from `modules/sd_samplers_extra.py::sample_unipc`. With **default settings the output is bit-identical to the stock `UniPC` sampler** (verified: max |delta| = 0.0 for `bh1`, `bh2` and `vary_coeff`).

## What it does

- Registers 4 selectable samplers instead of one hard-coded configuration
- Exposes `variant`, `order`, `lower_order_final`, `sigma_mode`, `denoise_to_zero` and `discard_next_to_last_sigma`
- Settings are read at sampling time, no restart needed
- Records the effective parameters in three independent places (console, PNG infotext, HTTP endpoint)
- Works on SDXL, SDXL v-pred and Anima

## Samplers registered

| Name | Variant |
| --- | --- |
| `UniPC Adv` | follows the settings page |
| `UniPC Adv bh1` | fixed `bh1` |
| `UniPC Adv bh2` | fixed `bh2` |
| `UniPC Adv vary` | fixed `vary_coeff` |

Aliases: `uniadv`, `uniadv_bh1`, `uniadv_bh2`, `uniadv_vary`.

The three fixed entries exist because the sampler **name** lands in the PNG infotext, so a folder of test images self-documents.

## Settings

`Settings -> UniPC Advanced`:

| Option | Default | Meaning |
| --- | --- | --- |
| `unipc_adv_variant` | `bh1` | used by the plain `UniPC Adv` entry |
| `unipc_adv_order` | `3` | multistep order, 1-4, clamped to `steps - 1` |
| `unipc_adv_lower_order_final` | on | stock behaviour |
| `unipc_adv_sigma_mode` | `scheduler` | `scheduler` = stock (use the sampler's sigmas) |
| `unipc_adv_denoise_to_zero` | off | experimental, costs 1 extra model eval |
| `unipc_adv_discard_next_to_last_sigma` | on | stock `UniPC` behaviour |
| `unipc_adv_log` | on | print the effective parameters to the console |

`sigma_mode` alternatives: `time_uniform`, `time_quadratic`, `logSNR`.

## How to confirm the parameters really arrive

Three independent channels:

1. **Console** - one line per sampling call:
   `[UniPC Adv] sampler=UniPC Adv bh2 variant=bh2 order=3(req 3) lower_order_final=True sigma_mode=scheduler denoise_to_zero=False discard_next_to_last_sigma=True steps=26 sigma=14.6000->0.0010 latent=[1,4,96,96]`
2. **PNG infotext** - six `UniPC Adv ...` keys are injected through a guarded monkeypatch of `KDiffusionSampler.add_infotext`
3. **HTTP** - `http://127.0.0.1:7860/unipc_adv/last` returns the registered samplers, the resolved settings and the parameters of the most recent call; `/unipc_adv/selfcheck` is a liveness probe

## What the variants do

| Variant | Behaviour |
| --- | --- |
| `bh1` | `B(h) = h`, first-order truncation |
| `bh2` | `B(h) = e^h - 1`, exact exponential |
| `vary_coeff` | different coefficient derivation; at `order = 3` it nearly discards the fresh evaluation and degrades towards a lower order method |

`bh1` and `bh2` differ in exactly one line and only where `rho = 0.5` is hard-coded, i.e. the `order == 2` predictor and the `order == 1` corrector. Both fire only during the first two updates. `bh1`'s first-step correction is larger by `h/(e^h - 1)`:

| Steps | 16 | 20 | 24 | 26 | 30 | 40 |
| --- | --- | --- | --- | --- | --- | --- |
| `bh1` larger by | +15.1% | +12.0% | +9.9% | +9.1% | +7.9% | +5.9% |

`vary_coeff` is expected to look worse, not merely different.

## Install / remove

Place this folder in `extensions/`. No dependencies beyond what Forge ships. To remove, delete the folder.

## Files

- `lib_unipc_adv.py` - pure torch logic, imports nothing from the WebUI, unit testable standalone
- `scripts/uni_pc_advanced.py` - registration, settings page, infotext hook, HTTP endpoints

## Testing

1. **Baseline gate.** One seed, 26 steps, stock `UniPC` vs `UniPC Adv bh1`. They must be pixel-identical; if not, the extension is not being picked up and nothing else is worth testing
2. **SDXL**: `bh1` vs `bh2` at 20 / 26 / 40 steps, CFG 5-7. The gap should shrink as steps grow
3. **SDXL v-pred**: note that UniPC's `SigmaConvert` assumes the VE (k-diffusion) convention; the stock sampler has the same limitation
4. **Anima**: UniPC works through the `lambda = -log(sigma)` identity, which is exact for the flow parameterisation. Worth trying `sigma_mode = logSNR` here
5. **Knobs**: `order` 1 vs 3 vs 4 (`4` is outside what the paper validated, treat it as experimental), `lower_order_final`, `discard_next_to_last_sigma`

## Notes

- UniPC costs **1 NFE per step**, the same as Euler; the corrector reuses the next step's evaluation, so there is no 2x penalty
- **TeaCache has no UniPC preset.** The shipped `ANIMA_COEFFICIENTS` polynomial is calibrated for the Euler a / ER SDE / Euler step-size distributions. TeaCache can still be enabled, but turn it off for the first round of comparisons
