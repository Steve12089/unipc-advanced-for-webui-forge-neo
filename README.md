# UniPC Advanced

A forge sampler that exposes the UniPC knobs the stock port hard-codes, so the
variants can be A/B tested on SDXL, SDXL v-pred and Anima without touching any
core file.

Everything the sampling math does is copied verbatim from
`modules/sd_samplers_extra.py::sample_unipc`. With **default settings the output
is bit-identical to the stock `UniPC` sampler** (verified: max |Δ| = 0.0 for
`bh1`, `bh2` and `vary_coeff`).

## What the stock port hides

| knob | stock `UniPC` | here |
|---|---|---|
| `variant` | hard-coded `"bh1"` (`sd_samplers_extra.py:126`) | 7-item dropdown / settings radio |
| `order` | `min(3, len(timesteps) - 2)` (`:125`) | slider 1..4 |
| `lower_order_final` | hard-coded `True` (`:127`) | checkbox |
| `skip_type` | dead argument - forge feeds the scheduler sigmas straight in | `sigma_mode`: `scheduler` / `time_uniform` / `time_quadratic` / `logSNR` |
| `denoise_to_zero` | dead argument in both A1111 and forge | checkbox (costs 1 extra model eval) |
| `discard_next_to_last_sigma` | registered as `True` for the `UniPC` entry | checkbox, live |

`uni_pc_variant / uni_pc_skip_type / uni_pc_order / uni_pc_lower_order_final`
still exist in A1111's `shared_options.py` but forge never ported them - that is
exactly the gap this extension fills.

## Samplers registered

| name | variant |
|---|---|
| `UniPC Adv` | follows the settings page |
| `UniPC Adv bh1` | fixed `bh1` |
| `UniPC Adv bh2` | fixed `bh2` |
| `UniPC Adv vary` | fixed `vary_coeff` |

Aliases: `uniadv`, `uniadv_bh1`, `uniadv_bh2`, `uniadv_vary` (and the
`unipc_adv_*` spellings).

The three fixed entries exist because the sampler **name** lands in the PNG
infotext, so a folder of test images self-documents.

## Settings

`Settings -> UniPC Advanced`:

| option | default | meaning |
|---|---|---|
| `unipc_adv_variant` | `bh1` | used by the plain `UniPC Adv` entry |
| `unipc_adv_order` | `3` | multistep order, clamped to `steps - 1` |
| `unipc_adv_lower_order_final` | on | stock behaviour |
| `unipc_adv_sigma_mode` | `scheduler` | `scheduler` == stock (use the sampler's sigmas) |
| `unipc_adv_denoise_to_zero` | off | experimental, +1 model eval |
| `unipc_adv_discard_next_to_last_sigma` | on | stock `UniPC` behaviour |
| `unipc_adv_log` | on | print the effective parameters to the console |

All of them are read at sampling time, so **no restart is needed** after a
change - except `order`, `sigma_mode`, `denoise_to_zero`, which are read when
the sampler runs anyway.

## How to confirm the parameters really arrive

Three independent channels:

1. **Console** - one line per sampling call:
   `[UniPC Adv] sampler=UniPC Adv bh2 variant=bh2 order=3(req 3) lower_order_final=True sigma_mode=scheduler denoise_to_zero=False discard_next_to_last_sigma=True steps=26 sigma=14.6000->0.0010 latent=[1,4,96,96]`
2. **PNG infotext** - six `UniPC Adv ...` keys are injected through a guarded
   monkeypatch of `KDiffusionSampler.add_infotext`. Drop an image into
   `PNG Info` and you will see them.
3. **HTTP** - `http://127.0.0.1:7860/unipc_adv/last` returns the registered
   samplers, the currently resolved settings and the parameters of the most
   recent call. `/unipc_adv/selfcheck` is a liveness probe.

## What the variants actually do (short version)

`variant` has only two code paths: `'bh' in variant` -> `multistep_uni_pc_bh_update`,
else `vary_coeff` -> `multistep_uni_pc_vary_update`.

`bh1` and `bh2` differ in exactly one line:

```python
if   variant == 'bh1': B_h = hh              # B(h) = h, first-order truncation
elif variant == 'bh2': B_h = torch.expm1(hh) # B(h) = e^h - 1, exact
```

`B_h` cancels algebraically everywhere the coefficients come from
`linalg.solve` (it sits in the denominator of `b` and is multiplied back
outside). It only survives in the two places that hard-code `rho = 0.5`:

* `uni_pc.py:441-444` - the predictor when `order == 2`
* `uni_pc.py:449-450` - the corrector when `order == 1`

Both of those fire **only during the first two updates**, i.e. in the high-sigma
composition regime, and only ~2 of N steps are affected. `bh1`'s first-step
correction is larger by `h/(e^h-1)`:

| steps | 16 | 20 | 24 | 26 | 30 | 40 |
|---|---|---|---|---|---|---|
| bh1 larger by | +15.1% | +12.0% | +9.9% | +9.1% | +7.9% | +5.9% |

So the honest expectation is *a small difference in early composition, not a
different sampler*. With CFG > 1 the x0 prediction is extrapolated, which
magnifies that early displacement - that is the plausible mechanism behind the
"bh2 is stronger with a prompt" folklore, but it is a mechanism, not a
measurement.

`vary_coeff` is a different beast: at `order == 3` the weight of the freshly
evaluated point is `-0.08357` for bh1/bh2 but only `-0.00946` for `vary_coeff`,
so it effectively throws that point away and degrades towards a lower order
method. In a synthetic ODE with a known solution it does not converge at all
(1e-2 error floor at 100 steps). Expect it to look *worse*, not different.

## How to test

Recommended protocol, in this order:

1. **Baseline gate.** Pick one seed, 26 steps, and generate with stock `UniPC`
   and with `UniPC Adv bh1`. They must be pixel-identical. If they are not, the
   extension is not being picked up the way you think and nothing else is worth
   testing yet.
2. **SDXL** (standard eps-pred): `bh1` vs `bh2` at 20 / 26 / 40 steps, CFG 5-7.
   The gap should shrink as steps grow, per the table above.
3. **SDXL v-pred**: same, but note that UniPC's `SigmaConvert` assumes the
   VE(k-diffusion) convention. Stock UniPC already has this limitation - if
   v-pred looks broken it is very likely broken identically for stock UniPC,
   which check 1 will tell you.
4. **Anima** (flow matching): UniPC works through the `lambda = -log(sigma)`
   identity, which is exact for the flow parameterisation, so it should be
   well-behaved. Worth trying `sigma_mode = logSNR` here since flow models are
   trained on a log-SNR schedule.
5. **Knobs**: `order` 1 vs 3 vs 4 (4 is outside what the paper validated, it
   runs but treat it as experimental), `lower_order_final` on/off,
   `discard_next_to_last_sigma` on/off.


## Install / remove

Already installed at `extensions/uni-pc-advanced/`. No dependencies beyond what
forge ships. To remove, delete the folder.

## Files

* `lib_unipc_adv.py` - pure torch logic, imports nothing from the WebUI, unit
  testable standalone.
* `scripts/uni_pc_advanced.py` - registration, settings page, infotext hook,
  HTTP endpoints.
* `README.md` - this file.
