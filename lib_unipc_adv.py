"""Pure-logic core for the "UniPC Adv" sampler (forge / A1111 UniPC fork).

This module deliberately imports nothing from the WebUI, so it can be unit
tested (and even driven) outside of a running forge instance.

The sampling math mirrors ``modules.sd_samplers_extra.sample_unipc`` so that
"UniPC Adv" with default settings is bit-comparable with the stock "UniPC"
sampler. Everything the stock implementation hard-codes is a parameter here:

    variant                     bh1 | bh2 | vary_coeff
    order                       1..4   (stock: min(3, steps-1))
    lower_order_final           bool   (stock: True)
    sigma_mode                  scheduler | time_uniform | time_quadratic | logSNR
                                       (A1111's uni_pc_skip_type; forge's port
                                       feeds the scheduler sigmas straight in,
                                       which is what "scheduler" reproduces)
    denoise_to_zero             bool   (dead argument in both A1111 and forge)
    discard_next_to_last_sigma  bool   (stock "UniPC" registers it as True)
"""

from __future__ import annotations

import math

import torch

VARIANTS = ("bh1", "bh2", "vary_coeff")
SIGMA_MODES = ("scheduler", "time_uniform", "time_quadratic", "logSNR")

ORDER_MIN = 1
ORDER_MAX = 4

DEFAULTS = {
    "variant": "bh1",
    "order": 3,
    "lower_order_final": True,
    "sigma_mode": "scheduler",
    "denoise_to_zero": False,
    "discard_next_to_last_sigma": True,
    "log": True,
}

#: maps our internal key -> the gradio option name registered by the extension
OPTION_NAMES = {
    "variant": "unipc_adv_variant",
    "order": "unipc_adv_order",
    "lower_order_final": "unipc_adv_lower_order_final",
    "sigma_mode": "unipc_adv_sigma_mode",
    "denoise_to_zero": "unipc_adv_denoise_to_zero",
    "discard_next_to_last_sigma": "unipc_adv_discard_next_to_last_sigma",
    "log": "unipc_adv_log",
}


class SigmaConvert:
    """VE(sigma) <-> VP(alpha, sigma) conversion, identical to forge's copy.

    ``marginal_log_mean_coeff(s) = 0.5*log(1/(1+s^2))`` gives
    ``alpha = 1/sqrt(1+s^2)`` and ``sigma_vp = s/sqrt(1+s^2)``, i.e.
    ``lambda = log(alpha/sigma_vp) = -log(s)`` which is exactly the
    k-diffusion VE lambda the rest of forge uses.
    """

    schedule = ""

    def marginal_log_mean_coeff(self, sigma):
        return 0.5 * torch.log(1 / ((sigma * sigma) + 1))

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std


def predict_eps_sigma(model, input, sigma_in, **kwargs):
    sigma = sigma_in.view(sigma_in.shape[:1] + (1,) * (input.ndim - 1))
    input = input * ((sigma ** 2 + 1.0) ** 0.5)
    return (input - model(input, sigma_in, **kwargs)) / sigma


# --------------------------------------------------------------------------- #
# parameter plumbing
# --------------------------------------------------------------------------- #
def _as_bool(value, default):
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _as_int(value, default, lo, hi):
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        number = int(default)
    return max(lo, min(hi, number))


def _as_choice(value, choices, default):
    try:
        value = str(value or "").strip()
    except Exception:
        value = ""
    return value if value in choices else default


def resolve_params(read_option, fixed_variant=None):
    """Collect the effective settings.

    ``read_option(name, default)`` is any callable; the extension passes
    ``lambda name, default: getattr(shared.opts, name, default)`` so that this
    module stays importable without the WebUI.
    """

    def read(key):
        return read_option(OPTION_NAMES[key], DEFAULTS[key])

    variant = fixed_variant if fixed_variant else read("variant")
    return {
        "variant": _as_choice(variant, VARIANTS, DEFAULTS["variant"]),
        "order": _as_int(read("order"), DEFAULTS["order"], ORDER_MIN, ORDER_MAX),
        "lower_order_final": _as_bool(read("lower_order_final"), DEFAULTS["lower_order_final"]),
        "sigma_mode": _as_choice(read("sigma_mode"), SIGMA_MODES, DEFAULTS["sigma_mode"]),
        "denoise_to_zero": _as_bool(read("denoise_to_zero"), DEFAULTS["denoise_to_zero"]),
        "discard_next_to_last_sigma": _as_bool(
            read("discard_next_to_last_sigma"), DEFAULTS["discard_next_to_last_sigma"]
        ),
        "log": _as_bool(read("log"), DEFAULTS["log"]),
    }


# --------------------------------------------------------------------------- #
# sigma grids
# --------------------------------------------------------------------------- #
def build_timesteps(sigmas, mode):
    """Turn the scheduler sigmas into the ``timesteps`` array UniPC consumes.

    ``scheduler`` (default) reproduces the stock forge behaviour exactly,
    including the ``0 -> 0.001`` clamp that ``sample_unipc`` performs.
    The other modes regenerate an evenly spaced grid, which is what A1111's
    ``uni_pc_skip_type`` used to do (in forge's port ``skip_type`` is dead
    because ``UniPC.sample`` indexes the supplied timesteps directly).
    """

    timesteps = sigmas.clone()
    if timesteps[-1] < 0.001:
        timesteps[-1] = 0.001

    if mode not in SIGMA_MODES or mode == "scheduler":
        return timesteps

    count = int(timesteps.shape[0])
    if count < 3:
        return timesteps

    device, dtype = timesteps.device, timesteps.dtype
    sigma_max = float(timesteps[0])
    sigma_min = float(timesteps[-1])
    if sigma_min <= 0.0 or sigma_max <= sigma_min:
        return timesteps

    if mode == "time_uniform":
        grid = torch.linspace(sigma_max, sigma_min, count, device=device, dtype=dtype)
    elif mode == "time_quadratic":
        grid = torch.linspace(
            sigma_max ** 0.5, sigma_min ** 0.5, count, device=device, dtype=dtype
        ).pow(2)
    else:  # logSNR / uniform lambda, lambda = -log(sigma)
        upper = -math.log(sigma_max)
        lower = -math.log(sigma_min)
        grid = torch.exp(-torch.linspace(upper, lower, count, device=device, dtype=dtype))

    # pin the endpoints so float drift cannot nudge them
    grid[0] = sigma_max
    grid[-1] = sigma_min
    return grid


def resolve_order(configured, n_timesteps):
    """Clamp the order so ``UniPC.sample``'s asserts stay satisfied."""

    steps = int(n_timesteps) - 1
    if steps < 2:
        return 1
    return max(ORDER_MIN, min(int(configured), steps - 1, ORDER_MAX))


# --------------------------------------------------------------------------- #
# the sampler itself
# --------------------------------------------------------------------------- #
def run_unipc(unipc_impl, params, model, x, sigmas, extra_args=None, callback=None, disable=False):
    """Run one UniPC solve.

    ``unipc_impl`` is ``modules.uni_pc.uni_pc`` injected by the caller so this
    function can be exercised from a plain python interpreter too.

    Returns ``(x, info)`` where ``info`` is a JSON friendly record of what was
    actually used.
    """

    extra_args = extra_args or {}
    timesteps = build_timesteps(sigmas, params["sigma_mode"])
    ns = SigmaConvert()

    # move the latent from the "clean" scale into the VP parameterisation
    x = x / torch.sqrt(1.0 + timesteps[0] ** 2.0)

    model_fn = unipc_impl.model_wrapper(
        lambda input, sigma, **kwargs: predict_eps_sigma(model, input, sigma, **kwargs),
        ns,
        model_type="noise",
        guidance_type="uncond",
        model_kwargs=extra_args,
    )

    order = resolve_order(params["order"], timesteps.shape[0])
    uni_pc = unipc_impl.UniPC(
        model_fn, ns, predict_x0=True, thresholding=False, variant=params["variant"]
    )
    x = uni_pc.sample(
        x,
        timesteps=timesteps,
        skip_type="time_uniform",
        method="multistep",
        order=order,
        lower_order_final=params["lower_order_final"],
        callback=callback,
        disable_pbar=disable,
    )

    if params["denoise_to_zero"]:
        # one extra model evaluation; yields x0 directly so no alpha rescale
        x = uni_pc.denoise_to_zero_fn(x, timesteps[-1].reshape(1))
    else:
        x = x / ns.marginal_alpha(timesteps[-1])

    info = {
        "sampler": None,
        "variant": params["variant"],
        "order": order,
        "requested_order": params["order"],
        "lower_order_final": params["lower_order_final"],
        "sigma_mode": params["sigma_mode"],
        "denoise_to_zero": params["denoise_to_zero"],
        "discard_next_to_last_sigma": params["discard_next_to_last_sigma"],
        "steps": int(timesteps.shape[0]) - 1,
        "sigma_first": float(timesteps[0]),
        "sigma_last": float(timesteps[-1]),
        "latent": [int(v) for v in x.shape],
    }
    return x, info


def describe(info):
    return (
        "[UniPC Adv] "
        f"sampler={info.get('sampler')} "
        f"variant={info['variant']} "
        f"order={info['order']}(req {info['requested_order']}) "
        f"lower_order_final={info['lower_order_final']} "
        f"sigma_mode={info['sigma_mode']} "
        f"denoise_to_zero={info['denoise_to_zero']} "
        f"discard_next_to_last_sigma={info['discard_next_to_last_sigma']} "
        f"steps={info['steps']} "
        f"sigma={info['sigma_first']:.4f}->{info['sigma_last']:.4f} "
        f"latent={info['latent']}"
    )


def infotext_pairs(info):
    return {
        "UniPC Adv variant": info["variant"],
        "UniPC Adv order": info["order"],
        "UniPC Adv lower order final": info["lower_order_final"],
        "UniPC Adv sigma mode": info["sigma_mode"],
        "UniPC Adv denoise to zero": info["denoise_to_zero"],
        "UniPC Adv discard next to last sigma": info["discard_next_to_last_sigma"],
    }