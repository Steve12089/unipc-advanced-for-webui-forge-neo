"""UniPC Advanced - expose the knobs that forge's UniPC port hard-codes.

forge registers exactly one UniPC sampler ("UniPC") and bakes in
``variant="bh1"``, ``lower_order_final=True``, ``order=min(3, steps-1)``,
``discard_next_to_last_sigma=True`` while ``skip_type`` and
``denoise_to_zero`` are dead arguments in the port.  This extension registers
parametrised clones so the variants can be A/B tested on SDXL, SDXL v-pred and
Anima without editing core files.

Registered samplers:
    UniPC Adv        follows the settings page (variant comes from settings)
    UniPC Adv bh1    variant fixed to bh1
    UniPC Adv bh2    variant fixed to bh2
    UniPC Adv vary   variant fixed to vary_coeff

Everything else (order, lower_order_final, sigma grid, denoise_to_zero,
discard_next_to_last_sigma) always comes from the settings page and can be
changed without restarting the WebUI.

Open ``/unipc_adv/last`` in the browser after a generation to confirm that the
parameters really reached the sampler; they are also written into the PNG
infotext and printed to the console.
"""

import os
import sys

import gradio as gr

from modules import script_callbacks, sd_samplers, sd_samplers_kdiffusion, shared
from modules.sd_samplers_common import SamplerData
from modules.uni_pc import uni_pc as unipc_impl

EXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXT_ROOT not in sys.path:
    sys.path.insert(0, EXT_ROOT)

import lib_unipc_adv as lib  # noqa: E402  (needs the sys.path tweak above)

SECTION = ("unipc_adv", "UniPC Advanced")

#: record of the most recent sampling call (for /unipc_adv/last)
LAST_CALL = {}
#: function object -> display name
FUNCS = {}


def _read_option(name, default):
    return getattr(shared.opts, name, default)


class _LiveOptions(dict):
    """``SamplerData.options`` with a live view of the settings page.

    ``KDiffusionSampler.get_sigmas`` reads ``self.config.options`` at sampling
    time, so a plain dict would require a restart after every toggle.
    """

    def get(self, key, default=None):
        if key == "discard_next_to_last_sigma":
            return bool(_read_option("unipc_adv_discard_next_to_last_sigma", True))
        return dict.get(self, key, default)


def _make_sampler(fixed_variant, display_name):
    """Build a sampler callable with the exact signature forge looks for.

    ``KDiffusionSampler`` inspects ``inspect.signature(self.func)`` and only
    forwards ``sigmas`` when the parameter exists, so the signature matters.
    """

    def _sample(model, x, sigmas, extra_args=None, callback=None, disable=False):
        params = lib.resolve_params(_read_option, fixed_variant)
        x, info = lib.run_unipc(
            unipc_impl, params, model, x, sigmas, extra_args, callback, disable
        )
        info["sampler"] = display_name
        LAST_CALL.clear()
        LAST_CALL.update(info)
        if params["log"]:
            print(lib.describe(info), flush=True)
        return x

    slug = display_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    _sample.__name__ = "sample_" + slug
    _sample.__doc__ = "UniPC Advanced ({} variant)".format(fixed_variant or "settings")
    FUNCS[_sample] = display_name
    return _sample


# (display name, forced variant or None to follow the settings page, aliases)
SAMPLERS = [
    ("UniPC Adv", None, ["uniadv", "unipc_adv", "uni_pc_adv"]),
    ("UniPC Adv bh1", "bh1", ["uniadv_bh1", "unipc_adv_bh1"]),
    ("UniPC Adv bh2", "bh2", ["uniadv_bh2", "unipc_adv_bh2"]),
    ("UniPC Adv vary", "vary_coeff", ["uniadv_vary", "unipc_adv_vary"]),
]


def _register_samplers():
    registered = []
    for display_name, fixed_variant, aliases in SAMPLERS:
        func = _make_sampler(fixed_variant, display_name)
        data = SamplerData(
            name=display_name,
            constructor=(lambda model, _f=func: sd_samplers_kdiffusion.KDiffusionSampler(_f, model)),
            aliases=list(aliases),
            options=_LiveOptions(discard_next_to_last_sigma=True),
        )
        try:
            sd_samplers.add_sampler(data)
        except Exception as exc:  # keep the sampler even if set_samplers() chokes
            print(f"[UniPC Adv] add_sampler({display_name!r}) raised {exc!r}; appending manually")
            if data.name not in [s.name for s in sd_samplers.all_samplers]:
                sd_samplers.all_samplers.append(data)
                sd_samplers.all_samplers_map = {s.name: s for s in sd_samplers.all_samplers}
        registered.append(display_name)
    print("[UniPC Adv] registered samplers: " + ", ".join(registered), flush=True)


def _install_infotext_hook():
    cls = sd_samplers_kdiffusion.KDiffusionSampler
    if getattr(cls.add_infotext, "_unipc_adv", False):
        return

    original = cls.add_infotext

    def add_infotext(self, p):
        original(self, p)
        if not LAST_CALL or self.funcname not in FUNCS:
            return
        try:
            p.extra_generation_params.update(lib.infotext_pairs(LAST_CALL))
        except Exception:
            pass

    add_infotext._unipc_adv = True
    cls.add_infotext = add_infotext


def on_ui_settings():
    section = SECTION
    shared.opts.add_option(
        "unipc_adv_variant",
        shared.OptionInfo(
            lib.DEFAULTS["variant"],
            "UniPC Adv: variant (used by the plain 'UniPC Adv' sampler)",
            gr.Radio,
            {"choices": list(lib.VARIANTS)},
            section=section,
        ),
    )
    shared.opts.add_option(
        "unipc_adv_order",
        shared.OptionInfo(
            lib.DEFAULTS["order"],
            "UniPC Adv: multistep order (stock UniPC uses min(3, steps-1))",
            gr.Slider,
            {"minimum": lib.ORDER_MIN, "maximum": lib.ORDER_MAX, "step": 1},
            section=section,
        ),
    )
    shared.opts.add_option(
        "unipc_adv_lower_order_final",
        shared.OptionInfo(
            lib.DEFAULTS["lower_order_final"],
            "UniPC Adv: lower the order on the final steps",
            gr.Checkbox,
            section=section,
        ),
    )
    shared.opts.add_option(
        "unipc_adv_sigma_mode",
        shared.OptionInfo(
            lib.DEFAULTS["sigma_mode"],
            "UniPC Adv: sigma grid (A1111's uni_pc_skip_type; 'scheduler' == stock behaviour)",
            gr.Radio,
            {"choices": list(lib.SIGMA_MODES)},
            section=section,
        ),
    )
    shared.opts.add_option(
        "unipc_adv_denoise_to_zero",
        shared.OptionInfo(
            lib.DEFAULTS["denoise_to_zero"],
            "UniPC Adv: denoise_to_zero (dead flag in stock forge; costs one extra model eval)",
            gr.Checkbox,
            section=section,
        ),
    )
    shared.opts.add_option(
        "unipc_adv_discard_next_to_last_sigma",
        shared.OptionInfo(
            lib.DEFAULTS["discard_next_to_last_sigma"],
            "UniPC Adv: discard the next-to-last sigma (stock 'UniPC' behaviour)",
            gr.Checkbox,
            section=section,
        ),
    )
    shared.opts.add_option(
        "unipc_adv_log",
        shared.OptionInfo(
            lib.DEFAULTS["log"],
            "UniPC Adv: print the effective parameters to the console",
            gr.Checkbox,
            section=section,
        ),
    )


def on_app_started(demo, app):
    try:

        @app.get("/unipc_adv/last")
        def unipc_adv_last():
            return {
                "registered": list(FUNCS.values()),
                "settings": lib.resolve_params(_read_option),
                "last_call": dict(LAST_CALL),
            }

        @app.get("/unipc_adv/selfcheck")
        def unipc_adv_selfcheck():
            return {"ok": bool(FUNCS), "samplers": list(FUNCS.values())}

        print("[UniPC Adv] API: /unipc_adv/last , /unipc_adv/selfcheck", flush=True)
    except Exception as exc:
        print(f"[UniPC Adv] failed to register API routes: {exc!r}")


_register_samplers()
_install_infotext_hook()
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_app_started(on_app_started)