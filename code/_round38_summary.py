# -*- coding: utf-8 -*-
"""Print a compact Round 38 summary from the generated result JSONs."""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    p = os.path.join(BASE, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    print("== #7 full routing regularizer ==")
    d = load("sol4_routing_fullreg_byfamily.json")
    if d:
        for k, v in d["by_family"].items():
            print(k, "none", v["none"]["delta_sample"], "fullreg", v["full_regularzier"]["delta_sample"],
                  "ded", v["none"]["systems_dedicated"], "->", v["full_regularzier"]["systems_dedicated"])
    else:
        print("missing")

    print("\n== #2 capacity sweep ==")
    d = load("c2_capacity_sweep_k8.json")
    if d:
        for k, v in d.items():
            print(k, "params", v["params"], "meanVRMSE", round(v["mean_vrmse"], 4),
                  "experts", v["experts_used_majority"])

    print("\n== #3 tokenizer 20-seed ==")
    d = load("c3_tokenizer_ablation_20seed.json")
    if d:
        for k, v in d["results"].items():
            print(k, "mean", v["mean"], "std", v["std"], "median", v["median"])
        for k, v in d["comparisons"].items():
            print(k, v)
    else:
        print("missing")

    print("\n== #5 adapter/LoRA ==")
    d = load("c5_adapter_lora_shear_to_turb.json")
    if d:
        for k, v in d["results"].items():
            print(k, "mean", v["mean"], "std", v["std"], "vals", v["vals"])
        print("params", d.get("trainable_params"))
    else:
        print("missing")

    print("\n== #8 spectral breadth ==")
    d = load("spectral_breadth_8systems.json")
    if d:
        for k, v in d.items():
            print(k, "alpha_asym", v["alpha_asym_median"], "[", v["alpha_asym_min"], ",", v["alpha_asym_max"], "]",
                  "full", v["alpha_full_median"], "r2", v["r2_asym_median"])

    print("\n== #3/#4 power/MDE ==")
    d = load("c3_c4_power_mde.json")
    if d:
        print(json.dumps(d, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
