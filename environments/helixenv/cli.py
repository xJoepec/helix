from __future__ import annotations

"""
helixenv CLI/TUI wrapper

This lightweight CLI forwards to the core Helix package's CLI/TUI so that the
Helix environment (used with verifiers) can also exercise the interactive and
demo workflows listed in uses.md without duplicating logic.

Usage examples:

  # Run the Helix demo (region counts, mass consistency, CP checks, Ulam PF)
  python -m environments.helixenv.cli demo --samples 4000 --noise 0.07 --plot

  # Launch the interactive TUI (requires extras: pip install '.[tui]')
  python -m environments.helixenv.cli tui
"""

import argparse
import os
import sys
from typing import List, Optional, Dict, Any, Tuple
import json
import shutil


def _ensure_helix_on_path() -> None:
    """Ensure the repository's `code/` directory (which contains the Helix package)
    is on sys.path so we can import `helix` without requiring installation.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    code_path = os.path.join(repo_root, "code")
    if code_path not in sys.path:
        sys.path.insert(0, code_path)


def run_demo(argv: Optional[List[str]] = None) -> int:
    _ensure_helix_on_path()
    from helix.cli import main as helix_main  # type: ignore

    # Delegate directly to Helix CLI. It already understands all demo args.
    return helix_main(argv)


def run_tui(argv: Optional[List[str]] = None) -> int:
    _ensure_helix_on_path()
    try:
        from helix.tui import run_tui as helix_tui  # type: ignore
    except Exception as e:
        print(
            "Helix TUI is unavailable. Install TUI extras in this repo: \n"
            "  pip install '.[tui]'\n"
            f"Details: {e}"
        )
        return 1
    return helix_tui(argv)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="helixenv-cli",
        description="Helixenv CLI wrapper that forwards to Helix demo/TUI",
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    # interactive
    sub.add_parser("interactive", help="Interactive menu (demo/analyze/TUI)")

    # demo: pass through remaining args to helix.cli
    p_demo = sub.add_parser("demo", help="Run Helix demo (diagnostics + optional plots)")
    p_demo.add_argument("rest", nargs=argparse.REMAINDER, help="Args passed to Helix CLI")

    # tui: forward to Helix TUI
    p_tui = sub.add_parser("tui", help="Launch Helix TUI (interactive)")
    p_tui.add_argument("rest", nargs=argparse.REMAINDER, help="Args passed to Helix TUI")

    # pro: launch Pro TUI in helixenv
    sub.add_parser("pro", help="Launch Helix Pro TUI (advanced)")

    args = parser.parse_args(argv)

    # Interactive menu implementation
    # Optional nice prompts if questionary is available
    def _has_questionary() -> bool:
        try:
            import questionary  # type: ignore
            return True
        except Exception:
            return False

    def _q_select(message: str, choices: List[str], default: Optional[str] = None) -> str:
        if _has_questionary():
            import questionary  # type: ignore
            return questionary.select(message, choices=choices, default=default).ask()  # type: ignore[attr-defined]
        else:
            # Fallback simple prompt
            if default and default in choices:
                default_idx = choices.index(default) + 1
            else:
                default_idx = 1
            print(message)
            for i, c in enumerate(choices, 1):
                print(f"  {i}) {c}")
            while True:
                s = input(f"[{default_idx}]> ").strip()
                if s == "":
                    return choices[default_idx - 1]
                if s.isdigit() and 1 <= int(s) <= len(choices):
                    return choices[int(s) - 1]
                print("Please choose a valid option.")

    def _q_text(message: str, default: Optional[str] = None) -> str:
        if _has_questionary():
            import questionary  # type: ignore
            return questionary.text(message, default=default or "").ask()  # type: ignore[attr-defined]
        sfx = f" [{default}]" if default is not None else ""
        s = input(f"{message}{sfx}: ").strip()
        return default if (s == "" and default is not None) else s

    def _q_confirm(message: str, default: bool = False) -> bool:
        if _has_questionary():
            import questionary  # type: ignore
            return bool(questionary.confirm(message, default=default).ask())  # type: ignore[attr-defined]
        d = "y" if default else "n"
        while True:
            s = _q_text(f"{message} [y/n]", d).lower()
            if s in ("y", "yes"): return True
            if s in ("n", "no"): return False
            print("Please answer y or n.")

    def _q_int(message: str, default: int) -> int:
        while True:
            s = _q_text(message, str(default))
            try:
                return int(s)
            except Exception:
                print("Please enter an integer.")

    def _q_float(message: str, default: float) -> float:
        while True:
            s = _q_text(message, str(default))
            try:
                return float(s)
            except Exception:
                print("Please enter a number.")

    # Persist last-used configs
    def _config_path() -> str:
        try:
            import platformdirs  # type: ignore
            cfg_dir = platformdirs.user_config_dir("helixenv", "helix")
        except Exception:
            cfg_dir = os.path.join(os.path.expanduser("~"), ".config", "helixenv")
        os.makedirs(cfg_dir, exist_ok=True)
        return os.path.join(cfg_dir, "cli_last.json")

    def _save_cfg(d: Dict[str, Any]) -> None:
        try:
            p = _config_path()
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2)
            os.replace(tmp, p)
        except Exception:
            pass

    def _load_cfg() -> Dict[str, Any]:
        try:
            with open(_config_path(), "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _prompt_file(message: str, default: str = "", must_exist: bool = True) -> str:
        while True:
            p = _q_text(message, default)
            if not p:
                return ""
            if not must_exist or os.path.exists(p):
                return p
            print("Path not found. Try again or leave blank to skip.")

    def _run_metrics(params: Dict[str, Any]) -> Dict[str, Any]:
        """Run Helix metrics using the core runner from helix.tui for structured results."""
        _ensure_helix_on_path()
        from helix.tui import _compute_metrics  # type: ignore
        return _compute_metrics(params, progress_cb=None)

    def _aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        import numpy as _np

        def _avg_list_of_lists(key: str) -> List[float]:
            # pad to max length with NaN and average per-index ignoring NaN
            max_len = max((len(r.get(key, [])) for r in runs), default=0)
            if max_len == 0:
                return []
            mat = _np.full((len(runs), max_len), _np.nan, dtype=_np.float64)
            for i, r in enumerate(runs):
                arr = _np.array(r.get(key, []), dtype=_np.float64)
                n = min(max_len, arr.shape[0])
                mat[i, :n] = arr[:n]
            return _np.nanmean(mat, axis=0).tolist()

        def _avg_cp_stats() -> List[Dict[str, float]]:
            # assume same number of depths; average fields elementwise
            max_depth = max((len(r.get("cp_stats", [])) for r in runs), default=0)
            out: List[Dict[str, float]] = []
            for d in range(max_depth):
                acc: Dict[str, List[float]] = {}
                for r in runs:
                    lst = r.get("cp_stats", [])
                    if d < len(lst):
                        for k, v in lst[d].items():
                            acc.setdefault(k, []).append(float(v))
                out.append({k: float(sum(vs) / max(1, len(vs))) for k, vs in acc.items()})
            return out

        avg_gap = float(sum(float(r.get("ulam_spectral_gap", 0.0)) for r in runs) / max(1, len(runs)))
        return {
            "region_counts_avg": _avg_list_of_lists("region_counts"),
            "mass_errors_avg": _avg_list_of_lists("mass_errors"),
            "cp_stats_avg": _avg_cp_stats(),
            "ulam_spectral_gap_avg": avg_gap,
            "runs": runs,
        }

    def run_interactive() -> int:
        _ensure_helix_on_path()
        from helix.cli import main as helix_main  # type: ignore
        from datetime import datetime as _dt

        def _offer_export(kind: str, arg_list: List[str]) -> None:
            if not _q_confirm("Export reproducible snippet?", False):
                return
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            # Bash command
            cmd = "./helix " + (kind + " " if kind else "") + " ".join(
                json.dumps(a) if (" " in a or a.startswith("-")) else a for a in arg_list
            )
            # Python script
            py = (
                "#!/usr/bin/env python3\n"
                "import os, sys\n"
                "sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))\n"
                "from helix.cli import main\n"
                f"args = {json.dumps(arg_list)}\n"
                "raise SystemExit(main(args))\n"
            )
            bash_fn = f"helix_repro_{ts}.sh"
            py_fn = f"helix_repro_{ts}.py"
            with open(bash_fn, "w", encoding="utf-8") as f:
                f.write(cmd + "\n")
            with open(py_fn, "w", encoding="utf-8") as f:
                f.write(py)
            try:
                os.chmod(bash_fn, 0o755)
            except Exception:
                pass
            print(f"Wrote {bash_fn} and {py_fn}")

        print("Helix Interactive CLI — choose a mode\n")
        while True:
            choice = _q_select("Select an option", [
                "Demo (built-in MLP)",
                "Analyze (built-in MLP sized to data)",
                "Analyze (custom model builder)",
                "Sweeps (seeds/widths) and summary",
                "TUI (interactive UI)",
                "Pro TUI (advanced)",
                "Load last config and re-run",
                "Quit",
            ], default="Demo (built-in MLP)")
            if choice in ("Quit",):
                return 0
            if choice.startswith("Demo"):
                # Collect demo params
                samples = _q_int("samples", 4000)
                noise = _q_float("noise", 0.07)
                seed = _q_int("seed", 1)
                width = _q_int("width", 16)
                epochs = _q_int("epochs", 100)
                no_train = _q_confirm("no_train", False)
                adv = _q_confirm("Advanced settings? (Ulam/plot)", False)
                ulam_bins = _q_int("ulam_bins", 25) if adv else 25
                ulam_spc = _q_int("ulam_samples_per_cell", 1) if adv else 1
                plot = _q_confirm("plot", False) if adv else False
                no_show = _q_confirm("no_show", False) if adv else False
                save_prefix = _q_text("save_prefix", "") if adv else ""
                args = [
                    "--samples", str(samples),
                    "--noise", str(noise),
                    "--seed", str(seed),
                    "--width", str(width),
                    "--epochs", str(epochs),
                    "--ulam-bins", str(ulam_bins),
                    "--ulam-samples-per-cell", str(ulam_spc),
                ]
                if no_train: args.append("--no-train")
                if plot: args.append("--plot")
                if no_show: args.append("--no-show")
                if save_prefix: args += ["--save-prefix", save_prefix]
                _save_cfg({"mode": "demo", "args": args})
                rc = helix_main(args)
                print(f"\n[done] exit code {rc}\n")
                _offer_export("", args)
                continue
            if choice.startswith("Analyze (built-in"):
                data_x = _prompt_file("data_x (.npy/.npz/.csv) [optional]", "", must_exist=True)
                data_y = _prompt_file("data_y (.npy/.npz/.csv) [optional]", "", must_exist=True)
                d_out = _q_int("d_out", 2)
                width = _q_int("width", 16)
                epochs = _q_int("epochs", 50)
                no_train = _q_confirm("no_train", True)
                ulam_bins = _q_int("ulam_bins", 25)
                ulam_spc = _q_int("ulam_samples_per_cell", 1)
                no_ulam = _q_confirm("no_ulam", True)
                seed = _q_int("seed", 1)

                args_list: List[str] = ["analyze",
                    "--d-out", str(d_out),
                    "--width", str(width),
                    "--epochs", str(epochs),
                    "--ulam-bins", str(ulam_bins),
                    "--ulam-samples-per-cell", str(ulam_spc),
                    "--seed", str(seed),
                ]
                if data_x: args_list += ["--data-x", data_x]
                if data_y: args_list += ["--data-y", data_y]
                if no_train: args_list.append("--no-train")
                if no_ulam: args_list.append("--no-ulam")
                _save_cfg({"mode": "analyze_builtin", "args": args_list})
                rc = helix_main(args_list)
                print(f"\n[done] exit code {rc}\n")
                _offer_export("analyze", args_list[1:])
                continue
            if choice.startswith("Analyze (custom model"):
                model_module = _prompt_file("model_module (.py)", must_exist=True)
                model_func = _q_text("model_func", "build_model")
                model_kwargs = _q_text("model_kwargs (JSON)", "")
                if model_kwargs:
                    try:
                        json.loads(model_kwargs)
                    except Exception:
                        print("[warn] Invalid JSON. Clearing model_kwargs.")
                        model_kwargs = ""
                weights = _prompt_file("weights (.pt/.pth) [optional]", "", must_exist=True)
                data_x = _prompt_file("data_x (.npy/.npz/.csv) [optional]", "", must_exist=True)
                data_y = _prompt_file("data_y (.npy/.npz/.csv) [optional]", "", must_exist=True)
                epochs = _q_int("epochs (if training)", 50)
                no_train = _q_confirm("no_train", True)
                ulam_bins = _q_int("ulam_bins", 25)
                ulam_spc = _q_int("ulam_samples_per_cell", 1)
                no_ulam = _q_confirm("no_ulam", True)
                seed = _q_int("seed", 1)

                args_list: List[str] = ["analyze",
                    "--model-module", model_module,
                    "--model-func", model_func,
                    "--epochs", str(epochs),
                    "--ulam-bins", str(ulam_bins),
                    "--ulam-samples-per-cell", str(ulam_spc),
                    "--seed", str(seed),
                ]
                if model_kwargs: args_list += ["--model-kwargs", model_kwargs]
                if weights: args_list += ["--weights", weights]
                if data_x: args_list += ["--data-x", data_x]
                if data_y: args_list += ["--data-y", data_y]
                if no_train: args_list.append("--no-train")
                if no_ulam: args_list.append("--no-ulam")
                _save_cfg({"mode": "analyze_custom", "args": args_list})
                rc = helix_main(args_list)
                print(f"\n[done] exit code {rc}\n")
                _offer_export("analyze", args_list[1:])
                continue
            if choice.startswith("Sweeps"):
                # Quick sweeps across seeds and/or widths with summary + export JSON
                print("Configure sweep (blank to use defaults)")
                use_seeds = _q_confirm("Sweep over seeds?", True)
                use_widths = _q_confirm("Sweep over widths?", False)
                base_samples = _q_int("base samples", 4000)
                base_noise = _q_float("base noise", 0.07)
                base_epochs = _q_int("base epochs", 50)
                no_train = _q_confirm("no_train", True)
                ulam_bins = _q_int("ulam_bins", 25)
                ulam_spc = _q_int("ulam_samples_per_cell", 1)
                # seeds list
                seeds: List[int] = [1, 2, 3] if use_seeds else [1]
                if use_seeds:
                    s_str = _q_text("seed list (comma-separated)", "1,2,3")
                    try:
                        seeds = [int(x.strip()) for x in s_str.split(",") if x.strip()]
                    except Exception:
                        print("[warn] Invalid seeds; using 1,2,3")
                        seeds = [1, 2, 3]
                # widths list
                widths: List[int] = [16] if not use_widths else [8, 16, 32]
                if use_widths:
                    w_str = _q_text("width list (comma-separated)", "8,16,32")
                    try:
                        widths = [int(x.strip()) for x in w_str.split(",") if x.strip()]
                    except Exception:
                        print("[warn] Invalid widths; using 8,16,32")
                        widths = [8, 16, 32]

                runs: List[Dict[str, Any]] = []
                total = len(seeds) * len(widths)
                print(f"Running {total} jobs…\n")
                for s in seeds:
                    for w in widths:
                        params = {
                            "samples": base_samples,
                            "noise": base_noise,
                            "seed": s,
                            "width": w,
                            "epochs": base_epochs,
                            "no_train": no_train,
                            "ulam_bins": ulam_bins,
                            "ulam_samples_per_cell": ulam_spc,
                        }
                        try:
                            m = _run_metrics(params)
                            m["_config"] = params
                            runs.append(m)
                            print(f"seed={s} width={w}: ok")
                        except Exception as e:
                            print(f"seed={s} width={w}: ERROR {e}")

                summary = _aggregate_runs(runs)
                # Pretty print summary
                print("\nSweep summary:")
                print("  region_counts_avg:", summary.get("region_counts_avg"))
                print("  mass_errors_avg:", summary.get("mass_errors_avg"))
                print("  ulam_spectral_gap_avg:", f"{summary.get('ulam_spectral_gap_avg', 0.0):.4f}")

                # Export JSON
                if _q_confirm("Export results to JSON?", True):
                    from datetime import datetime as _dt
                    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
                    out = {
                        "summary": summary,
                        "profile": {
                            "seeds": seeds,
                            "widths": widths,
                            "base": {
                                "samples": base_samples,
                                "noise": base_noise,
                                "epochs": base_epochs,
                                "no_train": no_train,
                                "ulam_bins": ulam_bins,
                                "ulam_samples_per_cell": ulam_spc,
                            },
                        },
                    }
                    fn = f"helix_sweep_{ts}.json"
                    with open(fn, "w", encoding="utf-8") as f:
                        json.dump(out, f, indent=2)
                    print(f"Saved {fn}")
                continue
            if choice.startswith("TUI (interactive"):
                return run_tui([])
            if choice.startswith("Pro TUI"):
                # Launch environments.helixenv.ui.run_pro_tui with robust import when run as script
                try:
                    from .ui import run_pro_tui  # type: ignore
                except Exception:
                    try:
                        here = os.path.dirname(os.path.abspath(__file__))
                        ui_path = os.path.join(here, "ui.py")
                        import importlib.util as _ilu
                        spec = _ilu.spec_from_file_location("helixenv_ui", ui_path)
                        if spec and spec.loader:
                            mod = _ilu.module_from_spec(spec)
                            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                            run_pro_tui = getattr(mod, "run_pro_tui")  # type: ignore[no-redef]
                        else:
                            raise ImportError("spec/loader not available for ui.py")
                    except Exception as e:
                        print(f"Pro TUI unavailable: {e}")
                        return 1
                return run_pro_tui([])
            if choice.startswith("Load last"):
                cfg = _load_cfg()
                mode = cfg.get("mode")
                args_list = cfg.get("args") or []
                if not mode or not args_list:
                    print("No saved config found.")
                    continue
                print(f"Re-running last config: {mode}\n{args_list}")
                rc = helix_main(args_list if isinstance(args_list, list) else [])
                print(f"\n[done] exit code {rc}\n")
                continue

    # If explicitly requested
    if args.cmd == "interactive":
        return run_interactive()

    # Default behavior: if no subcommand and in a TTY, go interactive; otherwise demo
    if args.cmd is None:
        try:
            if sys.stdin.isatty():
                return run_interactive()
        except Exception:
            pass
        # Fallback to demo
        return run_demo(None)

    # Default to demo when explicitly chosen
    if args.cmd == "demo":
        rest = getattr(args, "rest", None)
        return run_demo(rest)
    if args.cmd == "tui":
        return run_tui(getattr(args, "rest", None))
    if args.cmd == "pro":
        try:
            from .ui import run_pro_tui  # type: ignore
        except Exception:
            try:
                here = os.path.dirname(os.path.abspath(__file__))
                ui_path = os.path.join(here, "ui.py")
                import importlib.util as _ilu
                spec = _ilu.spec_from_file_location("helixenv_ui", ui_path)
                if spec and spec.loader:
                    mod = _ilu.module_from_spec(spec)
                    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                    run_pro_tui = getattr(mod, "run_pro_tui")  # type: ignore[no-redef]
                else:
                    raise ImportError("spec/loader not available for ui.py")
            except Exception as e:
                print(f"Pro TUI unavailable: {e}")
                return 1
        return run_pro_tui([])

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
