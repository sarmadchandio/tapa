"""Environment checks.

Most TAPA failures in the wild have been missing external programs rather than
bugs in the pipeline: MFA unable to find its OpenFst helpers, Dr.VOT unable to
find sox or a working Praat. Each surfaced minutes into a run, as a confusing
traceback or — worse — as a silent fallback to a cruder method. This module
checks the tools a given configuration needs, up front.

    python -m tapa.environment                 # check the default setup
    python -m tapa.environment --vot-backend drvot --drvot-repo /content/Dr.VOT
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

OK, MISSING, BROKEN = "ok", "missing", "broken"


def _run(cmd, timeout=30):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        return -1, str(e)


def check_ffmpeg():
    path = shutil.which("ffmpeg")
    if not path:
        return {"status": MISSING, "detail": "not on PATH",
                "fix": "apt-get install -y ffmpeg"}
    rc, out = _run([path, "-version"])
    if rc != 0:
        return {"status": BROKEN, "detail": out.strip()[:200], "fix": "reinstall ffmpeg"}
    return {"status": OK, "detail": out.splitlines()[0][:80], "path": path}


def check_praat():
    path = shutil.which("praat")
    if not path:
        return {"status": MISSING, "detail": "not on PATH",
                "fix": "apt-get install -y praat"}
    script = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".praat", delete=False) as fh:
            fh.write('writeInfoLine: "tapa_ok"\n')
            script = fh.name
        rc, out = _run([path, "--run", script])
    finally:
        if script:
            try:
                os.unlink(script)
            except OSError:
                pass
    if rc != 0 or "tapa_ok" not in out:
        fix = "apt-get install -y praat"
        if "libgtk" in out or "shared object" in out:
            fix = "apt-get install -y libgtk2.0-0 libglib2.0-0 libxtst6"
        elif "display" in out.lower():
            fix = "run under xvfb-run, or install a headless-capable Praat"
        return {"status": BROKEN, "detail": out.strip()[:200], "fix": fix}
    return {"status": OK, "detail": "runs headlessly", "path": path}


def check_sox():
    path = shutil.which("sox")
    if not path:
        return {"status": MISSING,
                "detail": "Dr.VOT's feature extraction shells out to sox",
                "fix": "apt-get install -y sox"}
    rc, out = _run([path, "--version"])
    if rc != 0:
        return {"status": BROKEN, "detail": out.strip()[:200], "fix": "reinstall sox"}
    return {"status": OK, "detail": out.strip().splitlines()[0][:80], "path": path}


def check_mfa(mfa_bin=None):
    path = mfa_bin if (mfa_bin and Path(mfa_bin).exists()) else shutil.which("mfa")
    if not path:
        for cand in ("/opt/miniforge/bin/mfa", "/usr/local/bin/mfa"):
            if Path(cand).exists():
                path = cand
                break
    if not path:
        return {"status": MISSING,
                "detail": "no MFA binary; alignment falls back to CMUdict",
                "fix": "conda install -c conda-forge montreal-forced-aligner"}
    # MFA calls OpenFst/Kaldi helpers by bare name; they live beside the binary.
    env = os.environ.copy()
    env["PATH"] = str(Path(path).resolve().parent) + os.pathsep + env.get("PATH", "")
    missing = [b for b in ("fstcompile", "fstarcsort", "compile-train-graphs")
               if shutil.which(b, path=env["PATH"]) is None]
    rc, out = _run([path, "version"])
    if rc != 0:
        return {"status": BROKEN, "detail": out.strip()[:200],
                "fix": "reinstall montreal-forced-aligner"}
    if missing:
        return {"status": BROKEN,
                "detail": f"helpers not found next to the binary: {', '.join(missing)}",
                "fix": "install openfst/kaldi into the same environment as mfa"}
    return {"status": OK, "detail": f"{out.strip()[:40]}, helpers present", "path": path}


def check_drvot(repo_dir=None):
    if not repo_dir:
        return {"status": MISSING, "detail": "drvot_repo_dir not set",
                "fix": "python -m tapa.drvot setup /content/Dr.VOT"}
    repo = Path(repo_dir)
    weights = repo / "final_models" / "adv_model.model"
    if not repo.exists():
        return {"status": MISSING, "detail": f"{repo} does not exist",
                "fix": f"python -m tapa.drvot setup {repo}"}
    if not weights.exists():
        return {"status": BROKEN, "detail": f"missing {weights.name}",
                "fix": f"python -m tapa.drvot setup {repo} --force"}
    pipeline_py = repo / "process_data_pipeline.py"
    if pipeline_py.exists():
        text = pipeline_py.read_text()
        if 'os.path.join(os.getcwd(),"linux_praat")' in text:
            return {"status": BROKEN,
                    "detail": "clone still points at the bundled GUI Praat binary",
                    "fix": f"python -m tapa.drvot setup {repo}  (re-patches in place)"}
    return {"status": OK, "detail": "weights present, Praat calls patched"}


def check_environment(cfg=None, vot_backend=None, drvot_repo=None, mfa_bin=None):
    """Check every tool the given configuration needs. Returns {name: result}."""
    if cfg is not None:
        vot_backend = vot_backend or getattr(cfg, "vot_backend", "tapa")
        drvot_repo = drvot_repo or getattr(cfg, "drvot_repo_dir", None)
        mfa_bin = mfa_bin or getattr(cfg, "mfa_bin", None)
    checks = {"ffmpeg": check_ffmpeg(), "mfa": check_mfa(mfa_bin)}
    if (vot_backend or "tapa") == "drvot":
        checks["praat"] = check_praat()
        checks["sox"] = check_sox()
        checks["drvot"] = check_drvot(drvot_repo)
    return checks


#: checks whose failure stops the pipeline outright, rather than degrading it
REQUIRED = {"ffmpeg"}


def report(checks, printer=print):
    """Print a readable summary. Returns True when nothing required is broken."""
    width = max(len(k) for k in checks)
    ok = True
    for name, r in checks.items():
        mark = {OK: "OK  ", MISSING: "-- ", BROKEN: "!! "}[r["status"]]
        printer(f"  {mark} {name:<{width}}  {r['detail']}")
        if r["status"] != OK:
            printer(f"       {'':<{width}}  fix: {r['fix']}")
            if name in REQUIRED:
                ok = False
    degraded = [n for n, r in checks.items() if r["status"] != OK and n not in REQUIRED]
    if degraded:
        printer(f"  note: {', '.join(degraded)} unavailable — the affected stage "
                f"will fall back to a cruder method or fail.")
    return ok


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="python -m tapa.environment",
                                description="Check TAPA's external dependencies.")
    p.add_argument("--vot-backend", default="tapa", choices=["tapa", "drvot"])
    p.add_argument("--drvot-repo", default=None)
    p.add_argument("--mfa-bin", default=None)
    a = p.parse_args(argv)
    checks = check_environment(vot_backend=a.vot_backend, drvot_repo=a.drvot_repo,
                               mfa_bin=a.mfa_bin)
    print("TAPA environment check")
    ok = report(checks)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
