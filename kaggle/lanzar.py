"""Lanza un lote de corridas en un kernel de Kaggle (CPU, 4 nucleos, hasta 12 h).

El kernel clona este repo (publico) en el commit indicado, corre los trabajos en paralelo (un hilo
cada uno) y deja resultados/ y logs/ como salida del kernel.

    python kaggle/lanzar.py --lote humo                 # prueba corta de todo
    python kaggle/lanzar.py --lote validacion           # Wagner N=19, notebook vs wagner_fit
    python kaggle/lanzar.py --lote enfoque1
    python kaggle/lanzar.py --lote enfoque2
    python kaggle/bajar.py  --lote enfoque1             # trae la salida a resultados/kaggle/<lote>/

Requiere la CLI de Kaggle (python -m kaggle.cli) autenticada. El commit tiene que estar en GitHub.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
REPO = "https://github.com/B3njjaman/rl-conjetura-stanley.git"
USUARIO = "benjacampos"

H = 3600
V = "scripts/01_wagner_conj21/correr.py"


def _extra(control):
    # control = el mismo algoritmo con lr = 0: la red no aprende (politica inicial fija)
    return (" --set lr=0", "_control") if control else ("", "")


def e1(N, s, iters, tmax, control=False):
    x, suf = _extra(control)
    return (f"e1_N{N}_s{s}{suf}", f"python -m stanley_rl.correr e1 --N {N} --seed {s} --iters {iters} "
                                  f"--tiempo-max {tmax} --salida resultados/02_enfoque1/N{N}_s{s}{suf}{x}")


def e2(N, k, s, iters, tmax, control=False):
    x, suf = _extra(control)
    return (f"e2_N{N}_k{k}_s{s}{suf}", f"python -m stanley_rl.correr e2 --N {N} --k {k} --seed {s} --iters {iters} "
                                       f"--tiempo-max {tmax} --salida resultados/03_enfoque2/N{N}_k{k}_s{s}{suf}{x}")


def val(modo, s, iters, tmax, pkl=1000):
    return (f"wagner_{modo}_s{s}", f"python {V} --modo {modo} --seed {s} --iters {iters} --tiempo-max {tmax} "
                                   f"--hilos 1 --guardar-cada 20 --pkl-cada {pkl}")


LOTES = {
    "humo": [val("notebook", 0, 4, 600), val("wagner_fit", 0, 3, 600),
             e1(5, 0, 15, 600), e1(7, 0, 3, 600),
             e2(12, 3, 0, 5, 600), e2(24, 4, 0, 3, 600)],
    "validacion": [val("wagner_fit", s, 30000, 10.4 * H) for s in (0, 1, 2)] +
                  [val("notebook", s, 3000, 3.3 * H) for s in (0, 1, 2)],
    "enfoque1": [e1(5, s, 300, 1 * H) for s in (0, 1, 2)] +
                [e1(6, s, 1500, 2 * H) for s in (0, 1, 2)] +
                [e1(7, s, 5000, 3 * H) for s in (0, 1, 2)] +
                [e1(N, s, it, 2 * H, control=True) for N, it in ((6, 1500), (7, 5000)) for s in (0, 1, 2)],
    "enfoque2": [e2(24, 4, s, 20000, 10.4 * H) for s in (0, 1)] +
                [e2(24, 3, s, 20000, 10.4 * H) for s in (0,)] +
                [e2(12, 3, s, 2000, 2.5 * H) for s in (0, 1, 2)] +
                [e2(24, 3, 1, 20000, 4 * H, control=True)],
}

RUN_PY = r'''
import json, os, subprocess, sys, time, shutil
REPO, COMMIT, PARALELOS = {repo!r}, {commit!r}, {paralelos}
TRABAJOS = {trabajos!r}
W = "/kaggle/working"
R = "/tmp/repo"
subprocess.run(["git", "clone", "-q", REPO, R], check=True)
subprocess.run(["git", "-C", R, "checkout", "-q", COMMIT], check=True)
shutil.rmtree(f"{{R}}/resultados", ignore_errors=True)
os.makedirs(f"{{W}}/resultados", exist_ok=True)
os.makedirs(f"{{W}}/logs", exist_ok=True)
os.symlink(f"{{W}}/resultados", f"{{R}}/resultados")
import numpy, numba, torch, networkx
print("commit", COMMIT, "| cpus", os.cpu_count(), "| numpy", numpy.__version__, "| numba", numba.__version__,
      "| torch", torch.__version__, "| networkx", networkx.__version__, flush=True)
env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1", PYTHONUNBUFFERED="1")
pendientes = list(TRABAJOS)
activos = {{}}
estado = {{}}
t0 = time.time()
while pendientes or activos:
    while pendientes and len(activos) < PARALELOS:
        nombre, cmd = pendientes.pop(0)
        f = open(f"{{W}}/logs/{{nombre}}.txt", "w")
        activos[nombre] = (subprocess.Popen(cmd, shell=True, cwd=R, env=env, stdout=f, stderr=subprocess.STDOUT), f, time.time())
        print(f"[{{time.time()-t0:7.0f}}s] inicia {{nombre}}: {{cmd}}", flush=True)
    time.sleep(10)
    for nombre in list(activos):
        p, f, ti = activos[nombre]
        if p.poll() is not None:
            f.close()
            estado[nombre] = dict(codigo=p.returncode, segundos=round(time.time() - ti))
            print(f"[{{time.time()-t0:7.0f}}s] termina {{nombre}} codigo {{p.returncode}}", flush=True)
            if p.returncode != 0:
                print(open(f"{{W}}/logs/{{nombre}}.txt").read()[-3000:], flush=True)
            del activos[nombre]
    json.dump(dict(commit=COMMIT, estado=estado, pendientes=[n for n, _ in pendientes], activos=list(activos)),
              open(f"{{W}}/estado.json", "w"), indent=1)
print("LISTO", json.dumps(estado), flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lote", required=True, choices=sorted(LOTES))
    ap.add_argument("--commit", default=None, help="default: HEAD (tiene que estar pusheado)")
    ap.add_argument("--paralelos", type=int, default=4)
    ap.add_argument("--solo-preparar", action="store_true")
    a = ap.parse_args()
    commit = a.commit or subprocess.check_output(["git", "-C", str(RAIZ), "rev-parse", "HEAD"], text=True).strip()
    remotos = subprocess.check_output(["git", "-C", str(RAIZ), "branch", "-r", "--contains", commit], text=True)
    if not remotos.strip():
        sys.exit(f"el commit {commit[:8]} no esta en GitHub: haz push antes de lanzar")
    slug = f"tesis-stanley-{a.lote}"
    d = AQUI / "kernels" / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(RUN_PY.format(repo=REPO, commit=commit, paralelos=a.paralelos,
                                            trabajos=LOTES[a.lote]), encoding="utf-8")
    meta = dict(id=f"{USUARIO}/{slug}", title=slug, code_file="run.py", language="python",
                kernel_type="script", is_private=True, enable_gpu=False, enable_tpu=False,
                enable_internet=True, dataset_sources=[], competition_sources=[], kernel_sources=[])
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{slug}: {len(LOTES[a.lote])} trabajos, commit {commit[:8]}")
    if not a.solo_preparar:
        subprocess.run([sys.executable, "-m", "kaggle.cli", "kernels", "push", "-p", str(d)], check=True)


if __name__ == "__main__":
    main()
