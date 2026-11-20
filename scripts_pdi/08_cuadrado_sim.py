#!/usr/bin/env python3
"""CapyTown scripts_pdi/08 — Experimentos con odometria: linea virtual y cuadrado 2 x 2 m.

Simula, SIN robot, la misma logica que corre en cuadrado_controller.py
(core_cuadrado.Runner). El controlador solo ve la pose (x, y, yaw) con la
latencia de /odom; no hay camara.

Experimento 1 (corregir una linea, recta de 2 m, arranca 10 cm al costado):
  python3 08_cuadrado_sim.py --experimento recta --controlador bangbang --bias 0.06
  python3 08_cuadrado_sim.py --experimento recta --kd 0 --bias 0.06          # solo P
  python3 08_cuadrado_sim.py --experimento recta --bias 0.06                 # P + D
  python3 08_cuadrado_sim.py --experimento recta --ki 0.5 --bias 0.06        # PID

Experimento 2 (giro de 90 deg):
  python3 08_cuadrado_sim.py --experimento giro90 --kp-yaw 10 --kd-yaw 0
  python3 08_cuadrado_sim.py --experimento giro90 --kp-yaw 10 --kd-yaw 1.2

Experimento 3 (cuadrado de 2 x 2 m; --chi = derrape de las 4 ruedas al girar):
  python3 08_cuadrado_sim.py --experimento cuadrado_abierto --v-scale 0.97 --chi 1.2
  python3 08_cuadrado_sim.py --experimento cuadrado_abierto --v-scale 0.97 --chi 1.2 --turn-scale 0.83   # giro calibrado
  python3 08_cuadrado_sim.py --experimento cuadrado         --v-scale 0.97 --chi 1.2
"""
import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core_cuadrado import Params, ROBOT, ruedas_rpm, simular, metricas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experimento", choices=["recta", "giro90", "cuadrado", "cuadrado_abierto"], default="cuadrado")
    ap.add_argument("--controlador", choices=["pid", "bangbang"], default="pid")
    ap.add_argument("--v", type=float, default=0.20, help="velocidad lineal (m/s)")
    ap.add_argument("--lado", type=float, default=2.0, help="largo del lado (m)")
    ap.add_argument("--kp", type=float, default=3.0, help="PID de la linea (error en m)")
    ap.add_argument("--ki", type=float, default=0.0)
    ap.add_argument("--kd", type=float, default=4.0)
    ap.add_argument("--kp-yaw", type=float, default=3.0, help="PID del giro (error en rad)")
    ap.add_argument("--ki-yaw", type=float, default=0.0)
    ap.add_argument("--kd-yaw", type=float, default=0.4)
    ap.add_argument("--offset", type=float, default=None, help="arranca a la izquierda de la linea (m); recta: 0.10")
    ap.add_argument("--bias", type=float, default=0.0, help="sesgo de motores inyectado (rad/s)")
    ap.add_argument("--v-scale", type=float, default=1.0, help="<1: las ruedas patinan en avance")
    ap.add_argument("--w-scale", type=float, default=1.0, help="<1: el robot gira menos de lo comandado")
    ap.add_argument("--chi", type=float, default=None, help="derrape de las 4 ruedas al girar (>= 1); w-scale = 1/chi")
    ap.add_argument("--turn-scale", type=float, default=1.0, help="calibracion del giro en lazo abierto (1/chi)")
    ap.add_argument("--ruido", type=float, default=0.0, help="ruido del yaw de /odom (rad)")
    ap.add_argument("--retardo", type=int, default=3, help="latencia de /odom en muestras (30 Hz)")
    ap.add_argument("--salida", default=None, help="nombre del PNG")
    a = ap.parse_args()

    offset = a.offset if a.offset is not None else (0.10 if a.experimento == "recta" else 0.0)
    p = Params(experimento=a.experimento, controlador=a.controlador, v=a.v, side=a.lado,
               kp=a.kp, ki=a.ki, kd=a.kd, kp_yaw=a.kp_yaw, ki_yaw=a.ki_yaw, kd_yaw=a.kd_yaw,
               line_offset=offset, bias_w=a.bias, turn_scale=a.turn_scale)
    w_scale = 1.0 / a.chi if a.chi else a.w_scale
    sim = simular(p, v_scale=a.v_scale, w_scale=w_scale, ruido_yaw=a.ruido, retardo=a.retardo)
    m = metricas(sim, p)

    print(f"experimento : {a.experimento}  ({a.controlador})")
    print(f"linea  kp={a.kp} ki={a.ki} kd={a.kd}   giro  kp={a.kp_yaw} ki={a.ki_yaw} kd={a.kd_yaw}")
    print(f"planta      : v_scale={a.v_scale} w_scale={w_scale:.2f} bias={a.bias} rad/s retardo={a.retardo} muestras")
    chi = a.chi or 1.0 / w_scale
    pico = max(max(abs(x) for x in ruedas_rpm(v, w, chi)) for v, w in zip(sim["v"], sim["w"]))
    print(f"4 ruedas    : rpm pico {pico:.0f} de {ROBOT['rpm_max']:.0f} ({pico / ROBOT['rpm_max'] * 100:.0f} %), pista efectiva {chi * ROBOT['pista'] * 100:.1f} cm (chi {chi:.2f})")
    print("terminado   :", "si" if sim["terminado"] else "NO (se acabo el tiempo)")
    for k, v in m.items():
        print(f"{k:26s}: " + ("no se asienta" if v is None else f"{float(v):.2f}"))

    t = sim["t"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax1.plot(sim["x"], sim["y"], c="#0277BD", lw=2, label="robot (odom)")
    if a.experimento in ("cuadrado", "cuadrado_abierto"):
        L = a.lado
        ax1.plot([0, L, L, 0, 0], [0, 0, L, L, 0], "--", c="gray", lw=1.2, label="cuadrado ideal")
    elif a.experimento == "recta":
        ax1.plot([0, a.lado], [-offset, -offset], "--", c="gray", lw=1.2, label="linea virtual")
    ax1.plot(0, 0, "o", c="#0F9D58", label="inicio")
    ax1.set_aspect("equal"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)"); ax1.set_title("Trayectoria")

    if a.experimento == "giro90":
        ax2.plot(t, np.degrees(sim["yaw"]), c="#0277BD", lw=2)
        ax2.axhline(90, ls="--", c="gray"); ax2.axhspan(88, 92, color="green", alpha=0.08)
        ax2.set_ylabel("yaw (deg)"); ax2.set_title("Giro de 90 deg")
    else:
        ax2.plot(t, sim["e_line"] * 100, c="#C49000", lw=1.5)
        ax2.axhline(0, ls="--", c="gray"); ax2.axhspan(-2, 2, color="green", alpha=0.08)
        ax2.set_ylabel("error a la linea virtual (cm)"); ax2.set_title("Error lateral")
    ax2.set_xlabel("tiempo (s)"); ax2.grid(alpha=0.3)
    fig.suptitle(f"CapyTown - {a.experimento} - solo odometria", fontsize=11)
    fig.tight_layout()
    out = a.salida or f"cuadrado_{a.experimento}_{a.controlador}.png"
    fig.savefig(out, dpi=130)
    print("grafica     :", out)


if __name__ == "__main__":
    main()
