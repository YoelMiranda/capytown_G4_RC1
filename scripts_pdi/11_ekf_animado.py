#!/usr/bin/env python3
"""CapyTown scripts_pdi/11 — El EKF en accion, ANIMADO (una sola ventana).

Es la version dinamica de las diapositivas "El EKF en accion" y "Como se afinan Q y R".
El robot recorre el cuadrado de 2 x 2 m una y otra vez. Tres estimadores intentan
saber DONDE esta, cada uno con su sensor de yaw y la misma odometria de distancia:

    azul   encoders : yaw absoluto, ruidoso y con un salto por derrape en cada giro
    verde  giroscopio: suave, pero con sesgo (deriva) que se acumula
    rojo   Kalman   : filtro de 2 estados [yaw, sesgo del giroscopio] que fusiona ambos
                      (el caso lineal del EKF; el robot real usa el EKF completo)

El dibujo muestra al robot REAL (negro) y la posicion que cree cada estimador. Con los
sliders cambias el sesgo del giroscopio, el ruido de los encoders, el derrape por giro y
la confianza del filtro: R (cuanto desconfia de los encoders) y Q (cuanto desconfia del modelo).

Uso:
    python 11_ekf_animado.py
    python 11_ekf_animado.py --guardar ekf.png --segundos 180      # sin ventana

Requisitos: numpy y matplotlib.
"""
import argparse
import math
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Rectangle
from matplotlib.transforms import Affine2D
from matplotlib.widgets import Button, Slider

from core_cuadrado import ROBOT

DT = 1.0 / 30.0
BLUE, GREEN, RED, GOLD = "#0277BD", "#0B7A43", "#DB4437", "#F4B400"
INK, GRAY = "#1F2A30", "#5B6770"
V = 0.20                      # m/s en las rectas
LADO = 2.0                    # m
W_GIRO = 0.5                  # rad/s en los giros
T_RECTA = LADO / V
T_GIRO = (math.pi / 2) / W_GIRO


class Mundo:
    """El robot real + tres estimadores de yaw (y de posicion por dead reckoning)."""

    def __init__(self, cfg, semilla=5):
        self.cfg = dict(cfg)
        self.semilla = semilla
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.semilla)
        self.t = 0.0
        self.x = self.y = self.th = 0.0
        self.fase, self.t_fase = "recta", 0.0
        self.slip = 0.0                                   # derrape acumulado de los encoders (rad)
        self.est = {k: dict(x=0.0, y=0.0, yaw=0.0) for k in ("enc", "gyro", "kf")}
        # Kalman [yaw, sesgo]
        self.xk = np.array([0.0, 0.0])
        self.P = np.diag([1e-3, 1e-4])
        self.sesgo_est = 0.0
        self.h = {k: [] for k in ("t", "e_enc", "e_gyro", "e_kf", "p_enc", "p_gyro", "p_kf", "sesgo", "sesgo_est")}
        self.vueltas = 0.0
        self.reiniciado = True

    def paso(self):
        c = self.cfg
        self.reiniciado = False
        # ---- verdad
        v, w = (V, 0.0) if self.fase == "recta" else (0.0, W_GIRO)
        self.th += w * DT
        self.x += v * math.cos(self.th) * DT
        self.y += v * math.sin(self.th) * DT
        self.t += DT
        self.t_fase += DT
        if self.fase == "recta" and self.t_fase >= T_RECTA:
            self.fase, self.t_fase = "giro", 0.0
        elif self.fase == "giro" and self.t_fase >= T_GIRO:
            self.fase, self.t_fase = "recta", 0.0
            self.slip += math.radians(c["slip"]) * self.rng.choice([-1.0, 1.0])
            self.vueltas += 0.25
        # ---- sensores
        sesgo = math.radians(c["bias"])
        w_gyro = w + sesgo + self.rng.normal(0.0, 0.004)
        z_enc = self.th + self.slip + self.rng.normal(0.0, math.radians(c["sig_enc"]))
        # ---- estimador 1: encoders (el yaw que miden, tal cual)
        self.est["enc"]["yaw"] = z_enc
        # ---- estimador 2: giroscopio integrado
        self.est["gyro"]["yaw"] += w_gyro * DT
        # ---- estimador 3: Kalman de 2 estados
        F = np.array([[1.0, -DT], [0.0, 1.0]])
        B = np.array([DT, 0.0])
        Q = np.diag([1e-5, 1e-8]) * c["Q"]
        R = math.radians(c["R"]) ** 2
        self.xk = F @ self.xk + B * w_gyro
        self.P = F @ self.P @ F.T + Q
        yk = z_enc - self.xk[0]
        S = self.P[0, 0] + R
        K = self.P[:, 0] / S
        self.xk = self.xk + K * yk
        self.P = (np.eye(2) - np.outer(K, [1.0, 0.0])) @ self.P
        self.est["kf"]["yaw"] = self.xk[0]
        self.sesgo_est = self.xk[1]
        # ---- posicion por dead reckoning con el yaw de cada estimador (misma distancia v)
        for e in self.est.values():
            e["x"] += v * math.cos(e["yaw"]) * DT
            e["y"] += v * math.sin(e["yaw"]) * DT
        # ---- historial (cada 3 pasos)
        if int(round(self.t / DT)) % 3 == 0:
            self.h["t"].append(self.t)
            for k in ("enc", "gyro", "kf"):
                self.h["e_" + k].append(math.degrees(self.est[k]["yaw"] - self.th))
                self.h["p_" + k].append(100.0 * math.hypot(self.est[k]["x"] - self.x, self.est[k]["y"] - self.y))
            self.h["sesgo"].append(c["bias"])
            self.h["sesgo_est"].append(math.degrees(self.sesgo_est))

    def rms(self, k):
        e = np.array(self.h["e_" + k])
        return float(np.sqrt(np.mean(e ** 2))) if len(e) else 0.0


CFG0 = dict(bias=0.25, sig_enc=2.0, slip=0.5, R=3.0, Q=1.0)


class App:
    def __init__(self):
        self.m = Mundo(CFG0)
        self.pausa = False
        self.fig = plt.figure(figsize=(14.2, 8.4))
        try:
            self.fig.canvas.manager.set_window_title("CapyTown - el EKF en accion")
        except Exception:
            pass
        self.ax = self.fig.add_axes([0.05, 0.40, 0.45, 0.54])
        self.a1 = self.fig.add_axes([0.58, 0.77, 0.39, 0.17])
        self.a2 = self.fig.add_axes([0.58, 0.57, 0.39, 0.14])
        self.a3 = self.fig.add_axes([0.58, 0.41, 0.39, 0.10])
        self._arena()
        self._plots()
        self._controles()
        self.anim = FuncAnimation(self.fig, self.actualizar, interval=30, blit=False, cache_frame_data=False)

    def _arena(self):
        ax = self.ax
        ax.set_aspect("equal")
        ax.set_xlim(-0.6, 2.7); ax.set_ylim(-0.6, 2.7)
        ax.grid(alpha=0.25)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title("Donde esta el robot (negro) y donde cree cada estimador que esta", fontsize=10)
        ax.plot([0, 2, 2, 0, 0], [0, 0, 2, 2, 0], color="#9AA5AD", lw=1.4, ls="--", zorder=1)
        self.tr = {}
        self.mk = {}
        for k, col, nom in (("enc", BLUE, "solo encoders"), ("gyro", GREEN, "solo giroscopio"), ("kf", RED, "Kalman (fusion)")):
            self.tr[k], = ax.plot([], [], color=col, lw=1.6, alpha=0.85, label=nom, zorder=3)
            self.mk[k], = ax.plot([], [], "o", color=col, ms=9, zorder=5)
        self.tr["real"], = ax.plot([], [], color=INK, lw=2.2, label="robot real", zorder=4)
        L, W = ROBOT["largo"], ROBOT["ancho"]
        self.cuerpo = Rectangle((-L / 2, -W / 2), L, W, fc="#CFE8F7", ec=INK, lw=1.6, zorder=6)
        self.frente = Polygon([[L / 2 - 0.004, 0], [L / 2 - 0.035, 0.026], [L / 2 - 0.035, -0.026]], fc=INK, zorder=7)
        ax.add_patch(self.cuerpo); ax.add_patch(self.frente)
        ax.legend(loc="lower right", fontsize=8)
        self.texto = ax.text(0.015, 0.985, "", transform=ax.transAxes, va="top", ha="left", family="monospace",
                             fontsize=8.5, bbox=dict(boxstyle="round", fc="white", ec="#BBBBBB", alpha=0.93), zorder=20)
        self.puntos = {k: deque(maxlen=6000) for k in ("enc", "gyro", "kf", "real")}

    def _plots(self):
        self.l1 = {k: self.a1.plot([], [], color=c, lw=lw, label=n)[0]
                   for k, c, lw, n in (("enc", BLUE, 0.8, "encoders"), ("gyro", GREEN, 1.8, "giroscopio"), ("kf", RED, 2.4, "Kalman"))}
        self.a1.axhline(0, color=GRAY, lw=0.9)
        self.a1.set_title("Error de yaw (grados)", fontsize=10)
        self.a1.legend(fontsize=7.5, loc="upper left", ncol=3)
        self.l2 = {k: self.a2.plot([], [], color=c, lw=lw)[0]
                   for k, c, lw in (("enc", BLUE, 1.2), ("gyro", GREEN, 1.8), ("kf", RED, 2.4))}
        self.a2.set_title("Error de POSICION por dead reckoning (cm)", fontsize=10)
        self.l3, = self.a3.plot([], [], color=RED, lw=2, label="sesgo estimado por el filtro")
        self.l3r, = self.a3.plot([], [], color=GRAY, lw=1.4, ls="--", label="sesgo real del giroscopio")
        self.a3.set_title("El filtro descubre el sesgo del giroscopio (grados/s)", fontsize=10)
        self.a3.set_xlabel("tiempo (s)")
        self.a3.legend(fontsize=7.5, loc="lower right")
        for a in (self.a1, self.a2, self.a3):
            a.grid(alpha=0.3)
            a.tick_params(labelsize=8)

    def _controles(self):
        f = self.fig

        def sl(rect, key, label, lo, hi, fmt):
            s = Slider(f.add_axes(rect), label, lo, hi, valinit=CFG0[key], valfmt=fmt)
            s.label.set_fontsize(8.5); s.valtext.set_fontsize(8.5)
            s.on_changed(lambda v, k=key: self.m.cfg.__setitem__(k, float(v)))
            return s
        self.sl = [
            sl([0.15, 0.30, 0.28, 0.025], "bias", "sesgo del giroscopio (grados/s)", 0, 1.0, "%.2f"),
            sl([0.15, 0.25, 0.28, 0.025], "sig_enc", "ruido de encoders (grados)", 0, 8, "%.1f"),
            sl([0.15, 0.20, 0.28, 0.025], "slip", "derrape por giro (grados)", 0, 15, "%.1f"),
            sl([0.68, 0.30, 0.26, 0.025], "R", "R: desconfianza de encoders", 0.5, 30, "%.1f"),
            sl([0.68, 0.25, 0.26, 0.025], "Q", "Q: desconfianza del modelo", 0.01, 100, "%.2f"),
        ]
        self.vel = Slider(f.add_axes([0.68, 0.20, 0.26, 0.025]), "velocidad de simulacion", 1, 30, valinit=6, valfmt="%d")
        self.vel.label.set_fontsize(8.5); self.vel.valtext.set_fontsize(8.5)
        self.b1 = Button(f.add_axes([0.15, 0.115, 0.09, 0.04]), "Reiniciar")
        self.b2 = Button(f.add_axes([0.25, 0.115, 0.09, 0.04]), "Pausa")
        self.b1.on_clicked(lambda _: self._reiniciar())
        self.b2.on_clicked(lambda _: self.alternar())
        f.text(0.05, 0.065, "Prueba: sube el derrape por giro: el filtro no puede distinguirlo de un giro real y se parece cada vez mas a los encoders. "
               "Con derrape 0 el Kalman gana por mucho.", fontsize=8.8, color=GRAY)
        f.text(0.05, 0.035, "R grande = ignora a los encoders (se parece al giroscopio); R chica = sigue su ruido. Q grande = desconfia del modelo. "
               "La posicion usa SOLO v y el yaw (dead reckoning).", fontsize=8.8, color=GRAY)

    def _reiniciar(self):
        self.m.reset()
        for d in self.puntos.values():
            d.clear()

    def alternar(self):
        self.pausa = not self.pausa
        self.b2.label.set_text("Seguir" if self.pausa else "Pausa")

    def actualizar(self, _):
        m = self.m
        if not self.pausa:
            for _ in range(int(self.vel.val)):
                m.paso()
                if int(round(m.t / DT)) % 2 == 0:
                    for k in ("enc", "gyro", "kf"):
                        self.puntos[k].append((m.est[k]["x"], m.est[k]["y"]))
                    self.puntos["real"].append((m.x, m.y))
        for k in ("enc", "gyro", "kf", "real"):
            if self.puntos[k]:
                xs, ys = zip(*self.puntos[k])
                self.tr[k].set_data(xs, ys)
            else:
                self.tr[k].set_data([], [])
        for k in ("enc", "gyro", "kf"):
            self.mk[k].set_data([m.est[k]["x"]], [m.est[k]["y"]])
        tr = Affine2D().rotate(m.th).translate(m.x, m.y) + self.ax.transData
        self.cuerpo.set_transform(tr); self.frente.set_transform(tr)
        h = m.h
        if h["t"]:
            t = h["t"]
            tmin = max(0.0, t[-1] - 120.0)
            for k in ("enc", "gyro", "kf"):
                self.l1[k].set_data(t, h["e_" + k]); self.l2[k].set_data(t, h["p_" + k])
            self.l3.set_data(t, h["sesgo_est"]); self.l3r.set_data(t, h["sesgo"])
            for a in (self.a1, self.a2, self.a3):
                a.set_xlim(tmin, max(tmin + 20, t[-1]))
            ee = np.array([h["e_enc"], h["e_gyro"], h["e_kf"]])[:, -4000:]
            self.a1.set_ylim(float(ee.min()) - 2, float(ee.max()) + 5)
            pp = np.array([h["p_enc"], h["p_gyro"], h["p_kf"]])[:, -4000:]
            self.a2.set_ylim(0, float(pp.max()) * 1.1 + 1)
            self.a3.set_ylim(-0.1, max(1.0, float(max(h["sesgo"])) * 1.3, float(max(h["sesgo_est"])) * 1.2))
        pe = h["p_enc"][-1] if h["t"] else 0.0
        pg = h["p_gyro"][-1] if h["t"] else 0.0
        pk = h["p_kf"][-1] if h["t"] else 0.0
        self.texto.set_text(
            f"t = {m.t:6.1f} s   vueltas = {m.vueltas:4.2f}\n"
            f"error de yaw (RMS, grados)\n"
            f"  encoders   {m.rms('enc'):6.2f}\n"
            f"  giroscopio {m.rms('gyro'):6.2f}\n"
            f"  Kalman     {m.rms('kf'):6.2f}\n"
            f"error de posicion ahora (cm)\n"
            f"  encoders   {pe:6.1f}\n"
            f"  giroscopio {pg:6.1f}\n"
            f"  Kalman     {pk:6.1f}")
        return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--guardar", default=None, help="corre sin ventana y guarda un PNG")
    ap.add_argument("--segundos", type=float, default=180.0, help="con --guardar: tiempo simulado")
    a = ap.parse_args()
    if a.guardar:
        import matplotlib
        matplotlib.use("Agg")
    app = App()
    if a.guardar:
        for _ in range(int(a.segundos / DT)):
            app.m.paso()
            if int(round(app.m.t / DT)) % 2 == 0:
                for k in ("enc", "gyro", "kf"):
                    app.puntos[k].append((app.m.est[k]["x"], app.m.est[k]["y"]))
                app.puntos["real"].append((app.m.x, app.m.y))
        app.pausa = True
        app.actualizar(0)
        app.fig.savefig(a.guardar, dpi=105)
        m = app.m
        print(f"RMS yaw (grados): encoders {m.rms('enc'):.2f}  giroscopio {m.rms('gyro'):.2f}  Kalman {m.rms('kf'):.2f}")
        print(f"posicion final (cm): encoders {m.h['p_enc'][-1]:.1f}  giroscopio {m.h['p_gyro'][-1]:.1f}  Kalman {m.h['p_kf'][-1]:.1f}")
        print("guardado:", a.guardar)
        return
    plt.show()


if __name__ == "__main__":
    main()
