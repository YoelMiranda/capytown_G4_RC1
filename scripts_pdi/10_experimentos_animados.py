#!/usr/bin/env python3
"""CapyTown scripts_pdi/10 — Experimentos 1, 2 y 3 ANIMADOS (una sola ventana).

Es la version dinamica de 08_cuadrado_sim.py: el robot de 4 ruedas se mueve en
tiempo real y puedes cambiar kp, ki, kd, el sesgo de motores o el derrape (chi)
MIENTRAS corre, para ver el efecto en vivo. Usa la misma logica que el nodo del
robot (core_cuadrado.Runner) y solo la odometria.

Escenarios (selector de la izquierda, los mismos de las diapositivas):
    1A Bang-bang   1B Solo P   1C P + D   1D PID      recta de 2 m, arranca 10 cm al costado
    2a/2b/2c       giro de 90 grados: kp bajo, kp alto, kp alto + kd
    3A / 3A+       cuadrado en lazo abierto: sin calibrar y con turn_scale calibrado
    3B / 3C        cuadrado con PID sobre /odom, sin y con Ki
    4              deriva por 1 grado de yaw (sin correccion)

Botones "Comparar ...": corren varios escenarios A LA VEZ para verlos superpuestos,
como las graficas de las diapositivas 66, 67 y 69.

Uso:
    python 10_experimentos_animados.py
    python 10_experimentos_animados.py --inicio "3B  Odom + PID"

Requisitos: numpy y matplotlib.
"""
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Rectangle
from matplotlib.transforms import Affine2D
from matplotlib.widgets import Button, RadioButtons, Slider

from core_cuadrado import Params, Planta, ROBOT, Runner, metricas, ruedas_rpm

DT = 1.0 / 30.0
BLUE, GREEN, RED, GOLD = "#0277BD", "#0B7A43", "#DB4437", "#F4B400"
DGOLD, INK, GRAY = "#7A5600", "#1F2A30", "#5B6770"

# ---- presets: los mismos valores de las diapositivas 65 a 69
BASE = dict(exp="recta", ctrl="pid", kp=3.0, ki=0.0, kd=4.0, kpy=3.0, kdy=0.4,
            offset=0.0, bias=0.0, chi=1.0, turn=1.0, vscale=1.0, ruido=0.0, yaw0=0.0)
PRESETS = {
    "1A  Bang-bang":         dict(BASE, exp="recta", ctrl="bangbang", offset=10.0, bias=0.06),
    "1B  Solo P":            dict(BASE, exp="recta", kd=0.0, offset=10.0, bias=0.06),
    "1C  P + D":             dict(BASE, exp="recta", offset=10.0, bias=0.06),
    "1D  PID":               dict(BASE, exp="recta", ki=0.5, offset=10.0, bias=0.06),
    "2a  Giro kp bajo":      dict(BASE, exp="giro90", kpy=1.0, kdy=0.0),
    "2b  Giro kp alto":      dict(BASE, exp="giro90", kpy=10.0, kdy=0.0),
    "2c  Giro kp alto + kd": dict(BASE, exp="giro90", kpy=10.0, kdy=1.2),
    "3A  Abierto (chi 1.2)": dict(BASE, exp="cuadrado_abierto", chi=1.2, vscale=0.97, ruido=0.5),
    "3A+ Abierto calibrado": dict(BASE, exp="cuadrado_abierto", chi=1.2, vscale=0.97, ruido=0.5, turn=0.83),
    "3B  Odom + PID":        dict(BASE, exp="cuadrado", chi=1.2, vscale=0.97, bias=0.03, ruido=0.5),
    "3C  Sesgo + Ki":        dict(BASE, exp="cuadrado", ki=0.5, bias=0.05, chi=1.2, vscale=0.97, ruido=0.5),
    "4   Deriva 1 grado":    dict(BASE, exp="recta", kp=0.0, ki=0.0, kd=0.0, yaw0=1.0),
}
NOTAS = {
    "1A  Bang-bang": "Zigzag permanente: nunca se asienta.",
    "1B  Solo P": "Oscila y queda lejos de la linea. Prueba subir/bajar kp.",
    "1C  P + D": "Kd amortigua el zigzag; queda un sesgo de ~2 cm.",
    "1D  PID": "Ki elimina el sesgo. Sube ki mucho y mira el wind-up.",
    "2a  Giro kp bajo": "Lento: tarda casi 3 s en llegar a 90 grados.",
    "2b  Giro kp alto": "Rapido, pero sobrepasa y oscila alrededor de 90.",
    "2c  Giro kp alto + kd": "Kd amortigua: llega rapido y casi sin pasarse.",
    "3A  Abierto (chi 1.2)": "Sin realimentacion y con derrape no cierra el cuadrado.",
    "3A+ Abierto calibrado": "turn_scale = 1/chi = 0.83 corrige los giros: cierra casi bien.",
    "3B  Odom + PID": "Hay sesgo y ruido de /odom: mueve kp, kd, ki y prueba Empujon.",
    "3C  Sesgo + Ki": "Compara ki = 0 con ki > 0 en las rectas (sesgo de 0.05).",
    "4   Deriva 1 grado": "Sin correccion, 1 grado de yaw = 3.5 cm a los 2 m: se sale del carril.",
}
COLOR = {"1A": RED, "1B": BLUE, "1C": GOLD, "1D": GREEN, "2a": BLUE, "2b": RED, "2c": GREEN,
         "3A": RED, "3A+": GOLD, "3B": GREEN, "3C": DGOLD, "4": RED}
GRUPOS = {
    "Comparar 1A-1D": ["1A  Bang-bang", "1B  Solo P", "1C  P + D", "1D  PID"],
    "Comparar giros": ["2a  Giro kp bajo", "2b  Giro kp alto", "2c  Giro kp alto + kd"],
    "Comparar 3A / 3B / 3C": ["3A  Abierto (chi 1.2)", "3A+ Abierto calibrado", "3B  Odom + PID", "3C  Sesgo + Ki"],
}


def color_de(nombre):
    return COLOR.get(nombre.split()[0], BLUE)


class Sim:
    """Un escenario corriendo: Runner (el controlador) + Planta (el robot)."""

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.reset()

    def _params(self):
        c = self.cfg
        return Params(experimento=c["exp"], controlador=c["ctrl"], kp=c["kp"], ki=c["ki"], kd=c["kd"],
                      kp_yaw=c["kpy"], kd_yaw=c["kdy"], line_offset=c["offset"] / 100.0,
                      bias_w=c["bias"], turn_scale=c["turn"])

    def reset(self):
        self.run = Runner(self._params())
        self.planta = Planta(v_scale=self.cfg["vscale"], w_scale=1.0 / self.cfg["chi"],
                             ruido_yaw=math.radians(self.cfg["ruido"]))
        self.planta.th = math.radians(self.cfg["yaw0"])
        self.h = {k: [] for k in ("t", "x", "y", "yaw", "yaw_obj", "e_line", "e_yaw", "estado", "rpm_i", "rpm_d")}
        self.done = False

    def aplicar(self):
        """Cambia parametros en vivo, sin reiniciar."""
        c, p = self.cfg, self.run.p
        p.controlador, p.kp, p.ki, p.kd = c["ctrl"], c["kp"], c["ki"], c["kd"]
        p.kp_yaw, p.kd_yaw = c["kpy"], c["kdy"]
        p.line_offset, p.bias_w, p.turn_scale = c["offset"] / 100.0, c["bias"], c["turn"]
        for pid, (a, b, d) in ((self.run.pid_line, (c["kp"], c["ki"], c["kd"])),
                               (self.run.pid_yaw, (c["kpy"], p.ki_yaw, c["kdy"]))):
            pid.kp, pid.ki, pid.kd = a, b, d
        self.planta.w_scale = 1.0 / c["chi"]
        self.planta.v_scale = c["vscale"]
        self.planta.ruido_yaw = math.radians(c["ruido"])

    def paso(self):
        if self.done:
            return
        ox, oy, oth = self.planta.medir()
        v, w, info = self.run.update(ox, oy, oth, DT)
        pl = self.planta
        ri, rd = ruedas_rpm(v, w, self.cfg["chi"])
        obj = self.run.yaw_target if self.run.state == "GIRA" else self.run.yaw_obj
        for k, val in (("t", self.run.t), ("x", pl.x), ("y", pl.y), ("yaw", pl.th), ("yaw_obj", obj),
                       ("e_line", info["e_line"]), ("e_yaw", info["e_yaw"]), ("estado", info["state"]),
                       ("rpm_i", ri), ("rpm_d", rd)):
            self.h[k].append(val)
        if self.run.done:
            self.done = True
            return
        pl.paso(v, w, DT)

    def metricas(self):
        if len(self.h["t"]) < 5:
            return {}
        sim = {k: np.array(v) for k, v in self.h.items()}
        sim["closure"] = self.run.closure_error
        sim["yaw_total_deg"] = math.degrees(self.run.yaw_total)
        try:
            return metricas(sim, self.run.p)
        except (ValueError, IndexError):
            return {}


class App:
    def __init__(self, inicio):
        self.nombre = inicio
        self.nombres = [inicio]
        self.sims = [Sim(PRESETS[inicio])]
        self.modo = "uno"
        self.pausa = False
        self.cargando = False
        self._arena = []
        self._dyn = []
        self.series = []

        self.fig = plt.figure(figsize=(14.2, 8.8))
        try:
            self.fig.canvas.manager.set_window_title("CapyTown - experimentos animados")
        except Exception:
            pass
        self.ax = self.fig.add_axes([0.045, 0.43, 0.46, 0.51])
        self.ax_e = self.fig.add_axes([0.575, 0.785, 0.40, 0.155])
        self.ax_y = self.fig.add_axes([0.575, 0.60, 0.40, 0.135])
        self.ax_r = self.fig.add_axes([0.575, 0.43, 0.40, 0.115])
        self.texto = self.ax.text(0.015, 0.985, "", transform=self.ax.transAxes, va="top", ha="left",
                                  family="monospace", fontsize=8.2,
                                  bbox=dict(boxstyle="round", fc="white", ec="#BBBBBB", alpha=0.93), zorder=20)
        self.fin = self.ax.text(0.5, 0.06, "", transform=self.ax.transAxes, ha="center", va="center", fontsize=12,
                                fontweight="bold", color=GREEN, zorder=21,
                                bbox=dict(boxstyle="round", fc="white", ec=GREEN, alpha=0.92))
        self.fin.set_visible(False)
        self._plots_fijos()
        self._controles()
        self.seleccionar(inicio)
        self.anim = FuncAnimation(self.fig, self.actualizar, interval=30, blit=False, cache_frame_data=False)

    # ------------------------------------------------------------ elementos fijos
    def _plots_fijos(self):
        self.ax_y.set_ylabel("grados")
        self.ax_y.set_title("Rumbo (yaw): real contra objetivo", fontsize=9)
        self.ax_r.set_ylabel("rpm")
        self.ax_r.set_xlabel("tiempo (s)")
        self.ax_r.set_title("rpm de las ruedas (limite del motor: 450)", fontsize=9)
        for a in (self.ax_e, self.ax_y, self.ax_r):
            a.grid(alpha=0.3)
            a.tick_params(labelsize=8)

    def _controles(self):
        f = self.fig
        ax_rad = f.add_axes([0.012, 0.02, 0.205, 0.37], facecolor="#F4F6F7")
        ax_rad.set_title("Escenario", fontsize=10, fontweight="bold")
        self.radio = RadioButtons(ax_rad, list(PRESETS), active=list(PRESETS).index(self.nombre))
        for t in self.radio.labels:
            t.set_fontsize(8.5)
        self.radio.on_clicked(lambda n: self.seleccionar(n))

        def sl(rect, key, label, lo, hi, fmt):
            a = f.add_axes(rect)
            s = Slider(a, label, lo, hi, valinit=self.sims[0].cfg.get(key, 0.0), valfmt=fmt)
            s.label.set_fontsize(8.5)
            s.valtext.set_fontsize(8.5)

            def cb(v, k=key):
                if self.cargando or self.modo != "uno":
                    return
                self.sims[0].cfg[k] = float(v)
                self.sims[0].aplicar()
            s.on_changed(cb)
            return s
        x1, x2, w = 0.30, 0.70, 0.21
        self.sl = {
            "kp": sl([x1, 0.340, w, 0.022], "kp", "kp  (linea)", 0, 10, "%.2f"),
            "ki": sl([x1, 0.300, w, 0.022], "ki", "ki  (linea)", 0, 2, "%.2f"),
            "kd": sl([x1, 0.260, w, 0.022], "kd", "kd  (linea)", 0, 8, "%.2f"),
            "kpy": sl([x1, 0.220, w, 0.022], "kpy", "kp  (giro)", 0, 15, "%.1f"),
            "kdy": sl([x1, 0.180, w, 0.022], "kdy", "kd  (giro)", 0, 3, "%.2f"),
            "offset": sl([x2, 0.340, w, 0.022], "offset", "arranque a la izq. (cm)", 0, 20, "%.0f"),
            "bias": sl([x2, 0.300, w, 0.022], "bias", "sesgo de motores (rad/s)", 0, 0.15, "%.3f"),
            "chi": sl([x2, 0.260, w, 0.022], "chi", "derrape chi", 1.0, 2.0, "%.2f"),
            "turn": sl([x2, 0.220, w, 0.022], "turn", "turn_scale (abierto)", 0.5, 1.2, "%.2f"),
            "ruido": sl([x2, 0.180, w, 0.022], "ruido", "ruido de /odom (grados)", 0, 3, "%.1f"),
        }
        self.vel = Slider(f.add_axes([x1, 0.140, w, 0.022]), "velocidad de simulacion", 1, 10, valinit=3, valfmt="%d")
        self.vel.label.set_fontsize(8.5); self.vel.valtext.set_fontsize(8.5)
        self.b_reset = Button(f.add_axes([0.30, 0.085, 0.075, 0.038]), "Reiniciar")
        self.b_pausa = Button(f.add_axes([0.385, 0.085, 0.075, 0.038]), "Pausa")
        self.b_push = Button(f.add_axes([0.470, 0.085, 0.075, 0.038]), "Empujon")
        self.b_reset.on_clicked(lambda _: self.reiniciar())
        self.b_pausa.on_clicked(lambda _: self.alternar_pausa())
        self.b_push.on_clicked(lambda _: self.empujar())
        self.b_cmp = []
        x = 0.585
        for nombre, ancho in (("Comparar 1A-1D", 0.11), ("Comparar giros", 0.11), ("Comparar 3A / 3B / 3C", 0.16)):
            b = Button(f.add_axes([x, 0.085, ancho, 0.038]), nombre, color="#E3F4EA", hovercolor="#CDEBD9")
            b.label.set_fontsize(8.5)
            b.on_clicked(lambda _, n=nombre: self.comparar(n))
            self.b_cmp.append(b)
            x += ancho + 0.01
        self.nota = f.text(0.30, 0.058, "", fontsize=9.5, color=DGOLD, fontweight="bold", va="center")
        f.text(0.30, 0.025, "MicroROS-Pi5 (4 ruedas, solo /odom). Los sliders cambian el controlador EN VIVO. "
               "Empujon: lo desplaza 6 cm y lo desvia 8 grados.", fontsize=8.5, color=GRAY)

    # ------------------------------------------------------------ acciones
    def alternar_pausa(self):
        self.pausa = not self.pausa
        self.b_pausa.label.set_text("Seguir" if self.pausa else "Pausa")

    def empujar(self):
        for s in self.sims:
            s.planta.empujar(0.06, math.radians(8.0))

    def reiniciar(self):
        for s in self.sims:
            s.reset()
            s.aplicar()

    def seleccionar(self, nombre):
        self.modo = "uno"
        self.nombre = nombre
        self.nombres = [nombre]
        cfg = PRESETS[nombre]
        self.cargando = True
        for k, s in self.sl.items():
            s.set_val(cfg[k])
        self.cargando = False
        self.sims = [Sim(cfg)]
        self.sims[0].aplicar()
        self.nota.set_text(NOTAS[nombre])
        self._armar()

    def comparar(self, grupo):
        self.modo = "comparar"
        self.nombre = grupo
        self.nombres = list(GRUPOS[grupo])
        self.sims = [Sim(PRESETS[n]) for n in self.nombres]
        self.nota.set_text("Modo comparar: varios escenarios a la vez. Los sliders solo cambian el modo individual.")
        self._armar()

    # ------------------------------------------------------------ armado de artistas
    def _nuevo_robot(self, color, relleno):
        L, W, wd, wa = ROBOT["largo"], ROBOT["ancho"], ROBOT["d_rueda"], ROBOT["a_rueda"]
        yb = (W - wa) / 2
        partes = [Rectangle((-L / 2, -W / 2), L, W, fc=relleno, ec=color, lw=1.6, zorder=4)]
        for xi in (-0.07, 0.07):
            for sg in (1, -1):
                partes.append(Rectangle((xi - wd / 2, sg * yb - wa / 2), wd, wa, fc="#333333", zorder=5))
        partes.append(Polygon([[L / 2 - 0.004, 0], [L / 2 - 0.035, 0.026], [L / 2 - 0.035, -0.026]], fc=color, zorder=6))
        for p in partes:
            self.ax.add_patch(p)
        return partes

    def _armar(self):
        for a in self._arena + self._dyn:
            try:
                a.remove()
            except Exception:
                pass
        self._arena, self._dyn, self.series = [], [], []
        exp = self.sims[0].cfg["exp"]
        self._dibujar_arena(exp)
        multiple = len(self.sims) > 1
        for sim, nom in zip(self.sims, self.nombres):
            col = color_de(nom)
            relleno = "#CFE8F7" if not multiple else "#FFFFFF"
            robot = self._nuevo_robot(col, relleno)
            trail, = self.ax.plot([], [], color=col, lw=2, alpha=0.8, zorder=3)
            ln_e, = self.ax_e.plot([], [], color=col, lw=1.8, label=nom.strip())
            ln_y, = self.ax_y.plot([], [], color=col, lw=1.8)
            ln_ri, = self.ax_r.plot([], [], color=GREEN, lw=1.4, label="ruedas izq.")
            ln_rd, = self.ax_r.plot([], [], color=RED, lw=1.4, label="ruedas der.")
            self._dyn += robot + [trail, ln_e, ln_y, ln_ri, ln_rd]
            self.series.append(dict(robot=robot, trail=trail, e=ln_e, y=ln_y, ri=ln_ri, rd=ln_rd))
        self.ln_obj, = self.ax_y.plot([], [], color=GRAY, lw=1.3, ls="--", label="objetivo")
        self.ln_cero, = self.ax_e.plot([], [], color=GRAY, lw=1.1, ls="--")
        self._dyn += [self.ln_obj, self.ln_cero]
        leg = self.ax_e.get_legend()
        if leg:
            leg.remove()
        self.ax_e.legend(handles=[s["e"] for s in self.series], fontsize=7.5, loc="upper right", ncol=2 if multiple else 1)
        yleg = self.ax_y.get_legend()
        if yleg:
            yleg.remove()
        self.ax_y.legend(handles=[self.ln_obj], fontsize=7.5, loc="upper left")
        self.ax_r.set_visible(not multiple)
        if not multiple:
            rleg = self.ax_r.get_legend()
            if rleg:
                rleg.remove()
            self.ax_r.legend(handles=[self.series[0]["ri"], self.series[0]["rd"]], fontsize=7.5, loc="upper left")
        self.ax_e.set_title("Error a la linea virtual (cm)" if exp != "giro90" else "Error de rumbo (grados)", fontsize=10)
        self.ax_e.set_ylabel("cm" if exp != "giro90" else "grados")

    def _dibujar_arena(self, exp):
        ax = self.ax
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        cfg = self.sims[0].cfg
        if exp == "recta":
            off = cfg["offset"] / 100.0
            ax.set_xlim(-0.2, 2.3); ax.set_ylim(-0.95, 0.95)
            self._arena += [ax.axhspan(-off - 0.0285, -off + 0.0285, color="#CDEBD9", alpha=0.8, zorder=1)]
            self._arena += ax.plot([0, 2], [-off, -off], color=GREEN, lw=2, ls="--", zorder=2)
            self._arena += [ax.axvline(2.0, color=GRAY, lw=1, ls=":")]
            ax.set_title("Corregir una linea (franja verde = holgura del carril, +-2.85 cm)", fontsize=10)
        elif exp == "giro90":
            ax.set_xlim(-0.55, 0.55); ax.set_ylim(-0.4, 0.7)
            self._arena += ax.plot([0, 0], [0, 0.5], color=GREEN, lw=2, ls="--", zorder=2)
            self._arena += [ax.text(0.02, 0.5, "objetivo +90 grados", color=GREEN, fontsize=9, fontweight="bold")]
            ax.set_title("Giro de 90 grados (el robot gira sobre su eje)", fontsize=10)
        else:
            ax.set_xlim(-0.45, 2.45); ax.set_ylim(-0.45, 2.45)
            for i in range(5):
                self._arena += ax.plot([0, 2], [i * 0.5, i * 0.5], color="#D0D0D0", lw=0.8, zorder=0)
                self._arena += ax.plot([i * 0.5, i * 0.5], [0, 2], color="#D0D0D0", lw=0.8, zorder=0)
            self._arena += ax.plot([0, 2, 2, 0, 0], [0, 0, 2, 2, 0], color=GREEN, lw=1.8, ls="--", zorder=2)
            self._arena += ax.plot([0], [0], "o", color=GREEN, ms=8, zorder=2)
            ax.set_title("Cuadrado de 2 x 2 m (16 tiles de 0.5 m)", fontsize=10)

    # ------------------------------------------------------------ animacion
    def actualizar(self, _):
        if not self.pausa:
            for _ in range(int(self.vel.val)):
                for s in self.sims:
                    s.paso()
        exp = self.sims[0].cfg["exp"]
        tmax, emax = 5.0, 3.0
        ymin, ymax = -5.0, 15.0
        for sim, ser in zip(self.sims, self.series):
            h, pl = sim.h, sim.planta
            tr = Affine2D().rotate(pl.th).translate(pl.x, pl.y) + self.ax.transData
            for p in ser["robot"]:
                p.set_transform(tr)
            ser["trail"].set_data(h["x"], h["y"])
            t = h["t"]
            if not t:
                continue
            tmax = max(tmax, t[-1])
            e = np.degrees(h["e_yaw"]) if exp == "giro90" else np.array(h["e_line"]) * 100.0
            ser["e"].set_data(t, e)
            emax = max(emax, float(np.abs(e).max()) * 1.2)
            yaw = np.degrees(h["yaw"])
            ser["y"].set_data(t, yaw)
            ymin, ymax = min(ymin, float(yaw.min()) - 5), max(ymax, float(yaw.max()) + 15)
            ser["ri"].set_data(t, h["rpm_i"]); ser["rd"].set_data(t, h["rpm_d"])
        s0 = self.sims[0]
        if s0.h["t"]:
            self.ln_obj.set_data(s0.h["t"], np.degrees(s0.h["yaw_obj"]))
            self.ln_cero.set_data([0, tmax], [0, 0])
            self.ax_e.set_ylim((-20, 100) if exp == "giro90" else (-emax, emax))
            self.ax_y.set_ylim(ymin, ymax)
            for a_ in (self.ax_e, self.ax_y, self.ax_r):
                a_.set_xlim(0, tmax)
            if self.modo == "uno":
                h0 = s0.h
                lim_r = max(120.0, max(max(np.abs(h0["rpm_i"])), max(np.abs(h0["rpm_d"]))) * 1.2)
                self.ax_r.set_ylim(-min(lim_r, 480), min(lim_r, 480))
        self.texto.set_text(self._texto_metricas())
        todos = all(s.done for s in self.sims)
        self.fin.set_text("FIN  ·  pulsa Reiniciar para repetir")
        self.fin.set_visible(todos)
        return []

    def _texto_metricas(self):
        if self.modo == "uno":
            sim = self.sims[0]
            lineas = [f"{self.nombre}   t = {sim.run.t:5.1f} s   estado: {sim.run.state}"]
            for k, v in sim.metricas().items():
                lineas.append(f"{k}: " + ("no se asienta" if v is None else f"{float(v):.2f}"))
            return "\n".join(lineas)
        lineas = [self.nombre]
        for sim, nom in zip(self.sims, self.nombres):
            m = sim.metricas()
            exp = sim.cfg["exp"]
            if exp == "recta":
                txt = f"|e| medio {m.get('|e| medio (cm)', 0):5.2f} cm"
            elif exp == "giro90":
                ts = m.get("t asentamiento +-2 deg (s)")
                txt = f"sobre {m.get('sobreimpulso (%)', 0):4.1f} %  asent. " + ("--" if ts is None else f"{float(ts):.2f} s")
            else:
                cz = m.get("error de cierre (cm)")
                txt = "cierre " + ("..." if cz is None else f"{float(cz):6.1f} cm")
            lineas.append(f"{nom.split()[0]:3s} {txt}")
        return "\n".join(lineas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inicio", default="3B  Odom + PID", choices=list(PRESETS), help="escenario con el que abre")
    ap.add_argument("--guardar", default=None, help="corre sin ventana hasta el final y guarda un PNG")
    ap.add_argument("--comparar", default=None, choices=list(GRUPOS), help="con --guardar: guarda un grupo comparado")
    a = ap.parse_args()
    if a.guardar:
        import matplotlib
        matplotlib.use("Agg")
    app = App(a.inicio)
    if a.guardar:
        if a.comparar:
            app.comparar(a.comparar)
        for _ in range(4000):
            if all(s.done for s in app.sims):
                break
            for s in app.sims:
                s.paso()
        app.actualizar(0)
        app.fig.savefig(a.guardar, dpi=110)
        print("guardado:", a.guardar)
        return
    plt.show()


if __name__ == "__main__":
    main()
