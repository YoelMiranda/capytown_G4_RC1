#!/usr/bin/env python3
"""Simulador del e-puck siguiendo una linea circular (sin Webots).

Es el mismo controlador PID de la diapositiva ("Implementacion"):
    error = izq_value - der_value
    vL = v + (kp*e + ki*E + kd*(e - e_ant))
    vR = v - (kp*e + ki*E + kd*(e - e_ant))
    velocidad de rueda = v_lineal / radio

El robot (e-puck): dos ruedas motrices con motores independientes, una rueda
loca atras (solo apoya, no restringe el movimiento) y dos sensores
infrarrojos de piso al frente. La pista es un anillo negro sobre piso blanco.

Para que kp, ki y kd se NOTEN, el modelo incluye lo que hace dificil el control
real: inercia del motor (retardo de primer orden), retardo de lectura del sensor,
ruido, una huella de sensor con borde suave y el limite de velocidad del motor.

Todo se muestra en UNA ventana de matplotlib: la pista con el robot, el error
de los sensores, las velocidades de rueda, indicadores de desempeno y sliders
para cambiar kp, ki, kd, v, radio, inercia y retardo mientras corre.

Uso:
    python epuck_seguidor_sim.py                 # ventana interactiva
    python epuck_seguidor_sim.py --kp 0.002      # otras ganancias de partida
    python epuck_seguidor_sim.py --headless 60 --guardar corrida.png   # sin ventana

Requisitos: numpy y matplotlib.
"""
import argparse
import math
from collections import deque

import numpy as np

# ---------------------------------------------------------------- constantes
# e-puck (datos del modelo de Webots)
R_RUEDA = 0.0205          # m, radio REAL de la rueda
L_EJE = 0.052             # m, separacion entre ruedas
R_CUERPO = 0.037          # m, radio del cuerpo (diametro 7.4 cm)
MAX_RUEDA = 6.28          # rad/s, velocidad maxima del motor (Webots la recorta)
X_CASTER = -0.030         # m, la rueda loca esta atras (solo dibujo)

# Sensores de piso, medidos desde el centro del robot (x adelante, y a la izquierda)
SENS_X = 0.030
SENS_Y = 0.015
SENS_R = 0.016            # m, radio de la huella que "ve" cada sensor (borde suave)

# Pista: anillo negro de radio R_PISTA y ancho ANCHO_LINEA
R_PISTA = 0.40
ANCHO_LINEA = 0.030

# Lectura del infrarrojo (unidades arbitrarias como las del sensor de Webots)
BLANCO = 1000.0
NEGRO = 300.0
RUIDO = 8.0

DT = 0.032                # s, paso de tiempo (el basicTimeStep por defecto de Webots)

# Dinamica no ideal (ajustables con sliders)
TAU_MOTOR = 0.10          # s, inercia del motor: la rueda tarda en alcanzar la orden
RETARDO = 2               # pasos de simulacion de retraso en la lectura (2 * 32 ms)

# Parametros del controlador, TAL CUAL en la diapositiva
KP, KI, KD = 0.001, 0.00001, 0.0001
V_LINEAL = 0.5
RADIO_CODIGO = 0.07       # "radio" del codigo de la diapositiva


# ------------------------------------------------------------- controlador
class Controlador:
    """El lazo de la diapositiva, linea por linea."""

    def __init__(self, kp=KP, ki=KI, kd=KD, v=V_LINEAL, radio=RADIO_CODIGO):
        self.kp, self.ki, self.kd, self.v, self.radio = kp, ki, kd, v, radio
        self.reset()

    def reset(self):
        self.enew = 0.0
        self.eold = 0.0
        self.E = 0.0

    def paso(self, izq_value, der_value):
        error = izq_value - der_value
        self.enew = error
        vL = self.v + self.kp * self.enew + self.ki * self.E + self.kd * (self.enew - self.eold)
        vR = self.v - (self.kp * self.enew + self.ki * self.E + self.kd * (self.enew - self.eold))
        self.E += self.enew
        izq_vel = vL / self.radio
        der_vel = vR / self.radio
        self.eold = self.enew
        return izq_vel, der_vel, error


# ----------------------------------------------------------------- mundo
def _puntos_huella(n_anillo=(6, 12, 18)):
    """Puntos de muestreo dentro de la huella circular del sensor."""
    pts = [(0.0, 0.0)]
    for k, n in enumerate(n_anillo, start=1):
        r = SENS_R * k / len(n_anillo)
        for j in range(n):
            a = 2 * math.pi * j / n
            pts.append((r * math.cos(a), r * math.sin(a)))
    return np.array(pts)


_HUELLA = _puntos_huella()


class Mundo:
    """Robot diferencial + pista circular."""

    def __init__(self, ctrl: Controlador, semilla=3, ruido=RUIDO, tau=TAU_MOTOR, retardo=RETARDO):
        self.ctrl = ctrl
        self.rng = np.random.default_rng(semilla)
        self.ruido = ruido
        self.tau = tau
        self.retardo = retardo
        self.reset()

    def reset(self):
        # abajo de la pista, mirando en sentido antihorario, 1 cm hacia adentro
        self.x, self.y, self.th = 0.0, -(R_PISTA - 0.010), 0.0
        self.t = 0.0
        self.ang_acum = 0.0
        self._ang_prev = math.atan2(self.y, self.x)
        self._buf = deque()                       # lecturas retrasadas
        self.izq = self.der = BLANCO
        self.err = 0.0
        self.w_izq = self.w_der = 0.0             # velocidad REAL de rueda (con inercia)
        self.cmd_izq = self.cmd_der = 0.0         # orden ya recortada al limite del motor
        self.v = self.w = 0.0
        self.dist_linea = abs(math.hypot(self.x, self.y) - R_PISTA)
        self.sat = False
        self._h_err = deque(maxlen=94)            # ~3 s
        self._h_sat = deque(maxlen=94)
        self._h_ang = deque(maxlen=94)
        self.ctrl.reset()

    # un sensor de piso: fraccion de su huella que cae sobre la linea negra
    def _leer(self, lado):
        c, s = math.cos(self.th), math.sin(self.th)
        cx = self.x + SENS_X * c - lado * SENS_Y * s
        cy = self.y + SENS_X * s + lado * SENS_Y * c
        px = cx + _HUELLA[:, 0]
        py = cy + _HUELLA[:, 1]
        r = np.hypot(px, py)
        cubierto = np.abs(r - R_PISTA) < ANCHO_LINEA / 2
        f = cubierto.mean()
        return BLANCO + f * (NEGRO - BLANCO) + self.rng.normal(0.0, self.ruido)

    def paso(self):
        # lectura con retardo (el dato llega `retardo` pasos despues)
        self._buf.append((self._leer(+1), self._leer(-1)))
        while len(self._buf) > int(self.retardo) + 1:
            self._buf.popleft()
        self.izq, self.der = self._buf[0]

        wi, wd, self.err = self.ctrl.paso(self.izq, self.der)
        # el motor no pasa de MAX_RUEDA (Webots recorta la velocidad)
        self.sat = abs(wi) > MAX_RUEDA or abs(wd) > MAX_RUEDA
        self.cmd_izq = max(-MAX_RUEDA, min(MAX_RUEDA, wi))
        self.cmd_der = max(-MAX_RUEDA, min(MAX_RUEDA, wd))
        # inercia del motor: respuesta de primer orden a la orden
        a = min(1.0, DT / max(self.tau, 1e-3))
        self.w_izq += (self.cmd_izq - self.w_izq) * a
        self.w_der += (self.cmd_der - self.w_der) * a
        # cinematica diferencial; la rueda loca no aporta restricciones
        v_izq = self.w_izq * R_RUEDA
        v_der = self.w_der * R_RUEDA
        v = 0.5 * (v_izq + v_der)
        w = (v_der - v_izq) / L_EJE
        th_med = self.th + 0.5 * w * DT
        self.x += v * math.cos(th_med) * DT
        self.y += v * math.sin(th_med) * DT
        self.th += w * DT
        self.t += DT
        # vueltas completadas y distancia a la linea
        ang = math.atan2(self.y, self.x)
        d = (ang - self._ang_prev + math.pi) % (2 * math.pi) - math.pi
        self.ang_acum += d
        self._ang_prev = ang
        self.dist_linea = abs(math.hypot(self.x, self.y) - R_PISTA)
        self.v = v
        self.w = w
        self._h_err.append(self.err)
        self._h_sat.append(self.sat)
        self._h_ang.append(d)

    # ---- indicadores de desempeno (ultimos ~3 s)
    @property
    def vueltas(self):
        return self.ang_acum / (2 * math.pi)

    @property
    def oscilacion(self):
        """Desviacion estandar del error de los sensores: zigzag."""
        return float(np.std(self._h_err)) if len(self._h_err) > 5 else 0.0

    @property
    def saturacion(self):
        """Fraccion del tiempo con alguna rueda pidiendo mas que el limite del motor."""
        return float(np.mean(self._h_sat)) if self._h_sat else 0.0

    @property
    def avance(self):
        """Velocidad angular alrededor de la pista (rad/s): cuanto avanza de verdad."""
        return sum(self._h_ang) / (len(self._h_ang) * DT) if self._h_ang else 0.0

    @property
    def fuera(self):
        return self.dist_linea > ANCHO_LINEA / 2 + 0.03

    @property
    def girando_en_sitio(self):
        return self.t > 4.0 and abs(self.v) < 0.02 and not self.fuera

    @property
    def diagnostico(self):
        if self.fuera:
            return "FUERA DE LA PISTA"
        if self.girando_en_sitio:
            return "GIRANDO SIN AVANZAR"
        return ""


# ------------------------------------------------------------ modo sin ventana
def correr_headless(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ctrl = Controlador(args.kp, args.ki, args.kd, args.v, args.radio)
    mundo = Mundo(ctrl, tau=args.tau, retardo=args.retardo)
    T, ERR, DIST, X, Y = [], [], [], [], []
    for _ in range(int(args.headless / DT)):
        mundo.paso()
        T.append(mundo.t); ERR.append(mundo.err); DIST.append(mundo.dist_linea)
        X.append(mundo.x); Y.append(mundo.y)
    DIST = np.array(DIST); ERR = np.array(ERR)
    print(f"tiempo simulado : {mundo.t:.1f} s")
    print(f"vueltas         : {mundo.vueltas:.2f}")
    print(f"dist. a la linea: media {DIST.mean() * 100:.2f} cm, maxima {DIST.max() * 100:.2f} cm")
    print(f"zigzag (std e)  : {ERR.std():.0f}")
    print(f"diagnostico     : {mundo.diagnostico or 'sigue la linea'}")
    if args.guardar:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
        t = np.linspace(0, 2 * math.pi, 200)
        a1.plot(R_PISTA * np.cos(t), R_PISTA * np.sin(t), "k", lw=3, alpha=0.5)
        a1.plot(X, Y, c="#0277BD"); a1.set_aspect("equal"); a1.grid(alpha=0.3)
        a1.set_title("Trayectoria")
        a2.plot(T, ERR, c="#C49000"); a2.set_xlabel("t (s)")
        a2.set_ylabel("error = izq - der"); a2.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(args.guardar, dpi=130)
        print("grafica:", args.guardar)


# ------------------------------------------------------------------ ventana
def correr_ventana(args):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle, Polygon, Rectangle, Wedge
    from matplotlib.transforms import Affine2D
    from matplotlib.widgets import Button, Slider

    ctrl = Controlador(args.kp, args.ki, args.kd, args.v, args.radio)
    mundo = Mundo(ctrl, tau=args.tau, retardo=args.retardo)

    fig = plt.figure(figsize=(13.5, 8.0))
    fig.canvas.manager.set_window_title("e-puck siguiendo una linea circular")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], left=0.05, right=0.97,
                          top=0.95, bottom=0.34, hspace=0.38, wspace=0.18)
    ax = fig.add_subplot(gs[:, 0])
    ax_e = fig.add_subplot(gs[0, 1])
    ax_w = fig.add_subplot(gs[1, 1])

    # ---- pista y robot
    ax.set_aspect("equal")
    lim = R_PISTA + 0.10
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_facecolor("#F4F4F0")
    ax.add_patch(Wedge((0, 0), R_PISTA + ANCHO_LINEA / 2, 0, 360, width=ANCHO_LINEA, color="#1F1F1F", zorder=1))
    ax.set_title("Pista circular y e-puck", fontsize=12)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.grid(alpha=0.25)

    cuerpo = Circle((0, 0), R_CUERPO, fc="#CFE8F7", ec="#0277BD", lw=1.5, zorder=3)
    rueda_i = Rectangle((-0.0205, L_EJE / 2 - 0.004), 0.041, 0.008, fc="#333333", zorder=4)
    rueda_d = Rectangle((-0.0205, -L_EJE / 2 - 0.004), 0.041, 0.008, fc="#333333", zorder=4)
    loca = Circle((X_CASTER, 0), 0.005, fc="#888888", ec="#444444", zorder=4)
    frente = Polygon([[R_CUERPO - 0.002, 0], [R_CUERPO - 0.014, 0.008], [R_CUERPO - 0.014, -0.008]],
                     fc="#0277BD", zorder=4)
    sens_i = Circle((SENS_X, SENS_Y), 0.0045, fc="white", ec="#C62828", lw=1.2, zorder=5)
    sens_d = Circle((SENS_X, -SENS_Y), 0.0045, fc="white", ec="#2E7D32", lw=1.2, zorder=5)
    partes = [cuerpo, rueda_i, rueda_d, loca, frente, sens_i, sens_d]
    for p in partes:
        ax.add_patch(p)
    estela, = ax.plot([], [], c="#0277BD", lw=1.2, alpha=0.6, zorder=2)
    traza = deque(maxlen=2500)

    texto = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                    family="monospace", fontsize=8.5,
                    bbox=dict(boxstyle="round", fc="white", ec="#BBBBBB", alpha=0.92), zorder=10)
    aviso = ax.text(0.5, 0.5, "", transform=ax.transAxes, ha="center", va="center",
                    fontsize=14, color="#C62828", fontweight="bold", zorder=11,
                    bbox=dict(boxstyle="round", fc="white", ec="#C62828", alpha=0.85))
    aviso.set_visible(False)

    # ---- graficas de la derecha
    N = 600
    t_hist, e_hist, wi_hist, wd_hist = deque(maxlen=N), deque(maxlen=N), deque(maxlen=N), deque(maxlen=N)
    ln_e, = ax_e.plot([], [], c="#C49000", lw=1.4)
    ax_e.axhline(0, ls="--", c="gray", lw=0.8)
    ax_e.set_ylabel("error = izq - der"); ax_e.set_title("Error de los sensores", fontsize=11)
    ax_e.grid(alpha=0.3)
    ln_wi, = ax_w.plot([], [], c="#C62828", lw=1.3, label="rueda izq")
    ln_wd, = ax_w.plot([], [], c="#2E7D32", lw=1.3, label="rueda der")
    ax_w.axhline(MAX_RUEDA, ls=":", c="gray", lw=0.9)
    ax_w.text(0.01, MAX_RUEDA, " limite del motor", va="bottom", fontsize=8, color="gray",
              transform=ax_w.get_yaxis_transform())
    ax_w.set_ylim(-MAX_RUEDA * 0.35, MAX_RUEDA * 1.18)
    ax_w.set_xlabel("tiempo (s)"); ax_w.set_ylabel("velocidad real (rad/s)")
    ax_w.set_title("Velocidad de cada rueda", fontsize=11)
    ax_w.legend(loc="lower left", fontsize=8); ax_w.grid(alpha=0.3)

    # ---- sliders (dos columnas) y botones
    def slider(rect, label, vmin, vmax, v0, fmt):
        a = fig.add_axes(rect)
        return Slider(a, label, vmin, vmax, valinit=v0, valfmt=fmt)
    s_kp = slider([0.10, 0.25, 0.28, 0.025], "kp", 0.0, 0.01, ctrl.kp, "%.5f")
    s_ki = slider([0.10, 0.21, 0.28, 0.025], "ki", 0.0, 0.0002, ctrl.ki, "%.6f")
    s_kd = slider([0.10, 0.17, 0.28, 0.025], "kd", 0.0, 0.01, ctrl.kd, "%.5f")
    s_v = slider([0.10, 0.13, 0.28, 0.025], "v", 0.0, 1.0, ctrl.v, "%.2f")
    s_rad = slider([0.10, 0.09, 0.28, 0.025], "radio (codigo)", 0.0205, 0.10, ctrl.radio, "%.4f")
    s_tau = slider([0.60, 0.25, 0.28, 0.025], "inercia motor (s)", 0.01, 0.40, mundo.tau, "%.2f")
    s_ret = slider([0.60, 0.21, 0.28, 0.025], "retardo sensor (pasos)", 0, 6, mundo.retardo, "%d")
    s_x = slider([0.60, 0.17, 0.28, 0.025], "pasos/cuadro", 1, 10, 2, "%d")
    s_kp.on_changed(lambda val: setattr(ctrl, "kp", val))
    s_ki.on_changed(lambda val: setattr(ctrl, "ki", val))
    s_kd.on_changed(lambda val: setattr(ctrl, "kd", val))
    s_v.on_changed(lambda val: setattr(ctrl, "v", val))
    s_rad.on_changed(lambda val: setattr(ctrl, "radio", val))
    s_tau.on_changed(lambda val: setattr(mundo, "tau", val))
    s_ret.on_changed(lambda val: setattr(mundo, "retardo", int(round(val))))

    estado = {"pausa": False}
    b_reset = Button(fig.add_axes([0.60, 0.10, 0.10, 0.04]), "Reiniciar")
    b_pausa = Button(fig.add_axes([0.72, 0.10, 0.10, 0.04]), "Pausa")

    def reiniciar(_=None):
        mundo.reset()
        traza.clear(); t_hist.clear(); e_hist.clear(); wi_hist.clear(); wd_hist.clear()

    def pausar(_=None):
        estado["pausa"] = not estado["pausa"]
        b_pausa.label.set_text("Seguir" if estado["pausa"] else "Pausa")

    b_reset.on_clicked(reiniciar)
    b_pausa.on_clicked(pausar)
    fig.text(0.05, 0.03,
             "Robot de 2 ruedas con rueda loca atras y sensores de piso al frente. Controlador de la diapositiva: "
             "vL = v + PID(e), vR = v - PID(e), e = izq - der.\n"
             "Prueba: kp bajo pierde la curva; kp alto gira en el sitio; kd amortigua el zigzag; "
             "ki alto acumula y se sale; mas retardo o v muy alta lo desestabilizan.",
             fontsize=8.5, color="#555555")

    def gris(valor):
        g = max(0.0, min(1.0, (valor - NEGRO) / (BLANCO - NEGRO)))
        return (g, g, g)

    def actualizar(_frame):
        if not estado["pausa"]:
            for _ in range(int(s_x.val)):
                mundo.paso()
                traza.append((mundo.x, mundo.y))
                t_hist.append(mundo.t); e_hist.append(mundo.err)
                wi_hist.append(mundo.w_izq); wd_hist.append(mundo.w_der)
        tr = Affine2D().rotate(mundo.th).translate(mundo.x, mundo.y) + ax.transData
        for p in partes:
            p.set_transform(tr)
        sens_i.set_facecolor(gris(mundo.izq))
        sens_d.set_facecolor(gris(mundo.der))
        if traza:
            xs, ys = zip(*traza)
            estela.set_data(xs, ys)
        texto.set_text(
            f"t = {mundo.t:6.1f} s   vueltas = {mundo.vueltas:5.2f}\n"
            f"sensor izq = {mundo.izq:6.0f}   der = {mundo.der:6.0f}   error = {mundo.err:+6.0f}\n"
            f"rueda izq = {mundo.w_izq:5.2f}   der = {mundo.w_der:5.2f} rad/s\n"
            f"v real = {mundo.v:5.3f} m/s   dist. a la linea = {mundo.dist_linea * 100:4.1f} cm\n"
            f"zigzag (std error, 3 s) = {mundo.oscilacion:5.0f}\n"
            f"motor al limite = {mundo.saturacion * 100:3.0f} % del tiempo\n"
            f"avance alrededor de la pista = {mundo.avance:5.2f} rad/s")
        diag = mundo.diagnostico
        aviso.set_text(diag)
        aviso.set_visible(bool(diag))
        if t_hist:
            ts = list(t_hist)
            ln_e.set_data(ts, list(e_hist))
            ln_wi.set_data(ts, list(wi_hist)); ln_wd.set_data(ts, list(wd_hist))
            for a in (ax_e, ax_w):
                a.set_xlim(ts[0], max(ts[0] + 5, ts[-1]))
            em = max(100.0, max(abs(v) for v in e_hist) * 1.2)
            ax_e.set_ylim(-em, em)
        return []

    anim = FuncAnimation(fig, actualizar, interval=30, blit=False, cache_frame_data=False)
    fig._anim = anim          # evita que el recolector de basura lo borre
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kp", type=float, default=KP)
    ap.add_argument("--ki", type=float, default=KI)
    ap.add_argument("--kd", type=float, default=KD)
    ap.add_argument("--v", type=float, default=V_LINEAL, help="velocidad lineal del codigo")
    ap.add_argument("--radio", type=float, default=RADIO_CODIGO, help="'radio' del codigo de la diapositiva")
    ap.add_argument("--tau", type=float, default=TAU_MOTOR, help="inercia del motor (s)")
    ap.add_argument("--retardo", type=int, default=RETARDO, help="retardo del sensor (pasos de 32 ms)")
    ap.add_argument("--headless", type=float, default=0.0, metavar="SEG",
                    help="simula SEG segundos sin ventana e imprime el resumen")
    ap.add_argument("--guardar", default=None, help="con --headless: guarda un PNG")
    args = ap.parse_args()
    if args.headless > 0:
        correr_headless(args)
    else:
        correr_ventana(args)


if __name__ == "__main__":
    main()
