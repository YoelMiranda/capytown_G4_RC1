# CapyTown G4 · RC-1 — Control PID: del simulador al robot real

Robótica (08079) · Universidad ESAN · 2026-2 · Sección S-003 · Robot Yahboom MicroROS-Pi5.
Nodo de la clase `scripts_pdi/cuadrado_controller.py` (sin modificar), probado en el simulador
`yahboomcar_sim` (repo `codeplai/microrospi5-sim`) y en el robot real.

**Informe:** [`informe.pdf`](informe.pdf) (1 página; fuente en `docs/informe.html`).

## Integrantes y roles

Cada integrante mantiene el rol del RC-0, como pide el enunciado.

| Rol | Integrante | Responsabilidad en RC-1 |
|---|---|---|
| Integración | Simón Contreras | Simulador y bringup (ROS_DOMAIN_ID), repositorio y tag, bags del robot real, hoja de resultados |
| Control | Aaron Cruzado | Ganancias Kp, Ki, Kd de la línea y del giro; orden de sintonía; `config/rc1_pid.yaml` |
| Planificación | Yoel Miranda | Predicciones, matriz de pruebas (un parámetro por prueba), secuencia de seguridad, mediciones con cinta |
| Percepción | Joaquín Rodríguez | Gráficas con `09_plot_csv.py` y `cierre_real.py`; comparación /odom frente a lo real |

## Configuración

| Dato | Valor |
|---|---|
| `ROS_DOMAIN_ID` | **20** |
| Simulador | `ros2 launch yahboomcar_sim sim.launch.py world:=arena_rc1 start:=1.0,1.0,0.0 robot_radius:=0.14 noise:=<nivel>` |
| Ruido usado | `high` en la Parte A · **`low` en las Partes B–D** · `medium` en la corrida de estrés del Exp. 3 |
| Robot real | `ros2 launch capytown_esan bringup.launch.py`; el nodo lee `/odom` (llega a ≈ 11 Hz según los bags) |

## Ganancias finales

| | Simulador | Robot real |
|---|---|---|
| Recta (línea) | Kp 3.0 · Ki 0.5 · Kd 4.0 · v 0.20 m/s | Kp 2.0 · Ki 0.0 · Kd 6.0 · v 0.08 m/s |
| Giro 90° | kp_yaw 5.0 · ki_yaw 0.0 · kd_yaw 0.4 | kp_yaw 3.0 · ki_yaw 0.0 · kd_yaw 0.4 · ω ≤ 1.0 |
| Cuadrado 3B | línea Kp 3 / Kd 4; giro 3.0 / 0.4 (valores del nodo) | línea Kp 2 / Kd 6; giro 3.0 / 0.4; ω ≤ 2.0 |

Detalle en [`config/rc1_pid.yaml`](config/rc1_pid.yaml).

## Resultados principales

| Prueba | Simulador | Robot real (medido) |
|---|---|---|
| 1D recta | error final 0.31 cm, asentamiento 5.77 s | desvío con cinta 3.5 / 1.8 / 1.4 / 2.2 cm |
| Giro 90° | prop2: 0 % sobreimpulso, 1.37 s | transportador 87° / 88° / 86° (odom ≈ 90°) |
| 3A lazo abierto | cierre real 26.9 cm (odom 2.5) | cinta 29.5 cm (odom 34.4) |
| 3B odom + PID | cierre real 32.5 cm (odom 0.3) | cinta 18.5 / 22.0 / 16.5 → **19.0 cm** (odom 2.0) |
| 3C PID + Ki | cierre real 33.1 cm (odom 0.3) | — |

Idea central: el nodo informa cierres de pocos cm porque corrige lo que ve `/odom`; lo que
`/odom` no ve (el empujón del piso en el simulador; el giro 2–4° corto y el residuo lateral
en el robot) sigue ahí.

## Estructura

```
informe.pdf            sim_log.txt            rqt_graph_sim.png
config/rc1_pid.yaml    bloques simulador y robot_real
csv/                   un CSV por prueba (exp*_sim*.csv, exp*_real*.csv) + hoja_resultados.csv
csv/desde_bag/         /odom y /cmd_vel extraídos de los bags (corridas sin CSV del nodo)
plots/                 PNG de 09_plot_csv.py y cierre_real.py (sim) y de tools/analizar.py (robot)
bags/                  bags del robot real (carpetas rosbag2, sqlite3)
logs/                  salida completa del nodo y de cierre_real.py por corrida del Exp. 3
tools/                 bag2csv.py (decodifica bags) y analizar.py (métricas y gráficas del robot)
scripts_pdi/           material del curso: cuadrado_controller.py + core_cuadrado.py (sin modificar),
                       09_plot_csv.py (copia con PARCHE RC-1) y cierre_real.py
```

## Correspondencia entre CSV, bag y medición (robot real)

Las horas salen del reloj de la Pi y de la pose inicial de `/odom` de cada archivo.

| Prueba | CSV del nodo | Bag | Medición |
|---|---|---|---|
| Giro int. 1 | `exp2_real.csv` | — | 87° |
| Giro int. 2 | `exp2_real_intento2.csv` | — | 88° |
| Giro (04:09) | — (`desde_bag/rc1_giro90_run1_desde_bag.csv`) | `rc1_giro90_run1` | — |
| Giro int. 3 | `exp2_real_intento3.csv` | — | 86° |
| 1D run1 | `exp1d_real_run1.csv` | — | 3.5 cm |
| 1D run2 | — (`desde_bag/rc1_real_1d_run1_desde_bag.csv`) | `rc1_real_1d_run1` | 1.8 cm |
| 1D run3 | — (`desde_bag/rc1_real_1d_run2_desde_bag.csv`) | `rc1_real_1d_run2` | 1.4 cm |
| 1D run4 | `exp1d_real_run2.csv` | `rc1_real_1d_run3` | 2.2 cm |
| 3A | `exp3a_real.csv` | **no se grabó** | 29.5 cm |
| 3B (07:29) | — (`desde_bag/rc1_real_3b_run1_desde_bag.csv`) | `rc1_real_3b_run1` | — |
| 3B run1 | `exp3b_real_run1.csv` | — | 18.5 cm |
| 3B run2 | `exp3b_real_run2.csv` | `rc1_real_3b_run2` | 22.0 cm |
| 3B run3 | `exp3b_real_run3.csv` | `rc1_real_3b_run3` | 16.5 cm |

## Evidencia faltante (declarada)

- No hay bag de 3A en el robot real (`bags/rc1_real_3a/`); la evidencia de 3A es el CSV del nodo y la medición con cinta.
- 3B run1 tiene CSV y cinta pero no bag; el bag `rc1_real_3b_run1` corresponde a otra corrida de 3B, sin medición.

## Reproducir las métricas del robot

```bash
pip install rosbags numpy matplotlib
python3 tools/bag2csv.py csv/desde_bag bags/rc1_*          # /odom y /cmd_vel de cada bag
python3 tools/analizar.py csv csv/desde_bag plots           # métricas (logs/metricas_real.json) y PNG
```
