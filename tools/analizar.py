import sys, csv, math, json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9,"axes.edgecolor":"#9a9993","axes.labelcolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e","axes.grid":True,"grid.color":"#e6e5e0","grid.linewidth":0.6,"figure.facecolor":"#fcfcfb","axes.facecolor":"#fcfcfb","axes.spines.top":False,"axes.spines.right":False})
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
SRC, BAGS, OUT = map(Path, sys.argv[1:4])
def norm(a): return math.atan2(math.sin(a), math.cos(a))
res = {}
def rel(t, x, y, th, i0=0):
    x0,y0,a0 = x[i0],y[i0],th[i0]; dx,dy = x-x0, y-y0
    X = dx*math.cos(a0)+dy*math.sin(a0); Y = -dx*math.sin(a0)+dy*math.cos(a0)
    return X, Y, np.unwrap([norm(v-a0) for v in th])
def settle(t, err, tol):
    out = np.nonzero(abs(err) > tol)[0]
    if not out.size: return float(t[0])
    return None if out[-1]+1 >= len(t) else float(t[out[-1]+1])
def save(fig, name): fig.tight_layout(); fig.savefig(OUT/name, dpi=130); plt.close(fig)

def from_csv(name):
    r = list(csv.DictReader(open(SRC/name))); est = np.array([x["estado"] for x in r])
    col = lambda k: np.array([float(x[k]) for x in r])
    t,x,y,th,w,e = col("t"),col("x"),col("y"),col("yaw"),col("w"),col("e_line")
    fin = np.nonzero(est=="FIN")[0]; k = int(fin[0]) if fin.size else len(t)
    return dict(t=t,x=x,y=y,th=th,w=w,e=e,est=est,k=k,v=col("v"))
def from_bag(name):
    r = list(csv.DictReader(open(BAGS/name)))
    od=[q for q in r if q["topic"]=="odom"]; cm=[q for q in r if q["topic"]=="cmd"]
    tc=np.array([float(q["t"]) for q in cm]); wc=np.array([float(q["w"]) for q in cm]); vc=np.array([float(q["v"]) for q in cm])
    act=np.nonzero((abs(vc)>1e-6)|(abs(wc)>1e-6))[0]; t0,t1=tc[act[0]],tc[act[-1]]
    t=np.array([float(q["t"]) for q in od]); x,y,th=[np.array([float(q[k]) for q in od]) for k in ("x","y","yaw")]
    i0=max(np.searchsorted(t,t0)-1,0); i1=min(np.searchsorted(t,t1+1.5),len(t)-1)
    s=slice(i0,i1+1)
    return dict(t=t[s]-t0,x=x[s],y=y[s],th=th[s],vmax=float(vc.max()),wmax=float(abs(wc).max()),dur=float(t1-t0),
                odom_hz=len(t)/(t[-1]-t[0]))

# ---- recta
def recta(tag, d, src):
    if src=="csv":
        k=d["k"]; t=d["t"][:k]; e=d["e"][:k]*100; X,Y,_=rel(d["t"],d["x"],d["y"],d["th"]); X,Y=X[:k],Y[:k]; vmax=float(d["v"].max()); wmax=float(abs(d["w"]).max())
    else:
        X,Y,_=rel(d["t"],d["x"],d["y"],d["th"]); t=d["t"]; e=(Y+0.10)*100; vmax=d["vmax"]; wmax=d["wmax"]
    m=dict(fuente=src, e_medio_cm=round(float(abs(e).mean()),2), e_max_cm=round(float(abs(e).max()),2),
           t_asent_s=settle(t,e,2.0), e_final_cm=round(float(e[-1]),2), avance_m=round(float(X[-1]),3), v=vmax, wmax=wmax)
    fig,(a1,a2)=plt.subplots(1,2,figsize=(9,3.4))
    a1.plot(X,Y*100,c=BLUE,lw=2); a1.axhline(-10,ls="--",c=GRAY,lw=1); a1.plot(0,0,"o",c=ORANGE,ms=6)
    a1.set_xlabel("avance x (m)"); a1.set_ylabel("lateral y (cm)"); a1.set_title("Trayectoria según /odom (línea virtual en y = −10 cm)",fontsize=9)
    a2.plot(t,e,c=BLUE,lw=2); a2.axhspan(-2,2,color="#1baf7a",alpha=0.10); a2.axhline(0,ls="--",c=GRAY,lw=1)
    a2.set_xlabel("tiempo (s)"); a2.set_ylabel("error a la línea (cm)"); a2.set_title("Error lateral e(t) — banda ±2 cm",fontsize=9)
    fig.suptitle(f"{tag} · recta 1D · robot real ({'CSV del nodo' if src=='csv' else 'reconstruido del bag, /odom ~11 Hz'})",fontsize=10)
    save(fig, tag+".png"); res[tag]=m

# ---- giro
def giro(tag, d, src):
    if src=="csv":
        k=d["k"]; t=d["t"]; _,_,yw=rel(d["t"],d["x"],d["y"],d["th"]); kend=min(len(t), k+90)  # FIN + 3 s
        t,yw=t[:kend],np.degrees(yw[:kend]); wmax=float(abs(d["w"]).max())
    else:
        _,_,yw=rel(d["t"],d["x"],d["y"],d["th"]); t=d["t"]; yw=np.degrees(yw); wmax=d["wmax"]
    m=dict(fuente=src, sobreimpulso_pct=round(max(0.0,(yw.max()-90)/90*100),2), t_asent_s=settle(t,yw-90,2.0),
           giro_final_deg=round(float(yw[-1]),2), e_final_deg=round(float(90-yw[-1]),2), wmax=wmax)
    fig,a=plt.subplots(figsize=(5.2,3.3))
    a.plot(t,yw,c=BLUE,lw=2); a.axhline(90,ls="--",c=GRAY,lw=1); a.axhspan(88,92,color="#1baf7a",alpha=0.10)
    a.set_xlabel("tiempo (s)"); a.set_ylabel("yaw según /odom (°)"); a.set_title(f"{tag} · giro 90° · robot real",fontsize=10)
    save(fig, tag+".png"); res[tag]=m

# ---- cuadrado
def cuadrado(tag, d, src, abierto=False):
    if src=="csv": X,Y,yw=rel(d["t"],d["x"],d["y"],d["th"]); dur=float(d["t"][d["k"]])
    else: X,Y,yw=rel(d["t"],d["x"],d["y"],d["th"]); dur=d["dur"]
    m=dict(fuente=src, cierre_odom_cm=round(100*math.hypot(X[-1],Y[-1]),1), yaw_odom_deg=round(math.degrees(yw[-1]),1), t_s=round(dur,1))
    fig,a=plt.subplots(figsize=(4.4,4.4))
    a.plot([0,2,2,0,0],[0,0,2,2,0],"--",c=GRAY,lw=1.2,label="cuadrado ideal")
    a.plot(X,Y,c=BLUE,lw=2,label="según /odom"); a.plot(0,0,"o",c=ORANGE,ms=6,label="inicio")
    a.set_aspect("equal"); a.legend(fontsize=7,loc="center"); a.set_xlabel("x (m)"); a.set_ylabel("y (m)")
    a.set_title(f"{tag} · {'lazo abierto' if abierto else 'odom + PID'}\ncierre /odom {m['cierre_odom_cm']} cm (el real se mide con cinta)",fontsize=9)
    save(fig, tag+".png"); res[tag]=m

for n in ("exp1d_real_run1","exp1d_real_run2"): recta(n, from_csv(n+".csv"), "csv")
for n in ("rc1_real_1d_run1","rc1_real_1d_run2"): recta(n+"_bag", from_bag(n+"_desde_bag.csv"), "bag")
for n in ("exp2_real","exp2_real_intento2","exp2_real_intento3"): giro(n, from_csv(n+".csv"), "csv")
giro("rc1_giro90_run1_bag", from_bag("rc1_giro90_run1_desde_bag.csv"), "bag")
cuadrado("exp3a_real", from_csv("exp3a_real.csv"), "csv", True)
for n in ("exp3b_real_run1","exp3b_real_run2","exp3b_real_run3"): cuadrado(n, from_csv(n+".csv"), "csv")
cuadrado("rc1_real_3b_run1_bag", from_bag("rc1_real_3b_run1_desde_bag.csv"), "bag")
json.dump(res, open(OUT/"metricas_real.json","w"), indent=1)
for k,v in res.items(): print(k, v)
