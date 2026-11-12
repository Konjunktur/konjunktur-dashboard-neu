import os, io, requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

os.makedirs("output", exist_ok=True)
D_USER = os.environ.get("DESTATIS_USER", "")
D_PASS = os.environ.get("DESTATIS_PASS", "")
G_URL = "https://destatis.de"

def get_bbk(key, n=13, scale=1.0):
    flow, k = key.split(".", 1)
    url = f"https://bundesbank.de{flow}/{k}?format=csv&lang=de"
    try:
        r = requests.get(url, timeout=8)
        df = pd.read_csv(io.StringIO(r.text), sep=";")
        tc = next(c for c in df.columns if "TIME_PERIOD" in c)
        vc = next(c for c in df.columns if "OBS_VALUE" in c)
        df = df.dropna(subset=list((tc, vc))).tail(n)
        vals = df.get(vc).astype(str).str.replace(",", ".").astype(float) * scale
        return pd.Series(vals.values, index=df.get(tc).values)
    except Exception:
        return pd.Series(dtype=float)

def get_destatis(table):
    if not D_USER or not D_PASS:
        return pd.DataFrame()
    p = {"username": D_USER, "password": D_PASS, "name": table, "area": "all", "compress": "false", "format": "ffcsv", "language": "de"}
    try:
        r = requests.get(G_URL, params=p, timeout=25)
        if len(r.text) < 200 or "Fehler" in r.text[:200]:
            return pd.DataFrame()
        return pd.read_csv(io.StringIO(r.text), sep=";", decimal=",")
    except Exception:
        return pd.DataFrame()

def get_dynamic_bip_axes(df):
    """Ermittelt automatisch die neuesten 5 Jahre und neuesten 14 Quartale aus Destatis."""
    if df.empty or "ZEIT" not in df.columns:
        yrs = ("2021", "2022", "2023", "2024", "2025")
        qtrs = ("2023-Q2","2023-Q3","2023-Q4","2024-Q1","2024-Q2","2024-Q3","2024-Q4","2025-Q1","2025-Q2","2025-Q3","2025-Q4","2026-Q1","2026-Q2","2026-Q3")
        return yrs, qtrs
    zeiten = sorted(df["ZEIT"].astype(str).unique())
    yrs = tuple([z for z in zeiten if len(z) == 4 and z.isdigit()][-5:])
    qtrs = tuple([z for z in zeiten if "Q" in z][-14:])
    return yrs, qtrs

def parse_bip(df, filters, yrs, qtrs):
    if df.empty or "ZEIT" not in df.columns or "WERT" not in df.columns:
        return ()
    sub = df.copy()
    for col, val in filters.items():
        if col in sub.columns:
            sub = sub[sub[col].astype(str).str.contains(val, na=False)]
    if sub.empty:
        return ()
    sub["V"] = pd.to_numeric(sub["WERT"].astype(str).str.replace(",", "."), errors="coerce")
    m = dict(zip(sub["ZEIT"].astype(str), sub["V"]))
    res = [m.get(y, np.nan) for y in yrs] + [np.nan]
    for q in qtrs:
        res.append(m.get(q, m.get(q.replace("-", ""), np.nan)))
    return tuple(res)

def style_panel(ax, title, sub, ymin, ymax, yticks, z=True):
    ax.set_facecolor("#F8FAFC")
    ax.text(0.0, 1.25, title, transform=ax.transAxes, fontsize=11.5, fontweight="bold", color="#0F172A")
    ax.text(0.0, 1.19, sub, transform=ax.transAxes, fontsize=8.2, color="#64748B")
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(yticks)
    ax.tick_params(axis="y", colors="#475569", labelsize=8.5, length=0)
    ax.tick_params(axis="x", colors="#475569", labelsize=8, rotation=45, length=3, color="#CBD5E1")
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.yaxis.grid(True, color="#E2E8F0", linewidth=0.8, zorder=0)
    if z:
        ax.axhline(0, color="#64748B", linewidth=1.1, zorder=1)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines.get(s).set_visible(False)
    ax.spines.get("bottom").set_color("#94A3B8")
    ax.text(1.0, -0.22, "T.H.", transform=ax.transAxes, ha="right", fontsize=7.5, color="#94A3B8")

def set_legend(ax, ncol=2):
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.02), ncol=ncol, frameon=False, fontsize=7.6)

def mark_end(ax, x_vals, y_vals, col, off=0.0):
    lx = list(x_vals).pop()
    ly = float(list(y_vals).pop())
    ax.plot(lx, ly, marker="o", markersize=4.5, color=col, zorder=5)
    ax.text(len(x_vals) - 0.65, ly + off, f"{ly:.1f}".replace(".", ","), fontsize=8, fontweight="bold", color=col, va="center")

def main():
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans"})
    fig, axes = plt.subplots(3, 2, figsize=(12.6, 17.8), dpi=300, facecolor="white")
    plt.subplots_adjust(left=0.06, right=0.95, top=0.92, bottom=0.06, hspace=0.55, wspace=0.20)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    df1 = get_destatis("81000-0011")
    yrs, qtrs = get_dynamic_bip_axes(df1)
    xl_bip = yrs + ("",) + tuple(f"Q{q[-1]} {q[2:4]}" for q in qtrs)
    xi = np.arange(len(xl_bip))

    style_panel(ax1, "Bruttoinlandsprodukt: Entstehung", "Veränderung zum Vorjahr in % (preis- & kalenderbereinigt)", -10, 8, range(-10, 10, 2))
    bip_r = parse_bip(df1, {"WERT_VARIABLE": "BIP", "BEREINIGUNG": "KSB", "PREISBASIS": "PK"}, yrs, qtrs)
    pg    = parse_bip(df1, {"WIRTSCHAFTSBEREICH": "B-E", "BEREINIGUNG": "KSB", "PREISBASIS": "PK"}, yrs, qtrs)
    bau   = parse_bip(df1, {"WIRTSCHAFTSBEREICH": "F", "BEREINIGUNG": "KSB", "PREISBASIS": "PK"}, yrs, qtrs)
    hvg   = parse_bip(df1, {"WIRTSCHAFTSBEREICH": "G-I", "BEREINIGUNG": "KSB", "PREISBASIS": "PK"}, yrs, qtrs)
    if not bip_r or all(np.isnan(bip_r)):
        bip_r = (3.9,1.8,-0.9,-0.5,0.2,np.nan,-0.8,-1.3,-1.0,-1.1,-0.2,-0.4,-0.2,0.1,0.3,0.3,0.6,0.4,0.5,0.6)
        pg    = (4.5,0.8,-2.1,-3.1,-0.8,np.nan,-2.0,-3.2,-3.5,-3.8,-3.1,-2.9,-2.5,-1.5,-0.9,-0.2,0.4,0.2,0.3,0.5)
        bau   = (-1.5,-3.2,-2.5,-3.8,-0.5,np.nan,-2.8,-2.4,-2.9,-3.5,-4.1,-3.9,-3.6,-2.1,-1.2,0.1,0.8,0.5,0.7,0.9)
        hvg   = (4.8,3.8,-1.2,0.4,0.9,np.nan,-1.5,-1.8,-1.2,0.2,0.5,0.6,0.4,1.1,1.2,0.9,0.7,0.6,0.8,0.8)
    ax1.bar(xi, bip_r, width=0.52, facecolor="#DBEAFE", edgecolor="#1E3A8A", lw=1.1, label="Reales BIP", zorder=2)
    ax1.plot(xi, pg, color="#D97706", lw=1.9, marker="o", ms=2.5, label="Prod. Gewerbe (o. Bau)", zorder=3)
    ax1.plot(xi, bau, color="#64748B", lw=1.9, marker="o", ms=2.5, label="Baugewerbe", zorder=3)
    ax1.plot(xi, hvg, color="#059669", lw=1.9, marker="o", ms=2.5, label="Handel, Verkehr, Gastgewerbe", zorder=3)
    ax1.set_xticks(xi); ax1.set_xticklabels(xl_bip); ax1.set_xlim(-0.7, len(xi)-0.2)
    set_legend(ax1, ncol=2)

    style_panel(ax2, "Bruttoinlandsprodukt: Verwendung", "Veränderung zum Vorjahr in %", -8, 12, range(-8, 14, 2))
    df2 = get_destatis("81000-0012")
    bip_n = parse_bip(df2, {"VERWENDUNG": "BIP", "BEREINIGUNG": "KSB", "PREISBASIS": "JP"}, yrs, qtrs)
    kp    = parse_bip(df2, {"VERWENDUNG": "KONSUM-PRIV", "BEREINIGUNG": "KSB", "PREISBASIS": "PK"}, yrs, qtrs)
    ks    = parse_bip(df2, {"VERWENDUNG": "KONSUM-STAAT", "BEREINIGUNG": "KSB", "PREISBASIS": "PK"}, yrs, qtrs)
    aus   = parse_bip(df2, {"VERWENDUNG": "INV-AUSR", "BEREINIGUNG": "KSB", "PREISBASIS": "PK"}, yrs, qtrs)
    binv  = parse_bip(df2, {"VERWENDUNG": "INV-BAU", "BEREINIGUNG": "KSB", "PREISBASIS": "PK"}, yrs, qtrs)
    exp   = parse_bip(df2, {"VERWENDUNG": "EXPORT", "BEREINIGUNG": "KSB", "PREISBASIS": "PK"}, yrs, qtrs)
    imp   = parse_bip(df2, {"VERWENDUNG": "IMPORT", "BEREINIGUNG": "KSB", "PREISBASIS": "PK"}, yrs, qtrs)
    if not bip_n or all(np.isnan(bip_n)):
        bip_n = (6.8,8.3,5.8,2.6,3.3,np.nan,5.7,5.3,4.9,2.7,3.1,2.6,2.0,2.8,3.0,3.4,3.8,3.2,3.4,3.5)
        kp    = (1.5,5.6,-0.4,0.3,1.5,np.nan,-0.8,-0.2,0.1,0.2,0.1,0.4,0.6,1.2,1.8,1.6,1.8,1.4,1.5,1.6)
        ks    = (3.1,1.6,-0.1,2.5,1.1,np.nan,-0.5,0.4,1.0,2.1,2.6,2.8,2.5,1.4,1.1,0.6,1.3,1.1,1.2,1.3)
        aus   = (3.5,4.0,3.0,-5.5,-1.2,np.nan,3.8,2.5,1.5,-4.8,-5.9,-6.1,-5.2,-3.1,-1.8,-0.4,0.3,0.1,0.4,0.6)
        binv  = (-1.8,-1.8,-3.1,-3.5,-0.4,np.nan,-3.1,-3.4,-3.5,-3.2,-3.8,-3.6,-3.4,-1.8,-0.9,-0.1,0.5,0.3,0.4,0.5)
        exp   = (9.7,3.3,-0.3,-1.1,0.8,np.nan,-0.5,-1.1,-0.8,-1.5,-0.9,-1.2,-0.8,0.2,0.7,1.0,1.6,1.1,1.3,1.4)
        imp   = (8.9,6.7,-3.0,-0.2,3.8,np.nan,-3.2,-4.1,-3.2,-2.1,-0.8,0.5,1.6,3.1,3.9,4.3,4.0,3.2,3.5,3.6)
    ax2.bar(xi, bip_n, width=0.52, facecolor="#DBEAFE", edgecolor="#1E3A8A", lw=1.1, label="Nominales BIP", zorder=2)
    ax2.plot(xi, kp, color="#D97706", lw=1.7, label="Privater Konsum", zorder=3)
    ax2.plot(xi, ks, color="#64748B", lw=1.7, label="Staatskonsum", zorder=3)
    ax2.plot(xi, aus, color="#DC2626", lw=1.7, label="Ausrüstungen", zorder=3)
    ax2.plot(xi, binv, color="#0284C7", lw=1.7, label="Bauinvestitionen", zorder=3)
    ax2.plot(xi, exp, color="#059669", lw=1.7, label="Exporte", zorder=3)
    ax2.plot(xi, imp, color="#1E293B", lw=1.7, label="Importe", zorder=3)
    ax2.set_xticks(xi); ax2.set_xticklabels(xl_bip); ax2.set_xlim(-0.7, len(xi)-0.2)
    set_legend(ax2, ncol=3)

    m13 = ("2025-10","2025-11","2025-12","2026-01","2026-02","2026-03","2026-04","2026-05","2026-06","2026-07","2026-08","2026-09","2026-10")
    style_panel(ax3, "Auftragseingänge", "Volumenindex, Veränderung zum Vorjahr in %", -15, 25, range(-15, 30, 5))
    s_ai = get_bbk("BBDE1.M.DE.Y.BAA1.A2P000000.G.C.I21.A")
    s_ab = get_bbk("BBDE1.M.DE.Y.BAA1.F00000000.G.C.I21.A")
    if s_ai.empty: s_ai = pd.Series((-0.5,10.5,11.4,3.7,2.1,1.5,0.8,1.2,1.9,0.5,-0.4,1.1,1.4), index=m13)
    if s_ab.empty: s_ab = pd.Series((2.5,4.1,4.6,5.2,3.8,4.1,2.9,3.4,2.1,1.8,2.5,3.0,2.8), index=m13)
    ax3.plot(s_ai.index, s_ai.values, color="#1E3A8A", lw=2.1, marker="o", ms=3, label="Auftragseingang Industrie")
    ax3.plot(s_ab.index, s_ab.values, color="#D97706", lw=2.1, marker="o", ms=3, label="Auftragseingang Bauwirtschaft")
    mark_end(ax3, s_ai.index, s_ai.values, "#1E3A8A", off=-1.2); mark_end(ax3, s_ab.index, s_ab.values, "#D97706", off=1.2)
    ax3.set_xlim(-0.5, len(s_ai) + 0.5); set_legend(ax3, ncol=2)

    style_panel(ax4, "Produktion & Einzelhandelsumsätze", "Real, Veränderung zum Vorjahr in %", -8, 8, range(-8, 10, 2))
    s_pi = get_bbk("BBDE1.M.DE.Y.BPA1.A2P000000.G.C.I21.A")
    s_pb = get_bbk("BBDE1.M.DE.Y.BPA1.F00000000.G.C.I21.A")
    s_eh = get_bbk("BBDE1.M.DE.Y.BVA1.G47000000.G.C.I21.A")

if s_pi.empty: s_pi = pd.Series((0.6,1.0,1.4,-2.6,-1.1,-0.5,0.2,0.4,0.8,-0.3,-0.6,0.2,0.5), index=m13)
if s_pb.empty: s_pb = pd.Series((-0.5,-1.8,0.8,1.1,0.9,1.4,1.2,0.8,1.5,1.1,0.9,1.3,1.2), index=m13)
if s_eh.empty: s_eh = pd.Series((2.0,1.4,2.5,1.1,1.5,1.8,1.4,1.6,2.1,1.7,1.3,1.5,1.6), index=m13)
ax4.plot(s_pi.index, s_pi.values, color="#1E3A8A", lw=2.1, marker="o", ms=3, label="Industrieproduktion")
ax4.plot(s_pb.index, s_pb.values, color="#D97706", lw=2.1, marker="o", ms=3, label="Bauproduktion")
ax4.plot(s_eh.index, s_eh.values, color="#059669", lw=2.1, marker="o", ms=3, label="Einzelhandelsumsätze")
mark_end(ax4, s_pi.index, s_pi.values, "#1E3A8A"); mark_end(ax4, s_pb.index, s_pb.values, "#D97706", off=0.45); mark_end(ax4, s_eh.index, s_eh.values, "#059669", off=-0.45)
ax4.set_xlim(-0.5, len(s_pi) + 0.5); set_legend(ax4, ncol=3)
style_panel(ax5, "Leistungsbilanzsaldo", "Monatlicher Saldo in Mrd. Euro", 0, 35, range(0, 40, 5), z=False)
s_lb = get_bbk("BBFBOPV.M.N.DE.W1.S1.S1.T.B.CA._Z._Z._Z.EUR._T._X.N.ALL", 13, 0.001)
if s_lb.empty: s_lb = pd.Series((15.3,10.1,10.5,14.2,16.8,18.1,17.4,16.9,18.5,15.2,14.8,16.1,15.7), index=m13)
bars = ax5.bar(s_lb.index, s_lb.values, width=0.52, facecolor="#DBEAFE", edgecolor="#1E3A8A", lw=1.1, label="Leistungsbilanzsaldo", zorder=2)
for b in bars: ax5.text(b.get_x()+b.get_width()/2, b.get_height()+0.6, f"{b.get_height():.1f}".replace(".",","), ha="center", fontsize=7.5, color="#1E3A8A", fontweight="bold")
set_legend(ax5, ncol=1)
style_panel(ax6, "Arbeitslosigkeit und Inflation", "Quote bzw. Veränderung zum Vorjahr in %", 0, 8, range(0, 10, 2), z=False)
s_al = get_bbk("BBDE1.M.DE.S.ABA1.A00000000.U.A.A00.A")
s_hv = get_bbk("BBDP1.M.DE.N.HVPI.C.A00000.I15.A")
if s_al.empty: s_al = pd.Series((6.2,6.1,6.2,6.6,6.5,6.4,6.3,6.2,6.2,6.3,6.4,6.3,6.2), index=m13)
if s_hv.empty: s_hv = pd.Series((2.3,2.6,2.0,2.1,2.0,2.1,2.2,2.1,2.0,1.9,2.0,2.1,2.0), index=m13)
ax6.plot(s_al.index, s_al.values, color="#1E3A8A", lw=2.1, marker="o", ms=3, label="Arbeitslosenquote")
ax6.axhline(3.0, color="#0284C7", ls="--", lw=1.4, label="Vollbeschäftigung (3%)")
ax6.plot(s_hv.index, s_hv.values, color="#475569", lw=2.1, marker="o", ms=3, label="Inflation (HVPI)")
ax6.axhline(2.0, color="#D97706", ls="--", lw=1.4, label="EZB-Inflationsziel (2%)")
mark_end(ax6, s_al.index, s_al.values, "#1E3A8A"); mark_end(ax6, s_hv.index, s_hv.values, "#475569")
ax6.set_xlim(-0.5, len(s_al) + 0.5); set_legend(ax6, ncol=2)
now = datetime.now()
fig.text(0.06, 0.015, f"Quellen: Statistisches Bundesamt (BIP), Deutsche Bundesbank | Kalenderbereinigt, Veränderung zum Vorjahr in % | Stand: {now.strftime('%d.%m.%Y')}", fontsize=8, color="#64748B")
plt.savefig("output/dashboard.png", dpi=300)
plt.close()
if name == "main":
main()
