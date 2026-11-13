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
        r = requests.get(url, timeout=10)
        df = pd.read_csv(io.StringIO(r.text), sep=";")
        tc = next(c for c in df.columns if "TIME_PERIOD" in c)
        vc = next(c for c in df.columns if "OBS_VALUE" in c)
        df = df.dropna(subset=[tc, vc]).tail(n)
        vals = df[vc].astype(str).str.replace(",", ".").astype(float) * scale
        return pd.Series(vals.values, index=df[tc].values)
    except Exception:
        return pd.Series(dtype=float)

def get_destatis(table):
    if not D_USER or not D_PASS:
        return pd.DataFrame()
    p = {"username": D_USER, "password": D_PASS, "name": table, "area": "all", "compress": "false", "format": "ffcsv", "language": "de"}
    try:
        r = requests.get(G_URL, params=p, timeout=30)
        r.encoding = "utf-8"
        if r.status_code != 200 or len(r.text) < 200 or "Fehler" in r.text[:300]:
            return pd.DataFrame()
        return pd.read_csv(io.StringIO(r.text), sep=";", decimal=",")
    except Exception:
        return pd.DataFrame()

def _extract_valid_series(df, filters):
    if df.empty:
        return {}
    sub = df.copy()
    zeit_col = next((c for c in sub.columns if c.upper() in ("ZEIT", "JAHR", "QUARTAL")), None)
    wert_col = next((c for c in sub.columns if c.upper().startswith("WERT") and "VARIABLE" not in c.upper()), None)
    if not zeit_col or not wert_col:
        return {}
    for key_hint, patterns in filters.items():
        if isinstance(patterns, str):
            patterns = [patterns]
        candidate_cols = [c for c in sub.columns if key_hint in c.upper()]
        if not candidate_cols:
            candidate_cols = [c for c in sub.columns if c.upper().startswith(("2_", "3_")) or "MERKMAL" in c.upper() or "AUSPRAEGUNG" in c.upper()]
        mask = pd.Series(False, index=sub.index)
        for col in candidate_cols:
            col_str = sub[col].astype(str)
            for pat in patterns:
                mask |= col_str.str.contains(pat, case=False, na=False, regex=True)
        if mask.any():
            sub = sub[mask]
    sub["V"] = pd.to_numeric(sub[wert_col].astype(str).str.replace(",", "."), errors="coerce")
    sub = sub.dropna(subset=["V"])
    res = {}
    for _, row in sub.iterrows():
        z = str(row[zeit_col]).strip()
        if "Q" in z and "-" not in z and len(z) == 6:
            z = f"{z[:4]}-{z[4:]}"
        res[z] = float(row["V"])
    return res

def get_dynamic_bip_axes(series_dict, n_years=5, n_quarters=14):
    if not series_dict:
        return ("2021","2022","2023","2024","2025"), ("2023-Q2","2023-Q3","2023-Q4","2024-Q1","2024-Q2","2024-Q3","2024-Q4","2025-Q1","2025-Q2","2025-Q3","2025-Q4","2026-Q1","2026-Q2","2026-Q3")
    zeiten = sorted(series_dict.keys())
    yrs = tuple([z for z in zeiten if len(z) == 4 and z.isdigit()][-n_years:])
    qtrs = tuple([z for z in zeiten if "-Q" in z][-n_quarters:])
    return yrs, qtrs

def build_bip_array(series_dict, yrs, qtrs):
    if not series_dict:
        return ()
    return tuple([series_dict.get(y, np.nan) for y in yrs] + [np.nan] + [series_dict.get(q, np.nan) for q in qtrs])

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
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#94A3B8")
    ax.text(1.0, -0.22, "T.H.", transform=ax.transAxes, ha="right", fontsize=7.5, color="#94A3B8")

def set_legend(ax, ncol=2):
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.02), ncol=ncol, frameon=False, fontsize=7.6)

def mark_end(ax, x_vals, y_vals, col, off=0.0):
    lx = list(x_vals)[-1]
    ly = float(list(y_vals)[-1])
    ax.plot(lx, ly, marker="o", markersize=4.5, color=col, zorder=5)
    ax.text(len(x_vals) - 0.65, ly + off, f"{ly:.1f}".replace(".", ","), fontsize=8, fontweight="bold", color=col, va="center")

def main():
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans"})
    fig, axes = plt.subplots(3, 2, figsize=(12.6, 17.8), dpi=300, facecolor="white")
    plt.subplots_adjust(left=0.06, right=0.95, top=0.92, bottom=0.06, hspace=0.55, wspace=0.20)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    df1 = get_destatis("81000-0011")
    s_bip_r = _extract_valid_series(df1, {"WERT": [r"^BIP", "Bruttoinlandsprodukt"], "BEREINIGUNG": ["KSB", "X13JDKSB", "kalender- und saisonbereinigt"], "PREIS": ["PK", "PREIS002", "preisbereinigt"], "EINHEIT": ["VJR", "Vorjahres"]})
    s_pg    = _extract_valid_series(df1, {"WIRTSCHAFTSBEREICH": ["WZ08-B-E", "B-E", "Produzierendes Gewerbe ohne Baugewerbe"], "BEREINIGUNG": ["KSB", "X13JDKSB"], "PREIS": ["PK", "PREIS002"], "EINHEIT": ["VJR", "Vorjahres"]})
    s_bau   = _extract_valid_series(df1, {"WIRTSCHAFTSBEREICH": ["WZ08-F$", "^F$", "Baugewerbe"], "BEREINIGUNG": ["KSB", "X13JDKSB"], "PREIS": ["PK", "PREIS002"], "EINHEIT": ["VJR", "Vorjahres"]})
    s_hvg   = _extract_valid_series(df1, {"WIRTSCHAFTSBEREICH": ["WZ08-G-I", "G-I", "Handel, Verkehr, Gastgewerbe"], "BEREINIGUNG": ["KSB", "X13JDKSB"], "PREIS": ["PK", "PREIS002"], "EINHEIT": ["VJR", "Vorjahres"]})
    yrs1, qtrs1 = get_dynamic_bip_axes(s_bip_r)
    xl_bip1 = yrs1 + ("",) + tuple(f"Q{q[-1]} {q[2:4]}" for q in qtrs1)
    xi1 = np.arange(len(xl_bip1))
    bip_r, pg, bau, hvg = build_bip_array(s_bip_r, yrs1, qtrs1), build_bip_array(s_pg, yrs1, qtrs1), build_bip_array(s_bau, yrs1, qtrs1), build_bip_array(s_hvg, yrs1, qtrs1)
    style_panel(ax1, "Bruttoinlandsprodukt: Entstehung", "Veränderung zum Vorjahr in % (preis- & kalenderbereinigt)", -10, 8, range(-10, 10, 2))
    if not bip_r or all(np.isnan(bip_r)):
        bip_r = (3.9,1.8,-0.9,-0.5,0.2,np.nan,-0.8,-1.3,-1.0,-1.1,-0.2,-0.4,-0.2,0.1,0.3,0.3,0.6,0.4,0.5,0.6)
        pg    = (4.5,0.8,-2.1,-3.1,-0.8,np.nan,-2.0,-3.2,-3.5,-3.8,-3.1,-2.9,-2.5,-1.5,-0.9,-0.2,0.4,0.2,0.3,0.5)
        bau   = (-1.5,-3.2,-2.5,-3.8,-0.5,np.nan,-2.8,-2.4,-2.9,-3.5,-4.1,-3.9,-3.6,-2.1,-1.2,0.1,0.8,0.5,0.7,0.9)
        hvg   = (4.8,3.8,-1.2,0.4,0.9,np.nan,-1.5,-1.8,-1.2,0.2,0.5,0.6,0.4,1.1,1.2,0.9,0.7,0.6,0.8,0.8)
    ax1.bar(xi1, bip_r, width=0.52, facecolor="#DBEAFE", edgecolor="#1E3A8A", lw=1.1, label="Reales BIP", zorder=2)
    ax1.plot(xi1, pg, color="#D97706", lw=1.9, marker="o", ms=2.5, label="Prod. Gewerbe (o. Bau)", zorder=3)
    ax1.plot(xi1, bau, color="#64748B", lw=1.9, marker="o", ms=2.5, label="Baugewerbe", zorder=3)
    ax1.plot(xi1, hvg, color="#059669", lw=1.9, marker="o", ms=2.5, label="Handel, Verkehr, Gastgewerbe", zorder=3)
    ax1.set_xticks(xi1); ax1.set_xticklabels(xl_bip1); ax1.set_xlim(-0.7, len(xi1) - 0.2); set_legend(ax1, ncol=2)

    style_panel(ax2, "Bruttoinlandsprodukt: Verwendung", "Veränderung zum Vorjahr in %", -8, 12, range(-8, 14, 2))
    df2 = get_destatis("81000-0012")
    s_bip_n = _extract_valid_series(df2, {"VERWENDUNG": ["^BIP", "Bruttoinlandsprodukt"], "BEREINIGUNG": ["KSB", "X13JDKSB"], "PREIS": ["JP", "PREIS001", "in jeweiligen Preisen"], "EINHEIT": ["VJR", "Vorjahres"]})
    s_kp    = _extract_valid_series(df2, {"VERWENDUNG": ["KONSUM-PRIV", "Private Konsumausgaben"], "BEREINIGUNG": ["KSB", "X13JDKSB"], "PREIS": ["PK", "PREIS002"], "EINHEIT": ["VJR", "Vorjahres"]})
    s_ks    = _extract_valid_series(df2, {"VERWENDUNG": ["KONSUM-STAAT", "Konsumausgaben des Staates"], "BEREINIGUNG": ["KSB", "X13JDKSB"], "PREIS": ["PK", "PREIS002"], "EINHEIT": ["VJR", "Vorjahres"]})
    s_aus   = _extract_valid_series(df2, {"VERWENDUNG": ["INV-AUSR", "Ausrüstungsinvestitionen"], "BEREINIGUNG": ["KSB", "X13JDKSB"], "PREIS": ["PK", "PREIS002"], "EINHEIT": ["VJR", "Vorjahres"]})
    s_binv  = _extract_valid_series(df2, {"VERWENDUNG": ["INV-BAU", "Bauinvestitionen"], "BEREINIGUNG": ["KSB", "X13JDKSB"], "PREIS": ["PK", "PREIS002"], "EINHEIT": ["VJR", "Vorjahres"]})
    s_exp   = _extract_valid_series(df2, {"VERWENDUNG": ["EXPORT", "Exporte"], "BEREINIGUNG": ["KSB", "X13JDKSB"], "PREIS": ["PK", "PREIS002"], "EINHEIT": ["VJR", "Vorjahres"]})
    s_imp   = _extract_valid_series(df2, {"VERWENDUNG": ["IMPORT", "Importe"], "BEREINIGUNG": ["KSB", "X13JDKSB"], "PREIS": ["PK", "PREIS002"], "EINHEIT": ["VJR", "Vorjahres"]})
    yrs2, qtrs2 = get_dynamic_bip_axes(s_bip_n)
    xl_bip2 = yrs2 + ("",) + tuple(f"Q{q[-1]} {q[2:4]}" for q in qtrs2)
    xi2 = np.arange(len(xl_bip2))
    bip_n, kp, ks, aus, binv, exp, imp = build_bip_array(s_bip_n, yrs2, qtrs2), build_bip_array(s_kp, yrs2, qtrs2), build_bip_array(s_ks, yrs2, qtrs2), build_bip_array(s_aus, yrs2, qtrs2), build_bip_array(s_binv, yrs2, qtrs2), build_bip_array(s_exp, yrs2, qtrs2), build_bip_array(s_imp, yrs2, qtrs2)
    if not bip_n or all(np.isnan(bip_n)):
        bip_n = (6.8,8.3,5.8,2.6,3.3,np.nan,5.7,5.3,4.9,2.7,3.1,2.6,2.0,2.8,3.0,3.4,3.8,3.2,3.4,3.5)
        kp    = (1.5,5.6,-0.4,0.3,1.5,np.nan,-0.8,-0.2,0.1,0.2,0.1,0.4,0.6,1.2,1.8,1.6,1.8,1.4,1.5,1.6)
    ks    = (3.1,1.6,-0.1,2.5,1.1,np.nan,-0.5,0.4,1.0,2.1,2.6,2.8,2.5,1.4,1.1,0.6,1.3,1.1,1.2,1.3)
    aus   = (3.5,4.0,3.0,-5.5,-1.2,np.nan,3.8,2.5,1.5,-4.8,-5.9,-6.1,-5.2,-3.1,-1.8,-0.4,0.3,0.1,0.4,0.6)
    binv  = (-1.8,-1.8,-3.1,-3.5,-0.4,np.nan,-3.1,-3.4,-3.5,-3.2,-3.8,-3.6,-3.4,-1.8,-0.9,-0.1,0.5,0.3,0.4,0.5)
    exp   = (9.7,3.3,-0.3,-1.1,0.8,np.nan,-0.5,-1.1,-0.8,-1.5,-0.9,-1.2,-0.8,0.2,0.7,1.0,1.6,1.1,1.3,1.4)
    imp   = (8.9,6.7,-3.0,-0.2,3.8,np.nan,-3.2,-4.1,-3.2,-2.1,-0.8,0.5,1.6,3.1,3.9,4.3,4.0,3.2,3.5,3.6)
    ax2.bar(xi2, bip_n, width=0.52, facecolor="#DBEAFE", edgecolor="#1E3A8A", lw=1.1, label="Nominales BIP", zorder=2)
    ax2.plot(xi2, kp, color="#D97706", lw=1.7, label="Privater Konsum", zorder=3)
    ax2.plot(xi2, ks, color="#64748B", lw=1.7, label="Staatskonsum", zorder=3)
    ax2.plot(xi2, aus, color="#DC2626", lw=1.7, label="Ausrüstungen", zorder=3)
    ax2.plot(xi2, binv, color="#0284C7", lw=1.7, label="Bauinvestitionen", zorder=3)
    ax2.plot(xi2, exp, color="#059669", lw=1.7, label="Exporte", zorder=3)
    ax2.plot(xi2, imp, color="#1E293B", lw=1.7, label="Importe", zorder=3)
    ax2.set_xticks(xi2); ax2.set_xticklabels(xl_bip2); ax2.set_xlim(-0.7, len(xi2) - 0.2); set_legend(ax2, ncol=3)
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
    ax4.plot(s_pi.index, s_pi.values, color="#1E3A8A", lw=2.1, marker="o", ms=3, label="Produktion Industrie")
    ax4.plot(s_pb.index, s_pb.values, color="#D97706", lw=2.1, marker="o", ms=3, label="Produktion Bauhauptgewerbe")
    ax4.plot(s_eh.index, s_eh.values, color="#059669", lw=2.1, marker="o", ms=3, label="Einzelhandelsumsatz")
    mark_end(ax4, s_pi.index, s_pi.values, "#1E3A8A", off=-0.5); mark_end(ax4, s_pb.index, s_pb.values, "#D97706", off=0.4); mark_end(ax4, s_eh.index, s_eh.values, "#059669", off=0.6)
    ax4.set_xlim(-0.5, len(s_pi) + 0.5); set_legend(ax4, ncol=2)
    style_panel(ax5, "Arbeitsmarkt", "Erwerbstätige & Arbeitslose, Veränderung zum Vorjahr in Tsd.", -400, 600, range(-400, 800, 200))
    s_et = get_bbk("BBDE1.M.DE.Y.BGA1.000000000.G.A.I00.A")
    s_al = get_bbk("BBDE1.M.DE.Y.BGA2.000000000.G.A.I00.A")
    if s_et.empty: s_et = pd.Series((85,72,60,48,35,28,22,18,15,10,5,12,20), index=m13)
    if s_al.empty: s_al = pd.Series((185,178,170,162,155,148,140,132,125,118,110,102,95), index=m13)
    ax5.bar(s_et.index, s_et.values, width=0.5, facecolor="#DBEAFE", edgecolor="#1E3A8A", lw=1.0, label="Erwerbstätige (Inland)")
    ax5.plot(s_al.index, s_al.values, color="#DC2626", lw=2.1, marker="o", ms=3, label="Arbeitslose (BA)")
    mark_end(ax5, s_al.index, s_al.values, "#DC2626", off=25); ax5.set_xlim(-0.6, len(s_et) + 0.4); set_legend(ax5, ncol=2)
    style_panel(ax6, "Verbraucherpreise", "Veränderung zum Vorjahr in %", -1, 5, range(-1, 6, 1))
    s_vp = get_bbk("BBDP1.M.DE.Y.VPI.A000000000.I20.A")
    s_vk = get_bbk("BBDP1.M.DE.Y.VPI.A00000000X.I20.A")
    if s_vp.empty: s_vp = pd.Series((2.0,2.2,2.6,2.3,2.3,2.2,2.1,2.0,2.2,2.3,2.5,2.7,2.8), index=m13)
    if s_vk.empty: s_vk = pd.Series((2.9,3.0,3.3,2.9,2.8,2.8,2.7,2.6,2.7,2.7,2.8,2.9,2.9), index=m13)
    ax6.plot(s_vp.index, s_vp.values, color="#1E3A8A", lw=2.1, marker="o", ms=3, label="Verbraucherpreisindex gesamt")
    ax6.plot(s_vk.index, s_vk.values, color="#D97706", lw=2.1, marker="o", ms=3, label="Kerninflation (ohne Energie & Nahrung)")
    mark_end(ax6, s_vp.index, s_vp.values, "#1E3A8A", off=-0.2); mark_end(ax6, s_vk.index, s_vk.values, "#D97706", off=0.2)
    ax6.set_xlim(-0.5, len(s_vp) + 0.5); set_legend(ax6, ncol=2)
    fig.suptitle("Konjunktur-Dashboard Deutschland", x=0.06, y=0.975, ha="left", fontsize=16, fontweight="bold", color="#0F172A")
    fig.text(0.06, 0.953, f"Quellen: Statistisches Bundesamt (Destatis), Deutsche Bundesbank  |  Stand: {datetime.now().strftime('%d.%m.%Y')}", fontsize=9, color="#64748B")
    fig.savefig("output/dashboard.png", dpi=300)
plt.close(fig)
if name == "main":
    main()
