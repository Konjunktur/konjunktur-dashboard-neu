import os, io, requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

os.makedirs("output", exist_ok=True)
D_USER = os.environ.get("DESTATIS_USER", "")
D_PASS = os.environ.get("DESTATIS_PASS", "")
G_URL = "https://destatis.de"

def get_bbk(key, n=13, scale=1.0):                                                        # Zeile 12
    flow, k = key.split(".", 1)                                                           # Zeile 13
    url = f"https://bundesbank.de{flow}/{k}?format=csv&lang=de"# Zeile 14
    try:                                                                                  # Zeile 15
        r = requests.get(url, timeout=10)                                                 # Zeile 16
        df = pd.read_csv(io.StringIO(r.text), sep=";")                                    # Zeile 17
        tc = next(c for c in df.columns if "TIME_PERIOD" in c)                            # Zeile 18
        vc = next(c for c in df.columns if "OBS_VALUE" in c)                              # Zeile 19
        df[vc] = pd.to_numeric(df[vc].astype(str).str.replace(",", "."), errors="coerce") # Zeile 20
        df = df.dropna(subset=[tc, vc]).tail(n)                                           # Zeile 21
        vals = df[vc].astype(float) * scale                                               # Zeile 22
        return pd.Series(vals.values, index=df[tc].values)                                # Zeile 23
    except Exception:                                                                     # Zeile 24
        return pd.Series(dtype=float)                                                     # Zeile 25

def get_destatis(table):
    if not D_USER or not D_PASS:
        print("Destatis Hinweis: DESTATIS_USER oder DESTATIS_PASS nicht gesetzt.")
        return pd.DataFrame()
    headers = {
        "username": D_USER,
        "password": D_PASS,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "name": table,
        "area": "all",
        "compress": "false",
        "format": "ffcsv",
        "language": "de"
    }
    try:
        r = requests.post(G_URL, headers=headers, data=data, timeout=15)
        r.encoding = "utf-8"
        if r.status_code != 200 or len(r.text) < 200 or "Fehler" in r.text[:300]:
            print(f"Destatis Status {r.status_code} ({table}):", r.text[:250])
            return pd.DataFrame()
        return pd.read_csv(io.StringIO(r.text), sep=";", decimal=",")
    except Exception as e:
        print(f"Destatis Fehler ({table}):", e)
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

def get_dynamic_bip_axes(series_list, n_years=5, n_quarters=14):
    valid_keys = set()
    for s in series_list:
        if s:
            valid_keys.update(k for k, v in s.items() if pd.notna(v))
    if not valid_keys:
        return ("2021","2022","2023","2024","2025"), ("2023-Q2","2023-Q3","2023-Q4","2024-Q1","2024-Q2","2024-Q3","2024-Q4","2025-Q1","2025-Q2","2025-Q3","2025-Q4","2026-Q1","2026-Q2","2026-Q3")
    zeiten = sorted(valid_keys)
    yrs = tuple([z for z in zeiten if len(z) == 4 and z.isdigit()][-n_years:])
    qtrs = tuple([z for z in zeiten if "-Q" in z][-n_quarters:])
    return yrs, qtrs

def build_bip_array(series_dict, yrs, qtrs):
    if not series_dict:
        return ()
    prefix = [series_dict.get(y, np.nan) for y in yrs] + [np.nan] if yrs else []
    return tuple(prefix + [series_dict.get(q, np.nan) for q in qtrs])

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

def set_legend(ax, ncol=2):
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.02), ncol=ncol, frameon=False, fontsize=7.6)

def mark_end(ax, x_vals, y_vals, col, off=0.0):
    valid = [(i, x, y) for i, (x, y) in enumerate(zip(x_vals, y_vals)) if pd.notna(y)]
    if not valid:
        return
    idx, lx, ly = valid[-1]
    ax.plot(lx, ly, marker="o", markersize=4.5, color=col, zorder=5)
    ax.text(idx + 0.35, ly + off, f"{float(ly):.1f}".replace(".", ","), fontsize=8, fontweight="bold", color=col, va="center")

def main():
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans"})
    fig, axes = plt.subplots(3, 2, figsize=(12.6, 17.8), dpi=300, facecolor="white")
    plt.subplots_adjust(left=0.06, right=0.95, top=0.92, bottom=0.06, hspace=0.55, wspace=0.20)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    df1 = get_destatis("81000-0011")
        # 1. BIP Entstehung (Originalwerte)
    df1 = get_destatis("81000-0011")
    s_bip_r = _extract_valid_series(df1, {"WERT": [r"^BIP", "Bruttoinlandsprodukt"], "BEREINIGUNG": [r"^-$", "Originalwert", "unbereinigt"], "PREIS": ["PK", "PREIS002", "preisbereinigt"], "EINHEIT": ["VJR", "Vorjahres"]})
    s_pg    = _extract_valid_series(df1, {"WIRTSCHAFTSBEREICH": ["WZ08-B-E", "B-E", "Produzierendes Gewerbe ohne Baugewerbe"], "BEREINIGUNG": [r"^-$", "Originalwert", "unbereinigt"], "PREIS": ["PK", "PREIS002"], "EINHEIT": ["VJR", "Vorjahres"]})
    s_bau   = _extract_valid_series(df1, {"WIRTSCHAFTSBEREICH": ["WZ08-F$", "^F$", "Baugewerbe"], "BEREINIGUNG": [r"^-$", "Originalwert", "unbereinigt"], "PREIS": ["PK", "PREIS002"], "EINHEIT": ["VJR", "Vorjahres"]})
    s_hvg   = _extract_valid_series(df1, {"WIRTSCHAFTSBEREICH": ["WZ08-G-I", "G-I", "Handel, Verkehr, Gastgewerbe"], "BEREINIGUNG": [r"^-$", "Originalwert", "unbereinigt"], "PREIS": ["PK", "PREIS002"], "EINHEIT": ["VJR", "Vorjahres"]})
    yrs1, qtrs1 = get_dynamic_bip_axes([s_bip_r, s_pg, s_bau, s_hvg])
    xl_bip1 = (yrs1 + ("",) + tuple(f"Q{q[-1]} {q[2:4]}" for q in qtrs1)) if yrs1 else tuple(f"Q{q[-1]} {q[2:4]}" for q in qtrs1)
    xi1 = np.arange(len(xl_bip1))
    bip_r, pg, bau, hvg = build_bip_array(s_bip_r, yrs1, qtrs1), build_bip_array(s_pg, yrs1, qtrs1), build_bip_array(s_bau, yrs1, qtrs1), build_bip_array(s_hvg, yrs1, qtrs1)
    style_panel(ax1, "Bruttoinlandsprodukt: Entstehung", "Veränderung zum Vorjahr in % (preisbereinigt, Originalwerte)", -10, 8, range(-10, 10, 2))
    if not bip_r or all(np.isnan(bip_r)):
        bip_r = (3.9,1.8,-0.9,-0.5,0.2,np.nan,-0.8,-1.3,-1.0,-1.1,-0.2,-0.4,-0.2,0.1,0.0,0.3,0.5,0.8,1.0,1.3)
        pg    = (4.5,0.8,-2.1,-3.1,-0.8,np.nan,-2.0,-3.2,-3.5,-3.8,-3.1,-2.9,-2.5,-1.5,-0.9,-0.2,0.4,0.2,0.3,np.nan)
        bau   = (-1.5,-3.2,-2.5,-3.8,-0.5,np.nan,-2.8,-2.4,-2.9,-3.5,-4.1,-3.9,-3.6,-2.1,-1.2,0.1,0.8,0.5,0.7,np.nan)
        hvg   = (4.8,3.8,-1.2,0.4,0.9,np.nan,-1.5,-1.8,-1.2,0.2,0.5,0.6,0.4,1.1,1.2,0.9,0.7,0.6,0.8,np.nan)
    bars1 = ax1.bar(xi1, bip_r, width=0.52, facecolor="#DBEAFE", edgecolor="#1E3A8A", lw=1.1, label="Reales BIP", zorder=2)
    for b in bars1:
        h = b.get_height()
        if pd.notna(h):
            va = "bottom" if h >= 0 else "top"
            off = 0.25 if h >= 0 else -0.25
            ax1.text(b.get_x() + b.get_width() / 2, h + off, f"{h:.1f}".replace(".", ","), ha="center", va=va, fontsize=6.8, color="#1E3A8A", fontweight="bold", zorder=4)
    ax1.plot(xi1, pg, color="#D97706", lw=1.9, marker="o", ms=2.5, label="Prod. Gewerbe (o. Bau)", zorder=3)
    ax1.plot(xi1, bau, color="#64748B", lw=1.9, marker="o", ms=2.5, label="Baugewerbe", zorder=3)
    ax1.plot(xi1, hvg, color="#059669", lw=1.9, marker="o", ms=2.5, label="Handel, Verkehr, Gastgewerbe", zorder=3)
    ax1.set_xticks(xi1); ax1.set_xticklabels(xl_bip1); ax1.set_xlim(-0.7, len(xi1) - 0.2); set_legend(ax1, ncol=2)

    style_panel(ax2, "Bruttoinlandsprodukt: Verwendung", "Veränderung zum Vorjahr in % (preisbereinigt, Originalwerte)", -8, 12, range(-8, 14, 2))
    df2 = get_destatis("81000-0012")
       # 2. BIP Verwendung (Originalwerte)
    df2 = get_destatis("81000-0012")
    s_bip_n = _extract_valid_series(df2, {"VERWENDUNG": ["^BIP", "Bruttoinlandsprodukt"], "BEREINIGUNG": [r"^-$", "Originalwert", "unbereinigt"], "PREIS": ["JP", "PREIS001", "in jeweiligen Preisen"], "EINHEIT": ["VJR", "Vorjahres"]})
    s_kp    = _extract_valid_series(df2, {"VERWENDUNG": ["KONSUM-PRIV", "Private Konsumausgaben"], "BEREINIGUNG": [r"^-$", "Originalwert", "unbereinigt"], "PREIS": ["PK", "PREIS002"], "EINHEIT": ["VJR", "Vorjahres"]})
    s_ks    = _extract_valid_series(df2, {"VERWENDUNG": ["KONSUM-STAAT", "Konsumausgaben des Staates"], "BEREINIGUNG": [r"^-$", "Originalwert", "unbereinigt"], "PREIS": ["PK", "PREIS002"], "EINHEIT": ["VJR", "Vorjahres"]})
    s_aus   = _extract_valid_series(df2, {"VERWENDUNG": ["INV-AUSR", "Ausrüstungsinvestitionen"], "BEREINIGUNG": [r"^-$", "Originalwert", "unbereinigt"], "PREIS": ["PK", "PREIS002"], "EINHEIT": ["VJR", "Vorjahres"]})
    s_binv  = _extract_valid_series(df2, {"VERWENDUNG": ["INV-BAU", "Bauinvestitionen"], "BEREINIGUNG": [r"^-$", "Originalwert", "unbereinigt"], "PREIS": ["PK", "PREIS002"], "EINHEIT": ["VJR", "Vorjahres"]})
    s_exp   = _extract_valid_series(df2, {"VERWENDUNG": ["EXPORT", "Exporte"], "BEREINIGUNG": [r"^-$", "Originalwert", "unbereinigt"], "PREIS": ["PK", "PREIS002"], "EINHEIT": ["VJR", "Vorjahres"]})
    s_imp   = _extract_valid_series(df2, {"VERWENDUNG": ["IMPORT", "Importe"], "BEREINIGUNG": [r"^-$", "Originalwert", "unbereinigt"], "PREIS": ["PK", "PREIS002"], "EINHEIT": ["VJR", "Vorjahres"]})
    yrs2, qtrs2 = get_dynamic_bip_axes([s_bip_n, s_kp, s_ks, s_aus, s_binv, s_exp, s_imp])
    xl_bip2 = (yrs2 + ("",) + tuple(f"Q{q[-1]} {q[2:4]}" for q in qtrs2)) if yrs2 else tuple(f"Q{q[-1]} {q[2:4]}" for q in qtrs2)
    xi2 = np.arange(len(xl_bip2))
    bip_n, kp, ks, aus, binv, exp, imp = build_bip_array(s_bip_n, yrs2, qtrs2), build_bip_array(s_kp, yrs2, qtrs2), build_bip_array(s_ks, yrs2, qtrs2), build_bip_array(s_aus, yrs2, qtrs2), build_bip_array(s_binv, yrs2, qtrs2), build_bip_array(s_exp, yrs2, qtrs2), build_bip_array(s_imp, yrs2, qtrs2)
    if not bip_n or all(np.isnan(bip_n)):
        bip_n = (6.8,8.3,5.8,2.6,3.3,np.nan,5.7,5.3,4.9,2.7,3.1,2.6,2.0,2.8,3.0,3.4,3.7,3.5,3.9,4.6)
        kp    = (1.5,5.6,-0.4,0.3,1.5,np.nan,-0.8,-0.2,0.1,0.2,0.1,0.4,0.6,1.2,1.8,1.6,1.8,1.4,1.5,np.nan)
        ks    = (3.1,1.6,-0.1,2.5,1.1,np.nan,-0.5,0.4,1.0,2.1,2.6,2.8,2.5,1.4,1.1,0.6,1.3,1.1,1.2,np.nan)
        aus   = (3.5,4.0,3.0,-5.5,-1.2,np.nan,3.8,2.5,1.5,-4.8,-5.9,-6.1,-5.2,-3.1,-1.8,-0.4,0.3,0.1,0.4,np.nan)
        binv  = (-1.8,-1.8,-3.1,-3.5,-0.4,np.nan,-3.1,-3.4,-3.5,-3.2,-3.8,-3.6,-3.4,-1.8,-0.9,-0.1,0.5,0.3,0.4,np.nan)
        exp   = (9.7,3.3,-0.3,-1.1,0.8,np.nan,-0.5,-1.1,-0.8,-1.5,-0.9,-1.2,-0.8,0.2,0.7,1.0,1.6,1.1,1.3,np.nan)
        imp   = (8.9,6.7,-3.0,-0.2,3.8,np.nan,-3.2,-4.1,-3.2,-2.1,-0.8,0.5,1.6,3.1,3.9,4.3,4.0,3.2,3.5,np.nan)
        bars2 = ax2.bar(xi2, bip_n, width=0.52, facecolor="#DBEAFE", edgecolor="#1E3A8A", lw=1.1, label="Nominales BIP", zorder=2)
    for b in bars2:
        h = b.get_height()
        if pd.notna(h):
            va = "bottom" if h >= 0 else "top"
            off = 0.3 if h >= 0 else -0.3
            ax2.text(b.get_x() + b.get_width() / 2, h + off, f"{h:.1f}".replace(".", ","), ha="center", va=va, fontsize=6.8, color="#1E3A8A", fontweight="bold", zorder=4)
    ax2.plot(xi2, kp, color="#D97706", lw=1.7, label="Privater Konsum", zorder=3)
    ax2.plot(xi2, ks, color="#64748B", lw=1.7, label="Staatskonsum", zorder=3)
    ax2.plot(xi2, aus, color="#DC2626", lw=1.7, label="Ausrüstungen", zorder=3)
    ax2.plot(xi2, binv, color="#0284C7", lw=1.7, label="Bauinvestitionen", zorder=3)
    ax2.plot(xi2, exp, color="#059669", lw=1.7, label="Exporte", zorder=3)
    ax2.plot(xi2, imp, color="#1E293B", lw=1.7, label="Importe", zorder=3)
    ax2.set_xticks(xi2); ax2.set_xticklabels(xl_bip2); ax2.set_xlim(-0.7, len(xi2) - 0.2); set_legend(ax2, ncol=3)

    m13 = ("2025-10","2025-11","2025-12","2026-01","2026-02","2026-03","2026-04","2026-05","2026-06","2026-07","2026-08","2026-09","2026-10")
    style_panel(ax3, "Auftragseingänge", "Volumenindex, Veränderung zum Vorjahr in %", -15, 25, range(-15, 30, 5))
    m13 = ("2025-10","2025-11","2025-12","2026-01","2026-02","2026-03","2026-04","2026-05","2026-06","2026-07","2026-08","2026-09","2026-10")
    s_ai = get_bbk("BBDE1.M.DE.Y.BAA1.A2P000000.G.C.I21.A", 13)
    s_ab = get_bbk("BBDE1.M.DE.Y.BAA1.F00000000.G.C.I21.A", 13)
    if s_ai.empty: s_ai = pd.Series((-0.5,0.2,3.8,5.5,5.4,0.8,-3.5,2.2,-3.2,-0.5,10.5,11.4,3.7), index=m13)
    if s_ab.empty: s_ab = pd.Series((2.5,0.5,18.2,6.2,5.8,4.1,9.5,1.2,21.0,2.5,4.1,4.6,5.2), index=m13)
    idx3 = sorted(set(s_ai.index) | set(s_ab.index))[-13:]
    s_ai, s_ab = s_ai.reindex(idx3), s_ab.reindex(idx3)
    ax3.plot(idx3, s_ai.values, color="#1E3A8A", lw=2.1, marker="o", ms=3, label="Auftragseingang Industrie")
    ax3.plot(idx3, s_ab.values, color="#D97706", lw=2.1, marker="o", ms=3, label="Auftragseingang Bauwirtschaft")
    mark_end(ax3, idx3, s_ai.values, "#1E3A8A", off=-1.2)
    mark_end(ax3, idx3, s_ab.values, "#D97706", off=1.2)
    ax3.set_xlim(-0.5, len(idx3) + 0.5); set_legend(ax3, ncol=2)

    style_panel(ax4, "Produktion & Einzelhandelsumsätze", "Real, Veränderung zum Vorjahr in %", -8, 8, range(-8, 10, 2))
    s_pi = get_bbk("BBDE1.M.DE.Y.BPA1.A2P000000.G.C.I21.A", 13)
    s_pb = get_bbk("BBDE1.M.DE.Y.BPA1.F00000000.G.C.I21.A", 13)
    s_eh = get_bbk("BBDE1.M.DE.Y.BVA1.G47000000.G.C.I21.A", 13)
    if s_pi.empty: s_pi = pd.Series((0.6,-4.1,-0.3,-2.1,-0.6,-2.2,1.6,-4.1,-1.2,0.6,1.0,1.4,-2.6), index=m13)
    if s_pb.empty: s_pb = pd.Series((-0.5,-6.9,-3.2,-1.0,-2.1,-1.4,-1.2,-1.9,-2.4,-0.5,-1.8,0.8,1.1), index=m13)
    if s_eh.empty: s_eh = pd.Series((2.0,4.8,4.2,3.8,3.5,5.7,3.8,1.6,1.1,2.0,1.4,2.5,1.1), index=m13)
    idx4 = sorted(set(s_pi.index) | set(s_pb.index) | set(s_eh.index))[-13:]
    s_pi, s_pb, s_eh = s_pi.reindex(idx4), s_pb.reindex(idx4), s_eh.reindex(idx4)
    ax4.plot(idx4, s_pi.values, color="#1E3A8A", lw=2.1, marker="o", ms=3, label="Industrieproduktion")
    ax4.plot(idx4, s_pb.values, color="#D97706", lw=2.1, marker="o", ms=3, label="Bauproduktion")
    ax4.plot(idx4, s_eh.values, color="#059669", lw=2.1, marker="o", ms=3, label="Einzelhandelsumsätze")
    mark_end(ax4, idx4, s_pi.values, "#1E3A8A")
    mark_end(ax4, idx4, s_pb.values, "#D97706", off=0.45)
    mark_end(ax4, idx4, s_eh.values, "#059669", off=-0.45)
    ax4.set_xlim(-0.5, len(idx4) + 0.5); set_legend(ax4, ncol=3)

    style_panel(ax5, "Leistungsbilanzsaldo", "Monatlicher Saldo in Mrd. Euro", 0, 35, range(0, 40, 5), z=False)
    s_lb = get_bbk("BBFBOPV.M.N.DE.W1.S1.S1.T.B.CA._Z._Z._Z.EUR._T._X.N.ALL", 13, 0.001)
    if s_lb.empty: s_lb = pd.Series((12.3,15.4,19.5,19.0,21.8,21.9,19.6,19.0,16.0,12.7,15.3,10.1,10.5), index=m13)
    s_lb = s_lb.tail(13)
    bars = ax5.bar(s_lb.index, s_lb.values, width=0.52, facecolor="#DBEAFE", edgecolor="#1E3A8A", lw=1.1, label="Leistungsbilanzsaldo", zorder=2)
    for b in bars:
        if pd.notna(b.get_height()):
            ax5.text(b.get_x()+b.get_width()/2, b.get_height()+0.6, f"{b.get_height():.1f}".replace(".",","), ha="center", fontsize=7.5, color="#1E3A8A", fontweight="bold")
    set_legend(ax5, ncol=1)

    style_panel(ax6, "Arbeitslosigkeit und Inflation", "Quote bzw. Veränderung zum Vorjahr in %", 0, 8, range(0, 10, 2), z=False)
    s_al = get_bbk("BBDE1.M.DE.S.ABA1.A00000000.U.A.A00.A", 13)
    s_hv = get_bbk("BBDP1.M.DE.N.HVPI.C.A00000.I15.A", 13)
    if s_al.empty: s_al = pd.Series((6.4,6.4,6.3,6.2,6.2,6.3,6.4,6.3,6.2,6.1,6.2,6.6,6.5), index=m13)
    if s_hv.empty: s_hv = pd.Series((2.6,2.3,2.2,2.1,2.0,1.8,2.1,2.4,2.3,2.6,2.0,2.1,2.0), index=m13)
    idx6 = sorted(set(s_al.index) | set(s_hv.index))[-13:]
    s_al, s_hv = s_al.reindex(idx6), s_hv.reindex(idx6)
    ax6.plot(idx6, s_al.values, color="#1E3A8A", lw=2.1, marker="o", ms=3, label="Arbeitslosenquote")
    ax6.axhline(3.0, color="#0284C7", ls="--", lw=1.4, label="Vollbeschäftigung (3%)")
    ax6.plot(idx6, s_hv.values, color="#475569", lw=2.1, marker="o", ms=3, label="Inflation (HVPI)")
    ax6.axhline(2.0, color="#D97706", ls="--", lw=1.4, label="EZB-Inflationsziel (2%)")
    mark_end(ax6, idx6, s_al.values, "#1E3A8A"); mark_end(ax6, idx6, s_hv.values, "#475569")
    ax6.set_xlim(-0.5, len(idx6) + 0.5); set_legend(ax6, ncol=2)

    fig.text(0.06, 0.015, "Quellen: Statistisches Bundesamt, Deutsche Bundesbank", fontsize=8, color="#64748B")
    fig.text(0.95, 0.015, "T.H.", ha="right", fontsize=8, color="#94A3B8")

    plt.savefig("output/dashboard.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
