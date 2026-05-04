import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler

# ============================================================
# PARAMETRI MODIFICABILI DALL'UTENTE
# ============================================================

# --- File di input ---
file_T   = 'RLC_Rres.txt'   # frequenza(kHz)  Vin(V)  Vout(V)  scala_Vin(V)  scala_Vout(V)
file_phi = 'RLC_phase.txt'  # frequenza(kHz)  Dt(ns)  scala_Dt(ns)

# --- Configurazione ---
# Ai capi di R:  T = 1/sqrt(1 + Q^2*(w/w0 - w0/w)^2)
#                phi_R = arctan((1/(wC) - wL)/R)  --> va da +pi/2 a -pi/2, zero a f0
#
# f1 < f0 < f2  (frequenze di taglio a T_max/sqrt(2) e phi = ±pi/4)

# Errore sistematico di calibrazione per T (1.2%)
sigma_kv = 0.012

# ---- Range di fit per T ----
T_f1_fmin = 122    # kHz
T_f1_fmax = 127    # kHz
T_f2_fmin = 133.5    # kHz
T_f2_fmax = 139    # kHz

# ---- Range di fit per phi ----
phi_f1_fmin = 120  # kHz
phi_f1_fmax = 127  # kHz
phi_f2_fmin = 134.6  # kHz
phi_f2_fmax = 138.3  # kHz

# ============================================================
# RANGE DEGLI ASSI E SCALA LOG PER OGNI GRAFICO
# ============================================================
# Per ogni grafico imposta:
#   xmin, xmax : limiti asse x  (None = automatico)
#   ymin, ymax : limiti asse y  (None = automatico)
#   xlog       : True per scala logaritmica sull'asse x
#   ylog       : True per scala logaritmica sull'asse y
#
# I range si applicano al pannello dati (superiore).
# Il pannello residui condivide sempre lo stesso asse x.

# --- Grafico T intorno a f1 ---
T_f1_xmin, T_f1_xmax = 118, 130
T_f1_ymin, T_f1_ymax = 0.15, 0.45
T_f1_xlog, T_f1_ylog = False, False

# --- Grafico T intorno a f2 ---
T_f2_xmin, T_f2_xmax = 130, 142
T_f2_ymin, T_f2_ymax = 0.1, 0.45
T_f2_xlog, T_f2_ylog = False, False

# --- Grafico phi intorno a f1 ---
phi_f1_xmin, phi_f1_xmax = 122.5, 129
phi_f1_ymin, phi_f1_ymax = -1.5, 0
phi_f1_xlog, phi_f1_ylog = False, False

# --- Grafico phi intorno a f2 ---
phi_f2_xmin, phi_f2_xmax = 133.5, 140
phi_f2_ymin, phi_f2_ymax = 0, 1.5
phi_f2_xlog, phi_f2_ylog = False, False

# --- Grafico panoramico: pannello T ---
pan_T_xmin, pan_T_xmax = 100, 160 
pan_T_ymin, pan_T_ymax = None, None
pan_T_xlog, pan_T_ylog = False, False

# --- Grafico panoramico: pannello phi ---
pan_phi_xmin, pan_phi_xmax = None, None
pan_phi_ymin, pan_phi_ymax = None, None
pan_phi_xlog, pan_phi_ylog = True, False

# ============================================================
# SETTAGGIO GRAFICI
# ============================================================
plt.style.use(hep.style.ROOT)
params = {
    'legend.fontsize': '9',
    'legend.loc': 'best',
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.facecolor': 'w',
    'legend.edgecolor': 'gray',
    'figure.figsize': (6, 5),
    'axes.labelsize': '10',
    'figure.titlesize': '13',
    'axes.titlesize': '11',
    'xtick.labelsize': '9',
    'ytick.labelsize': '9',
    'lines.linewidth': '1.2',
    'text.usetex': True,
    'axes.formatter.min_exponent': '2',
    'figure.constrained_layout.use': True,
}
plt.rcParams.update(params)
plt.rcParams['axes.prop_cycle'] = cycler(color=['b', 'r', 'g', 'c', 'm', 'y', 'k'])

# ============================================================
# FUNZIONE DI UTILITA': applica range e scala agli assi
# ============================================================
def apply_axis_settings(ax, xmin=None, xmax=None, ymin=None, ymax=None,
                        xlog=False, ylog=False):
    if xmin is not None or xmax is not None:
        ax.set_xlim(xmin, xmax)
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')

# ============================================================
# FUNZIONI DI UTILITA' FIT
# ============================================================
def linear(x, m, q):
    return m * x + q

def weighted_linfit(x, y, sy):
    w   = 1.0 / sy**2
    S   = np.sum(w);  Sx = np.sum(w*x);  Sy = np.sum(w*y)
    Sxx = np.sum(w*x**2);  Sxy = np.sum(w*x*y)
    D   = S*Sxx - Sx**2
    m   = (S*Sxy - Sx*Sy) / D
    q   = (Sxx*Sy - Sx*Sxy) / D
    sm  = np.sqrt(S/D);  sq = np.sqrt(Sxx/D)
    cov_mq  = -Sx/D
    residui = y - linear(x, m, q)
    chi2    = np.sum((residui/sy)**2)
    ndf     = len(x) - 2
    chi2rid = chi2/ndf if ndf > 0 else np.nan
    return m, q, sm, sq, cov_mq, chi2rid, chi2, ndf

def freq_from_linear(y_target, m, q, sm, sq, cov_mq):
    f   = (y_target - q) / m
    dfm = -f / m;  dfq = -1.0 / m
    sf  = np.sqrt(dfm**2*sm**2 + dfq**2*sq**2 + 2*dfm*dfq*cov_mq)
    return f, sf

# ============================================================
# CARICAMENTO DATI: FUNZIONE DI TRASFERIMENTO
# ============================================================
print("=" * 60)
print("CARICAMENTO DATI")
print("=" * 60)

data_T = np.loadtxt(file_T)
fr_T  = data_T[:, 0];  Vin = data_T[:, 1];  Vout = data_T[:, 2]
sVin  = data_T[:, 3];  sVout = data_T[:, 4]

A     = Vout / Vin
sig_Vin_let  = 0.41 * sVin  / 10.0
sig_Vout_let = 0.41 * sVout / 10.0
sA_let = A * np.sqrt((sig_Vin_let/Vin)**2 + (sig_Vout_let/Vout)**2)

# ============================================================
# STIMA DI T_max DAI DATI
# ============================================================
T_max    = np.max(A)
T_target = T_max / np.sqrt(2)
print(f"\n  T_max sperimentale        = {T_max:.5f}")
print(f"  T_target = T_max/sqrt(2)  = {T_target:.5f}")

# ============================================================
# CARICAMENTO DATI: SFASAMENTO
# ============================================================
data_phi = np.loadtxt(file_phi)
fr_phi  = data_phi[:, 0];  Dt = data_phi[:, 1];  sDt_sc = data_phi[:, 2]

phi_rad = 2.0 * np.pi * (fr_phi * 1e3) * (Dt * 1e-9)
sig_Dt  = 0.41 * (sDt_sc / 10.0) * np.sqrt(2)
sphi    = 2.0 * np.pi * (fr_phi * 1e3) * (sig_Dt * 1e-9)

# ============================================================
# FIT LINEARI – FUNZIONE DI TRASFERIMENTO
# ============================================================
print("\n" + "=" * 60)
print("FIT LINEARE – FUNZIONE DI TRASFERIMENTO")
print("=" * 60)

mask_T1 = (fr_T >= T_f1_fmin) & (fr_T <= T_f1_fmax)
fr_T1, A_T1, sA_T1 = fr_T[mask_T1], A[mask_T1], sA_let[mask_T1]
if len(fr_T1) < 2:
    raise ValueError("Troppo pochi punti in T_f1. Modifica T_f1_fmin/T_f1_fmax.")

m_T1, q_T1, sm_T1, sq_T1, cov_mq_T1, chi2r_T1, chi2_T1, ndf_T1 = \
    weighted_linfit(fr_T1, A_T1, sA_T1)
sm_T1_tot = np.sqrt(sm_T1**2 + 2*m_T1**2*sigma_kv**2)
f1_T, sf1_T = freq_from_linear(T_target, m_T1, q_T1, sm_T1_tot, sq_T1, cov_mq_T1)

print(f"\n--- Intorno a f1 ---")
print(f"  Punti: {len(fr_T1)},  NDF = {ndf_T1}")
print(f"  m = {m_T1:.5e} ± {sm_T1:.1e}  (solo lettura)  |  ± {sm_T1_tot:.1e} (con cal.)")
print(f"  q = {q_T1:.5e} ± {sq_T1:.1e}")
print(f"  chi2/ndf = {chi2r_T1:.3f}  (chi2 = {chi2_T1:.2f}, ndf = {ndf_T1})")
print(f"  f1(T) = ({f1_T:.4f} ± {sf1_T:.4f}) kHz")

mask_T2 = (fr_T >= T_f2_fmin) & (fr_T <= T_f2_fmax)
fr_T2, A_T2, sA_T2 = fr_T[mask_T2], A[mask_T2], sA_let[mask_T2]
if len(fr_T2) < 2:
    raise ValueError("Troppo pochi punti in T_f2. Modifica T_f2_fmin/T_f2_fmax.")

m_T2, q_T2, sm_T2, sq_T2, cov_mq_T2, chi2r_T2, chi2_T2, ndf_T2 = \
    weighted_linfit(fr_T2, A_T2, sA_T2)
sm_T2_tot = np.sqrt(sm_T2**2 + 2*m_T2**2*sigma_kv**2)
f2_T, sf2_T = freq_from_linear(T_target, m_T2, q_T2, sm_T2_tot, sq_T2, cov_mq_T2)

print(f"\n--- Intorno a f2 ---")
print(f"  Punti: {len(fr_T2)},  NDF = {ndf_T2}")
print(f"  m = {m_T2:.5e} ± {sm_T2:.1e}  (solo lettura)  |  ± {sm_T2_tot:.1e} (con cal.)")
print(f"  q = {q_T2:.5e} ± {sq_T2:.1e}")
print(f"  chi2/ndf = {chi2r_T2:.3f}  (chi2 = {chi2_T2:.2f}, ndf = {ndf_T2})")
print(f"  f2(T) = ({f2_T:.4f} ± {sf2_T:.4f}) kHz")

f0_T = np.sqrt(f1_T * f2_T);  Delta_f_T = f2_T - f1_T;  Q_T = f0_T / Delta_f_T
sf0_T      = 0.5*f0_T*np.sqrt((sf1_T/f1_T)**2 + (sf2_T/f2_T)**2)
sDelta_f_T = np.sqrt(sf1_T**2 + sf2_T**2)
sQ_T       = Q_T*np.sqrt((sf0_T/f0_T)**2 + (sDelta_f_T/Delta_f_T)**2)

print(f"\n--- Grandezze fisiche da T ---")
print(f"  f0      = ({f0_T:.4f} ± {sf0_T:.4f}) kHz")
print(f"  Delta_f = ({Delta_f_T:.4f} ± {sDelta_f_T:.4f}) kHz")
print(f"  Q       = {Q_T:.3f} ± {sQ_T:.3f}")

# ============================================================
# FIT LINEARI – SFASAMENTO
# ============================================================
# Ai capi di R con la convenzione di segno dei dati sperimentali:
# phi cresce da -pi/2 a +pi/2 con la frequenza, quindi:
#   f1 (fronte sinistro) -> phi = -pi/4
#   f2 (fronte destro)   -> phi = +pi/4
phi_target_f1 = -np.pi / 4.0
phi_target_f2 = +np.pi / 4.0

print("\n" + "=" * 60)
print("FIT LINEARE – SFASAMENTO (ai capi di R)")
print("=" * 60)

mask_p1 = (fr_phi >= phi_f1_fmin) & (fr_phi <= phi_f1_fmax)
fr_p1, phi_p1, sphi_p1 = fr_phi[mask_p1], phi_rad[mask_p1], sphi[mask_p1]
if len(fr_p1) < 2:
    raise ValueError("Troppo pochi punti in phi_f1. Modifica phi_f1_fmin/phi_f1_fmax.")

m_p1, q_p1, sm_p1, sq_p1, cov_mq_p1, chi2r_p1, chi2_p1, ndf_p1 = \
    weighted_linfit(fr_p1, phi_p1, sphi_p1)
f1_phi, sf1_phi = freq_from_linear(phi_target_f1, m_p1, q_p1, sm_p1, sq_p1, cov_mq_p1)

print(f"\n--- Intorno a f1 ---")
print(f"  Punti: {len(fr_p1)},  NDF = {ndf_p1}")
print(f"  m = {m_p1:.5e} ± {sm_p1:.1e}  rad/kHz")
print(f"  q = {q_p1:.5e} ± {sq_p1:.1e}  rad")
print(f"  chi2/ndf = {chi2r_p1:.3f}  (chi2 = {chi2_p1:.2f}, ndf = {ndf_p1})")
print(f"  f1(phi) = ({f1_phi:.4f} ± {sf1_phi:.4f}) kHz   [target: +pi/4]")

mask_p2 = (fr_phi >= phi_f2_fmin) & (fr_phi <= phi_f2_fmax)
fr_p2, phi_p2, sphi_p2 = fr_phi[mask_p2], phi_rad[mask_p2], sphi[mask_p2]
if len(fr_p2) < 2:
    raise ValueError("Troppo pochi punti in phi_f2. Modifica phi_f2_fmin/phi_f2_fmax.")

m_p2, q_p2, sm_p2, sq_p2, cov_mq_p2, chi2r_p2, chi2_p2, ndf_p2 = \
    weighted_linfit(fr_p2, phi_p2, sphi_p2)
f2_phi, sf2_phi = freq_from_linear(phi_target_f2, m_p2, q_p2, sm_p2, sq_p2, cov_mq_p2)

print(f"\n--- Intorno a f2 ---")
print(f"  Punti: {len(fr_p2)},  NDF = {ndf_p2}")
print(f"  m = {m_p2:.5e} ± {sm_p2:.1e}  rad/kHz")
print(f"  q = {q_p2:.5e} ± {sq_p2:.1e}  rad")
print(f"  chi2/ndf = {chi2r_p2:.3f}  (chi2 = {chi2_p2:.2f}, ndf = {ndf_p2})")
print(f"  f2(phi) = ({f2_phi:.4f} ± {sf2_phi:.4f}) kHz   [target: -pi/4]")

f0_phi = np.sqrt(f1_phi * f2_phi);  Delta_f_phi = f2_phi - f1_phi;  Q_phi = f0_phi / Delta_f_phi
sf0_phi      = 0.5*f0_phi*np.sqrt((sf1_phi/f1_phi)**2 + (sf2_phi/f2_phi)**2)
sDelta_f_phi = np.sqrt(sf1_phi**2 + sf2_phi**2)
sQ_phi       = Q_phi*np.sqrt((sf0_phi/f0_phi)**2 + (sDelta_f_phi/Delta_f_phi)**2)

print(f"\n--- Grandezze fisiche da phi ---")
print(f"  f0      = ({f0_phi:.4f} ± {sf0_phi:.4f}) kHz")
print(f"  Delta_f = ({Delta_f_phi:.4f} ± {sDelta_f_phi:.4f}) kHz")
print(f"  Q       = {Q_phi:.3f} ± {sQ_phi:.3f}")

# ============================================================
# RIEPILOGO FINALE
# ============================================================
print("\n" + "=" * 60)
print("RIEPILOGO RISULTATI")
print("=" * 60)
print(f"  T_max = {T_max:.5f}   -->   T_target = T_max/sqrt(2) = {T_target:.5f}")
print(f"")
print(f"  Grandezza       T (funz. trasf.)           phi (sfasamento)")
print(f"  f1  (kHz)    {f1_T:.4f} ± {sf1_T:.4f}         {f1_phi:.4f} ± {sf1_phi:.4f}")
print(f"  f2  (kHz)    {f2_T:.4f} ± {sf2_T:.4f}         {f2_phi:.4f} ± {sf2_phi:.4f}")
print(f"  f0  (kHz)    {f0_T:.4f} ± {sf0_T:.4f}         {f0_phi:.4f} ± {sf0_phi:.4f}")
print(f"  Df  (kHz)    {Delta_f_T:.4f} ± {sDelta_f_T:.4f}         {Delta_f_phi:.4f} ± {sDelta_f_phi:.4f}")
print(f"  Q            {Q_T:.3f}  ± {sQ_T:.3f}            {Q_phi:.3f}  ± {sQ_phi:.3f}")

# ============================================================
# GRAFICI – FUNZIONE DI TRASFERIMENTO
# ============================================================

# ---- T intorno a f1 ----
fig, axes = plt.subplots(2, 1, figsize=(5, 4), sharex=True,
                          height_ratios=[3, 1], constrained_layout=True)
fig.suptitle(r'Funzione di trasferimento -- intorno a $f_1$')
ax = axes[0]
ax.errorbar(fr_T, A, yerr=sA_let, fmt='o', ms=2, color='lightgray', label=r'Tutti i dati', zorder=1)
ax.errorbar(fr_T1, A_T1, yerr=sA_T1, fmt='o', ms=3, color='blue', label=r'Dati nel fit', zorder=3)
ax.plot(np.linspace(T_f1_fmin, T_f1_fmax, 200),
        linear(np.linspace(T_f1_fmin, T_f1_fmax, 200), m_T1, q_T1),
        'r-', lw=1.5, label='Fit lineare', zorder=4)
ax.axhline(T_target, color='green', ls='dashed', lw=1,
           label=fr'$T_{{max}}/\sqrt{{2}} = {T_target:.4f}$')
ax.axvline(f1_T, color='orange', ls='dotted', lw=1.2,
           label=fr'$f_1 = {f1_T:.3f}$ kHz')
ax.set_ylabel(r'$A = V_{out}/V_{in}$');  ax.legend(fontsize=8)
apply_axis_settings(ax, T_f1_xmin, T_f1_xmax, T_f1_ymin, T_f1_ymax, T_f1_xlog, T_f1_ylog)
ax2 = axes[1]
ax2.errorbar(fr_T1, A_T1 - linear(fr_T1, m_T1, q_T1), yerr=sA_T1, fmt='o', ms=3, color='blue')
ax2.axhline(0, color='black', lw=0.8)
ax2.set_xlabel(r'Frequenza (kHz)');  ax2.set_ylabel(r'Residui')
ax2.text(0.02, 0.92, fr'$\chi^2/\mathrm{{ndf}} = {chi2r_T1:.2f}$',
         transform=ax2.transAxes, fontsize=8, va='top')
apply_axis_settings(ax2, T_f1_xmin, T_f1_xmax, xlog=T_f1_xlog)
plt.show()

# ---- T intorno a f2 ----
fig, axes = plt.subplots(2, 1, figsize=(5, 4), sharex=True,
                          height_ratios=[3, 1], constrained_layout=True)
fig.suptitle(r'Funzione di trasferimento -- intorno a $f_2$')
ax = axes[0]
ax.errorbar(fr_T, A, yerr=sA_let, fmt='o', ms=2, color='lightgray', label=r'Tutti i dati', zorder=1)
ax.errorbar(fr_T2, A_T2, yerr=sA_T2, fmt='o', ms=3, color='blue', label=r'Dati nel fit', zorder=3)
ax.plot(np.linspace(T_f2_fmin, T_f2_fmax, 200),
        linear(np.linspace(T_f2_fmin, T_f2_fmax, 200), m_T2, q_T2),
        'r-', lw=1.5, label='Fit lineare', zorder=4)
ax.axhline(T_target, color='green', ls='dashed', lw=1,
           label=fr'$T_{{max}}/\sqrt{{2}} = {T_target:.4f}$')
ax.axvline(f2_T, color='orange', ls='dotted', lw=1.2,
           label=fr'$f_2 = {f2_T:.3f}$ kHz')
ax.set_ylabel(r'$A = V_{out}/V_{in}$');  ax.legend(fontsize=8)
apply_axis_settings(ax, T_f2_xmin, T_f2_xmax, T_f2_ymin, T_f2_ymax, T_f2_xlog, T_f2_ylog)
ax2 = axes[1]
ax2.errorbar(fr_T2, A_T2 - linear(fr_T2, m_T2, q_T2), yerr=sA_T2, fmt='o', ms=3, color='blue')
ax2.axhline(0, color='black', lw=0.8)
ax2.set_xlabel(r'Frequenza (kHz)');  ax2.set_ylabel(r'Residui')
ax2.text(0.02, 0.92, fr'$\chi^2/\mathrm{{ndf}} = {chi2r_T2:.2f}$',
         transform=ax2.transAxes, fontsize=8, va='top')
apply_axis_settings(ax2, T_f2_xmin, T_f2_xmax, xlog=T_f2_xlog)
plt.show()

# ============================================================
# GRAFICI – SFASAMENTO
# ============================================================

# ---- phi intorno a f1  (target = +pi/4) ----
fig, axes = plt.subplots(2, 1, figsize=(5, 4), sharex=True,
                          height_ratios=[3, 1], constrained_layout=True)
fig.suptitle(r'Sfasamento $\Delta\phi_R$ -- intorno a $f_1$')
ax = axes[0]
ax.errorbar(fr_phi, phi_rad, yerr=sphi, fmt='o', ms=2, color='lightgray', label=r'Tutti i dati', zorder=1)
ax.errorbar(fr_p1, phi_p1, yerr=sphi_p1, fmt='o', ms=3, color='blue', label=r'Dati nel fit', zorder=3)
ax.plot(np.linspace(phi_f1_fmin, phi_f1_fmax, 200),
        linear(np.linspace(phi_f1_fmin, phi_f1_fmax, 200), m_p1, q_p1),
        'r-', lw=1.5, label='Fit lineare', zorder=4)
# phi_target_f1 = +pi/4  (fronte sinistro, alta frequenza relativa al target)
ax.axhline(phi_target_f1, color='green', ls='dashed', lw=1,
           label=r'$\Delta\phi = -\pi/4$')
ax.axhline(0, color='gray', ls='dotted', lw=0.8)
ax.axvline(f1_phi, color='orange', ls='dotted', lw=1.2,
           label=fr'$f(+\pi/4) = {f1_phi:.3f}$ kHz')
ax.set_ylabel(r'$\Delta\phi_R$ (rad)');  ax.legend(fontsize=8)
apply_axis_settings(ax, phi_f1_xmin, phi_f1_xmax, phi_f1_ymin, phi_f1_ymax,
                    phi_f1_xlog, phi_f1_ylog)
ax2 = axes[1]
ax2.errorbar(fr_p1, phi_p1 - linear(fr_p1, m_p1, q_p1), yerr=sphi_p1, fmt='o', ms=3, color='blue')
ax2.axhline(0, color='black', lw=0.8)
ax2.set_xlabel(r'Frequenza (kHz)');  ax2.set_ylabel(r'Residui (rad)')
ax2.text(0.02, 0.92, fr'$\chi^2/\mathrm{{ndf}} = {chi2r_p1:.2f}$',
         transform=ax2.transAxes, fontsize=8, va='top')
apply_axis_settings(ax2, phi_f1_xmin, phi_f1_xmax, xlog=phi_f1_xlog)
plt.show()

# ---- phi intorno a f2  (target = -pi/4) ----
fig, axes = plt.subplots(2, 1, figsize=(5, 4), sharex=True,
                          height_ratios=[3, 1], constrained_layout=True)
fig.suptitle(r'Sfasamento $\Delta\phi_R$ -- intorno a $f_2$')
ax = axes[0]
ax.errorbar(fr_phi, phi_rad, yerr=sphi, fmt='o', ms=2, color='lightgray', label=r'Tutti i dati', zorder=1)
ax.errorbar(fr_p2, phi_p2, yerr=sphi_p2, fmt='o', ms=3, color='blue', label=r'Dati nel fit', zorder=3)
ax.plot(np.linspace(phi_f2_fmin, phi_f2_fmax, 200),
        linear(np.linspace(phi_f2_fmin, phi_f2_fmax, 200), m_p2, q_p2),
        'r-', lw=1.5, label='Fit lineare', zorder=4)
# phi_target_f2 = -pi/4  (fronte destro, bassa frequenza relativa al target)
ax.axhline(phi_target_f2, color='green', ls='dashed', lw=1,
           label=r'$\Delta\phi = +\pi/4$')
ax.axhline(0, color='gray', ls='dotted', lw=0.8)
ax.axvline(f2_phi, color='orange', ls='dotted', lw=1.2,
           label=fr'$f(-\pi/4) = {f2_phi:.3f}$ kHz')
ax.set_ylabel(r'$\Delta\phi_R$ (rad)');  ax.legend(fontsize=8)
apply_axis_settings(ax, phi_f2_xmin, phi_f2_xmax, phi_f2_ymin, phi_f2_ymax,
                    phi_f2_xlog, phi_f2_ylog)
ax2 = axes[1]
ax2.errorbar(fr_p2, phi_p2 - linear(fr_p2, m_p2, q_p2), yerr=sphi_p2, fmt='o', ms=3, color='blue')
ax2.axhline(0, color='black', lw=0.8)
ax2.set_xlabel(r'Frequenza (kHz)');  ax2.set_ylabel(r'Residui (rad)')
ax2.text(0.02, 0.92, fr'$\chi^2/\mathrm{{ndf}} = {chi2r_p2:.2f}$',
         transform=ax2.transAxes, fontsize=8, va='top')
apply_axis_settings(ax2, phi_f2_xmin, phi_f2_xmax, xlog=phi_f2_xlog)
plt.show()

# ============================================================
# GRAFICI PANORAMICI – DUE FIGURE SEPARATE
# ============================================================

# ----- Prima figura: Modulo A -----
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(fr_T, A, yerr=sA_let, fmt='o', ms=2.5, color='blue', label=r'$A = V_{out}/V_{in}$')
ax.axhline(T_target, color='green', ls='dashed', lw=1,
           label=fr'$T_{{\mathrm{{max}}}}/\sqrt{{2}} = {T_target:.4f}$')
ax.axvspan(T_f1_fmin, T_f1_fmax, alpha=0.10, color='grey', label='Range fit $f_1$')
ax.axvspan(T_f2_fmin, T_f2_fmax, alpha=0.10, color='grey',  label='Range fit $f_2$')
ax.axvline(f1_T, color='orange', ls='dotted', lw=1.2)
ax.axvline(f2_T, color='orange',  ls='dotted', lw=1.2)
ax.set_ylabel(r'$A = V_{out}/V_{in}$')
ax.set_xlabel(r'Frequenza (kHz)')
ax.set_title(r'RLC – Funzione di trasferimento (modulo)')
ax.legend(fontsize=8)
apply_axis_settings(ax, pan_T_xmin, pan_T_xmax, pan_T_ymin, pan_T_ymax,
                    pan_T_xlog, pan_T_ylog)
plt.show()

# ----- Seconda figura: Sfasamento -----
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(fr_phi, phi_rad, yerr=sphi, fmt='o', ms=2.5, color='red', label=r'$\Delta\phi_R$')
ax.axhline(phi_target_f1, color='green',     ls='dashed', lw=1, label=r'$-\pi/4$')
ax.axhline(phi_target_f2, color='darkgreen', ls='dashed', lw=1, label=r'$+\pi/4$')
ax.axhline(0, color='gray', ls='dotted', lw=0.8)
ax.axvspan(phi_f1_fmin, phi_f1_fmax, alpha=0.10, color='grey')
ax.axvspan(phi_f2_fmin, phi_f2_fmax, alpha=0.10, color='grey')
ax.axvline(f1_phi, color='orange', ls='dotted', lw=1.2)
ax.axvline(f2_phi, color='orange',  ls='dotted', lw=1.2)
ax.set_ylabel(r'$\Delta\phi_R$ (rad)')
ax.set_xlabel(r'Frequenza (kHz)')
ax.set_title(r'RLC – Sfasamento')
ax.legend(fontsize=8)
# IMPORTANTE: qui non tocco l’asse x della fase (le impostazioni restano come da variabili)
apply_axis_settings(ax, pan_phi_xmin, pan_phi_xmax, pan_phi_ymin, pan_phi_ymax,
                    pan_phi_xlog, pan_phi_ylog)
plt.show()