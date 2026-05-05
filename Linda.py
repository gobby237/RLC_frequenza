# ANALISI DELLO SFASAMENTO AI CAPI DI R IN UN CIRCUITO RLC
# File di input atteso: colonne -> frequenza (kHz), Delta_t, scala_tempi
# La fase e' calcolata come phi = 2*pi*f*Delta_t [radianti].
# Scegli l'unita' di Delta_t e scala_tempi tramite TIME_UNIT = 'ns' oppure 'us'.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit

try:
    import mplhep as hep
    HAVE_HEP = True
except Exception:
    HAVE_HEP = False

# ============================================================
# PARAMETRI MODIFICABILI DALL'UTENTE
# ============================================================
file = 'RLC_phase'
inputname = file + '.txt'

# 'ns' se Delta_t e scala_tempi sono in nanosecondi, 'us' se sono in microsecondi
TIME_UNIT = 'ns'

# Convenzione del segno della fase.
# Modello: phi = PHASE_SIGN * arctan[Q(omega/omega0 - omega0/omega)].
# Se il fit e' specchiato rispetto ai dati, cambia +1 in -1.
PHASE_SIGN = +1.0

# True: fit con offset phi0. False: fit ideale senza offset.
FIT_PHI0 = True
phi0_init = 0.0  # rad

# Fa anche il fit senza offset per confronto
DO_COMPARE_NO_OFFSET = True

# Guess iniziale per Q. Se la stima automatica fallisce, usa Q_init_manual.
USE_AUTO_GUESS_Q = True
Q_init_manual = 10.0

# Errori temporali.
# Errore sui due cursori: sigma_t = sqrt(2) * (scala_t/10) * cursor_factor.
cursor_factor = 0.41

# Eventuale contributo relativo sulla misura di Delta_t; metti 0.0 se non vuoi usarlo.
errscala_dt = 0.03

# Sistematico additivo opzionale sulla fase [rad]
PHASE_SYS_RAD = 0.0

# Range di fit [kHz]
frfit0 = 0.0
frfit1 = 1.0e6

# Scansione chi2 per la mappa (phi0, omega0, Q)
MAKE_CHI2_MAP = True
NSI = 2.0
step_scan = 101  # meglio dispari: la griglia contiene il centro

# R esterna per ricavare L e C dalla fase
R_ext = 60.20     # ohm
eR_ext = 1.38     # ohm

# Grafica
SAVE_FIG = True
SHOW_FIG = True
USE_TEX = False
DEB = False

# ============================================================
# SETTAGGIO GRAFICI
# ============================================================
if HAVE_HEP:
    plt.style.use(hep.style.ROOT)

params = {
    'legend.fontsize': 10,
    'legend.loc': 'best',
    'legend.frameon': True,
    'legend.framealpha': 0.85,
    'legend.facecolor': 'w',
    'legend.edgecolor': '0.85',
    'figure.figsize': (6, 4),
    'axes.labelsize': 11,
    'figure.titlesize': 14,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.2,
    'text.usetex': USE_TEX,
    'axes.formatter.use_mathtext': True,
    'figure.constrained_layout.use': True,
}
plt.rcParams.update(params)

# ============================================================
# MODELLI
# ============================================================
def fitf_phase_R(x, omega0, Q):
    """Sfasamento ai capi di R in radianti."""
    omega = 2.0 * np.pi * np.asarray(x, dtype=float) * 1e3
    return PHASE_SIGN * np.arctan(Q * (omega / omega0 - omega0 / omega))


def fitf_with_phi0(x, phi0, omega0, Q):
    """Modello con offset additivo di fase phi0 [rad]."""
    return phi0 + fitf_phase_R(x, omega0, Q)


def model_selected(x, *par):
    if FIT_PHI0:
        return fitf_with_phi0(x, *par)
    return fitf_phase_R(x, *par)


def chi2_value(y, yerr, ymodel):
    return float(np.sum(((y - ymodel) / yerr) ** 2))


def savefig(name):
    if SAVE_FIG:
        plt.savefig(name, bbox_inches='tight', pad_inches=1, transparent=True,
                    facecolor='w', edgecolor='w', orientation='Portrait', dpi=120)
    if SHOW_FIG:
        plt.show()
    else:
        plt.close()

# ============================================================
# UTILITY CHI2
# ============================================================
def make_grid(center, sigma, n_sigma, n_step, positive=False):
    if n_step % 2 == 0:
        n_step += 1
    if (not np.isfinite(sigma)) or sigma <= 0:
        sigma = 0.02 * abs(center) if center != 0 else 1.0
    lo = center - n_sigma * sigma
    hi = center + n_sigma * sigma
    if positive:
        lo = max(lo, np.finfo(float).eps)
        hi = max(hi, lo * 1.01)
    grid = np.linspace(lo, hi, n_step)
    idx = np.argmin(np.abs(grid - center))
    grid[idx] = center
    grid.sort()
    return grid


def find_crossings(grid, prof, level):
    idx_min = int(np.nanargmin(prof))
    left = np.nan
    right = np.nan
    for i in range(idx_min, 0, -1):
        y1 = prof[i - 1] - level
        y2 = prof[i] - level
        if y1 * y2 <= 0:
            x1, x2 = grid[i - 1], grid[i]
            left = x1 - y1 * (x2 - x1) / (y2 - y1)
            break
    for i in range(idx_min, len(grid) - 1):
        y1 = prof[i] - level
        y2 = prof[i + 1] - level
        if y1 * y2 <= 0:
            x1, x2 = grid[i], grid[i + 1]
            right = x1 - y1 * (x2 - x1) / (y2 - y1)
            break
    return left, right


def build_chi2_map(fr, phase, ephase, popt, perr):
    """Costruisce una mappa chi2(phi0, omega0, Q). Richiede FIT_PHI0=True."""
    phi0_bf, om_bf, q_bf = popt
    e_phi0, e_om, e_q = perr

    phi0_grid = make_grid(phi0_bf, e_phi0, NSI, step_scan, positive=False)
    om_grid = make_grid(om_bf, e_om, NSI, step_scan, positive=True)
    q_grid = make_grid(q_bf, e_q, NSI, step_scan, positive=True)

    chi2_map = np.empty((len(phi0_grid), len(om_grid), len(q_grid)), dtype=float)

    # Vettorizzo su omega0 e Q, poi ciclo solo su phi0 per limitare la memoria.
    omega = (2.0 * np.pi * fr * 1e3)[:, None, None]
    om = om_grid[None, :, None]
    qq = q_grid[None, None, :]
    base = PHASE_SIGN * np.arctan(qq * (omega / om - om / omega))
    y = phase[:, None, None]
    s = ephase[:, None, None]

    for i, phi0 in enumerate(phi0_grid):
        chi2_map[i, :, :] = np.sum(((y - (base + phi0)) / s) ** 2, axis=0)

    chi2_min = float(np.min(chi2_map))
    idx_min = np.unravel_index(np.argmin(chi2_map), chi2_map.shape)

    # Profilazioni 2D e 1D
    chi2D_om_q = chi2_map.min(axis=0).T       # righe Q, colonne omega0
    prof_phi0 = chi2_map.min(axis=(1, 2))
    prof_om = chi2_map.min(axis=(0, 2))
    prof_q = chi2_map.min(axis=(0, 1))

    lvl = chi2_min + 1.0
    phi_l, phi_r = find_crossings(phi0_grid, prof_phi0, lvl)
    om_l, om_r = find_crossings(om_grid, prof_om, lvl)
    q_l, q_r = find_crossings(q_grid, prof_q, lvl)

    return {
        'phi0_grid': phi0_grid,
        'omega0_grid': om_grid,
        'Q_grid': q_grid,
        'chi2_map': chi2_map,
        'chi2D_omega0_Q': chi2D_om_q,
        'chi2_min': chi2_min,
        'idx_min': idx_min,
        'prof_phi0': prof_phi0,
        'prof_omega0': prof_om,
        'prof_Q': prof_q,
        'cross_phi0': (phi_l, phi_r),
        'cross_omega0': (om_l, om_r),
        'cross_Q': (q_l, q_r),
    }

# ============================================================
# CARICAMENTO DATI
# ============================================================
data = np.loadtxt(inputname)
if data.ndim == 1:
    data = data.reshape(1, -1)
if data.shape[1] < 3:
    raise RuntimeError('Il file deve contenere almeno 3 colonne: f(kHz), Delta_t, scala_tempi.')

fr_all = data[:, 0].astype(float)
dt_all = data[:, 1].astype(float)
scale_t_all = data[:, 2].astype(float)

idx_sort = np.argsort(fr_all)
fr_all = fr_all[idx_sort]
dt_all = dt_all[idx_sort]
scale_t_all = scale_t_all[idx_sort]

if TIME_UNIT.lower() == 'ns':
    time_factor = 1e-9
    time_label = 'ns'
elif TIME_UNIT.lower() == 'us':
    time_factor = 1e-6
    time_label = r'\mu s'
else:
    raise ValueError("TIME_UNIT deve essere 'ns' oppure 'us'.")

# Errore su Delta_t: cursori + eventuale contributo relativo sulla misura dt.
e_dt_all = np.sqrt(
    (np.sqrt(2.0) * (scale_t_all / 10.0) * cursor_factor) ** 2
    + (errscala_dt * np.abs(dt_all)) ** 2
)

phase_all = 2.0 * np.pi * (fr_all * 1e3) * (dt_all * time_factor)
ephase_all = 2.0 * np.pi * (fr_all * 1e3) * (e_dt_all * time_factor)

if PHASE_SYS_RAD > 0:
    ephase_all = np.sqrt(ephase_all ** 2 + PHASE_SYS_RAD ** 2)

positive = ephase_all[np.isfinite(ephase_all) & (ephase_all > 0)]
if len(positive) == 0:
    raise RuntimeError('Tutte le incertezze sulla fase sono nulle o non finite.')
ephase_all = np.where((ephase_all <= 0) | (~np.isfinite(ephase_all)),
                      np.nanmedian(positive), ephase_all)

mask_fit = (
    (fr_all >= frfit0) & (fr_all <= frfit1) & (fr_all > 0)
    & np.isfinite(phase_all) & np.isfinite(ephase_all) & (ephase_all > 0)
)
fr = fr_all[mask_fit]
dt = dt_all[mask_fit]
e_dt = e_dt_all[mask_fit]
phase = phase_all[mask_fit]
ephase = ephase_all[mask_fit]
N = len(fr)

if N < (4 if FIT_PHI0 else 3):
    raise RuntimeError('Troppi pochi punti nel range di fit.')

# Non uso np.unwrap: la fase teorica su R sta naturalmente tra -pi/2 e +pi/2.

# ============================================================
# STIMA INIZIALE
# ============================================================
idx_zero = int(np.argmin(np.abs(phase)))
f0_est = fr[idx_zero]
Binit = 2.0 * np.pi * f0_est * 1e3

if USE_AUTO_GUESS_Q:
    try:
        f_left = fr[fr < f0_est]
        ph_left = phase[fr < f0_est]
        f_right = fr[fr > f0_est]
        ph_right = phase[fr > f0_est]
        target_left = -PHASE_SIGN * np.pi / 4.0
        target_right = PHASE_SIGN * np.pi / 4.0
        f1 = f_left[np.argmin(np.abs(ph_left - target_left))]
        f2 = f_right[np.argmin(np.abs(ph_right - target_right))]
        Cinit = abs(f0_est / (f2 - f1)) if f2 > f1 else Q_init_manual
        if not np.isfinite(Cinit) or Cinit <= 0:
            Cinit = Q_init_manual
    except Exception:
        f1 = np.nan
        f2 = np.nan
        Cinit = Q_init_manual
else:
    f1 = np.nan
    f2 = np.nan
    Cinit = Q_init_manual

print(f'Stima iniziale: f0 = {f0_est:.6g} kHz, omega0 = {Binit:.4e} rad/s, Q = {Cinit:.3g}')
if np.isfinite(f1) and np.isfinite(f2):
    print(f'  Stima da +-pi/4: f1 = {f1:.6g} kHz, f2 = {f2:.6g} kHz')

# ============================================================
# GRAFICO 1: Delta_t e fase
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(9, 3), sharex=True,
                       constrained_layout=True, width_ratios=[1, 1])
ax[0].errorbar(fr, dt, yerr=e_dt, fmt='o', ms=2, label=r'$\Delta t$')
ax[0].set_ylabel(rf'$\Delta t$ ({time_label})')
ax[0].set_xlabel(r'Frequenza (kHz)')
ax[0].legend(loc='best')
ax[1].errorbar(fr, phase, yerr=ephase, fmt='o', ms=2, color='red',
               label=r'$\phi = 2\pi f \Delta t$')
ax[1].set_ylabel(r'Sfasamento $\phi$ (rad)')
ax[1].set_xlabel(r'Frequenza (kHz)')
ax[1].yaxis.set_ticks_position('right')
ax[1].yaxis.set_label_position('right')
ax[1].legend(loc='best')
savefig(file + '_1.png')

# ============================================================
# FIT CON SCIPY
# ============================================================
if FIT_PHI0:
    p0 = [phi0_init, Binit, Cinit]
    bounds = ([-np.pi, np.finfo(float).eps, np.finfo(float).eps],
              [ np.pi, np.inf, np.inf])
else:
    p0 = [Binit, Cinit]
    bounds = ([np.finfo(float).eps, np.finfo(float).eps], [np.inf, np.inf])

popt, pcov = curve_fit(model_selected, fr, phase, p0=p0,
                       sigma=ephase, absolute_sigma=True,
                       method='trf', bounds=bounds, max_nfev=200000)
perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
residuA = phase - model_selected(fr, *popt)
chisq = np.sum((residuA / ephase) ** 2)
df = N - len(popt)
chisq_rid = chisq / df

if FIT_PHI0:
    A_BF, B_BF, C_BF = popt
    eA_BF, eB_BF, eC_BF = perr
else:
    A_BF, eA_BF = 0.0, 0.0
    B_BF, C_BF = popt
    eB_BF, eC_BF = perr

print('============== BEST FIT con SciPy ====================')
print(f'FIT_PHI0 = {FIT_PHI0}')
if FIT_PHI0:
    print(f'phi0 = ({A_BF:.3e} +/- {eA_BF:.1e}) rad')
print(f'omega0 = ({B_BF:.5e} +/- {eB_BF:.1e}) rad/s')
print(f'f0     = ({B_BF/(2*np.pi*1e3):.4f} +/- {eB_BF/(2*np.pi*1e3):.4f}) kHz')
print(f'Q      = ({C_BF:.3f} +/- {eC_BF:.3f})')
print(f'chi2 = {chisq:.2f}, chi2/dof = {chisq_rid:.2f}')
print('=======================================================')

x_fit = np.linspace(min(fr), max(fr), 1000)

# ============================================================
# R, L, C da fit fase usando R esterna
# ============================================================
L_scipy = C_BF * R_ext / B_BF
C_scipy = 1.0 / (B_BF * C_BF * R_ext)
eL_scipy = L_scipy * np.sqrt((eC_BF/C_BF)**2 + (eR_ext/R_ext)**2 + (eB_BF/B_BF)**2)
eC_scipy = C_scipy * np.sqrt((eC_BF/C_BF)**2 + (eR_ext/R_ext)**2 + (eB_BF/B_BF)**2)
print('============== R, L, C da SciPy (fase) ====================')
print(f'R  = ({R_ext:.2f} +/- {eR_ext:.2f}) ohm  [da fit TR]')
print(f'L  = ({L_scipy*1e3:.4f} +/- {eL_scipy*1e3:.4f}) mH')
print(f'C  = ({C_scipy*1e9:.4f} +/- {eC_scipy*1e9:.4f}) nF')
print('============================================================')

# ============================================================
# GRAFICO 2: Fit + residui
# ============================================================
fig, ax = plt.subplots(2, 1, figsize=(5, 4), sharex=True,
                       constrained_layout=True, height_ratios=[2, 1])
ax[0].plot(x_fit, model_selected(x_fit, *popt), label='Fit SciPy', linestyle='--', color='black')
if FIT_PHI0:
    ax[0].plot(x_fit, fitf_with_phi0(x_fit, phi0_init, Binit, Cinit),
               label='Init guess', linestyle='dashed', color='green')
else:
    ax[0].plot(x_fit, fitf_phase_R(x_fit, Binit, Cinit),
               label='Init guess', linestyle='dashed', color='green')
ax[0].errorbar(fr, phase, yerr=ephase, fmt='o', ms=2, color='red', label=r'$\phi$')
ax[0].legend(loc='upper left')
ax[0].set_ylabel(r'Sfasamento $\phi$ (rad)')
ax[1].errorbar(fr, residuA, yerr=ephase, fmt='o', ms=2, color='red', label='Residui')
ax[1].axhline(0, color='black')
ax[1].set_ylabel('Residui (rad)')
ax[1].set_xlabel('Frequenza (kHz)')
savefig(file + '_2.png')

# ============================================================
# Fit senza offset per confronto
# ============================================================
if DO_COMPARE_NO_OFFSET:
    popt2, pcov2 = curve_fit(fitf_phase_R, fr, phase, p0=[Binit, Cinit],
                             sigma=ephase, absolute_sigma=True,
                             method='trf', bounds=([np.finfo(float).eps, np.finfo(float).eps], [np.inf, np.inf]),
                             max_nfev=200000)
    perr2 = np.sqrt(np.maximum(np.diag(pcov2), 0.0))
    res2 = phase - fitf_phase_R(fr, *popt2)
    chi2_2par = np.sum((res2/ephase)**2)
    chi2_rid_2par = chi2_2par / (N - 2)
    print('====== FIT 2 parametri (offset forzato a 0) ======')
    print(f'f0 = {popt2[0]/(2*np.pi*1e3):.4f} +/- {perr2[0]/(2*np.pi*1e3):.4f} kHz')
    print(f'Q  = {popt2[1]:.3f} +/- {perr2[1]:.3f}')
    print(f'chi2 = {chi2_2par:.2f}, chi2/dof = {chi2_rid_2par:.2f}')
    print('==================================================')

# ============================================================
# Scansione chi2 e grafici solo se FIT_PHI0=True
# ============================================================
if MAKE_CHI2_MAP and FIT_PHI0:
    scan = build_chi2_map(fr, phase, ephase, popt, perr)
    chi2_min = scan['chi2_min']
    idx = scan['idx_min']
    A_chi = scan['phi0_grid']
    B_chi = scan['omega0_grid']
    C_chi = scan['Q_grid']
    chi2D = scan['chi2D_omega0_Q']
    prof_A = scan['prof_phi0']
    prof_B = scan['prof_omega0']
    prof_C = scan['prof_Q']
    A_left, A_right = scan['cross_phi0']
    B_left, B_right = scan['cross_omega0']
    C_left, C_right = scan['cross_Q']

    residui_chi2 = phase - fitf_with_phi0(fr, A_chi[idx[0]], B_chi[idx[1]], C_chi[idx[2]])
    print(f'\nchi2 minimo dalla mappa: {chi2_min:.4f}  (posizione: {idx})')
    print(f'Verifica chi2 residui:   {np.sum((residui_chi2/ephase)**2):.4f}')

    errA = A_chi[idx[0]] - A_left if np.isfinite(A_left) else np.nan
    errAA = A_right - A_chi[idx[0]] if np.isfinite(A_right) else np.nan
    errB = B_chi[idx[1]] - B_left if np.isfinite(B_left) else np.nan
    errBB = B_right - B_chi[idx[1]] if np.isfinite(B_right) else np.nan
    errC = C_chi[idx[2]] - C_left if np.isfinite(C_left) else np.nan
    errCC = C_right - C_chi[idx[2]] if np.isfinite(C_right) else np.nan

    print('\n============== BEST FIT con chi2 ====================')
    print(f'phi0  = ({A_chi[idx[0]]:.3e} - {errA:.1e} + {errAA:.1e}) rad')
    print(f'omega0 = ({B_chi[idx[1]]:.5e} - {errB:.1e} + {errBB:.1e}) rad/s')
    print(f'f0     = ({B_chi[idx[1]]/(2*np.pi*1e3):.4f} - {errB/(2*np.pi*1e3):.4f} + {errBB/(2*np.pi*1e3):.4f}) kHz')
    print(f'Q      = ({C_chi[idx[2]]:.3e} - {errC:.1e} + {errCC:.1e})')
    print(f'chi2   = {chi2_min:.2f}')
    print(f'chi2_rid = {chi2_min/(N-3):.2f}')
    print('=======================================================')

    # Grafico 3: fit da minimo chi2
    fig, ax = plt.subplots(2, 1, figsize=(5, 4), sharex=True,
                           constrained_layout=True, height_ratios=[2, 1])
    ax[0].plot(x_fit, fitf_with_phi0(x_fit, A_chi[idx[0]], B_chi[idx[1]], C_chi[idx[2]]),
               label='Fit chi2', linestyle='--', color='blue')
    ax[0].errorbar(fr, phase, yerr=ephase, fmt='o', ms=2, color='red', label=r'$\phi$')
    ax[0].legend(loc='upper left')
    ax[0].set_ylabel(r'Sfasamento $\phi$ (rad)')
    ax[1].errorbar(fr, residui_chi2, yerr=ephase, fmt='o', ms=2, color='red')
    ax[1].axhline(0, color='black')
    ax[1].set_ylabel('Residui (rad)')
    ax[1].set_xlabel('Frequenza (kHz)')
    savefig(file + '_3.png')

    # Grafico 4: mappa omega0-Q con profili
    cmap = mpl.colormaps['plasma'].reversed()
    level = np.linspace(np.min(chi2D), np.max(chi2D), 100)
    line_c = 'gray'
    lvl = chi2_min + 1.0
    fig, ax = plt.subplots(2, 2, figsize=(5.5, 5), constrained_layout=True,
                           height_ratios=[3, 1], width_ratios=[1, 3],
                           sharex='col', sharey='row')
    fig.suptitle(r'$\chi^2 (\omega_0, Q)$')
    im = ax[0, 1].contourf(B_chi, C_chi, chi2D, levels=level, cmap=cmap)
    cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=ax[0, 1])
    cbar.set_label(r'$\chi^2$', rotation=360)
    contour_levels = [v for v in [chi2_min+1, chi2_min+2.3, chi2_min+3.8] if np.min(chi2D) < v < np.max(chi2D)]
    if contour_levels:
        CS = ax[0, 1].contour(B_chi, C_chi, chi2D, levels=contour_levels,
                              linewidths=1, colors='k', alpha=0.5, linestyles='dotted')
        ax[0, 1].clabel(CS, inline=True, fontsize=9, fmt='%.1f')
    ax[0, 1].plot(B_chi[idx[1]], C_chi[idx[2]], 'wo', mec='k', ms=4)
    if np.isfinite(C_left): ax[0, 1].axhline(C_left, color=line_c, ls='dashed')
    if np.isfinite(C_right): ax[0, 1].axhline(C_right, color=line_c, ls='dashed')
    if np.isfinite(B_left): ax[0, 1].axvline(B_left, color=line_c, ls='dashed')
    if np.isfinite(B_right): ax[0, 1].axvline(B_right, color=line_c, ls='dashed')

    ax[0, 0].plot(prof_C, C_chi, ls='-')
    ax[0, 0].axvline(lvl, color=line_c, ls='dashed')
    if np.isfinite(C_left): ax[0, 0].axhline(C_left, color=line_c, ls='dashed')
    if np.isfinite(C_right): ax[0, 0].axhline(C_right, color=line_c, ls='dashed')
    ax[0, 0].set_xlim(chi2_min - 1, chi2_min + 4)
    ax[0, 0].set_ylabel(r'$Q$-valore')

    ax[1, 1].plot(B_chi, prof_B)
    ax[1, 1].axhline(lvl, color=line_c, ls='dashed')
    if np.isfinite(B_left): ax[1, 1].axvline(B_left, color=line_c, ls='dashed')
    if np.isfinite(B_right): ax[1, 1].axvline(B_right, color=line_c, ls='dashed')
    ax[1, 1].set_ylim(chi2_min - 1, chi2_min + 4)
    ax[1, 1].set_xlabel(r'$\omega_0\;(\mathrm{rad/s})$', loc='center')
    ax[1, 0].set_axis_off()
    savefig(file + '_4.png')

    # Grafico 5: offset
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    ax.plot(A_chi, prof_A)
    ax.axhline(lvl, color='gray', ls='dashed', label=r'$\chi^2_{min}+1$')
    ax.axvline(A_chi[idx[0]], color='k', ls='dashed', label=rf'$\phi_0$ BF = {A_chi[idx[0]]:.3f} rad')
    if np.isfinite(A_left): ax.axvline(A_left, color='gray', ls='dotted')
    if np.isfinite(A_right): ax.axvline(A_right, color='gray', ls='dotted')
    ax.set_xlabel(r'Offset $\phi_0$ (rad)')
    ax.set_ylabel(r'$\chi^2$ profilato')
    ax.legend(fontsize=8)
    savefig(file + '_5.png')
else:
    print('Mappa chi2 3D saltata: MAKE_CHI2_MAP=False oppure FIT_PHI0=False.')
