#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mappe del chi quadro nel piano (L, C) per un circuito RLC serie,
ai capi della resistenza R, usando due dataset:

  1) fase:   frequenza [kHz]   Delta_t [ns]   scala_Dt [ns]
  2) moduli: frequenza [kHz]   Vin [V]   Vout [V]   V/div Vin   V/div Vout

Questa versione costruisce le mappe chi2(L,C) profilando/minimizzando su R,
NON su Q, senza prior su R. La griglia e' centrata sui valori veri attesi. Per ogni punto della griglia (L,C), la forma della risonanza e'
calcolata con:

    omega0(L,C) = 1/sqrt(L*C)
    Q(L,C,R)   = sqrt(L/C)/R

Parametri di disturbo ancora profilati per descrivere bene le curve reali:

  - Modulo: normalizzazione A, ed eventualmente offset additivo.
  - Fase: nessun parametro additivo o di ritardo viene fittato.
           L'offset iniziale phi0 e il delay temporale sono fissati a zero.

Output:
  - chi2_LC_moduli_profile_R.png
  - chi2_LC_phase_profile_R.png
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

try:
    import mplhep as hep
    plt.style.use(hep.style.ROOT)
except Exception:
    pass

# ============================================================
# PARAMETRI MODIFICABILI DALL'UTENTE
# ============================================================

# --- File di input ---
PHASE_FILE = 'RLC_phase.txt'
MODULI_FILE = 'RLC_Rres_nuovo.txt'

# --- Intervalli di frequenza per i fit ---
PHASE_FR_MIN_KHZ = 0.0
PHASE_FR_MAX_KHZ = 1.0e6

MOD_FR_MIN_KHZ = 4.0
MOD_FR_MAX_KHZ = 9.0e5

# --- Errori per i moduli, come nel codice originale ---
READING_ERROR_DIV = 0.1
SCALE_ERROR_FRAC = 0.03

# --- Stime iniziali per i fit globali preliminari ---
# Il fit preliminare serve solo a centrare bene la mappa e stampare un controllo.
F0_INIT_KHZ = None       # se None, usa il massimo del modulo
Q_INIT = 10.0
A_INIT = None            # se None, usa max(T_R)

# --- Modello del modulo ---
# T_R(f) = offset + A/sqrt(1 + Q(L,C,R)^2 * (... )^2)
FIT_MOD_OFFSET = False
MOD_OFFSET_INIT = 0.0

# --- Modello della fase ---
# phi(f) = sign*atan[Q(L,C,R)*(...)]
# In questa versione NON si fittano ne' l'offset iniziale phi0 ne' il delay temporale.
PHASE_SIGN = 'auto'       # 'auto', +1, -1
FIT_PHASE_OFFSET = False
FIT_PHASE_DELAY = False
PHASE_OFFSET_INIT_RAD = 0.0
PHASE_DELAY_INIT_NS = 0.0
PHASE_DELAY_BOUND_NS = 5000.0

# --- Griglia L-C ---
# Scegli L_REF_H vicino al valore atteso. Se C_REF_F=None, C_ref viene scelto
# usando omega0 del fit preliminare: L_ref*C_ref = 1/omega0_fit^2.
# Valori veri/attesi per centrare la visualizzazione della mappa.
# C_TRUE_F = 5.2e-9 corrisponde a 5.2 nF; L_TRUE_H = 0.30e-3 corrisponde a 0.30 mH.
L_TRUE_H = 0.30e-3
C_TRUE_F = 5.2e-9
R_TRUE_OHM = 20.0

L_REF_H = L_TRUE_H
C_REF_F = C_TRUE_F

# Tipo di griglia per L e C: 'linear' produce mappe simili all'esempio inviato;
# 'log' e' utile se vuoi mostrare una regione molto ampia.
LC_GRID_SCALE = 'linear'       # 'linear' oppure 'log'
LC_RANGE_FACTOR = 2.0          # intorno ai valori veri: centro/factor ... centro*factor
N_LC = 220

# --- Profilazione su R ---
# La mappa NON scansiona Q. Scansiona R e calcola Q(L,C,R)=sqrt(L/C)/R.
# Se R_REF_OHM=None, il centro della griglia R e' stimato dal Q del fit globale
# e dai valori centrali L_ref, C_ref. Se vuoi centrare su circa 20 ohm, lascia 20.
R_REF_OHM = R_TRUE_OHM
R_SCAN_FACTOR = 1.6            # intorno a R vera: circa 12.5 ... 32 ohm se R_TRUE=20
N_R_SCAN = 240
R_MIN_ABS_OHM = 0.05
R_MAX_ABS_OHM = 500.0

# --- Grafica ---
USE_TEX = False
FONT_SIZE = 12
CMAP_NAME = 'jet'              # simile all'esempio; prova anche 'plasma_r'
COLOR_MODE = 'log_chi2'        # 'log_chi2' oppure 'delta_chi2'
DELTA_CHI2_COLOR_MAX = None    # usato se COLOR_MODE='delta_chi2'
CONTOUR_LEVELS_DELTA = [1.0, 2.30, 6.18, 11.83, 25.0, 50.0, 100.0]
PLOT_UNITS = 'SI'              # 'SI' -> L [H], C [F]; 'lab' -> L [mH], C [nF]
SHOW_LC_CONSTANT_LINE = True

# --- Output ---
OUT_MODULI_MAP = 'chi2_LC_moduli_profile_R_true_window.png'
OUT_PHASE_MAP = 'chi2_LC_phase_profile_R_true_window.png'

# ============================================================
# STILE GRAFICO
# ============================================================
plt.rcParams.update({
    'text.usetex': USE_TEX,
    'font.size': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 1,
    'legend.fontsize': FONT_SIZE - 1,
    'xtick.labelsize': FONT_SIZE - 1,
    'ytick.labelsize': FONT_SIZE - 1,
    'figure.constrained_layout.use': True,
})

# ============================================================
# MODELLI
# ============================================================

def omega_from_khz(f_khz):
    return 2.0 * np.pi * np.asarray(f_khz, dtype=float) * 1.0e3


def q_from_lcr(L, C, R):
    """Fattore di merito del RLC serie: Q = sqrt(L/C)/R."""
    return np.sqrt(L / C) / R


def mod_R_omegaQ(f_khz, A, omega0, Q):
    """Vecchia parametrizzazione stabile usata solo per il fit preliminare."""
    omega = omega_from_khz(f_khz)
    x = omega / omega0 - omega0 / omega
    return A / np.sqrt(1.0 + (Q * x) ** 2)


def mod_R_omegaQ_offset(f_khz, A, omega0, Q, offset):
    return offset + mod_R_omegaQ(f_khz, A, omega0, Q)


def phase_R_omegaQ(f_khz, omega0, Q, phi0=0.0, t_delay_ns=0.0, sign=+1.0):
    """Vecchia parametrizzazione stabile usata solo per il fit preliminare."""
    omega = omega_from_khz(f_khz)
    x = omega / omega0 - omega0 / omega
    return phi0 + sign * np.arctan(Q * x) + omega * (t_delay_ns * 1.0e-9)



def print_fit_formulas():
    """Stampa a terminale le formule usate per le mappe chi2(L,C)."""
    print('\n================ FORMULE USATE NELLE MAPPE ================')
    print('Definizioni comuni:')
    print('  omega = 2*pi*f')
    print('  omega0(L,C) = 1/sqrt(L*C)')
    print('  Q(L,C,R) = sqrt(L/C)/R')
    print('')
    if FIT_MOD_OFFSET:
        print('Modulo ai capi di R:')
        print('  T_R(f) = offset + A / sqrt(1 + Q(L,C,R)^2 * (omega/omega0(L,C) - omega0(L,C)/omega)^2)')
        print('  Nella mappa chi2(L,C) si profila su R, A e offset.')
    else:
        print('Modulo ai capi di R:')
        print('  T_R(f) = A / sqrt(1 + Q(L,C,R)^2 * (omega/omega0(L,C) - omega0(L,C)/omega)^2)')
        print('  Nella mappa chi2(L,C) si profila su R e A.')
    print('')
    print('Fase ai capi di R:')
    print('  Delta_phi_R(f) = sign * arctan[ Q(L,C,R) * (omega/omega0(L,C) - omega0(L,C)/omega) ]')
    print('  Nella mappa chi2(L,C) si profila solo su R.')
    print('  Nota: phi0 = 0 e t_delay = 0, cioe non vengono fittati offset o delay.')
    print('')
    print('Intervallo di visualizzazione/scansione:')
    print(f'  L centrata su {L_TRUE_H:.4g} H, range = [{L_TRUE_H/LC_RANGE_FACTOR:.4g}, {L_TRUE_H*LC_RANGE_FACTOR:.4g}] H')
    print(f'  C centrata su {C_TRUE_F:.4g} F, range = [{C_TRUE_F/LC_RANGE_FACTOR:.4g}, {C_TRUE_F*LC_RANGE_FACTOR:.4g}] F')
    print(f'  R centrata su {R_TRUE_OHM:.4g} ohm, range circa = [{R_TRUE_OHM/R_SCAN_FACTOR:.4g}, {R_TRUE_OHM*R_SCAN_FACTOR:.4g}] ohm')
    print('  Nessun prior su R, L o C: si restringe solo la regione scansionata/visualizzata.')
    print('===========================================================\n')

# ============================================================
# CARICAMENTO DATI E ERRORI
# ============================================================

def load_phase_data(filename):
    data = np.loadtxt(filename)
    fr_all = data[:, 0]
    dt_ns = data[:, 1]
    scala_dt_ns = data[:, 2]

    phi_all = 2.0 * np.pi * (fr_all * 1.0e3) * (dt_ns * 1.0e-9)

    # Errore del codice fase originale: distribuzione triangolare,
    # differenza di due letture.
    e_dt_ns = np.sqrt(2.0) * (scala_dt_ns / 10.0) * 0.41
    e_phi_all = 2.0 * np.pi * (fr_all * 1.0e3) * (e_dt_ns * 1.0e-9)

    mask = (
        (fr_all >= PHASE_FR_MIN_KHZ) &
        (fr_all <= PHASE_FR_MAX_KHZ) &
        (fr_all > 0.0) &
        np.isfinite(phi_all) &
        np.isfinite(e_phi_all) &
        (e_phi_all > 0.0)
    )
    return fr_all[mask], phi_all[mask], e_phi_all[mask], dt_ns[mask], e_dt_ns[mask]


def load_moduli_data(filename):
    data = np.loadtxt(filename)
    fr_all = data[:, 0]
    vin = data[:, 1]
    vout = data[:, 2]
    vdiv_in = data[:, 3]
    vdiv_out = data[:, 4]

    # Errori del codice moduli originale.
    e_vin = np.sqrt((READING_ERROR_DIV * vdiv_in) ** 2 + (SCALE_ERROR_FRAC * vin) ** 2)
    e_vout = np.sqrt((READING_ERROR_DIV * vdiv_out) ** 2 + (SCALE_ERROR_FRAC * vout) ** 2)

    tr = vout / vin
    e_tr = tr * np.sqrt((e_vout / vout) ** 2 + (e_vin / vin) ** 2)

    mask = (
        (fr_all >= MOD_FR_MIN_KHZ) &
        (fr_all <= MOD_FR_MAX_KHZ) &
        (fr_all > 0.0) &
        np.isfinite(tr) &
        np.isfinite(e_tr) &
        (e_tr > 0.0) &
        (vin != 0.0) &
        (vout != 0.0)
    )
    return fr_all[mask], tr[mask], e_tr[mask]

# ============================================================
# FIT GLOBALI PRELIMINARI
# ============================================================

def estimate_f0_from_peak(fr_khz, y):
    idx = int(np.nanargmax(y))
    return float(fr_khz[idx])


def fit_moduli_global(fr, tr, e_tr):
    f0_init_khz = F0_INIT_KHZ if F0_INIT_KHZ is not None else estimate_f0_from_peak(fr, tr)
    omega0_init = 2.0 * np.pi * f0_init_khz * 1.0e3
    A0 = A_INIT if A_INIT is not None else float(np.nanmax(tr))

    if FIT_MOD_OFFSET:
        p0 = [A0, omega0_init, Q_INIT, MOD_OFFSET_INIT]
        bounds = ([0.0, 1.0, 0.01, -np.inf], [np.inf, np.inf, 500.0, np.inf])
        popt, pcov = curve_fit(
            mod_R_omegaQ_offset, fr, tr, sigma=e_tr, absolute_sigma=True,
            p0=p0, bounds=bounds, method='trf', maxfev=100000,
        )
        model = mod_R_omegaQ_offset(fr, *popt)
        names = ['A', 'omega0', 'Q', 'offset']
    else:
        p0 = [A0, omega0_init, Q_INIT]
        bounds = ([0.0, 1.0, 0.01], [np.inf, np.inf, 500.0])
        popt, pcov = curve_fit(
            mod_R_omegaQ, fr, tr, sigma=e_tr, absolute_sigma=True,
            p0=p0, bounds=bounds, method='trf', maxfev=100000,
        )
        model = mod_R_omegaQ(fr, *popt)
        names = ['A', 'omega0', 'Q']

    perr = np.sqrt(np.diag(pcov))
    chi2 = float(np.sum(((tr - model) / e_tr) ** 2))
    ndf = len(fr) - len(popt)
    return popt, perr, chi2, ndf, names


def fit_phase_global_for_sign(fr, phi, e_phi, sign, omega0_guess, q_guess):
    def model_full(f_khz, omega0, Q, phi0, t_delay_ns):
        return phase_R_omegaQ(f_khz, omega0, Q, phi0, t_delay_ns, sign=sign)

    if FIT_PHASE_OFFSET and FIT_PHASE_DELAY:
        p0 = [omega0_guess, q_guess, PHASE_OFFSET_INIT_RAD, PHASE_DELAY_INIT_NS]
        bounds = ([1.0, 0.01, -4.0*np.pi, -PHASE_DELAY_BOUND_NS],
                  [np.inf, 500.0, 4.0*np.pi, PHASE_DELAY_BOUND_NS])
        popt, pcov = curve_fit(
            model_full, fr, phi, sigma=e_phi, absolute_sigma=True,
            p0=p0, bounds=bounds, method='trf', maxfev=100000,
        )
        model = model_full(fr, *popt)
        names = ['omega0', 'Q', 'phi0', 't_delay_ns']
    elif FIT_PHASE_OFFSET and not FIT_PHASE_DELAY:
        def model_no_delay(f_khz, omega0, Q, phi0):
            return phase_R_omegaQ(f_khz, omega0, Q, phi0, 0.0, sign=sign)
        p0 = [omega0_guess, q_guess, PHASE_OFFSET_INIT_RAD]
        bounds = ([1.0, 0.01, -4.0*np.pi], [np.inf, 500.0, 4.0*np.pi])
        popt, pcov = curve_fit(
            model_no_delay, fr, phi, sigma=e_phi, absolute_sigma=True,
            p0=p0, bounds=bounds, method='trf', maxfev=100000,
        )
        model = model_no_delay(fr, *popt)
        names = ['omega0', 'Q', 'phi0']
    elif (not FIT_PHASE_OFFSET) and FIT_PHASE_DELAY:
        def model_no_offset(f_khz, omega0, Q, t_delay_ns):
            return phase_R_omegaQ(f_khz, omega0, Q, 0.0, t_delay_ns, sign=sign)
        p0 = [omega0_guess, q_guess, PHASE_DELAY_INIT_NS]
        bounds = ([1.0, 0.01, -PHASE_DELAY_BOUND_NS], [np.inf, 500.0, PHASE_DELAY_BOUND_NS])
        popt, pcov = curve_fit(
            model_no_offset, fr, phi, sigma=e_phi, absolute_sigma=True,
            p0=p0, bounds=bounds, method='trf', maxfev=100000,
        )
        model = model_no_offset(fr, *popt)
        names = ['omega0', 'Q', 't_delay_ns']
    else:
        def model_simple(f_khz, omega0, Q):
            return phase_R_omegaQ(f_khz, omega0, Q, 0.0, 0.0, sign=sign)
        p0 = [omega0_guess, q_guess]
        bounds = ([1.0, 0.01], [np.inf, 500.0])
        popt, pcov = curve_fit(
            model_simple, fr, phi, sigma=e_phi, absolute_sigma=True,
            p0=p0, bounds=bounds, method='trf', maxfev=100000,
        )
        model = model_simple(fr, *popt)
        names = ['omega0', 'Q']

    perr = np.sqrt(np.diag(pcov))
    chi2 = float(np.sum(((phi - model) / e_phi) ** 2))
    ndf = len(fr) - len(popt)
    return popt, perr, chi2, ndf, names, model


def fit_phase_global(fr, phi, e_phi, omega0_guess, q_guess):
    signs = [+1.0, -1.0] if PHASE_SIGN == 'auto' else [float(PHASE_SIGN)]
    results = []
    for sgn in signs:
        try:
            res = fit_phase_global_for_sign(fr, phi, e_phi, sgn, omega0_guess, q_guess)
            results.append((sgn, *res))
        except Exception as exc:
            print(f'Fit fase con sign={sgn:+.0f} non riuscito: {exc}')

    if not results:
        raise RuntimeError('Nessun fit di fase riuscito. Controlla stime iniziali e dati.')

    results.sort(key=lambda item: item[3])  # item[3] = chi2
    return results[0]

# ============================================================
# GRIGLIE L-C E R
# ============================================================

def make_axis(center, factor, npts, scale):
    if scale == 'log':
        return np.logspace(np.log10(center / factor), np.log10(center * factor), npts)
    if scale == 'linear':
        return np.linspace(center / factor, center * factor, npts)
    raise ValueError("LC_GRID_SCALE deve essere 'linear' oppure 'log'.")


def make_lc_grid(omega0_best):
    L_center = float(L_REF_H)
    C_center = 1.0 / (omega0_best ** 2 * L_center) if C_REF_F is None else float(C_REF_F)

    L_vals = make_axis(L_center, LC_RANGE_FACTOR, N_LC, LC_GRID_SCALE)
    C_vals = make_axis(C_center, LC_RANGE_FACTOR, N_LC, LC_GRID_SCALE)
    L_grid, C_grid = np.meshgrid(L_vals, C_vals, indexing='xy')
    omega0_grid = 1.0 / np.sqrt(L_grid * C_grid)
    return L_vals, C_vals, L_grid, C_grid, omega0_grid, L_center, C_center


def make_r_grid(q_best, L_center, C_center):
    if R_REF_OHM is None:
        q0 = max(float(q_best), 0.01)
        r0 = np.sqrt(L_center / C_center) / q0
    else:
        r0 = float(R_REF_OHM)

    r0 = np.clip(r0, R_MIN_ABS_OHM, R_MAX_ABS_OHM)
    r_low = max(r0 / R_SCAN_FACTOR, R_MIN_ABS_OHM)
    r_high = min(r0 * R_SCAN_FACTOR, R_MAX_ABS_OHM)
    if r_high <= r_low:
        r_low, r_high = R_MIN_ABS_OHM, R_MAX_ABS_OHM
    return np.logspace(np.log10(r_low), np.log10(r_high), N_R_SCAN)

# ============================================================
# PROFILAZIONE SU R
# ============================================================

def profile_moduli_chi2_LC_over_R(fr, tr, e_tr, L_grid, C_grid, omega0_grid, r_values, use_offset=False):
    """
    Per ogni (L,C) scansiona R. A ogni R corrisponde
        Q(L,C,R) = sqrt(L/C)/R.
    Per ogni R profila analiticamente sui parametri lineari A e, opzionalmente, offset.
    """
    shape = omega0_grid.shape
    best_chi2 = np.full(shape, np.inf)
    best_R = np.full(shape, np.nan)
    best_Q = np.full(shape, np.nan)
    best_A = np.full(shape, np.nan)
    best_offset = np.zeros(shape)

    omega = omega_from_khz(fr)
    y = tr
    sig = e_tr
    wgt = 1.0 / sig ** 2
    y2 = np.sum(wgt * y ** 2)
    W = np.sum(wgt)
    Wy = np.sum(wgt * y)
    sqrt_L_over_C = np.sqrt(L_grid / C_grid)

    for ir, R in enumerate(r_values):
        Q_grid = sqrt_L_over_C / R
        S2 = np.zeros(shape)
        Sy = np.zeros(shape)
        if use_offset:
            S1 = np.zeros(shape)

        for wi, yi, om in zip(wgt, y, omega):
            x = om / omega0_grid - omega0_grid / om
            s = 1.0 / np.sqrt(1.0 + (Q_grid * x) ** 2)
            S2 += wi * s ** 2
            Sy += wi * yi * s
            if use_offset:
                S1 += wi * s

        if use_offset:
            det = S2 * W - S1 ** 2
            good = np.abs(det) > 0.0
            A_hat = np.full(shape, np.nan)
            off_hat = np.full(shape, np.nan)
            A_hat[good] = (Sy[good] * W - Wy * S1[good]) / det[good]
            off_hat[good] = (S2[good] * Wy - S1[good] * Sy[good]) / det[good]
            chi2 = y2 - (A_hat * Sy + off_hat * Wy)
        else:
            good = S2 > 0.0
            A_hat = np.full(shape, np.nan)
            A_hat[good] = Sy[good] / S2[good]
            off_hat = np.zeros(shape)
            chi2 = y2 - Sy ** 2 / S2

        # Evita normalizzazioni negative o NaN.
        chi2 = np.where((A_hat > 0.0) & np.isfinite(chi2), chi2, np.inf)
        update = chi2 < best_chi2
        best_chi2[update] = chi2[update]
        best_R[update] = R
        best_Q[update] = Q_grid[update]
        best_A[update] = A_hat[update]
        best_offset[update] = off_hat[update]

        if (ir + 1) % max(1, N_R_SCAN // 5) == 0:
            print(f'  moduli: profilo R {ir + 1}/{N_R_SCAN}')

    return best_chi2, best_R, best_Q, best_A, best_offset


def profile_phase_chi2_LC_over_R(fr, phi, e_phi, L_grid, C_grid, omega0_grid, r_values,
                                  sign=+1.0, fit_offset=False, fit_delay=False):
    """
    Per ogni (L,C) scansiona R. A ogni R corrisponde
        Q(L,C,R) = sqrt(L/C)/R.
    In questa versione per la fase non si profila su nessun parametro aggiuntivo:
        phi0 = 0 e t_delay = 0.
    """
    shape = omega0_grid.shape
    best_chi2 = np.full(shape, np.inf)
    best_R = np.full(shape, np.nan)
    best_Q = np.full(shape, np.nan)
    best_phi0 = np.zeros(shape)
    best_tdelay_ns = np.zeros(shape)

    omega = omega_from_khz(fr)
    y = phi
    sig = e_phi
    wgt = 1.0 / sig ** 2
    S00 = np.sum(wgt)
    S01 = np.sum(wgt * omega)
    S11 = np.sum(wgt * omega ** 2)
    sqrt_L_over_C = np.sqrt(L_grid / C_grid)

    for ir, R in enumerate(r_values):
        Q_grid = sqrt_L_over_C / R
        b0 = np.zeros(shape)
        b1 = np.zeros(shape)
        r2 = np.zeros(shape)

        for wi, yi, om in zip(wgt, y, omega):
            x = om / omega0_grid - omega0_grid / om
            g = sign * np.arctan(Q_grid * x)
            resid_for_linear = yi - g
            b0 += wi * resid_for_linear
            b1 += wi * om * resid_for_linear
            r2 += wi * resid_for_linear ** 2

        if fit_offset and fit_delay:
            det = S00 * S11 - S01 ** 2
            phi0 = (b0 * S11 - b1 * S01) / det
            tdelay_s = (S00 * b1 - S01 * b0) / det
            chi2 = r2 - (phi0 * b0 + tdelay_s * b1)
            tdelay_ns = tdelay_s * 1.0e9
        elif fit_offset and not fit_delay:
            phi0 = b0 / S00
            tdelay_ns = np.zeros(shape)
            chi2 = r2 - b0 ** 2 / S00
        elif (not fit_offset) and fit_delay:
            phi0 = np.zeros(shape)
            tdelay_s = b1 / S11
            tdelay_ns = tdelay_s * 1.0e9
            chi2 = r2 - b1 ** 2 / S11
        else:
            phi0 = np.zeros(shape)
            tdelay_ns = np.zeros(shape)
            chi2 = r2

        if fit_delay and PHASE_DELAY_BOUND_NS is not None:
            chi2 = np.where(np.abs(tdelay_ns) <= PHASE_DELAY_BOUND_NS, chi2, np.inf)

        chi2 = np.where(np.isfinite(chi2), chi2, np.inf)
        update = chi2 < best_chi2
        best_chi2[update] = chi2[update]
        best_R[update] = R
        best_Q[update] = Q_grid[update]
        best_phi0[update] = phi0[update]
        best_tdelay_ns[update] = tdelay_ns[update]

        if (ir + 1) % max(1, N_R_SCAN // 5) == 0:
            print(f'  fase: profilo R {ir + 1}/{N_R_SCAN}')

    return best_chi2, best_R, best_Q, best_phi0, best_tdelay_ns

# ============================================================
# GRAFICI
# ============================================================

def axes_values_for_plot(L_vals, C_vals):
    if PLOT_UNITS == 'SI':
        return L_vals, C_vals, r'$L$ [H]', r'$C$ [F]'
    if PLOT_UNITS == 'lab':
        return L_vals * 1.0e3, C_vals * 1.0e9, r'$L$ [mH]', r'$C$ [nF]'
    raise ValueError("PLOT_UNITS deve essere 'SI' oppure 'lab'.")


def plot_LC_profile(L_vals, C_vals, chi2_map, best_R_map, best_Q_map, title, outfile,
                    omega0_ref=None, extra_text=None):
    chi2_min = float(np.nanmin(chi2_map))
    idx_min = np.unravel_index(np.nanargmin(chi2_map), chi2_map.shape)
    iC_min, iL_min = idx_min

    L_min = float(L_vals[iL_min])
    C_min = float(C_vals[iC_min])
    R_min = float(best_R_map[idx_min])
    Q_min = float(best_Q_map[idx_min])
    omega0_min = 1.0 / np.sqrt(L_min * C_min)
    f0_min_khz = omega0_min / (2.0 * np.pi * 1.0e3)

    delta = chi2_map - chi2_min
    finite_chi2 = chi2_map[np.isfinite(chi2_map)]
    finite_delta = delta[np.isfinite(delta)]
    if finite_chi2.size == 0:
        raise RuntimeError('La mappa chi2 contiene solo valori non finiti.')

    X, Y, xlabel, ylabel = axes_values_for_plot(L_vals, C_vals)
    cmap = mpl.colormaps[CMAP_NAME]

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 5.8), constrained_layout=True)

    if COLOR_MODE == 'log_chi2':
        positive = finite_chi2[finite_chi2 > 0.0]
        if positive.size == 0:
            raise RuntimeError('La mappa chi2 non contiene valori positivi per LogNorm.')
        vmin = max(float(np.nanmin(positive)), 1.0e-12)
        vmax = float(np.nanpercentile(positive, 99.0))
        if vmax <= vmin:
            vmax = float(np.nanmax(positive))
        im = ax.pcolormesh(X, Y, chi2_map, shading='auto', cmap=cmap,
                           norm=LogNorm(vmin=vmin, vmax=vmax))
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r'$\chi^2$ [-]')
    elif COLOR_MODE == 'delta_chi2':
        if DELTA_CHI2_COLOR_MAX is None:
            zmax = float(np.nanpercentile(finite_delta, 98.0))
            zmax = max(zmax, 10.0)
        else:
            zmax = float(DELTA_CHI2_COLOR_MAX)
        levels = np.linspace(0.0, zmax, 120)
        im = ax.contourf(X, Y, delta, levels=levels, cmap=cmap, extend='max')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r'$\Delta\chi^2 = \chi^2 - \chi^2_{min}$ [-]')
    else:
        raise ValueError("COLOR_MODE deve essere 'log_chi2' oppure 'delta_chi2'.")

    # Contorni in Delta chi2: utili anche se la scala colore e' log(chi2).
    usable_contours = [lv for lv in CONTOUR_LEVELS_DELTA if lv < np.nanmax(finite_delta)]
    if usable_contours:
        cs = ax.contour(X, Y, delta, levels=usable_contours, colors='k',
                        linewidths=0.9, linestyles='dotted')
        ax.clabel(cs, fmt='%.2g', fontsize=FONT_SIZE - 2)

    ax.plot(L_min if PLOT_UNITS == 'SI' else L_min * 1.0e3,
            C_min if PLOT_UNITS == 'SI' else C_min * 1.0e9,
            marker='x', ms=9, mew=2.0, color='black', label='Minimo profilo')

    # Linee verticali/orizzontali al minimo, simili all'esempio.
    x_min_plot = L_min if PLOT_UNITS == 'SI' else L_min * 1.0e3
    y_min_plot = C_min if PLOT_UNITS == 'SI' else C_min * 1.0e9
    ax.axvline(x_min_plot, color='k', lw=1.0, ls='--', alpha=0.75)
    ax.axhline(y_min_plot, color='k', lw=1.0, ls='--', alpha=0.75)

    if SHOW_LC_CONSTANT_LINE:
        omega_line = omega0_min if omega0_ref is None else omega0_ref
        L_line = np.linspace(L_vals.min(), L_vals.max(), 600)
        C_line = 1.0 / (omega_line ** 2 * L_line)
        mask_line = (C_line >= C_vals.min()) & (C_line <= C_vals.max())
        if np.any(mask_line):
            if PLOT_UNITS == 'SI':
                ax.plot(L_line[mask_line], C_line[mask_line], color='white',
                        lw=1.5, ls='-', alpha=0.95, label=r'$LC=1/\omega_0^2$')
            else:
                ax.plot(L_line[mask_line] * 1.0e3, C_line[mask_line] * 1.0e9,
                        color='white', lw=1.5, ls='-', alpha=0.95,
                        label=r'$LC=1/\omega_0^2$')

    if LC_GRID_SCALE == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')

    text = (
        rf'$\chi^2_{{min}}={chi2_min:.3g}$' + '\n' +
        rf'$L_{{min}}={L_min:.4g}\,\mathrm{{H}}$' + '\n' +
        rf'$C_{{min}}={C_min:.4g}\,\mathrm{{F}}$' + '\n' +
        rf'$f_0={f0_min_khz:.4g}\,\mathrm{{kHz}}$' + '\n' +
        rf'$R_{{prof}}={R_min:.4g}\,\Omega$' + '\n' +
        rf'$Q={Q_min:.4g}$'
    )
    if extra_text:
        text += '\n' + extra_text

    ax.text(0.03, 0.97, text, transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.86, edgecolor='0.5'),
            fontsize=FONT_SIZE - 1)

    fig.savefig(outfile, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return {
        'chi2_min': chi2_min,
        'L_min_H': L_min,
        'C_min_F': C_min,
        'R_min_ohm': R_min,
        'Q_min': Q_min,
        'f0_min_kHz': f0_min_khz,
        'idx_min': idx_min,
    }

# ============================================================
# STAMPE
# ============================================================

def print_moduli_fit(popt, perr, chi2, ndf, names):
    print('\n============== FIT GLOBALE MODULI ==============')
    for name, val, err in zip(names, popt, perr):
        if name == 'omega0':
            print(f'omega0 = ({val:.6g} +/- {err:.2g}) rad/s')
            print(f'B_eq   = ({val/1.0e3:.6g} +/- {err/1.0e3:.2g}) 10^3 rad/s')
            print(f'f0     = {val/(2*np.pi*1.0e3):.6g} kHz')
        else:
            print(f'{name:7s}= ({val:.6g} +/- {err:.2g})')
    print(f'chi2 = {chi2:.3f}, chi2/ndf = {chi2/ndf:.3f}, ndf = {ndf}')
    print('================================================')


def print_phase_fit(sign, popt, perr, chi2, ndf, names):
    print('\n============== FIT GLOBALE FASE ==============')
    print(f'segno scelto = {sign:+.0f}')
    for name, val, err in zip(names, popt, perr):
        if name == 'omega0':
            print(f'omega0 = ({val:.6g} +/- {err:.2g}) rad/s')
            print(f'B_eq   = ({val/1.0e3:.6g} +/- {err/1.0e3:.2g}) 10^3 rad/s')
            print(f'f0     = {val/(2*np.pi*1.0e3):.6g} kHz')
        elif name == 't_delay_ns':
            print(f'{name:12s}= ({val:.6g} +/- {err:.2g}) ns')
        else:
            print(f'{name:12s}= ({val:.6g} +/- {err:.2g})')
    print(f'chi2 = {chi2:.3f}, chi2/ndf = {chi2/ndf:.3f}, ndf = {ndf}')
    print('==============================================')

# ============================================================
# MAIN
# ============================================================

def main():
    t_start = time.time()
    print_fit_formulas()

    # ----------------------------
    # Dati moduli e fit globale preliminare
    # ----------------------------
    fr_mod, tr, e_tr = load_moduli_data(MODULI_FILE)
    popt_mod, perr_mod, chi2_mod, ndf_mod, names_mod = fit_moduli_global(fr_mod, tr, e_tr)
    print_moduli_fit(popt_mod, perr_mod, chi2_mod, ndf_mod, names_mod)

    omega0_mod_best = float(popt_mod[names_mod.index('omega0')])
    q_mod_best = float(popt_mod[names_mod.index('Q')])

    # ----------------------------
    # Dati fase e fit globale preliminare
    # ----------------------------
    fr_ph, phi, e_phi, dt_ns, e_dt_ns = load_phase_data(PHASE_FILE)
    sign_phase, popt_phase, perr_phase, chi2_phase, ndf_phase, names_phase, model_phase = fit_phase_global(
        fr_ph, phi, e_phi, omega0_guess=omega0_mod_best, q_guess=q_mod_best
    )
    print_phase_fit(sign_phase, popt_phase, perr_phase, chi2_phase, ndf_phase, names_phase)

    omega0_phase_best = float(popt_phase[names_phase.index('omega0')])
    q_phase_best = float(popt_phase[names_phase.index('Q')])

    # ----------------------------
    # Mappa L-C per i moduli, profilando su R
    # ----------------------------
    print(f'\nCostruzione mappa moduli L-C: {N_LC} x {N_LC}, profiling su {N_R_SCAN} valori di R ...')
    L_vals_m, C_vals_m, L_grid_m, C_grid_m, omega0_grid_m, Lc_m, Cc_m = make_lc_grid(omega0_mod_best)
    r_grid_mod = make_r_grid(q_mod_best, Lc_m, Cc_m)
    print(f'Range R moduli: {r_grid_mod[0]:.4g} ... {r_grid_mod[-1]:.4g} ohm')

    t0 = time.time()
    chi2_map_mod, R_map_mod, Q_map_mod, A_map_mod, off_map_mod = profile_moduli_chi2_LC_over_R(
        fr_mod, tr, e_tr, L_grid_m, C_grid_m, omega0_grid_m, r_grid_mod, use_offset=FIT_MOD_OFFSET
    )
    print(f'Mappa moduli completata in {time.time() - t0:.2f} s')

    idx_mod = np.unravel_index(np.nanargmin(chi2_map_mod), chi2_map_mod.shape)
    extra_mod = rf'$A_{{prof}}={A_map_mod[idx_mod]:.4g}$'
    if FIT_MOD_OFFSET:
        extra_mod += '\n' + rf'$offset_{{prof}}={off_map_mod[idx_mod]:.3g}$'

    res_mod = plot_LC_profile(
        L_vals_m, C_vals_m, chi2_map_mod, R_map_mod, Q_map_mod,
        title=r'$\chi^2$ vs $(L,C)$ R resonance',
        outfile=OUT_MODULI_MAP,
        omega0_ref=omega0_mod_best,
        extra_text=extra_mod,
    )

    print('\nMinimo mappa moduli:')
    print(f"chi2_min = {res_mod['chi2_min']:.4f}")
    print(f"L = {res_mod['L_min_H']:.6g} H, C = {res_mod['C_min_F']:.6g} F")
    print(f"R = {res_mod['R_min_ohm']:.6g} ohm, Q = {res_mod['Q_min']:.6g}")
    print(f"f0 = {res_mod['f0_min_kHz']:.6g} kHz")
    print(f'Grafico salvato in: {OUT_MODULI_MAP}')

    # ----------------------------
    # Mappa L-C per la fase, profilando su R
    # ----------------------------
    print(f'\nCostruzione mappa fase L-C: {N_LC} x {N_LC}, profiling su {N_R_SCAN} valori di R ...')
    L_vals_p, C_vals_p, L_grid_p, C_grid_p, omega0_grid_p, Lc_p, Cc_p = make_lc_grid(omega0_phase_best)
    r_grid_phase = make_r_grid(q_phase_best, Lc_p, Cc_p)
    print(f'Range R fase: {r_grid_phase[0]:.4g} ... {r_grid_phase[-1]:.4g} ohm')

    t0 = time.time()
    chi2_map_phase, R_map_phase, Q_map_phase, phi0_map_phase, td_map_phase = profile_phase_chi2_LC_over_R(
        fr_ph, phi, e_phi, L_grid_p, C_grid_p, omega0_grid_p, r_grid_phase,
        sign=sign_phase, fit_offset=FIT_PHASE_OFFSET, fit_delay=FIT_PHASE_DELAY
    )
    print(f'Mappa fase completata in {time.time() - t0:.2f} s')

    idx_phase = np.unravel_index(np.nanargmin(chi2_map_phase), chi2_map_phase.shape)
    extra_phase = ''
    if FIT_PHASE_OFFSET:
        extra_phase += rf'$\phi_{{0,prof}}={phi0_map_phase[idx_phase]:.3g}\,\mathrm{{rad}}$'
    if FIT_PHASE_DELAY:
        if extra_phase:
            extra_phase += '\n'
        extra_phase += rf'$t_{{delay,prof}}={td_map_phase[idx_phase]:.3g}\,\mathrm{{ns}}$'

    res_phase = plot_LC_profile(
        L_vals_p, C_vals_p, chi2_map_phase, R_map_phase, Q_map_phase,
        title=r'$\chi^2$ vs $(L,C)$ R phase',
        outfile=OUT_PHASE_MAP,
        omega0_ref=omega0_phase_best,
        extra_text=extra_phase,
    )

    print('\nMinimo mappa fase:')
    print(f"chi2_min = {res_phase['chi2_min']:.4f}")
    print(f"L = {res_phase['L_min_H']:.6g} H, C = {res_phase['C_min_F']:.6g} F")
    print(f"R = {res_phase['R_min_ohm']:.6g} ohm, Q = {res_phase['Q_min']:.6g}")
    print(f"f0 = {res_phase['f0_min_kHz']:.6g} kHz")
    print(f'Grafico salvato in: {OUT_PHASE_MAP}')

    print(f'\nTempo totale: {time.time() - t_start:.2f} s')


if __name__ == '__main__':
    main()
