import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from cycler import cycler

try:
    import mplhep as hep
    HAVE_HEP = True
except Exception:
    HAVE_HEP = False

# ============================================================
# PARAMETRI MODIFICABILI DALL'UTENTE
# ============================================================

# File funzione di trasferimento:
# colonne: frequenza(kHz)  Vin(V)  Vout(V)  scala_Vin(V/div)  scala_Vout(V/div)
file_T = 'RLC_Rres.txt'

# File sfasamento:
# colonne: frequenza(kHz)  Delta_t(ns)  scala_tempi(ns/div)
file_phi = 'RLC_phase.txt'

# Range di frequenza usato nei fit e nelle mappe
frfit0_T   = 0.0
frfit1_T   = 1.0e6
frfit0_phi = 0.0
frfit1_phi = 1.0e6

# Guess iniziali fisici
R_init = 20.0       # Ohm
L_init = 0.30e-3    # H
C_init = 5.2e-9     # F

# Il modulo misurato puo avere un massimo diverso da 1.
# Il modello del modulo usa quindi A*T_R ideale.
A_init = 0.43

# Segno della fase.
# Se AUTO_PHASE_SIGN=True, il codice prova sia +1 sia -1 e tiene quello con chi2 minore.
AUTO_PHASE_SIGN = True
PHASE_SIGN = +1.0

# Range manuale per le mappe L,C.
# Consigliato per vedere la degenerazione L*C circa costante.
USE_MANUAL_LC_RANGE = True
L_min_manual = 0.15e-3
L_max_manual = 0.70e-3
C_min_manual = 2.0e-9
C_max_manual = 12.0e-9

# Se USE_MANUAL_LC_RANGE=False, usa best fit +/- n_sigma_map*sigma.
n_sigma_map = 3.0

# Numero di punti per lato della griglia 2D
step_LC = 220

# Scala logaritmica per il colore del chi2
USE_LOG_COLOR = True

# Errori funzione di trasferimento
reading_error_div = 0.1
scale_error_frac  = 0.03

# Errori sfasamento
cursor_factor = 0.41
PHASE_SYS_RAD = 0.0

# Fattori di scala opzionali degli errori
# Non cambiano il best fit, ma solo chi2 assoluto e barre d'errore.
eT_scale   = 1.0
ephi_scale = 1.0

# Grafica
USE_TEX  = False
SAVE_FIG = True
SHOW_FIG = True
OUTFILE  = 'RLC_chi2_LC.png'

# Disegna anche i grafici dati+fit preliminari
PLOT_FIT_CHECKS = True

# ============================================================
# STILE GRAFICI
# ============================================================
if HAVE_HEP:
    plt.style.use(hep.style.ROOT)

params = {
    'legend.fontsize': 9,
    'legend.loc': 'best',
    'legend.frameon': True,
    'legend.framealpha': 0.85,
    'legend.facecolor': 'w',
    'legend.edgecolor': '0.85',
    'figure.figsize': (11, 4.8),
    'axes.labelsize': 11,
    'figure.titlesize': 14,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.linewidth': 1.2,
    'text.usetex': USE_TEX,
    'axes.formatter.use_mathtext': True,
    'figure.constrained_layout.use': True,
}
plt.rcParams.update(params)
plt.rcParams['axes.prop_cycle'] = cycler(
    color=['tab:blue', 'tab:red', 'tab:green', 'tab:purple'])

# ============================================================
# MODELLI FISICI RLC SERIE, USCITA AI CAPI DI R
# ============================================================
def omega_from_khz(f_khz):
    return 2.0 * np.pi * np.asarray(f_khz, dtype=float) * 1e3


def transfer_R_RLC_unit(f_khz, R, L, C):
    """
    Modulo ideale della funzione di trasferimento ai capi di R:
        T_R = R / sqrt(R^2 + (omega L - 1/(omega C))^2)
    Picco massimo = 1.
    """
    omega = omega_from_khz(f_khz)
    X = omega * L - 1.0 / (omega * C)
    return R / np.sqrt(R**2 + X**2)


def transfer_R_RLC_scaled(f_khz, A, R, L, C):
    """
    Modulo con fattore di scala A:
        T_R = A * R/sqrt(R^2 + X^2)
    """
    return A * transfer_R_RLC_unit(f_khz, R, L, C)


def phase_R_RLC_signed(f_khz, R, L, C, sign):
    """
    Sfasamento ai capi di R, in radianti.
    sign serve per adattarsi alla convenzione sperimentale di Delta_t.
    """
    omega = omega_from_khz(f_khz)
    X = omega * L - 1.0 / (omega * C)
    return sign * np.arctan(X / R)


def chi2_val(y, yerr, ymodel):
    return float(np.sum(((y - ymodel) / yerr) ** 2))

# ============================================================
# FIT IN PARAMETRI LOGARITMICI
# ============================================================
# Fit diretto in R,L,C spesso non converge perché le scale numeriche sono molto diverse.
# Fittare log-parametri mantiene positivi R,L,C,A e rende il problema più stabile.
def transfer_log_model(f_khz, logA, logR, logL, logC):
    A, R, L, C = np.exp(logA), np.exp(logR), np.exp(logL), np.exp(logC)
    return transfer_R_RLC_scaled(f_khz, A, R, L, C)


def phase_log_model_factory(sign):
    def phase_log_model(f_khz, logR, logL, logC):
        R, L, C = np.exp(logR), np.exp(logL), np.exp(logC)
        return phase_R_RLC_signed(f_khz, R, L, C, sign)
    return phase_log_model


def unpack_T_log(p):
    return np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), np.exp(p[3])


def unpack_phi_log(p):
    return np.exp(p[0]), np.exp(p[1]), np.exp(p[2])


def log_cov_to_linear_errors(values, pcov_log):
    sig_log = np.sqrt(np.maximum(np.diag(pcov_log), 0.0))
    return np.asarray(values, dtype=float) * sig_log


def make_log_guesses_T():
    guesses = []
    for Afac in [0.6, 0.8, 1.0, 1.2, 1.5]:
        for Rfac in [0.4, 0.7, 1.0, 1.5, 2.5]:
            for Lfac in [0.6, 0.85, 1.0, 1.2, 1.6]:
                for Cfac in [0.6, 0.85, 1.0, 1.2, 1.6]:
                    guesses.append(np.log([
                        max(A_init * Afac, 1e-12),
                        max(R_init * Rfac, 1e-12),
                        max(L_init * Lfac, 1e-15),
                        max(C_init * Cfac, 1e-18),
                    ]))
    return guesses


def make_log_guesses_phi():
    guesses = []
    for Rfac in [0.4, 0.7, 1.0, 1.5, 2.5]:
        for Lfac in [0.6, 0.85, 1.0, 1.2, 1.6]:
            for Cfac in [0.6, 0.85, 1.0, 1.2, 1.6]:
                guesses.append(np.log([
                    max(R_init * Rfac, 1e-12),
                    max(L_init * Lfac, 1e-15),
                    max(C_init * Cfac, 1e-18),
                ]))
    return guesses


def robust_curve_fit(model, x, y, yerr, guesses, max_nfev=300000):
    best = None
    best_chi2 = np.inf
    last_exc = None

    for p0 in guesses:
        try:
            popt, pcov = curve_fit(
                model, x, y,
                p0=p0,
                sigma=yerr,
                absolute_sigma=True,
                method='trf',
                max_nfev=max_nfev,
            )
            chisq = chi2_val(y, yerr, model(x, *popt))
            if np.isfinite(chisq) and chisq < best_chi2:
                best = (popt, pcov)
                best_chi2 = chisq
        except Exception as exc:
            last_exc = exc

    if best is None:
        raise RuntimeError(f'Fit non riuscito con nessuna guess. Ultimo errore: {last_exc}')
    return best[0], best[1], best_chi2

# ============================================================
# CARICAMENTO E PREPARAZIONE DATI: MODULO T
# ============================================================
data_T = np.loadtxt(file_T)
if data_T.ndim == 1:
    data_T = data_T.reshape(1, -1)
if data_T.shape[1] < 5:
    raise RuntimeError('file_T deve avere 5 colonne: f, Vin, Vout, scala_Vin, scala_Vout.')

fr_T_all  = data_T[:, 0]
Vin       = data_T[:, 1]
Vout      = data_T[:, 2]
Vdiv_in   = data_T[:, 3]
Vdiv_out  = data_T[:, 4]

# Errore su Vin e Vout: lettura + calibrazione.
# Se vuoi usare un errore di lettura di 0.04 div invece di 0.1 div, cambia reading_error_div.
eVin  = np.sqrt((reading_error_div * Vdiv_in) ** 2 + (scale_error_frac * Vin) ** 2)
eVout = np.sqrt((reading_error_div * Vdiv_out) ** 2 + (scale_error_frac * Vout) ** 2)

T_all  = Vout / Vin
eT_all = T_all * np.sqrt((eVout / Vout) ** 2 + (eVin / Vin) ** 2)
eT_all *= eT_scale

mask_T = (
    (fr_T_all >= frfit0_T) & (fr_T_all <= frfit1_T)
    & np.isfinite(fr_T_all) & np.isfinite(T_all) & np.isfinite(eT_all)
    & (eT_all > 0) & (fr_T_all > 0) & (Vin > 0) & (Vout > 0)
)
fr_T = fr_T_all[mask_T]
T    = T_all[mask_T]
eT   = eT_all[mask_T]

# ============================================================
# CARICAMENTO E PREPARAZIONE DATI: FASE
# ============================================================
data_phi = np.loadtxt(file_phi)
if data_phi.ndim == 1:
    data_phi = data_phi.reshape(1, -1)
if data_phi.shape[1] < 3:
    raise RuntimeError('file_phi deve avere 3 colonne: f, Delta_t, scala_tempi.')

fr_phi_all = data_phi[:, 0]
Dt_ns      = data_phi[:, 1]
scala_t_ns = data_phi[:, 2]

phi_all = 2.0 * np.pi * (fr_phi_all * 1e3) * (Dt_ns * 1e-9)

sigma_t_ns = np.sqrt(2.0) * (scala_t_ns / 10.0) * cursor_factor
ephi_all = 2.0 * np.pi * (fr_phi_all * 1e3) * (sigma_t_ns * 1e-9)

if PHASE_SYS_RAD > 0:
    ephi_all = np.sqrt(ephi_all**2 + PHASE_SYS_RAD**2)
ephi_all *= ephi_scale

positive = ephi_all[np.isfinite(ephi_all) & (ephi_all > 0)]
if len(positive) == 0:
    raise RuntimeError('Tutte le incertezze sulla fase sono nulle o non finite.')
ephi_all = np.where((ephi_all <= 0) | ~np.isfinite(ephi_all),
                    np.nanmedian(positive), ephi_all)

mask_phi = (
    (fr_phi_all >= frfit0_phi) & (fr_phi_all <= frfit1_phi)
    & np.isfinite(fr_phi_all) & np.isfinite(phi_all) & np.isfinite(ephi_all)
    & (ephi_all > 0) & (fr_phi_all > 0)
)
fr_phi = fr_phi_all[mask_phi]
phi    = phi_all[mask_phi]
ephi   = ephi_all[mask_phi]

if len(fr_T) < 5:
    raise RuntimeError('Troppi pochi punti per il fit del modulo.')
if len(fr_phi) < 4:
    raise RuntimeError('Troppi pochi punti per il fit della fase.')

# ============================================================
# FIT ROBUSTO MODULO
# ============================================================
print('=' * 70)
print('FIT MODULO T_R con parametri [A, R, L, C]')
print('=' * 70)

popt_T_log, pcov_T_log, chi2_T_fit = robust_curve_fit(
    transfer_log_model, fr_T, T, eT, make_log_guesses_T()
)
A_T, R_T, L_T, C_T = unpack_T_log(popt_T_log)
eA_T, eR_T, eL_T, eC_T = log_cov_to_linear_errors([A_T, R_T, L_T, C_T], pcov_T_log)
ndof_T = len(fr_T) - 4

print(f'  A = {A_T:.6g} +/- {eA_T:.2g}')
print(f'  R = {R_T:.6g} +/- {eR_T:.2g} Ohm')
print(f'  L = {L_T:.6e} +/- {eL_T:.2e} H')
print(f'  C = {C_T:.6e} +/- {eC_T:.2e} F')
print(f'  chi2 = {chi2_T_fit:.3f}, ndof = {ndof_T}, chi2/ndof = {chi2_T_fit/ndof_T:.3f}')
if chi2_T_fit / ndof_T > 5:
    print(f'  ATTENZIONE: chi2/ndof alto. Per le sole barre, prova eT_scale = {np.sqrt(chi2_T_fit/ndof_T):.1f}')
print(f'  f0 = {1/(2*np.pi*np.sqrt(L_T*C_T))*1e-3:.4f} kHz')

# ============================================================
# FIT ROBUSTO FASE, PROVANDO ENTRAMBI I SEGNI SE RICHIESTO
# ============================================================
print()
print('=' * 70)
print('FIT FASE phi_R con parametri [R, L, C]')
print('=' * 70)

signs_to_try = [+1.0, -1.0] if AUTO_PHASE_SIGN else [PHASE_SIGN]
best_phase = None
for sign in signs_to_try:
    model_phi = phase_log_model_factory(sign)
    popt_log, pcov_log, chisq = robust_curve_fit(
        model_phi, fr_phi, phi, ephi, make_log_guesses_phi()
    )
    if best_phase is None or chisq < best_phase['chi2']:
        best_phase = {
            'sign': sign,
            'model': model_phi,
            'popt_log': popt_log,
            'pcov_log': pcov_log,
            'chi2': chisq,
        }

PHASE_SIGN_USED = best_phase['sign']
popt_phi_log = best_phase['popt_log']
pcov_phi_log = best_phase['pcov_log']
chi2_phi_fit = best_phase['chi2']

R_phi, L_phi, C_phi = unpack_phi_log(popt_phi_log)
eR_phi, eL_phi, eC_phi = log_cov_to_linear_errors([R_phi, L_phi, C_phi], pcov_phi_log)
ndof_phi = len(fr_phi) - 3

print(f'  segno fase usato = {PHASE_SIGN_USED:+.0f}')
print(f'  R = {R_phi:.6g} +/- {eR_phi:.2g} Ohm')
print(f'  L = {L_phi:.6e} +/- {eL_phi:.2e} H')
print(f'  C = {C_phi:.6e} +/- {eC_phi:.2e} F')
print(f'  chi2 = {chi2_phi_fit:.3f}, ndof = {ndof_phi}, chi2/ndof = {chi2_phi_fit/ndof_phi:.3f}')
if chi2_phi_fit / ndof_phi > 5:
    print(f'  ATTENZIONE: chi2/ndof alto. Per le sole barre, prova ephi_scale = {np.sqrt(chi2_phi_fit/ndof_phi):.1f}')
print(f'  f0 = {1/(2*np.pi*np.sqrt(L_phi*C_phi))*1e-3:.4f} kHz')

# ============================================================
# GRAFICI DI CONTROLLO DATI + FIT
# ============================================================
if PLOT_FIT_CHECKS:
    f_plot = np.linspace(fr_T.min(), fr_T.max(), 1400)
    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True,
                             height_ratios=[3, 1], constrained_layout=True)
    axes[0].errorbar(fr_T, T, yerr=eT, fmt='o', ms=2.5, color='blue', label='Dati')
    axes[0].plot(f_plot, transfer_R_RLC_scaled(f_plot, A_T, R_T, L_T, C_T), 'r-', lw=1.5,
                 label=rf'Fit: $A={A_T:.3f}$, $R={R_T:.2f}\,\Omega$')
    axes[0].set_ylabel(r'$T_R = V_{out}/V_{in}$')
    axes[0].legend(fontsize=8)
    axes[0].set_title(r'Funzione di trasferimento -- fit robusto')

    res_T = T - transfer_R_RLC_scaled(fr_T, A_T, R_T, L_T, C_T)
    axes[1].errorbar(fr_T, res_T, yerr=eT, fmt='o', ms=2.5, color='blue')
    axes[1].axhline(0, color='k', lw=0.8)
    axes[1].set_ylabel('Residui')
    axes[1].set_xlabel('Frequenza (kHz)')
    axes[1].text(0.02, 0.92, rf'$\chi^2$/ndof = {chi2_T_fit/ndof_T:.2f}',
                 transform=axes[1].transAxes, fontsize=9, va='top')
    if SAVE_FIG:
        plt.savefig('RLC_fit_T.png', bbox_inches='tight', dpi=160, facecolor='w')
    if SHOW_FIG:
        plt.show()
    else:
        plt.close()

    f_plot_phi = np.linspace(fr_phi.min(), fr_phi.max(), 1400)
    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True,
                             height_ratios=[3, 1], constrained_layout=True)
    axes[0].errorbar(fr_phi, phi, yerr=ephi, fmt='o', ms=2.5, color='red', label='Dati')
    axes[0].plot(f_plot_phi, phase_R_RLC_signed(f_plot_phi, R_phi, L_phi, C_phi, PHASE_SIGN_USED),
                 'b-', lw=1.5, label=rf'Fit: sign={PHASE_SIGN_USED:+.0f}, $R={R_phi:.2f}\,\Omega$')
    axes[0].axhline(0, color='gray', ls='dotted', lw=0.8)
    axes[0].set_ylabel(r'$\phi_R$ (rad)')
    axes[0].legend(fontsize=8)
    axes[0].set_title(r'Sfasamento -- fit robusto')

    res_phi = phi - phase_R_RLC_signed(fr_phi, R_phi, L_phi, C_phi, PHASE_SIGN_USED)
    axes[1].errorbar(fr_phi, res_phi, yerr=ephi, fmt='o', ms=2.5, color='red')
    axes[1].axhline(0, color='k', lw=0.8)
    axes[1].set_ylabel('Residui (rad)')
    axes[1].set_xlabel('Frequenza (kHz)')
    axes[1].text(0.02, 0.92, rf'$\chi^2$/ndof = {chi2_phi_fit/ndof_phi:.2f}',
                 transform=axes[1].transAxes, fontsize=9, va='top')
    if SAVE_FIG:
        plt.savefig('RLC_fit_phi.png', bbox_inches='tight', dpi=160, facecolor='w')
    if SHOW_FIG:
        plt.show()
    else:
        plt.close()

# ============================================================
# GRIGLIE L,C
# ============================================================
def make_grid(center, sigma, lo_manual, hi_manual, n_step):
    if USE_MANUAL_LC_RANGE:
        return np.linspace(lo_manual, hi_manual, n_step)
    if (not np.isfinite(sigma)) or sigma <= 0:
        sigma = 0.15 * center
    lo = max(center - n_sigma_map * sigma, center * 0.01, np.finfo(float).eps)
    hi = center + n_sigma_map * sigma
    if hi <= lo:
        hi = lo * 1.01
    return np.linspace(lo, hi, n_step)

L_grid_T   = make_grid(L_T,   eL_T,   L_min_manual, L_max_manual, step_LC)
C_grid_T   = make_grid(C_T,   eC_T,   C_min_manual, C_max_manual, step_LC)
L_grid_phi = make_grid(L_phi, eL_phi, L_min_manual, L_max_manual, step_LC)
C_grid_phi = make_grid(C_phi, eC_phi, C_min_manual, C_max_manual, step_LC)

# ============================================================
# MAPPE CHI2(L,C)
# ============================================================
print()
print('Calcolo mappe chi2(L,C)...')

def build_chi2_map_T(fr, y, yerr, A_fixed, R_fixed, L_grid, C_grid):
    LL, CC = np.meshgrid(L_grid, C_grid, indexing='ij')
    omega = omega_from_khz(fr)

    LL3 = LL[:, :, np.newaxis]
    CC3 = CC[:, :, np.newaxis]
    omega3 = omega[np.newaxis, np.newaxis, :]

    X3 = omega3 * LL3 - 1.0 / (omega3 * CC3)
    ymod = A_fixed * R_fixed / np.sqrt(R_fixed**2 + X3**2)
    residuals = (y[np.newaxis, np.newaxis, :] - ymod) / yerr[np.newaxis, np.newaxis, :]
    return np.sum(residuals**2, axis=2)


def build_chi2_map_phi(fr, y, yerr, R_fixed, L_grid, C_grid, sign):
    LL, CC = np.meshgrid(L_grid, C_grid, indexing='ij')
    omega = omega_from_khz(fr)

    LL3 = LL[:, :, np.newaxis]
    CC3 = CC[:, :, np.newaxis]
    omega3 = omega[np.newaxis, np.newaxis, :]

    X3 = omega3 * LL3 - 1.0 / (omega3 * CC3)
    ymod = sign * np.arctan(X3 / R_fixed)
    residuals = (y[np.newaxis, np.newaxis, :] - ymod) / yerr[np.newaxis, np.newaxis, :]
    return np.sum(residuals**2, axis=2)

chi2_T_map   = build_chi2_map_T(fr_T,   T,   eT,   A_T, R_T,   L_grid_T,   C_grid_T)
chi2_phi_map = build_chi2_map_phi(fr_phi, phi, ephi, R_phi, L_grid_phi, C_grid_phi, PHASE_SIGN_USED)

idx_T   = np.unravel_index(np.argmin(chi2_T_map),   chi2_T_map.shape)
idx_phi = np.unravel_index(np.argmin(chi2_phi_map), chi2_phi_map.shape)

L_T_map   = L_grid_T[idx_T[0]]
C_T_map   = C_grid_T[idx_T[1]]
L_phi_map = L_grid_phi[idx_phi[0]]
C_phi_map = C_grid_phi[idx_phi[1]]

chi2_T_min   = float(chi2_T_map[idx_T])
chi2_phi_min = float(chi2_phi_map[idx_phi])

print(f'Mappa modulo -- minimo: L={L_T_map:.4e} H, C={C_T_map:.4e} F, chi2={chi2_T_min:.2f}')
print(f'Mappa fase   -- minimo: L={L_phi_map:.4e} H, C={C_phi_map:.4e} F, chi2={chi2_phi_min:.2f}')

# ============================================================
# GRAFICO: DUE MAPPE AFFIANCATE
# ============================================================
cmap = mpl.colormaps['jet']

maps = [
    {
        'ax_idx': 0,
        'data':     chi2_T_map,
        'L_grid':   L_grid_T,
        'C_grid':   C_grid_T,
        'title':    r'$\chi^2(L,C)$ -- modulo $T_R$',
        'R':        R_T,
        'extra':    rf'$A$ fissato = {A_T:.3g}',
        'L_fit':    L_T,   'C_fit':   C_T,
        'L_map':    L_T_map, 'C_map': C_T_map,
        'chi2min':  chi2_T_min,
    },
    {
        'ax_idx': 1,
        'data':     chi2_phi_map,
        'L_grid':   L_grid_phi,
        'C_grid':   C_grid_phi,
        'title':    r'$\chi^2(L,C)$ -- sfasamento $\phi_R$',
        'R':        R_phi,
        'extra':    rf'segno fase = {PHASE_SIGN_USED:+.0f}',
        'L_fit':    L_phi,  'C_fit':  C_phi,
        'L_map':    L_phi_map, 'C_map': C_phi_map,
        'chi2min':  chi2_phi_min,
    },
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2),
                         constrained_layout=True, sharey=False)

for item in maps:
    ax  = axes[item['ax_idx']]
    z   = item['data']
    Lg  = item['L_grid']
    Cg  = item['C_grid']

    zmin = max(float(np.nanmin(z)), 1e-12)
    zmax = float(np.nanmax(z))
    z_plot = np.asarray(z.T, dtype=float)
    z_plot = np.where(z_plot > 0, z_plot, np.nan)

    norm = LogNorm(vmin=zmin, vmax=zmax) if USE_LOG_COLOR else mpl.colors.Normalize(vmin=zmin, vmax=zmax)
    im = ax.pcolormesh(Lg, Cg, z_plot, shading='auto', cmap=cmap, norm=norm)

    chi2min = item['chi2min']
    contour_levels = [chi2min + dv for dv in [1.0, 2.30, 5.99, 9.21, 25.0, 100.0]]
    valid_levels = [lv for lv in contour_levels if zmin < lv < zmax]
    if valid_levels:
        cs = ax.contour(Lg, Cg, z.T,
                        levels=valid_levels,
                        colors='k', linewidths=0.9, alpha=0.65)
        ax.clabel(cs, inline=True, fontsize=8, fmt=lambda v, m=chi2min: f'+{v - m:.2g}')

    ax.axvline(item['L_fit'], color='black', lw=1.2, ls='-', label='Best fit')
    ax.axhline(item['C_fit'], color='black', lw=1.2, ls='-')
    ax.axvline(item['L_map'], color='black', lw=1.0, ls='--', label='Min mappa')
    ax.axhline(item['C_map'], color='black', lw=1.0, ls='--')
    ax.plot(item['L_map'], item['C_map'], marker='o', ms=5,
            color='white', mec='black', mew=1.0, zorder=5)

    ax.set_xlim(Lg[0], Lg[-1])
    ax.set_ylim(Cg[0], Cg[-1])
    ax.set_title(item['title'])
    ax.set_xlabel(r'$L$ (H)')
    ax.set_ylabel(r'$C$ (F)')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-9, 9))
    ax.legend(fontsize=8, loc='upper right')

    txt = (
        rf"$R$ fissato = {item['R']:.3g} $\Omega$" + '\n'
        + item['extra'] + '\n'
        + rf"$\chi^2_{{min}}$ = {chi2min:.2f}" + '\n'
        + rf"$L_{{BF}}$ = {item['L_fit']*1e3:.4f} mH" + '\n'
        + rf"$C_{{BF}}$ = {item['C_fit']*1e9:.4f} nF"
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            va='top', ha='left', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.4', alpha=0.85))

    cbar = fig.colorbar(im, ax=ax, pad=0.015, shrink=0.93)
    cbar.set_label(r'$\chi^2$')

if SAVE_FIG:
    plt.savefig(OUTFILE, bbox_inches='tight', dpi=160, facecolor='w')
if SHOW_FIG:
    plt.show()
else:
    plt.close()
