import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import least_squares
from cycler import cycler

try:
    import mplhep as hep
    HAVE_HEP = True
except Exception:
    HAVE_HEP = False

# ============================================================
# PARAMETRI MODIFICABILI DALL'UTENTE
# ============================================================

# File input: colonne = frequenza (kHz), delta_t sfasamento (ns), scala_tempi (ns)
file = 'RLC_phase.txt'

# Modalita di fit: 'R' fase sul resistore, 'C' fase sul condensatore, 'L' fase sull'induttore
fit_mode = 'R'

# Guess iniziali
R_init = 10.0       # Ohm
L_init = 290e-6     # H
C_init = 5.2e-9     # F

# Intervallo di frequenza per il fit (kHz)
frfit0 = 40.0
frfit1 = 150000.0

# Priori gaussiani esterni
R_prior = 10.08
R_prior_err = 0.06
C_prior = 5.2e-9
C_prior_err = 0.02e-9

# Scansione mappe chi2 attorno al minimo globale
n_sigma_scan = 4.0
step_scan = 60

# Opzioni grafiche
USE_TEX = False
DEB = False

# ============================================================
# STILE GRAFICI
# ============================================================
if HAVE_HEP:
    plt.style.use(hep.style.ROOT)

params = {
    'legend.fontsize': 10,
    'legend.loc': 'upper right',
    'legend.frameon': True,
    'legend.framealpha': 0.85,
    'legend.facecolor': 'w',
    'legend.edgecolor': '0.85',
    'figure.figsize': (6.2, 4.2),
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
plt.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:green', 'tab:red', 'tab:purple'])


# ============================================================
# MODELLI DI FASE
# ============================================================
def fitf_R(x_khz, R, L, C):
    omega = 2.0 * np.pi * np.asarray(x_khz) * 1e3
    return -np.degrees(np.arctan((1.0 / (omega * C) - omega * L) / R))


def fitf_C(x_khz, R, L, C):
    omega = 2.0 * np.pi * np.asarray(x_khz) * 1e3
    return -np.degrees(np.arctan2(omega**2 * L * C - 1.0, omega * R * C))


def fitf_L(x_khz, R, L, C):
    omega = 2.0 * np.pi * np.asarray(x_khz) * 1e3
    return -np.degrees(np.arctan2(omega**2 * L * C - 1.0, -omega * R * C))


if fit_mode == 'R':
    phase_model = fitf_R
elif fit_mode == 'C':
    phase_model = fitf_C
else:
    phase_model = fitf_L


# ============================================================
# FUNZIONI DI SERVIZIO
# ============================================================
def set_compact_axis(ax, which='both'):
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    if which in ('x', 'both'):
        ax.xaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
    if which in ('y', 'both'):
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))


def chi2_phase(fr, PH, ePH, theta):
    model = phase_model(fr, *theta)
    return np.sum(((PH - model) / ePH) ** 2)



def chi2_prior(theta):
    R, _, C = theta
    return ((R - R_prior) / R_prior_err) ** 2 + ((C - C_prior) / C_prior_err) ** 2



def chi2_total(fr, PH, ePH, theta):
    return chi2_phase(fr, PH, ePH, theta) + chi2_prior(theta)



def residuals_with_priors(theta, fr, PH, ePH):
    R, L, C = theta
    res_phase = (PH - phase_model(fr, R, L, C)) / ePH
    res_prior_R = np.array([(R - R_prior) / R_prior_err])
    res_prior_C = np.array([(C - C_prior) / C_prior_err])
    return np.concatenate([res_phase, res_prior_R, res_prior_C])



def robust_global_fit(fr, PH, ePH, p0):
    lower = np.array([1e-12, 1e-12, 1e-15], dtype=float)
    upper = np.array([np.inf, np.inf, np.inf], dtype=float)

    guesses = [
        np.array(p0, dtype=float),
        np.array(p0, dtype=float) * np.array([0.8, 0.8, 0.8]),
        np.array(p0, dtype=float) * np.array([1.2, 1.2, 1.2]),
        np.array(p0, dtype=float) * np.array([0.8, 1.2, 1.0]),
        np.array(p0, dtype=float) * np.array([1.2, 0.8, 1.0]),
        np.array([R_prior, L_init, C_prior], dtype=float),
    ]

    best = None
    best_cost = np.inf

    for guess in guesses:
        guess = np.maximum(guess, lower * 10.0)
        try:
            result = least_squares(
                residuals_with_priors,
                x0=guess,
                bounds=(lower, upper),
                args=(fr, PH, ePH),
                method='trf',
                x_scale='jac',
                loss='linear',
                max_nfev=200000,
            )
            if result.success and np.isfinite(result.cost) and result.cost < best_cost:
                best = result
                best_cost = result.cost
        except Exception:
            pass

    if best is None:
        raise RuntimeError('Il fit globale con priori non e riuscito.')

    return best



def covariance_from_jacobian(result):
    jac = result.jac
    jtj = jac.T @ jac
    try:
        cov = np.linalg.inv(jtj)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(jtj)
    return cov



def covariance_to_correlation(cov):
    d = np.sqrt(np.diag(cov))
    corr = np.zeros_like(cov)
    for i in range(len(d)):
        for j in range(len(d)):
            if d[i] > 0 and d[j] > 0:
                corr[i, j] = cov[i, j] / (d[i] * d[j])
    return corr



def numerical_gradient(func, theta, rel_step=1e-6):
    theta = np.asarray(theta, dtype=float)
    grad = np.zeros_like(theta, dtype=float)
    for i in range(len(theta)):
        h = rel_step * max(abs(theta[i]), 1.0)
        tp = theta.copy()
        tm = theta.copy()
        tp[i] += h
        tm[i] -= h
        if tm[i] <= 0:
            tm[i] = max(theta[i] - 0.5 * h, np.finfo(float).eps)
        grad[i] = (func(tp) - func(tm)) / (tp[i] - tm[i])
    return grad



def derived_quantities(theta):
    R, L, C = theta
    omega0 = 1.0 / np.sqrt(L * C)
    f0_hz = omega0 / (2.0 * np.pi)
    delta = R / (2.0 * L)
    tau = 1.0 / delta
    Q = omega0 * L / R
    return {
        'R_ohm': R,
        'L_H': L,
        'C_F': C,
        'LC_HF': L * C,
        'omega0_rad_s': omega0,
        'f0_hz': f0_hz,
        'f0_khz': f0_hz / 1e3,
        'delta_s_inv': delta,
        'tau_s': tau,
        'Q': Q,
    }



def derived_errors(theta, cov):
    values = derived_quantities(theta)
    errs = {}
    for key in values:
        grad = numerical_gradient(lambda t: derived_quantities(t)[key], theta)
        var = float(grad @ cov @ grad)
        errs[key] = np.sqrt(max(var, 0.0))
    return errs



def fmt_value(name, val, err):
    if name == 'R_ohm':
        return f'R       = ({val:.5f} +/- {err:.5f}) Ohm'
    if name == 'L_H':
        return f'L       = ({val * 1e6:.5f} +/- {err * 1e6:.5f}) uH'
    if name == 'C_F':
        return f'C       = ({val * 1e9:.5f} +/- {err * 1e9:.5f}) nF'
    if name == 'LC_HF':
        return f'L*C     = ({val:.6e} +/- {err:.2e}) H F'
    if name == 'omega0_rad_s':
        return f'omega0  = ({val:.6e} +/- {err:.2e}) rad/s'
    if name == 'f0_hz':
        return f'f0      = ({val:.6f} +/- {err:.6f}) Hz'
    if name == 'f0_khz':
        return f'f0      = ({val:.6f} +/- {err:.6f}) kHz'
    if name == 'delta_s_inv':
        return f'delta   = ({val:.6e} +/- {err:.2e}) s^-1'
    if name == 'tau_s':
        return f'tau     = ({val:.6e} +/- {err:.2e}) s'
    if name == 'Q':
        return f'Q       = ({val:.6f} +/- {err:.6f})'
    return f'{name} = {val} +/- {err}'



def make_scan_grid(center, sigma, positive=True):
    sigma_eff = sigma
    if not np.isfinite(sigma_eff) or sigma_eff <= 0:
        sigma_eff = 0.05 * abs(center)
    sigma_eff = max(sigma_eff, 0.03 * abs(center), 1e-18)
    lo = center - n_sigma_scan * sigma_eff
    hi = center + n_sigma_scan * sigma_eff
    if positive:
        lo = max(lo, np.finfo(float).eps)
        hi = max(hi, lo * 1.02)
    return np.linspace(lo, hi, step_scan)



def find_profile_crossings(grid, profile, level=1.0):
    idx_min = int(np.argmin(profile))
    left = np.nan
    right = np.nan

    for i in range(idx_min, 0, -1):
        y1 = profile[i - 1] - level
        y2 = profile[i] - level
        if y1 == 0:
            left = grid[i - 1]
            break
        if y1 * y2 <= 0:
            left = grid[i - 1] - y1 * (grid[i] - grid[i - 1]) / (y2 - y1)
            break

    for i in range(idx_min, len(grid) - 1):
        y1 = profile[i] - level
        y2 = profile[i + 1] - level
        if y2 == 0:
            right = grid[i + 1]
            break
        if y1 * y2 <= 0:
            right = grid[i] - y1 * (grid[i + 1] - grid[i]) / (y2 - y1)
            break

    return left, right


# ============================================================
# SCANSIONE 3D DEL CHI2 TOTALE CON PROFILAZIONE
# ============================================================


def format_profile_errors(center, interval, unit):
    left, right = interval
    if not np.isfinite(left) or not np.isfinite(right):
        return f'intervallo 1 sigma non chiuso nella finestra di scansione ({unit})'
    return f'-{center - left:.5e}  +{right - center:.5e} {unit}'

def chi2_volume_profiled(fr, PH, ePH, theta_best, cov):
    sigR, sigL, sigC = np.sqrt(np.diag(cov))
    R_grid = make_scan_grid(theta_best[0], sigR, positive=True)
    L_grid = make_scan_grid(theta_best[1], sigL, positive=True)
    C_grid = make_scan_grid(theta_best[2], sigC, positive=True)

    omega = 2.0 * np.pi * fr * 1e3
    y = PH[:, None]
    yerr = ePH[:, None]

    volume = np.empty((len(R_grid), len(L_grid), len(C_grid)), dtype=float)

    for i, R in enumerate(R_grid):
        prior_R = ((R - R_prior) / R_prior_err) ** 2
        for j, L in enumerate(L_grid):
            Cvec = C_grid[None, :]
            omg = omega[:, None]

            if fit_mode == 'R':
                model = -np.degrees(np.arctan((1.0 / (omg * Cvec) - omg * L) / R))
            elif fit_mode == 'C':
                model = -np.degrees(np.arctan2(omg**2 * L * Cvec - 1.0, omg * R * Cvec))
            else:
                model = -np.degrees(np.arctan2(omg**2 * L * Cvec - 1.0, -omg * R * Cvec))

            chi2_data = np.sum(((y - model) / yerr) ** 2, axis=0)
            chi2_prior_C = ((C_grid - C_prior) / C_prior_err) ** 2
            volume[i, j, :] = chi2_data + prior_R + chi2_prior_C

    chi2_min = np.min(volume)
    idx_min = np.unravel_index(np.argmin(volume), volume.shape)

    map_RL = volume.min(axis=2) - chi2_min
    map_RC = volume.min(axis=1) - chi2_min
    map_LC = volume.min(axis=0) - chi2_min

    prof_R = volume.min(axis=2).min(axis=1) - chi2_min
    prof_L = volume.min(axis=2).min(axis=0) - chi2_min
    prof_C = volume.min(axis=1).min(axis=0) - chi2_min

    return {
        'R_grid': R_grid,
        'L_grid': L_grid,
        'C_grid': C_grid,
        'chi2_min': chi2_min,
        'idx_min': idx_min,
        'map_RL': map_RL,
        'map_RC': map_RC,
        'map_LC': map_LC,
        'prof_R': prof_R,
        'prof_L': prof_L,
        'prof_C': prof_C,
    }


# ============================================================
# GRAFICI
# ============================================================
def plot_data_full(fr_all, PH_all, ePH_all):
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.errorbar(fr_all, PH_all, yerr=ePH_all, fmt='o', ms=3, color='tab:blue', label=r'$\phi(f)$')
    ax.set_xlabel('Frequenza (kHz)')
    ax.set_ylabel('Sfasamento (gradi)')
    ax.set_title('Dati completi')
    ax.legend(loc='best')
    plt.show()



def plot_fit_and_residuals(fr, PH, ePH, theta_best):
    xfit = np.linspace(np.min(fr), np.max(fr), 1500)
    yfit = phase_model(xfit, *theta_best)
    resid = PH - phase_model(fr, *theta_best)

    fig, ax = plt.subplots(2, 1, figsize=(6.8, 5.2), sharex=True,
                           constrained_layout=True, height_ratios=[2.2, 1.0])
    ax[0].plot(xfit, yfit, color='black', ls='--', label='Best fit globale con priori')
    ax[0].errorbar(fr, PH, yerr=ePH, fmt='o', ms=3, color='tab:red', label='Dati')
    ax[0].set_ylabel('Sfasamento (gradi)')
    ax[0].set_title('Fit globale con priori gaussiani su R e C')
    ax[0].legend(loc='best')

    ax[1].axhline(0.0, color='black', lw=1.0)
    ax[1].errorbar(fr, resid, yerr=ePH, fmt='o', ms=3, color='tab:red')
    ax[1].set_xlabel('Frequenza (kHz)')
    ax[1].set_ylabel('Residui')
    plt.show()



def plot_profile_map(x, y, dchi2_map, prof_x, prof_y, best_x, best_y,
                     x_cross, y_cross, xlabel, ylabel, title):
    eps = 1e-3
    zplot = dchi2_map + eps
    vmax = max(float(np.max(zplot)), 10.0)
    fill_levels = np.geomspace(eps, vmax, 90)
    contour_levels = [1.0, 2.30, 5.99, 9.21]

    fig = plt.figure(figsize=(8.3, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.30, 4.00], height_ratios=[4.00, 1.35],
                          wspace=0.04, hspace=0.05)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[0, 1], sharey=ax_left)
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_main)
    ax_empty = fig.add_subplot(gs[1, 0])
    ax_empty.axis('off')

    im = ax_main.contourf(
        x, y, zplot.T,
        levels=fill_levels,
        cmap=mpl.colormaps['turbo'],
        norm=LogNorm(vmin=eps, vmax=vmax),
        extend='max',
    )
    cs = ax_main.contour(x, y, dchi2_map.T, levels=contour_levels, colors='k', linewidths=1.0)
    ax_main.clabel(cs, inline=True, fontsize=8, fmt=lambda v: f'{v:.2g}')

    ax_main.axvline(best_x, color='k', lw=1.2)
    ax_main.axhline(best_y, color='k', lw=1.2)

    if np.isfinite(x_cross[0]):
        ax_main.axvline(x_cross[0], color='0.30', ls='--', lw=1.0)
    if np.isfinite(x_cross[1]):
        ax_main.axvline(x_cross[1], color='0.30', ls='--', lw=1.0)
    if np.isfinite(y_cross[0]):
        ax_main.axhline(y_cross[0], color='0.30', ls='--', lw=1.0)
    if np.isfinite(y_cross[1]):
        ax_main.axhline(y_cross[1], color='0.30', ls='--', lw=1.0)

    ax_main.plot(best_x, best_y, 'wo', ms=4, mec='k', mew=0.8)
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    ax_main.set_title(title)
    set_compact_axis(ax_main, 'both')

    ax_left.plot(prof_y + eps, y, color='tab:blue')
    ax_left.set_xscale('log')
    ax_left.invert_xaxis()
    ax_left.axhline(best_y, color='k', lw=1.1)
    if np.isfinite(y_cross[0]):
        ax_left.axhline(y_cross[0], color='0.30', ls='--', lw=1.0)
    if np.isfinite(y_cross[1]):
        ax_left.axhline(y_cross[1], color='0.30', ls='--', lw=1.0)
    ax_left.set_xlabel(r'$\Delta\chi^2$')
    ax_left.set_ylabel(ylabel)
    set_compact_axis(ax_left, 'y')

    ax_bottom.plot(x, prof_x + eps, color='tab:blue')
    ax_bottom.set_yscale('log')
    ax_bottom.axvline(best_x, color='k', lw=1.1)
    if np.isfinite(x_cross[0]):
        ax_bottom.axvline(x_cross[0], color='0.30', ls='--', lw=1.0)
    if np.isfinite(x_cross[1]):
        ax_bottom.axvline(x_cross[1], color='0.30', ls='--', lw=1.0)
    ax_bottom.set_xlabel(xlabel)
    ax_bottom.set_ylabel(r'$\Delta\chi^2$')
    set_compact_axis(ax_bottom, 'x')

    cbar = fig.colorbar(im, ax=ax_main, pad=0.02)
    cbar.set_label(r'$\Delta\chi^2$', rotation=270, labelpad=12)
    plt.show()


# ============================================================
# PROGRAMMA PRINCIPALE
# ============================================================
data = np.loadtxt(file)
fr_all = data[:, 0]
delta_t_ns = data[:, 1]
scala_tempi_ns = data[:, 2]

delta_t = delta_t_ns * 1e-9
scala_tempi = scala_tempi_ns * 1e-9
PH_all = np.degrees(2.0 * np.pi * fr_all * 1e3 * delta_t)
sigma_t = np.sqrt(2.0) * (scala_tempi / 10.0) * 0.41
ePH_all = np.degrees(2.0 * np.pi * fr_all * 1e3 * sigma_t)

mask_fit = (fr_all >= frfit0) & (fr_all <= frfit1)
fr = fr_all[mask_fit]
PH = PH_all[mask_fit]
ePH = ePH_all[mask_fit]

if DEB:
    print('Prime frequenze:', fr[:5])
    print('Prime fasi:', PH[:5])
    print('Primi errori:', ePH[:5])

plot_data_full(fr_all, PH_all, ePH_all)

# Fit globale con priori gaussiani
result = robust_global_fit(fr, PH, ePH, p0=[R_init, L_init, C_init])
theta_best = result.x
cov_theta = covariance_from_jacobian(result)
corr_theta = covariance_to_correlation(cov_theta)
errs = derived_errors(theta_best, cov_theta)
vals = derived_quantities(theta_best)

chi2_data_best = chi2_phase(fr, PH, ePH, theta_best)
chi2_prior_best = chi2_prior(theta_best)
chi2_total_best = chi2_data_best + chi2_prior_best
ndof_data = len(fr) - 3
ndof_total = len(fr) + 2 - 3

print('\n================ FIT GLOBALE CON PRIORI =================')
print(f'chi2 fase      = {chi2_data_best:.4f}')
print(f'chi2 prior     = {chi2_prior_best:.4f}')
print(f'chi2 totale    = {chi2_total_best:.4f}')
print(f'chi2 rid fase  = {chi2_data_best / ndof_data:.4f}   con ndof = {ndof_data}')
print(f'chi2 rid tot   = {chi2_total_best / ndof_total:.4f}   con ndof = {ndof_total}')
print(f'pull R prior   = {(theta_best[0] - R_prior) / R_prior_err:.4f}')
print(f'pull C prior   = {(theta_best[2] - C_prior) / C_prior_err:.4f}')
print('---------------------------------------------------------')
for key in ['R_ohm', 'L_H', 'C_F', 'LC_HF', 'omega0_rad_s', 'f0_khz', 'delta_s_inv', 'tau_s', 'Q']:
    print(fmt_value(key, vals[key], errs[key]))
print('---------------------------------------------------------')
print('Matrice di correlazione (R, L, C):')
print(corr_theta)
print('=========================================================\n')

plot_fit_and_residuals(fr, PH, ePH, theta_best)

# Mappe profilo del chi2 totale
scan = chi2_volume_profiled(fr, PH, ePH, theta_best, cov_theta)
R_grid = scan['R_grid']
L_grid = scan['L_grid']
C_grid = scan['C_grid']

R_cross = find_profile_crossings(R_grid, scan['prof_R'], level=1.0)
L_cross = find_profile_crossings(L_grid, scan['prof_L'], level=1.0)
C_cross = find_profile_crossings(C_grid, scan['prof_C'], level=1.0)

print('============== ERRORI DA PROFILO DEL CHI2 ==============')
print(f'R profilo   : {format_profile_errors(theta_best[0], R_cross, "Ohm")}')
print(f'L profilo   : {format_profile_errors(theta_best[1], L_cross, "H")}')
print(f'C profilo   : {format_profile_errors(theta_best[2], C_cross, "F")}')
print('========================================================\n')

plot_profile_map(
    R_grid, L_grid, scan['map_RL'], scan['prof_R'], scan['prof_L'],
    theta_best[0], theta_best[1], R_cross, L_cross,
    r'$R$ (Ohm)', r'$L$ (H)', r'$\Delta\chi^2(R, L)$ profilando su $C$'
)

plot_profile_map(
    R_grid, C_grid, scan['map_RC'], scan['prof_R'], scan['prof_C'],
    theta_best[0], theta_best[2], R_cross, C_cross,
    r'$R$ (Ohm)', r'$C$ (F)', r'$\Delta\chi^2(R, C)$ profilando su $L$'
)

plot_profile_map(
    L_grid, C_grid, scan['map_LC'], scan['prof_L'], scan['prof_C'],
    theta_best[1], theta_best[2], L_cross, C_cross,
    r'$L$ (H)', r'$C$ (F)', r'$\Delta\chi^2(L, C)$ profilando su $R$'
)
