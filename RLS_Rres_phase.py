import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
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

# --- File di input ---
# colonne: frequenza (kHz), delta_t sfasamento (ns), scala_tempi (ns)
file = 'RLC_phase.txt'

# --- Modalita di fit ---
# 'R' = fase sulla resistenza, 'C' = fase sul condensatore, 'L' = fase sull'induttore
fit_mode = 'R'

# --- Guess iniziali ---
R_init = 10.0
L_init = 290e-6
C_init = 5.2e-9

# --- Range di frequenza per il fit (kHz) ---
frfit0 = 40.0
frfit1 = 150000.0

# --- Prior esterno sul parametro fissato ---
C_prior = 5.2e-9
C_prior_err = 0.02e-9

# --- Scansione delle mappe chi2 ---
n_sigma_scan = 2.0
step_scan = 5000

# --- Grafica ---
USE_TEX = True
DEB = True

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
# UTILITY NUMERICHE
# ============================================================
def chi2_value(y, yerr, ymodel):
    return np.sum(((y - ymodel) / yerr) ** 2)


class FitFailure(RuntimeError):
    pass


def robust_curve_fit(model, x, y, yerr, p0, bounds):
    guesses = [
        np.array(p0, dtype=float),
        np.array(p0, dtype=float) * np.array([0.7, 0.7]),
        np.array(p0, dtype=float) * np.array([1.3, 1.3]),
        np.array(p0, dtype=float) * np.array([0.7, 1.3]),
        np.array(p0, dtype=float) * np.array([1.3, 0.7]),
    ]

    best = None
    best_chi2 = np.inf

    for guess in guesses:
        try:
            popt, pcov = curve_fit(
                model,
                x,
                y,
                p0=guess,
                sigma=yerr,
                absolute_sigma=True,
                method='trf',
                bounds=bounds,
                max_nfev=200000,
            )
            chisq = chi2_value(y, yerr, model(x, *popt))
            if np.isfinite(chisq) and chisq < best_chi2:
                best = (popt, pcov)
                best_chi2 = chisq
        except Exception:
            pass

    if best is None:
        raise FitFailure('Il fit non e riuscito con nessuna guess iniziale.')

    return best[0], best[1], best_chi2



def make_scan_grid(best, sigma, positive=True):
    sigma_eff = sigma
    if not np.isfinite(sigma_eff) or sigma_eff <= 0:
        sigma_eff = 0.10 * abs(best)
    sigma_eff = max(sigma_eff, 0.03 * abs(best))
    lo = best - n_sigma_scan * sigma_eff
    hi = best + n_sigma_scan * sigma_eff
    if positive:
        lo = max(lo, np.finfo(float).eps)
        hi = max(hi, lo * 1.01)
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
            x1, x2 = grid[i - 1], grid[i]
            left = x1 - y1 * (x2 - x1) / (y2 - y1)
            break

    for i in range(idx_min, len(grid) - 1):
        y1 = profile[i] - level
        y2 = profile[i + 1] - level
        if y2 == 0:
            right = grid[i + 1]
            break
        if y1 * y2 <= 0:
            x1, x2 = grid[i], grid[i + 1]
            right = x1 - y1 * (x2 - x1) / (y2 - y1)
            break

    return left, right



def numerical_gradient(func, theta, rel_step=1e-6):
    theta = np.asarray(theta, dtype=float)
    grad = np.zeros_like(theta, dtype=float)
    for i in range(len(theta)):
        h = rel_step * abs(theta[i])
        if h == 0:
            h = rel_step
        tp = theta.copy()
        tm = theta.copy()
        tp[i] += h
        tm[i] -= h
        if tm[i] <= 0:
            tm[i] = max(theta[i] - 0.5 * h, np.finfo(float).eps)
            h = tp[i] - tm[i]
        grad[i] = (func(tp) - func(tm)) / h
    return grad



def covariance_to_correlation(cov):
    d = np.sqrt(np.diag(cov))
    out = np.zeros_like(cov)
    for i in range(len(d)):
        for j in range(len(d)):
            if d[i] > 0 and d[j] > 0:
                out[i, j] = cov[i, j] / (d[i] * d[j])
    return out



def derived_quantities(theta):
    R, L, C = theta
    omega0 = 1.0 / np.sqrt(L * C)
    f0_hz = omega0 / (2.0 * np.pi)
    f0_khz = f0_hz / 1e3
    delta = R / (2.0 * L)
    tau = 1.0 / delta
    Q = omega0 * L / R
    return {
        'R_ohm': R,
        'L_H': L,
        'C_F': C,
        'omega0_rad_s': omega0,
        'f0_hz': f0_hz,
        'f0_khz': f0_khz,
        'delta_s_inv': delta,
        'tau_s': tau,
        'Q': Q,
    }



def derived_errors(theta, cov):
    keys = ['R_ohm', 'L_H', 'C_F', 'omega0_rad_s', 'f0_hz', 'f0_khz', 'delta_s_inv', 'tau_s', 'Q']
    errs = {}
    for key in keys:
        grad = numerical_gradient(lambda t: derived_quantities(t)[key], theta)
        var = float(grad @ cov @ grad)
        errs[key] = np.sqrt(max(var, 0.0))
    return errs



def fmt_value(name, val, err):
    if name == 'C_F':
        return f'C      = ({val * 1e9:.5f} +/- {err * 1e9:.5f}) nF'
    if name == 'L_H':
        return f'L      = ({val * 1e6:.5f} +/- {err * 1e6:.5f}) uH'
    if name == 'R_ohm':
        return f'R      = ({val:.5f} +/- {err:.5f}) Ohm'
    if name == 'omega0_rad_s':
        return f'omega0 = ({val:.6e} +/- {err:.2e}) rad/s'
    if name == 'f0_hz':
        return f'f0     = ({val:.5f} +/- {err:.5f}) Hz'
    if name == 'f0_khz':
        return f'f0     = ({val:.5f} +/- {err:.5f}) kHz'
    if name == 'delta_s_inv':
        return f'delta  = ({val:.6e} +/- {err:.2e}) s^-1'
    if name == 'tau_s':
        return f'tau    = ({val:.6e} +/- {err:.2e}) s'
    if name == 'Q':
        return f'Q      = ({val:.6f} +/- {err:.6f})'
    return f'{name} = {val} +/- {err}'



def set_compact_axis(ax, which='both'):
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    if which in ('x', 'both'):
        ax.xaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
    if which in ('y', 'both'):
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))



def format_param_value(value, kind):
    if kind == 'R':
        return f'{value:.5f} Ohm'
    if kind == 'L':
        return f'{value * 1e6:.5f} uH'
    if kind == 'C':
        return f'{value * 1e9:.5f} nF'
    return f'{value:.5e}'


# ============================================================
# COSTRUZIONE MODELLI CON PARAMETRO FISSATO
# ============================================================
def prepare_fixed_analysis(fixed_name, fixed_value):
    fixed_name = fixed_name.upper()

    if fixed_name == 'C':
        def model(x, p1, p2):
            return phase_model(x, p1, p2, fixed_value)
        p0 = [R_init, L_init]
        bounds = ([np.finfo(float).eps, np.finfo(float).eps], [np.inf, np.inf])
        free_names = ('R', 'L')
        free_units = ('Ohm', 'H')
        axis_labels = (r'R (Ohm)', r'L (H)')

        def pack_theta(free_params):
            return np.array([free_params[0], free_params[1], fixed_value], dtype=float)

    else:
        raise ValueError("In questa versione e supportato solo fixed_name = 'C'.")

    return model, p0, bounds, free_names, free_units, axis_labels, pack_theta



def refit_with_shifted_fixed(fr, PH, ePH, fixed_name, fixed_value_shifted, p0):
    model, _, bounds, _, _, _, _ = prepare_fixed_analysis(fixed_name, fixed_value_shifted)
    popt, _, _ = robust_curve_fit(model, fr, PH, ePH, p0=p0, bounds=bounds)
    return popt



def build_full_covariance(fixed_name, fixed_sigma, pcov_free, dpfree_dfixed):
    fixed_name = fixed_name.upper()
    s2 = fixed_sigma ** 2

    cov_free_total = pcov_free + np.outer(dpfree_dfixed, dpfree_dfixed) * s2
    cross = dpfree_dfixed * s2

    if fixed_name == 'C':
        cov = np.zeros((3, 3), dtype=float)
        cov[:2, :2] = cov_free_total
        cov[:2, 2] = cross
        cov[2, :2] = cross
        cov[2, 2] = s2
    else:
        raise ValueError("In questa versione e supportato solo fixed_name = 'C'.")

    return cov



def run_fixed_analysis(fr, PH, ePH, fixed_name, fixed_value, fixed_sigma):
    model, p0, bounds, free_names, free_units, axis_labels, pack_theta = prepare_fixed_analysis(fixed_name, fixed_value)

    popt_free, pcov_free, chisq = robust_curve_fit(model, fr, PH, ePH, p0=p0, bounds=bounds)
    yfit = model(fr, *popt_free)
    resid = PH - yfit
    ndof = len(fr) - len(popt_free)

    dpfree_dfixed = np.zeros(len(popt_free), dtype=float)
    if fixed_sigma is not None and fixed_sigma > 0:
        p_plus = refit_with_shifted_fixed(fr, PH, ePH, fixed_name, fixed_value + fixed_sigma, p0=popt_free)
        p_minus = refit_with_shifted_fixed(fr, PH, ePH, fixed_name, max(fixed_value - fixed_sigma, np.finfo(float).eps), p0=popt_free)
        denom = (fixed_value + fixed_sigma) - max(fixed_value - fixed_sigma, np.finfo(float).eps)
        dpfree_dfixed = (p_plus - p_minus) / denom

    cov_theta = build_full_covariance(fixed_name, fixed_sigma, pcov_free, dpfree_dfixed)
    theta_best = pack_theta(popt_free)
    errs = derived_errors(theta_best, cov_theta)
    vals = derived_quantities(theta_best)
    corr = covariance_to_correlation(cov_theta)

    sig_free_total = np.sqrt(np.diag(cov_theta)[[0, 1]])

    grid_x = make_scan_grid(popt_free[0], sig_free_total[0], positive=True)
    grid_y = make_scan_grid(popt_free[1], sig_free_total[1], positive=True)

    chi2_map = np.empty((len(grid_x), len(grid_y)), dtype=float)
    for i, xv in enumerate(grid_x):
        for j, yv in enumerate(grid_y):
            chi2_map[i, j] = chi2_value(PH, ePH, model(fr, xv, yv))

    chi2_min = float(np.min(chi2_map))
    dchi2_map = chi2_map - chi2_min
    prof_x = dchi2_map.min(axis=1)
    prof_y = dchi2_map.min(axis=0)
    x_left, x_right = find_profile_crossings(grid_x, prof_x, level=1.0)
    y_left, y_right = find_profile_crossings(grid_y, prof_y, level=1.0)

    return {
        'fixed_name': fixed_name.upper(),
        'fixed_value': fixed_value,
        'fixed_sigma': fixed_sigma,
        'free_names': free_names,
        'free_units': free_units,
        'axis_labels': axis_labels,
        'model': model,
        'popt_free': popt_free,
        'pcov_free': pcov_free,
        'theta_best': theta_best,
        'cov_theta': cov_theta,
        'corr_theta': corr,
        'values': vals,
        'errors': errs,
        'chi2': chisq,
        'chi2_min': chi2_min,
        'ndof': ndof,
        'chi2_red': chisq / ndof,
        'resid': resid,
        'yfit': yfit,
        'grid_x': grid_x,
        'grid_y': grid_y,
        'chi2_map': chi2_map,
        'dchi2_map': dchi2_map,
        'profile_x': prof_x,
        'profile_y': prof_y,
        'profile_1sigma': {'x': (x_left, x_right), 'y': (y_left, y_right)},
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



def plot_fit_and_residuals(fr, PH, ePH, result, title):
    xfit = np.linspace(np.min(fr), np.max(fr), 1200)
    yfit_line = result['model'](xfit, *result['popt_free'])

    fig, ax = plt.subplots(2, 1, figsize=(6.8, 5.2), sharex=True,
                           constrained_layout=True, height_ratios=[2.2, 1.0])

    ax[0].plot(xfit, yfit_line, color='black', ls='--', label='Best fit')
    ax[0].errorbar(fr, PH, yerr=ePH, fmt='o', ms=3, color='tab:red', label='Dati')
    ax[0].set_ylabel('Sfasamento (gradi)')
    ax[0].set_title(title)
    ax[0].legend(loc='best')

    ax[1].axhline(0.0, color='black', lw=1.0)
    ax[1].errorbar(fr, result['resid'], yerr=ePH, fmt='o', ms=3, color='tab:red')
    ax[1].set_xlabel('Frequenza (kHz)')
    ax[1].set_ylabel('Residui')
    plt.show()



def plot_chi2_panel(result, title):
    x = result['grid_x']
    y = result['grid_y']
    chi2_map = result['chi2_map']
    prof_x = result['profile_x'] + result['chi2_min']
    prof_y = result['profile_y'] + result['chi2_min']
    best_x = result['popt_free'][0]
    best_y = result['popt_free'][1]
    chi2_min = result['chi2_min']
    x_left, x_right = result['profile_1sigma']['x']
    y_left, y_right = result['profile_1sigma']['y']
    xlabel, ylabel = result['axis_labels']

    chi2_line_1sigma = chi2_min + 1.0
    contour_levels = [chi2_min + 1.0, chi2_min + 2.30, chi2_min + 5.99, chi2_min + 9.21]

    vmin = max(chi2_min, 1e-12)
    vmax = max(float(np.max(chi2_map)), chi2_min + 10.0)
    levels_fill = np.geomspace(vmin, vmax, 80)

    fig = plt.figure(figsize=(9.0, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 4.0], height_ratios=[4.0, 1.5],
                          wspace=0.05, hspace=0.05)

    ax_left = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[0, 1], sharey=ax_left)
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_main)
    ax_empty = fig.add_subplot(gs[1, 0])
    ax_empty.axis('off')

    im = ax_main.contourf(
        x, y, chi2_map.T,
        levels=levels_fill,
        cmap=mpl.colormaps['turbo'],
        norm=LogNorm(vmin=vmin, vmax=vmax),
        extend='max',
    )
    ax_main.contour(x, y, chi2_map.T, levels=contour_levels, colors='k', linewidths=1.0)

    # solo il punto del minimo nella mappa centrale
    ax_main.plot(best_x, best_y, marker='o', ms=5, color='white', mec='black', mew=0.9, zorder=5)

    # etichetta del minimo, spostata rispetto al punto
    dx = 0.05 * (x.max() - x.min())
    dy = 0.06 * (y.max() - y.min())
    x_text = np.clip(best_x + dx, x.min() + 0.01 * (x.max() - x.min()), x.max() - 0.28 * (x.max() - x.min()))
    y_text = np.clip(best_y + dy, y.min() + 0.08 * (y.max() - y.min()), y.max() - 0.08 * (y.max() - y.min()))
    ax_main.text(
        x_text,
        y_text,
        f'$\\chi^2_{{min}}$ = {chi2_min:.2f}',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.3', alpha=0.9),
        zorder=6,
    )

    # riquadro riassuntivo fuori dalle curve principali
    err_x_minus = best_x - x_left if np.isfinite(x_left) else np.nan
    err_x_plus = x_right - best_x if np.isfinite(x_right) else np.nan
    err_y_minus = best_y - y_left if np.isfinite(y_left) else np.nan
    err_y_plus = y_right - best_y if np.isfinite(y_right) else np.nan

    info_lines = [
        f'$\\chi^2_{{min}}$ = {chi2_min:.2f}',
        f'$\\chi^2_{{min}}+1$ = {chi2_line_1sigma:.2f}',
        f'R* = {format_param_value(best_x, "R")}',
        f'R- = {format_param_value(x_left, "R")}' if np.isfinite(x_left) else 'R- = n.d.',
        f'R+ = {format_param_value(x_right, "R")}' if np.isfinite(x_right) else 'R+ = n.d.',
        f'$\\sigma_R^-$ = {format_param_value(err_x_minus, "R")}' if np.isfinite(err_x_minus) else '$\\sigma_R^-$ = n.d.',
        f'$\\sigma_R^+$ = {format_param_value(err_x_plus, "R")}' if np.isfinite(err_x_plus) else '$\\sigma_R^+$ = n.d.',
        f'L* = {format_param_value(best_y, "L")}',
        f'L- = {format_param_value(y_left, "L")}' if np.isfinite(y_left) else 'L- = n.d.',
        f'L+ = {format_param_value(y_right, "L")}' if np.isfinite(y_right) else 'L+ = n.d.',
        f'$\\sigma_L^-$ = {format_param_value(err_y_minus, "L")}' if np.isfinite(err_y_minus) else '$\\sigma_L^-$ = n.d.',
        f'$\\sigma_L^+$ = {format_param_value(err_y_plus, "L")}' if np.isfinite(err_y_plus) else '$\\sigma_L^+$ = n.d.',
        'contorni: $\\chi^2_{min}+1$, +2.30, +5.99, +9.21',
    ]
    ax_main.text(
        0.02,
        0.98,
        '\n'.join(info_lines),
        transform=ax_main.transAxes,
        va='top',
        ha='left',
        fontsize=8.6,
        bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.35', alpha=0.93),
        zorder=6,
    )

    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    ax_main.set_title(title)
    set_compact_axis(ax_main, 'both')

    # profilo in R (sotto): linee solo fuori dalla mappa
    ax_bottom.plot(x, prof_x, color='tab:blue')
    ax_bottom.axhline(chi2_min, color='k', lw=1.0)
    ax_bottom.axhline(chi2_line_1sigma, color='0.35', lw=1.0)
    ax_bottom.axvline(best_x, color='k', lw=1.1)
    if np.isfinite(x_left):
        ax_bottom.axvline(x_left, color='0.35', lw=1.0)
    if np.isfinite(x_right):
        ax_bottom.axvline(x_right, color='0.35', lw=1.0)
    ax_bottom.set_xlabel(xlabel)
    ax_bottom.set_ylabel(r'$\chi^2$')
    ymax_bottom = max(chi2_line_1sigma + 2.0, float(np.nanmax(prof_x[np.isfinite(prof_x)])) * 1.03)
    ax_bottom.set_ylim(chi2_min - 0.05, ymax_bottom)
    set_compact_axis(ax_bottom, 'x')

    # profilo in L (sinistra): linee solo fuori dalla mappa
    ax_left.plot(prof_y, y, color='tab:blue')
    ax_left.axvline(chi2_min, color='k', lw=1.0)
    ax_left.axvline(chi2_line_1sigma, color='0.35', lw=1.0)
    ax_left.axhline(best_y, color='k', lw=1.1)
    if np.isfinite(y_left):
        ax_left.axhline(y_left, color='0.35', lw=1.0)
    if np.isfinite(y_right):
        ax_left.axhline(y_right, color='0.35', lw=1.0)
    ax_left.set_xlabel(r'$\chi^2$')
    ax_left.set_ylabel(ylabel)
    xmax_left = max(chi2_line_1sigma + 2.0, float(np.nanmax(prof_y[np.isfinite(prof_y)])) * 1.03)
    ax_left.set_xlim(xmax_left, chi2_min - 0.05)
    set_compact_axis(ax_left, 'y')

    cbar = fig.colorbar(im, ax=ax_main, pad=0.02)
    cbar.set_label(r'$\chi^2$')

    plt.setp(ax_main.get_yticklabels(), visible=False)
    plt.show()


# ============================================================
# STAMPA RISULTATI
# ============================================================
def print_result(result):
    fixed_name = result['fixed_name']
    fixed_value = result['fixed_value']
    fixed_sigma = result['fixed_sigma']
    vals = result['values']
    errs = result['errors']

    print('===========================================================')
    print(f'FIT CON C FISSATO: C = ({fixed_value * 1e9:.5f} +/- {fixed_sigma * 1e9:.5f}) nF')
    print('-----------------------------------------------------------')
    print(fmt_value('R_ohm', vals['R_ohm'], errs['R_ohm']))
    print(fmt_value('L_H', vals['L_H'], errs['L_H']))
    print(fmt_value('C_F', vals['C_F'], errs['C_F']))
    print(fmt_value('omega0_rad_s', vals['omega0_rad_s'], errs['omega0_rad_s']))
    print(fmt_value('f0_khz', vals['f0_khz'], errs['f0_khz']))
    print(fmt_value('delta_s_inv', vals['delta_s_inv'], errs['delta_s_inv']))
    print(fmt_value('tau_s', vals['tau_s'], errs['tau_s']))
    print(fmt_value('Q', vals['Q'], errs['Q']))
    print(f"chi2 = {result['chi2']:.3f}")
    print(f"ndof = {result['ndof']}")
    print(f"chi2 ridotto = {result['chi2_red']:.3f}")
    print('-----------------------------------------------------------')
    print('Matrice di correlazione su [R, L, C]:')
    print(np.array2string(result['corr_theta'], formatter={'float_kind': lambda x: f'{x: .4f}'}))
    print('===========================================================')


# ============================================================
# MAIN
# ============================================================
def main():
    data = np.loadtxt(file)
    fr_all = data[:, 0]
    delta_t_ns = data[:, 1]
    scala_tempi_ns = data[:, 2]

    delta_t = delta_t_ns * 1e-9
    scala_tempi = scala_tempi_ns * 1e-9

    PH_all = np.degrees(2.0 * np.pi * fr_all * 1e3 * delta_t)
    sigma_t = np.sqrt(2.0) * (scala_tempi / 10.0) * 0.41
    ePH_all = np.degrees(2.0 * np.pi * fr_all * 1e3 * sigma_t)

    if DEB:
        print('Prime frequenze (kHz):', fr_all[:5])
        print('Prime fasi (deg):', PH_all[:5])
        print('Prime sigma_fase (deg):', ePH_all[:5])

    plot_data_full(fr_all, PH_all, ePH_all)

    mask = (fr_all >= frfit0) & (fr_all <= frfit1)
    fr = fr_all[mask]
    PH = PH_all[mask]
    ePH = ePH_all[mask]

    if len(fr) < 4:
        raise RuntimeError('Troppi pochi punti nel range di fit.')

    result_C = run_fixed_analysis(fr, PH, ePH, fixed_name='C', fixed_value=C_prior, fixed_sigma=C_prior_err)
    print_result(result_C)
    plot_fit_and_residuals(fr, PH, ePH, result_C, title='Fit della fase con C fissato')
    plot_chi2_panel(result_C, title=r'Mappa di $\chi^2$ nel piano (R, L) con C fissato')


if __name__ == '__main__':
    main()
