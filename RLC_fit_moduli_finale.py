import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler
import multiprocessing.pool

# ============================================================
# PARAMETRI MODIFICABILI DALL'UTENTE
# ============================================================

# --- File di input ---
# Il file deve contenere 5 colonne separate da spazio:
# frequenza (kHz)   Vin (V)   Vout (V)   V/div canale Vin   V/div canale Vout
file = 'RLC_Rres_nuovo.txt'

# --- Modalita di fit ---
fit_mode = 'R'  # 'R' per trasferimento su resistore, 'C' su condensatore

# Le stime iniziali per il fit
if fit_mode == 'R':
    A_init = 0.43
    B_init = 2.0 * np.pi * 130.0e3
    C_init = 10.5
else:
    A_init = 10.0
    B_init = 2.0 * np.pi * 18000.0
    C_init = 10.0

# --- Intervallo di frequenza per il fit (kHz) ---
frfit0 = 4.0
frfit1 = 900000.0

# --- Errori di misura ---
reading_error_div = 0.1
scale_error_frac = 0.03

# --- Scansione del chi2 ---
n_sigma_scan = 2
step_scan = 100

# --- Debug ---
DEB = False

# ============================================================
# RANGE DEI GRAFICI
# ============================================================
# Imposta None per lasciare il limite automatico.
# Ogni coppia xmin/xmax agisce sul pannello superiore e sui residui
# quando i due pannelli condividono l'asse x.

# Grafico 1: dati completi, pannello tensioni
g1_volt_xmin, g1_volt_xmax = None, None
g1_volt_ymin, g1_volt_ymax = None, None
g1_volt_xlog, g1_volt_ylog = False, False

# Grafico 1: dati completi, pannello funzione di trasferimento
g1_T_xmin, g1_T_xmax = None, None
g1_T_ymin, g1_T_ymax = None, None
g1_T_xlog, g1_T_ylog = False, False

# Grafico 2: fit + residui, asse lineare
g2_xmin, g2_xmax = 30, None
g2_ymin, g2_ymax = None, None
g2_res_ymin, g2_res_ymax = None, None

# Grafico extra: fit + residui, asse x logaritmico
g2log_xmin, g2log_xmax = 4.3, None
g2log_ymin, g2log_ymax = None, None
g2log_res_ymin, g2log_res_ymax = None, None

# Zoom laterale del grafico logaritmico, centrato attorno alla risonanza.
# Se g2log_zoom_center_khz = None, il centro viene preso dal fit: f0 = omega0/(2*pi).
MAKE_LOG_RESONANCE_ZOOM = True
g2log_zoom_center_khz = None
g2log_zoom_half_width_khz = 10.0
g2log_zoom_xmin, g2log_zoom_xmax = None, None
g2log_zoom_ymin, g2log_zoom_ymax = None, None

# Grafico 3: fit dal minimo chi2 + residui
g3_xmin, g3_xmax = None, None
g3_ymin, g3_ymax = None, None
g3_res_ymin, g3_res_ymax = None, None

# Grafico 4: mappa chi2 e profili
g4_B_xmin, g4_B_xmax = None, None
g4_C_ymin, g4_C_ymax = None, None
g4_prof_xmin, g4_prof_xmax = None, None
g4_prof_ymin, g4_prof_ymax = None, None


# ============================================================
# SETTAGGIO GRAFICI
# ============================================================
plt.style.use(hep.style.ROOT)
params = {
    'legend.fontsize': '10',
    'legend.loc': 'upper right',
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.facecolor': 'w',
    'legend.edgecolor': 'w',
    'figure.figsize': (6, 4),
    'axes.labelsize': '10',
    'figure.titlesize': '14',
    'axes.titlesize': '12',
    'xtick.labelsize': '10',
    'ytick.labelsize': '10',
    'lines.linewidth': '1',
    'text.usetex': True,
    'axes.formatter.min_exponent': '2',
    'figure.subplot.left': '0.125',
    'figure.subplot.bottom': '0.125',
    'figure.subplot.right': '0.925',
    'figure.subplot.top': '0.925',
    'figure.subplot.wspace': '0.1',
    'figure.subplot.hspace': '0.1',
    'figure.constrained_layout.use': True,
}
plt.rcParams.update(params)
plt.rcParams['axes.prop_cycle'] = cycler(color=['b', 'g', 'r', 'c', 'm', 'y', 'k'])

# ============================================================
# DEFINIZIONE FUNZIONI DI FIT
# ============================================================
def fitf_C(x, A, B, C):
    """Funzione di trasferimento per uscita su condensatore."""
    omega = 2.0 * np.pi * x * 1e3
    return A / np.sqrt((1 - omega**2 / B**2)**2 + (1 / C**2) * omega**2 / B**2)


def fitf_R(x, A, B, C):
    """Funzione di trasferimento per uscita su resistore."""
    omega = 2.0 * np.pi * x * 1e3
    return A / np.sqrt(1 + C**2 * (omega / B - B / omega)**2)


if fit_mode == 'R':
    fit_func = fitf_R
else:
    fit_func = fitf_C

# ============================================================
# FUNZIONE PER IL CALCOLO DELLA MAPPA DEL CHI2
# ============================================================
def fitchi2(i, j, k):
    """Calcola chi2 per la terna di parametri (i,j,k) e lo salva in mappa."""
    global fr, TR, eTR, A_chi, B_chi, C_chi, mappa
    AA, BB, CC = A_chi[i], B_chi[j], C_chi[k]
    residuals = TR - fit_func(fr, AA, BB, CC)
    chi2 = np.sum((residuals / eTR)**2)
    mappa[i, j, k] = chi2

# ============================================================
# FUNZIONI DI PROFILAZIONE
# ============================================================
def profi2D(axis, matrix3D):
    """
    Profila matrix3D eliminando l'asse indicato.
    axis=1 -> elimina A -> profilo (B,C), restituito come (C,B)
    axis=2 -> elimina B -> profilo (A,C), restituito come (C,A)
    axis=3 -> elimina C -> profilo (A,B), restituito come (B,A)
    """
    if axis == 1:
        return matrix3D.min(axis=0).T
    elif axis == 2:
        return matrix3D.min(axis=1).T
    elif axis == 3:
        return matrix3D.min(axis=2).T
    else:
        raise ValueError('axis deve essere 1, 2 oppure 3')


def profi1D(axes_to_remove, matrix3D):
    """
    Profila matrix3D minimizzando sugli assi indicati.
    axes_to_remove: tupla/lista di interi (1=A, 2=B, 3=C) da rimuovere.
    """
    real_axes = tuple(a - 1 for a in axes_to_remove)
    return matrix3D.min(axis=real_axes)

# ============================================================
# FUNZIONI UTILI PER RANGE E SCALE DEI GRAFICI
# ============================================================
def apply_axis_settings(ax, xmin=None, xmax=None, ymin=None, ymax=None,
                        xlog=False, ylog=False):
    """Applica range e scala a un asse, lasciando automatici i limiti None."""
    if xmin is not None or xmax is not None:
        ax.set_xlim(xmin, xmax)
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')


# ============================================================
# CARICAMENTO DATI
# ============================================================
data = np.loadtxt(file)
fr = data[:, 0]
Vin = data[:, 1]
Vo = data[:, 2]
Vdiv_in = data[:, 3]
VdivR = data[:, 4]

N = len(fr[fr > 0])

# ============================================================
# CALCOLO INCERTEZZE
# ============================================================
eVin = np.sqrt((reading_error_div * Vdiv_in)**2 + (scale_error_frac * Vin)**2)
eVo = np.sqrt((reading_error_div * VdivR)**2 + (scale_error_frac * Vo)**2)

TR = Vo / Vin
eTR = TR * np.sqrt((eVo / Vo)**2 + (eVin / Vin)**2)

# ============================================================
# GRAFICO 1A: DATI COMPLETI Vin e Vout
# ============================================================
# Qui aumento di 3 punti la dimensione di tutte le scritte rispetto allo stile globale.
fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.4), constrained_layout=True)

ax.errorbar(fr, Vin, yerr=eVin, fmt='o', label=r'$V_{in}$',
            ms=3, color="darkgreen")
ax.errorbar(fr, Vo, yerr=eVo, fmt='o', label=r'$V_{out}$',
            ms=3, color='purple')

ax.set_xlabel(r'Frequenza [kHz]', fontsize=13)
ax.set_ylabel(r'Voltaggio [V]', fontsize=13)
ax.set_title(r'Tensioni misurate $V_{in}$ e $V_{out}$', fontsize=15)
ax.legend(prop={'size': 13}, loc='best')
ax.tick_params(axis='both', which='major', labelsize=13)
ax.tick_params(axis='both', which='minor', labelsize=13)

apply_axis_settings(ax, g1_volt_xmin, g1_volt_xmax, g1_volt_ymin, g1_volt_ymax,
                    g1_volt_xlog, g1_volt_ylog)

plt.savefig(file.replace('.txt', '') + '_1_Vin_Vout.png',
            bbox_inches='tight', pad_inches=1, transparent=True,
            facecolor='w', edgecolor='w', orientation='Portrait', dpi=120)
plt.show()

# ============================================================
# GRAFICO 1B: DATI COMPLETI funzione di trasferimento
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.2), constrained_layout=True)

ax.errorbar(fr, TR, yerr=eTR, fmt='o',
            label=r'$T_R=\frac{V_{out}}{V_{in}}$', ms=2, color='darkred')
ax.legend(prop={'size': 10}, loc='best')
ax.set_ylabel(r'Funzione di trasferimento $T_R$')
ax.set_xlabel(r'Frequenza [kHz]')

apply_axis_settings(ax, g1_T_xmin, g1_T_xmax, g1_T_ymin, g1_T_ymax,
                    g1_T_xlog, g1_T_ylog)

plt.savefig(file.replace('.txt', '') + '_1_T.png',
            bbox_inches='tight', pad_inches=1, transparent=True,
            facecolor='w', edgecolor='w', orientation='Portrait', dpi=120)
plt.show()

# ============================================================
# APPLICAZIONE MASCHERA DI FREQUENZA PER IL FIT
# ============================================================
mask_fit = (fr >= frfit0) & (fr <= frfit1) & (fr > 0) & np.isfinite(TR) & np.isfinite(eTR) & (eTR > 0)
fr = fr[mask_fit]
TR = TR[mask_fit]
eTR = eTR[mask_fit]
N = len(fr)

if N < 4:
    raise RuntimeError('Troppi pochi punti nel range di fit.')

# ============================================================
# FIT CON SCIPY
# ============================================================
popt, pcov = curve_fit(
    fit_func,
    fr,
    TR,
    p0=[A_init, B_init, C_init],
    method='lm',
    sigma=eTR,
    absolute_sigma=True,
)
perr = np.sqrt(np.diag(pcov))

print(
    ' ampiezza = {a:.3f} +/- {b:.3f} \n omega0 = {c:.1f} +/- {d:.1f} kHz \n Q-valore = {e:.1f} +/- {f:.1f}'.format(
        a=popt[0], b=perr[0],
        c=popt[1] / 1000, d=perr[1] / 1000,
        e=popt[2], f=perr[2],
    )
)

residuA = TR - fit_func(fr, *popt)
chisq = np.sum((residuA / eTR)**2)
df = N - 3
chisq_rid = chisq / df

# ============================================================
# GRAFICO 2: FIT + RESIDUI, ASSE LINEARE
# ============================================================
x_fit = np.linspace(min(fr), max(fr), 1000)

fig, ax = plt.subplots(2, 1, figsize=(5, 4), sharex=True,
                       constrained_layout=True, height_ratios=[2, 1])
ax[0].plot(x_fit, fit_func(x_fit, *popt), label='Fit', linestyle='--', color='black')
ax[0].plot(x_fit, fit_func(x_fit, A_init, B_init, C_init),
           label='init guess', linestyle='dashed', color='green')
ax[0].errorbar(fr, TR, yerr=eTR, fmt='o',
               label=r'$T_R=\frac{V_{out}}{V_{in}}$', ms=2, color='darkred')
ax[0].legend(loc='upper left')
ax[0].set_ylabel(r'Funzione di trasferimento $T_R$')
apply_axis_settings(ax[0], g2_xmin, g2_xmax, g2_ymin, g2_ymax)

ax[1].errorbar(fr, residuA, yerr=eTR, fmt='o', label=r'Residui', ms=2, color='darkred')
ax[1].set_ylabel(r'Residui')
ax[1].set_xlabel(r'Frequenza [kHz]')
ax[1].plot(fr, np.zeros(len(fr)), color='black')
apply_axis_settings(ax[1], g2_xmin, g2_xmax, g2_res_ymin, g2_res_ymax)

plt.savefig(file.replace('.txt', '') + '_2.png',
            bbox_inches='tight', pad_inches=1, transparent=True,
            facecolor='w', edgecolor='w', orientation='Portrait', dpi=100)
plt.show()

# ============================================================
# GRAFICO EXTRA: FIT + RESIDUI, ASSE X LOGARITMICO
# con zoom separato a destra, entrambi con residui sotto
# ============================================================
mask_log = fr > 0
fr_log = fr[mask_log]
TR_log = TR[mask_log]
eTR_log = eTR[mask_log]
residu_log = TR_log - fit_func(fr_log, *popt)

x_fit_log = np.logspace(
    np.log10(np.min(fr_log)),
    np.log10(np.max(fr_log)),
    1500,
)

# Frequenza di risonanza dal fit: B = omega0
f0_fit_khz = popt[1] / (2.0 * np.pi * 1e3)

# Range dello zoom.
# Se g2log_zoom_center_khz = None, centro automaticamente sul valore f0 del fit.
if MAKE_LOG_RESONANCE_ZOOM:
    if g2log_zoom_center_khz is None:
        zoom_center = f0_fit_khz
    else:
        zoom_center = g2log_zoom_center_khz

    if g2log_zoom_xmin is None:
        zoom_xmin = zoom_center - g2log_zoom_half_width_khz
    else:
        zoom_xmin = g2log_zoom_xmin

    if g2log_zoom_xmax is None:
        zoom_xmax = zoom_center + g2log_zoom_half_width_khz
    else:
        zoom_xmax = g2log_zoom_xmax

    zoom_xmin = max(zoom_xmin, np.min(fr_log[fr_log > 0]))
    zoom_xmax = min(zoom_xmax, np.max(fr_log))
    if zoom_xmax <= zoom_xmin:
        zoom_xmin = max(zoom_center * 0.9, np.min(fr_log[fr_log > 0]))
        zoom_xmax = min(zoom_center * 1.1, np.max(fr_log))

# Se non vuoi lo zoom separato, questo blocco produce solo il grafico totale.
if MAKE_LOG_RESONANCE_ZOOM:
    fig = plt.figure(figsize=(10.2, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[2, 1],
        width_ratios=[1.55, 1.0],
        hspace=0.05,
        wspace=0.12,
    )

    ax_main = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_zoom_res = fig.add_subplot(gs[1, 1], sharex=ax_zoom)
else:
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(6.8, 5.0), sharex=True,
        constrained_layout=True, height_ratios=[2, 1]
    )
    ax_zoom = None
    ax_zoom_res = None

# Dimensioni aumentate di 3 punti per i due grafici in scala logaritmica
log_label_fs = 13
log_title_fs = 15
log_tick_fs = 13
log_legend_fs = 13

# ----------------------------
# Grafico totale, asse x log
# ----------------------------
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

ax_main.plot(x_fit_log, fit_func(x_fit_log, *popt),
             label='Fit', linestyle='--', color='black')
ax_main.errorbar(fr_log, TR_log, yerr=eTR_log, fmt='o',
                 label=r'$T_R=\frac{V_{out}}{V_{in}}$', ms=2, color='darkred')
ax_main.set_xscale('log')
ax_main.set_ylabel(r'Funzione di trasferimento $T_R$', fontsize=log_label_fs)
ax_main.legend(loc='best', fontsize=log_legend_fs)
ax_main.set_title(r'Fit funzione di trasferimento con asse logaritmico', fontsize=log_title_fs)
apply_axis_settings(ax_main, g2log_xmin, g2log_xmax, g2log_ymin, g2log_ymax, xlog=True)

ax_res.axhline(0, color='black', lw=0.8)
ax_res.errorbar(fr_log, residu_log, yerr=eTR_log, fmt='o',
                label=r'Residui', ms=2, color='darkred')
ax_res.set_xscale('log')
ax_res.set_ylabel(r'Residui', fontsize=log_label_fs)
ax_res.set_xlabel(r'Frequenza [kHz]', fontsize=log_label_fs)
apply_axis_settings(ax_res, g2log_xmin, g2log_xmax, g2log_res_ymin, g2log_res_ymax, xlog=True)

# ----------------------------
# Zoom separato a destra
# ----------------------------
if MAKE_LOG_RESONANCE_ZOOM:
    # Anche lo zoom usa scala logaritmica sull'asse x.
    x_fit_zoom = np.logspace(np.log10(zoom_xmin), np.log10(zoom_xmax), 1000)
    mask_zoom_data = (fr_log >= zoom_xmin) & (fr_log <= zoom_xmax)

    ax_zoom.plot(x_fit_zoom, fit_func(x_fit_zoom, *popt),
                 label='Fit', linestyle='--', color='black')
    ax_zoom.errorbar(fr_log[mask_zoom_data], TR_log[mask_zoom_data],
                     yerr=eTR_log[mask_zoom_data], fmt='o',
                     label=r'$T_R$', ms=2.2, color='darkred')

    ax_zoom.set_xlim(zoom_xmin, zoom_xmax)
    if g2log_zoom_ymin is not None or g2log_zoom_ymax is not None:
        ax_zoom.set_ylim(g2log_zoom_ymin, g2log_zoom_ymax)
    else:
        y_values = []
        if np.any(mask_zoom_data):
            y_values.extend((TR_log[mask_zoom_data] - eTR_log[mask_zoom_data]).tolist())
            y_values.extend((TR_log[mask_zoom_data] + eTR_log[mask_zoom_data]).tolist())
        y_values.extend(fit_func(x_fit_zoom, *popt).tolist())
        y_values = np.asarray(y_values, dtype=float)
        y_values = y_values[np.isfinite(y_values)]
        if len(y_values) > 0:
            ymin = np.min(y_values)
            ymax = np.max(y_values)
            pad = 0.08 * (ymax - ymin) if ymax > ymin else 0.05 * max(abs(ymax), 1.0)
            ax_zoom.set_ylim(ymin - pad, ymax + pad)

    ax_zoom.set_xscale('log')
    ax_zoom.axvline(f0_fit_khz, color='0.35', ls=':', lw=0.9)
    ax_zoom.set_ylabel(r'Funzione di trasferimento $T_R$', fontsize=log_label_fs)
    ax_zoom.set_title(r'Zoom vicino a $f_0$', fontsize=log_title_fs)
    ax_zoom.legend(loc='best', fontsize=log_legend_fs)

    # Rettangolo grigio sul grafico principale: indica la zona riportata nello zoom.
    zoom_ymin_rect, zoom_ymax_rect = ax_zoom.get_ylim()
    rect = Rectangle(
        (zoom_xmin, zoom_ymin_rect),
        zoom_xmax - zoom_xmin,
        zoom_ymax_rect - zoom_ymin_rect,
        fill=False,
        edgecolor='0.45',
        linewidth=1.0,
        linestyle='-',
        zorder=10,
        label='Area dello zoom',
    )
    ax_main.add_patch(rect)
    ax_main.legend(loc='best', fontsize=log_legend_fs)

    ax_zoom_res.axhline(0, color='black', lw=0.8)
    ax_zoom_res.errorbar(fr_log[mask_zoom_data], residu_log[mask_zoom_data],
                         yerr=eTR_log[mask_zoom_data], fmt='o',
                         ms=2.2, color='darkred')
    ax_zoom_res.set_xlim(zoom_xmin, zoom_xmax)
    ax_zoom_res.set_xscale('log')
    ax_zoom_res.set_ylabel(r'Residui', fontsize=log_label_fs)
    ax_zoom_res.set_xlabel(r'Frequenza [kHz]', fontsize=log_label_fs)

    # Range dei residui nello zoom: se non specificato, autoscale sui residui locali.
    if g2log_res_ymin is not None or g2log_res_ymax is not None:
        ax_zoom_res.set_ylim(g2log_res_ymin, g2log_res_ymax)
    else:
        rz = residu_log[mask_zoom_data]
        erz = eTR_log[mask_zoom_data]
        if len(rz) > 0:
            yres = np.concatenate([rz - erz, rz + erz, [0.0]])
            ymin = np.nanmin(yres)
            ymax = np.nanmax(yres)
            pad = 0.10 * (ymax - ymin) if ymax > ymin else 0.02
            ax_zoom_res.set_ylim(ymin - pad, ymax + pad)


# Tick label piu grandi per i pannelli in scala logaritmica
for axx in [ax_main, ax_res]:
    axx.tick_params(axis='both', which='major', labelsize=log_tick_fs)
    axx.tick_params(axis='both', which='minor', labelsize=log_tick_fs)

if MAKE_LOG_RESONANCE_ZOOM:
    for axx in [ax_zoom, ax_zoom_res]:
        axx.tick_params(axis='both', which='major', labelsize=log_tick_fs)
        axx.tick_params(axis='both', which='minor', labelsize=log_tick_fs)

# Riduci il numero di etichette sull'asse x, ma lasciandone alcune visibili.
# Nel grafico principale mostro poche frequenze scelte in kHz.
main_xticks = [5, 10, 30, 100, 300, 900]
main_xticks = [v for v in main_xticks if np.min(fr_log) <= v <= np.max(fr_log)]

for axx in [ax_main, ax_res]:
    axx.set_xticks(main_xticks)
    axx.set_xticklabels([f'{v:g}' for v in main_xticks], fontsize=log_tick_fs)
    axx.xaxis.set_minor_formatter(NullFormatter())

# Nello zoom mostro 6 etichette equidistanti nel range selezionato.
# Le etichette sono stampate con una sola cifra decimale, tipo 130.1.
if MAKE_LOG_RESONANCE_ZOOM:
    zoom_xticks = np.linspace(zoom_xmin, zoom_xmax, 6)

    for axx in [ax_zoom, ax_zoom_res]:
        axx.set_xticks(zoom_xticks)
        axx.set_xticklabels([f'{v:.1f}' for v in zoom_xticks], fontsize=log_tick_fs)
        axx.xaxis.set_minor_formatter(NullFormatter())

plt.savefig(file.replace('.txt', '') + '_2_logx_zoom_side.png',
            bbox_inches='tight', pad_inches=1, transparent=True,
            facecolor='w', edgecolor='w', orientation='Portrait', dpi=120)
plt.show()

# ============================================================
# STAMPA BEST FIT SCIPY
# ============================================================
A_BF, B_BF, C_BF = popt
eA_BF, eB_BF, eC_BF = perr

print('============== BEST FIT with SciPy ====================')
print(r'A = ({a:.3e} +/- {b:.1e})'.format(a=A_BF, b=eA_BF))
print(r'B = ({c:.5e} +/- {d:.1e}) kHz'.format(c=B_BF * 1e-3, d=eB_BF * 1e-3))
print(r'C = ({e:.3e} +/- {f:.1e})'.format(e=C_BF, f=eC_BF))
print(r'chisq = {m:.2f}'.format(m=chisq))
print('=======================================================')

# ============================================================
# SCANSIONE DEL CHI2 ATTORNO AL MINIMO
# ============================================================
A0, A1 = A_BF - n_sigma_scan * eA_BF, A_BF + n_sigma_scan * eA_BF
B0, B1 = B_BF - n_sigma_scan * eB_BF, B_BF + n_sigma_scan * eB_BF
C0, C1 = C_BF - n_sigma_scan * eC_BF, C_BF + n_sigma_scan * eC_BF

step = step_scan
A_chi = np.linspace(A0, A1, step)
B_chi = np.linspace(B0, B1, step)
C_chi = np.linspace(C0, C1, step)

mappa = np.zeros((step, step, step))
item = [(i, j, k) for i in range(step) for j in range(step) for k in range(step)]

pool = multiprocessing.pool.ThreadPool(100)
pool.starmap(fitchi2, item, chunksize=10)
pool.close()
pool.join()

mappa = np.asarray(mappa)

chi2_min = np.min(mappa)
argchi2_min = np.unravel_index(np.argmin(mappa), mappa.shape)

residui_chi2 = TR - fit_func(
    fr,
    A_chi[argchi2_min[0]],
    B_chi[argchi2_min[1]],
    C_chi[argchi2_min[2]],
)
chisq_res = np.sum((residui_chi2 / eTR)**2)
print(chi2_min, argchi2_min, chisq_res)

# ============================================================
# GRAFICO 3: FIT DAL MINIMO CHI2 + RESIDUI, ASSE LINEARE
# ============================================================
fig, ax = plt.subplots(2, 1, figsize=(3, 5), sharex=True,
                       constrained_layout=True, height_ratios=[2, 1])
ax[0].plot(
    x_fit,
    fit_func(x_fit, A_chi[argchi2_min[0]], B_chi[argchi2_min[1]], C_chi[argchi2_min[2]]),
    label='Fit',
    linestyle='--',
    color='blue',
)
ax[0].errorbar(fr, TR, yerr=eTR, fmt='o', label=r'$V_{out}$', ms=2, color='darkred')
ax[0].legend(loc='upper left')
ax[0].set_ylabel(r'Funzione di trasferimento $T_R$')
apply_axis_settings(ax[0], g3_xmin, g3_xmax, g3_ymin, g3_ymax)

ax[1].errorbar(fr, residui_chi2, yerr=eTR, fmt='o', label=r'Residui', ms=2, color='darkred')
ax[1].set_ylabel(r'Residui')
ax[1].set_xlabel(r'Frequenza [kHz]')
ax[1].plot(fr, np.zeros(N), color='black')
apply_axis_settings(ax[1], g3_xmin, g3_xmax, g3_res_ymin, g3_res_ymax)

plt.savefig(file.replace('.txt', '') + '_3.png',
            bbox_inches='tight', pad_inches=1, transparent=True,
            facecolor='w', edgecolor='w', orientation='Portrait', dpi=100)
plt.show()

# ============================================================
# PROFILI DEL CHI2 E CALCOLO DEGLI ERRORI
# ============================================================
chi2D = profi2D(1, mappa)

prof_A = profi1D([2, 3], mappa)
prof_B = profi1D([1, 3], mappa)
prof_C = profi1D([1, 2], mappa)

lvl = chi2_min + 1.0

diff_A = abs(prof_A - lvl)
diff_B = abs(prof_B - lvl)
diff_C = abs(prof_C - lvl)

B_dx = np.argmin(diff_B[B_chi < B_BF])
B_sx = np.argmin(diff_B[B_chi > B_BF]) + len(B_chi[B_chi < B_BF])
C_dx = np.argmin(diff_C[C_chi < C_BF])
C_sx = np.argmin(diff_C[C_chi > C_BF]) + len(C_chi[C_chi < C_BF])
A_dx = np.argmin(diff_A[A_chi < A_BF])
A_sx = np.argmin(diff_A[A_chi > A_BF]) + len(A_chi[A_chi < A_BF])

errA = A_chi[argchi2_min[0]] - A_chi[A_dx]
errAA = A_chi[A_sx] - A_chi[argchi2_min[0]]
errB = B_chi[argchi2_min[1]] - B_chi[B_dx]
errBB = B_chi[B_sx] - B_chi[argchi2_min[1]]
errC = C_chi[argchi2_min[2]] - C_chi[C_dx]
errCC = C_chi[C_sx] - C_chi[argchi2_min[2]]

print('============== BEST FIT with chi2 ====================')
print(r'A = ({a:.3e} - {b:.1e} + {c:.1e})'.format(a=A_chi[argchi2_min[0]], b=errA, c=errAA))
print(r'B = ({d:.5e} - {e:.1e} + {f:.1e}) kHz'.format(d=B_chi[argchi2_min[1]] * 1e-3, e=errB * 1e-3, f=errBB * 1e-3))
print(r'C = ({g:.3e} - {h:.1e} + {n:.1e}) '.format(g=C_chi[argchi2_min[2]], h=errC, n=errCC))
print(r'chisq = {m:.2f}'.format(m=np.min(mappa)))
print('=======================================================')

# ============================================================
# GRAFICO 4: PROFILI CHI2
# ============================================================
cmap = mpl.colormaps['plasma'].reversed()
level = np.linspace(np.min(chi2D), np.max(chi2D), 100)
line_c = 'gray'

fig, ax = plt.subplots(2, 2, figsize=(5.5, 5), constrained_layout=True,
                       height_ratios=[3, 1], width_ratios=[1, 3],
                       sharex='col', sharey='row')
fig.suptitle(r'$\chi^2 \left(\omega_0, Q \right)$')

im = ax[0, 1].contourf(B_chi, C_chi, chi2D, levels=level, cmap=cmap)
cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=ax[0, 1],
                    ticks=[int(chi2_min), int(chi2_min + 2), int(chi2_min + 4), int(chi2_min + 6)])
cbar.set_label(r'$\chi^2$', rotation=360)

CS = ax[0, 1].contour(
    B_chi,
    C_chi,
    chi2D,
    levels=[chi2_min + 0.0001, chi2_min + 1, chi2_min + 2.3, chi2_min + 3.8],
    linewidths=1,
    colors='k',
    alpha=0.5,
    linestyles='dotted',
)
ax[0, 1].clabel(CS, inline=True, fontsize=9, fmt='%.1f')
ax[0, 1].text(
    B_chi[np.argmin(prof_B)],
    C_chi[np.argmin(prof_C)],
    r'{g:.0f}'.format(g=chi2_min),
    color='k',
    alpha=0.5,
    fontsize=9,
)

ax[0, 1].plot([B0, B1], [C_chi[C_sx], C_chi[C_sx]], color=line_c, ls='dashed')
ax[0, 1].plot([B0, B1], [C_chi[C_dx], C_chi[C_dx]], color=line_c, ls='dashed')
ax[0, 1].plot([B_chi[B_sx], B_chi[B_sx]], [C0, C1], color=line_c, ls='dashed')
ax[0, 1].plot([B_chi[B_dx], B_chi[B_dx]], [C0, C1], color=line_c, ls='dashed')

ax[0, 0].plot(prof_C, C_chi, ls='-')
ax[0, 0].plot([int(chi2_min - 1), int(chi2_min + 4)], [C_chi[C_sx], C_chi[C_sx]], color=line_c, ls='dashed')
ax[0, 0].plot([int(chi2_min - 1), int(chi2_min + 4)], [C_chi[C_dx], C_chi[C_dx]], color=line_c, ls='dashed')
ax[0, 0].set_xticks([int(chi2_min), int(chi2_min + 1), int(chi2_min + 4), int(chi2_min + 6)])
ax[0, 0].text(int(chi2_min + 1), C_chi[np.argmin(prof_C)],
              r'{g:.2f}'.format(g=C_chi[np.argmin(prof_C)]), color='k', alpha=0.5, fontsize=9)
ax[0, 0].text(int(chi2_min + 2), C_chi[C_sx],
              r'{g:.2f}'.format(g=errCC), color='b', alpha=0.5, fontsize=9)
ax[0, 0].text(int(chi2_min + 2), C_chi[C_dx],
              r'{g:.2f}'.format(g=-errC), color='r', alpha=0.5, fontsize=9)

ax[1, 1].plot(B_chi, prof_B)
ax[1, 1].plot([B_chi[B_sx], B_chi[B_sx]], [int(chi2_min - 1), int(chi2_min + 4)], color=line_c, ls='dashed')
ax[1, 1].plot([B_chi[B_dx], B_chi[B_dx]], [int(chi2_min - 1), int(chi2_min + 4)], color=line_c, ls='dashed')
ax[1, 1].text(B_chi[np.argmin(prof_B)], int(chi2_min + 1),
              r'{g:.3e}'.format(g=B_chi[np.argmin(prof_B)]), color='k', alpha=0.5, fontsize=9)
ax[1, 1].text(B_chi[B_sx], int(chi2_min + 2),
              r'{g:.0e}'.format(g=errBB), color='b', alpha=0.5, fontsize=9)
ax[1, 1].text(B_chi[B_dx], int(chi2_min + 2),
              r'{g:.0e}'.format(g=-errB), color='r', alpha=0.5, fontsize=9)
ax[1, 1].set_yticks([int(chi2_min), int(chi2_min + 4), int(chi2_min + 6)])

ax[1, 0].set_axis_off()
ax[0, 0].set_ylabel(r'$Q$-valore')
ax[1, 1].set_xlabel(r'$\omega_0$ [Hz]', loc='center')

# Range automatici predefiniti per i profili, modificabili dalle variabili g4_*.
default_prof_xmin = int(chi2_min - 1)
default_prof_xmax = int(chi2_min + 4)
default_prof_ymin = int(chi2_min - 1)
default_prof_ymax = int(chi2_min + 4)

ax[0, 0].set_xlim(
    default_prof_xmin if g4_prof_xmin is None else g4_prof_xmin,
    default_prof_xmax if g4_prof_xmax is None else g4_prof_xmax,
)
ax[1, 1].set_ylim(
    default_prof_ymin if g4_prof_ymin is None else g4_prof_ymin,
    default_prof_ymax if g4_prof_ymax is None else g4_prof_ymax,
)

# Range della mappa centrale e dei pannelli condivisi.
if g4_B_xmin is not None or g4_B_xmax is not None:
    ax[0, 1].set_xlim(g4_B_xmin, g4_B_xmax)
    ax[1, 1].set_xlim(g4_B_xmin, g4_B_xmax)
if g4_C_ymin is not None or g4_C_ymax is not None:
    ax[0, 1].set_ylim(g4_C_ymin, g4_C_ymax)
    ax[0, 0].set_ylim(g4_C_ymin, g4_C_ymax)

plt.savefig(file.replace('.txt', '') + '_4.png',
            bbox_inches='tight', pad_inches=1, transparent=True,
            facecolor='w', edgecolor='w', orientation='Portrait', dpi=100)
plt.show()
