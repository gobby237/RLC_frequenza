import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler
import matplotlib.colors as colors
import multiprocessing.pool

# ============================================================
# PARAMETRI MODIFICABILI DALL'UTENTE
# ============================================================

# --- File di input ---
# Il file deve contenere 5 colonne separate da spazio:
# frequenza (kHz)   Vin (V)   Vout (V)   V/div canale Vin   V/div canale Vout
file = 'RLC_Rres.txt'               # nome del file con i dati

# --- Modalità di fit ---
fit_mode = 'R'                       # 'R' per trasferimento su resistore, 'C' su condensatore
# Le stime iniziali per il fit (possono essere aggiustate)
if fit_mode == 'R':
    A_init = 0.43                    # ampiezza (adimensionale)
    B_init = 2.0 * np.pi * 130.0e3  # omega0 in rad/s (18 kHz)
    C_init = 10.5                    # fattore di qualità Q
else:  # fit_mode == 'C'
    A_init = 10.0                    # per il fit su C, A ~ Q (picco)
    B_init = 2.0 * np.pi * 18000.0
    C_init = 10.0

# --- Intervallo di frequenza per il fit (kHz) ---
frfit0 = 40.0                        # limite inferiore (usa 0 per prendere tutto)
frfit1 = 150000.0                       # limite superiore

# --- Errori di misura ---
# Incertezza di lettura della scala verticale (in divisioni)
reading_error_div = 0.1              # tipicamente 0.1 divisioni
# Incertezza di scala (guadagno) data in frazione (es. 3% -> 0.03)
scale_error_frac = 0.03              # errore di calibrazione verticale

# --- Scansione del chi2 ---
n_sigma_scan = 2                     # semiampiezza della scansione in unità di sigma del fit
step_scan = 100                      # numero di punti per parametro

# --- Debug ---
DEB = False                          # True per stampe aggiuntive

# ============================================================
# SETTAGGIO GRAFICI (invariato)
# ============================================================
plt.style.use(hep.style.ROOT)
params = {'legend.fontsize': '10',
          'legend.loc': 'upper right',
          'legend.frameon': 'True',
          'legend.framealpha': '0.8',
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
          'figure.constrained_layout.use': True
          }
plt.rcParams.update(params)
plt.rcParams['axes.prop_cycle'] = cycler(color=['b','g','r','c','m','y','k'])

# ============================================================
# DEFINIZIONE FUNZIONI DI FIT
# ============================================================
def fitf_C(x, A, B, C):
    """Funzione di trasferimento per uscita su condensatore."""
    omega = 2.0 * np.pi * x * 1e3   # x in kHz -> omega in rad/s
    return A / np.sqrt((1 - omega**2/B**2)**2 + (1/C**2) * omega**2/B**2)

def fitf_R(x, A, B, C):
    """Funzione di trasferimento per uscita su resistore."""
    omega = 2.0 * np.pi * x * 1e3
    return A / np.sqrt(1 + C**2 * (omega/B - B/omega)**2)

# Scelta automatica della funzione da usare per il fit
if fit_mode == 'R':
    fit_func = fitf_R
else:
    fit_func = fitf_C

# ============================================================
# FUNZIONE PER IL CALCOLO DELLA MAPPA DEL CHI2 (corretta)
# ============================================================
def fitchi2(i, j, k):
    """Calcola chi2 per la terna di parametri (i,j,k) e lo salva in mappa."""
    global fr, TR, eTR, A_chi, B_chi, C_chi, mappa
    x = fr
    y = TR
    y_err = eTR
    AA, BB, CC = A_chi[i], B_chi[j], C_chi[k]
    residuals = y - fit_func(x, AA, BB, CC)   #### CORREZIONE: usa la funzione scelta
    chi2 = np.sum((residuals / y_err)**2)
    mappa[i, j, k] = chi2

# ============================================================
# FUNZIONI DI PROFILAZIONE (corrette)
# ============================================================
def profi2D(axis, matrix3D):
    """
    Profila matrix3D (shape step×step×step) eliminando l'asse 'axis'.
    axis=1 -> elimina A -> profilo (B,C) memorizzato come (C,B) per contour.
    axis=2 -> elimina B -> (C,A)
    axis=3 -> elimina C -> (B,A)
    """
    if axis == 1:          # profila via A (asse 0) -> rimane (B,C)
        return matrix3D.min(axis=0).T   # .T per avere righe C, colonne B
    elif axis == 2:        # profila via B (asse 1) -> rimane (A,C) -> restituito (C,A)
        return matrix3D.min(axis=1).T
    elif axis == 3:        # profila via C (asse 2) -> rimane (A,B) -> restituito (B,A)
        return matrix3D.min(axis=2).T
    else:
        raise ValueError("axis deve essere 1,2,3")

def profi1D(axes_to_remove, matrix3D):
    """
    Profila matrix3D minimizzando sugli assi indicati.
    axes_to_remove: tupla di interi (1=A, 2=B, 3=C) da rimuovere.
    Restituisce un array 1D con i minimi del chi2 rispetto al parametro rimanente.
    """
    # converto in indici 0-based
    real_axes = tuple(a - 1 for a in axes_to_remove)
    return matrix3D.min(axis=real_axes)

# ============================================================
# CARICAMENTO DATI
# ============================================================
data = np.loadtxt(file)
fr     = data[:, 0]      # frequenza (kHz)
Vin    = data[:, 1]      # tensione d'ingresso (V)
Vo     = data[:, 2]      # tensione d'uscita (V)
Vdiv_in = data[:, 3]     # sensibilità verticale canale Vin (V/div)
VdivR  = data[:, 4]      # sensibilità verticale canale Vout (V/div)   #### CORREZIONE: colonna 4

N = len(fr[fr > 0])      # numero di punti con frequenza positiva

# ============================================================
# CALCOLO INCERTEZZE (corretto)
# ============================================================
# Errore su Vin e Vout
eVin = np.sqrt((reading_error_div * Vdiv_in)**2 + (scale_error_frac * Vin)**2)
eVo  = np.sqrt((reading_error_div * VdivR)**2  + (scale_error_frac * Vo)**2)

# Funzione di trasferimento e sua incertezza (senza termine spurio)
TR = Vo / Vin
# La formula per l'errore relativo (errori indipendenti)
eTR = TR * np.sqrt((eVo / Vo)**2 + (eVin / Vin)**2)   #### CORREZIONE: tolto termine aggiuntivo

# ============================================================
# PRIMI GRAFICI: dati completi
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(5, 4), sharex=True,
                       constrained_layout=True, width_ratios=[1, 1])
ax[0].errorbar(fr, Vin, yerr=eVin, fmt='o', label=r'$V_{in}$', ms=2)
ax[0].errorbar(fr, Vo,  yerr=eVo,  fmt='o', label=r'$V_{out}$', ms=2)
ax[0].legend(prop={'size': 10}, loc='best')
ax[0].set_ylabel(r'Voltaggio (V)')

ax[1].errorbar(fr, TR, yerr=eTR, fmt='o',
               label=r'$T=\frac{V_{out}}{V_{in}}$', ms=2, color='red')
ax[1].legend(prop={'size': 10}, loc='best')
ax[1].set_ylabel(r'Funzione di trasferimento $T_R$')
ax[1].set_xlabel(r'Frequenza (kHz)')
ax[1].yaxis.set_ticks_position('right')
ax[1].yaxis.set_label_position('right')

plt.savefig(file.replace('.txt','') + '_1.png',
            bbox_inches='tight', pad_inches=1, transparent=True,
            facecolor='w', edgecolor='w', orientation='Portrait', dpi=100)
plt.show()

# ============================================================
# APPLICAZIONE MASCHERA DI FREQUENZA PER IL FIT
# ============================================================
mask_fit = (fr >= frfit0) & (fr <= frfit1)
fr   = fr[mask_fit]          # d'ora in poi si lavora solo sui dati selezionati
TR   = TR[mask_fit]
eTR  = eTR[mask_fit]
# Ricalcolo il numero di punti per il fit
N = len(fr)

# ============================================================
# FIT CON SCIPY
# ============================================================
popt, pcov = curve_fit(fit_func, fr, TR, p0=[A_init, B_init, C_init],
                       method='lm', sigma=eTR, absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))
print(' ampiezza = {a:.3f} +/- {b:.3f} \n omega0 = {c:.1f} +/- {d:.1f} kHz \n Q-valore = {e:.1f} +/- {f:.1f}'.format(
    a=popt[0], b=perr[0],
    c=popt[1]/1000, d=perr[1]/1000,
    e=popt[2], f=perr[2]))

# Residui e chi2
residuA = TR - fit_func(fr, *popt)   #### CORREZIONE: usa la stessa funzione del fit
chisq = np.sum((residuA / eTR)**2)
df = N - 3
chisq_rid = chisq / df

# Grafico del fit e dei residui
x_fit = np.linspace(min(fr), max(fr), 1000)

fig, ax = plt.subplots(2, 1, figsize=(5, 4), sharex=True,
                       constrained_layout=True, height_ratios=[2, 1])
ax[0].plot(x_fit, fit_func(x_fit, *popt), label='Fit', linestyle='--', color='black')
ax[0].plot(x_fit, fit_func(x_fit, A_init, B_init, C_init),
           label='init guess', linestyle='dashed', color='green')
ax[0].errorbar(fr, TR, yerr=eTR, fmt='o',
               label=r'$T=\frac{V_{out}}{V_{in}}$', ms=2, color='red')
ax[0].legend(loc='upper left')
ax[0].set_ylabel(r'Funzione di trasferimento $T_R$')

ax[1].errorbar(fr, residuA, yerr=eTR, fmt='o', label=r'Residui', ms=2, color='red')
ax[1].set_ylabel(r'Residui')
ax[1].set_xlabel(r'Frequenza (kHz)')
ax[1].plot(fr, np.zeros(len(fr)), color='black')

plt.savefig(file.replace('.txt','') + '_2.png',
            bbox_inches='tight', pad_inches=1, transparent=True,
            facecolor='w', edgecolor='w', orientation='Portrait', dpi=100)
plt.show()

# Stampa dei migliori parametri di fit
A_BF, B_BF, C_BF = popt
eA_BF, eB_BF, eC_BF = perr

print("============== BEST FIT with SciPy ====================")
print(r'A = ({a:.3e} +/- {b:.1e})'.format(a=A_BF, b=eA_BF))
print(r'B = ({c:.5e} +/- {d:.1e}) kHz'.format(c=B_BF*1e-3, d=eB_BF*1e-3))
print(r'C = ({e:.3e} +/- {f:.1e})'.format(e=C_BF, f=eC_BF))
print(r'chisq = {m:.2f}'.format(m=chisq))
print("=======================================================")

# ============================================================
# SCANSIONE DEL CHI2 ATTORNO AL MINIMO
# ============================================================
# Intervalli di scansione
A0, A1 = A_BF - n_sigma_scan * eA_BF, A_BF + n_sigma_scan * eA_BF
B0, B1 = B_BF - n_sigma_scan * eB_BF, B_BF + n_sigma_scan * eB_BF
C0, C1 = C_BF - n_sigma_scan * eC_BF, C_BF + n_sigma_scan * eC_BF

step = step_scan
A_chi = np.linspace(A0, A1, step)
B_chi = np.linspace(B0, B1, step)
C_chi = np.linspace(C0, C1, step)

# Mappa 3D del chi2
mappa = np.zeros((step, step, step))
item = [(i, j, k) for i in range(step) for j in range(step) for k in range(step)]

pool = multiprocessing.pool.ThreadPool(100)
pool.starmap(fitchi2, item, chunksize=10)
pool.close()

mappa = np.asarray(mappa)   # già ndarray

chi2_min = np.min(mappa)
argchi2_min = np.unravel_index(np.argmin(mappa), mappa.shape)

# Residui calcolati con i parametri del minimo del chi2 (controllo coerenza)
residui_chi2 = TR - fit_func(fr, A_chi[argchi2_min[0]], B_chi[argchi2_min[1]], C_chi[argchi2_min[2]])
chisq_res = np.sum((residui_chi2 / eTR)**2)
print(chi2_min, argchi2_min, chisq_res)

# Grafico del fit ottenuto dal minimo del chi2
fig, ax = plt.subplots(2, 1, figsize=(3, 5), sharex=True,
                       constrained_layout=True, height_ratios=[2, 1])
ax[0].plot(x_fit, fit_func(x_fit, A_chi[argchi2_min[0]], B_chi[argchi2_min[1]], C_chi[argchi2_min[2]]),
           label='Fit', linestyle='--', color='blue')
ax[0].errorbar(fr, TR, yerr=eTR, fmt='o', label=r'$V_{out}$', ms=2, color='red')
ax[0].legend(loc='upper left')
ax[0].set_ylabel(r'Funzione di trasferimento $T_R$')

ax[1].errorbar(fr, residui_chi2, yerr=eTR, fmt='o', label=r'Residui', ms=2, color='red')
ax[1].set_ylabel(r'Residui')
ax[1].set_xlabel(r'Frequenza (kHz)')
ax[1].plot(fr, np.zeros(N))

plt.savefig(file.replace('.txt','') + '_3.png',
            bbox_inches='tight', pad_inches=1, transparent=True,
            facecolor='w', edgecolor='w', orientation='Portrait', dpi=100)
plt.show()

# ============================================================
# PROFILI DEL CHI2 E CALCOLO DEGLI ERRORI (corretto)
# ============================================================
# Profili 2D e 1D
chi2D = profi2D(1, mappa)                       # profilo di B e C (elimina A)

prof_A = profi1D([2, 3], mappa)                 # profilo di A (eliminati B e C)
prof_B = profi1D([1, 3], mappa)                 # profilo di B (eliminati A e C)
prof_C = profi1D([1, 2], mappa)                 # profilo di C (eliminati A e B)

# Chi2_min + 1 per errori (1 parametro di interesse)
lvl = chi2_min + 1.0

diff_A = abs(prof_A - lvl)
diff_B = abs(prof_B - lvl)
diff_C = abs(prof_C - lvl)

# Indici degli estremi di confidenza per ogni parametro
# Nota: ora diff_B è funzione di B_chi, diff_C di C_chi, diff_A di A_chi
B_dx = np.argmin(diff_B[B_chi < B_BF])
B_sx = np.argmin(diff_B[B_chi > B_BF]) + len(B_chi[B_chi < B_BF])
C_dx = np.argmin(diff_C[C_chi < C_BF])
C_sx = np.argmin(diff_C[C_chi > C_BF]) + len(C_chi[C_chi < C_BF])
A_dx = np.argmin(diff_A[A_chi < A_BF])
A_sx = np.argmin(diff_A[A_chi > A_BF]) + len(A_chi[A_chi < A_BF])

# Errori asimmetrici
errA  = A_chi[argchi2_min[0]] - A_chi[A_dx]
errAA = A_chi[A_sx] - A_chi[argchi2_min[0]]
errB  = B_chi[argchi2_min[1]] - B_chi[B_dx]
errBB = B_chi[B_sx] - B_chi[argchi2_min[1]]
errC  = C_chi[argchi2_min[2]] - C_chi[C_dx]
errCC = C_chi[C_sx] - C_chi[argchi2_min[2]]

print("============== BEST FIT with chi2 ====================")
print(r'A = ({a:.3e} - {b:.1e} + {c:.1e})'.format(a=A_chi[argchi2_min[0]], b=errA, c=errAA))
print(r'B = ({d:.5e} - {e:.1e} + {f:.1e}) kHz'.format(d=B_chi[argchi2_min[1]]*1e-3, e=errB*1e-3, f=errBB*1e-3))
print(r'C = ({g:.3e} - {h:.1e} + {n:.1e}) '.format(g=C_chi[argchi2_min[2]], h=errC, n=errCC))
print(r'chisq = {m:.2f}'.format(m=np.min(mappa)))
print("=======================================================")

# ============================================================
# GRAFICI DELLE PROFILAZIONI (corretti)
# ============================================================
cmap = mpl.colormaps['plasma'].reversed()
level = np.linspace(np.min(chi2D), np.max(chi2D), 100)
line_c = 'gray'

fig, ax = plt.subplots(2, 2, figsize=(5.5, 5), constrained_layout=True,
                       height_ratios=[3, 1], width_ratios=[1, 3],
                       sharex='col', sharey='row')
fig.suptitle(r'$\chi^2 \left(\omega_0, Q \right)$')

# Contour 2D (B, C)
im = ax[0,1].contourf(B_chi, C_chi, chi2D, levels=level, cmap=cmap)
cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=ax[0,1],
                    ticks=[int(chi2_min), int(chi2_min+2), int(chi2_min+4), int(chi2_min+6)])
cbar.set_label(r'$\chi^2$', rotation=360)
CS = ax[0,1].contour(B_chi, C_chi, chi2D,
                     levels=[chi2_min+0.0001, chi2_min+1, chi2_min+2.3, chi2_min+3.8],
                     linewidths=1, colors='k', alpha=0.5, linestyles='dotted')
ax[0,1].clabel(CS, inline=True, fontsize=9, fmt='%.1f')
ax[0,1].text(B_chi[np.argmin(prof_B)], C_chi[np.argmin(prof_C)],
             r'{g:.0f}'.format(g=chi2_min), color='k', alpha=0.5, fontsize=9)

# Linee di errore sul contour (B,C)
ax[0,1].plot([B0, B1], [C_chi[C_sx], C_chi[C_sx]], color=line_c, ls='dashed')
ax[0,1].plot([B0, B1], [C_chi[C_dx], C_chi[C_dx]], color=line_c, ls='dashed')
ax[0,1].plot([B_chi[B_sx], B_chi[B_sx]], [C0, C1], color=line_c, ls='dashed')
ax[0,1].plot([B_chi[B_dx], B_chi[B_dx]], [C0, C1], color=line_c, ls='dashed')

# Pannello sinistro in alto: profilo di C (verticale)
ax[0,0].plot(prof_C, C_chi, ls='-')   #### CORREZIONE: prof_C contro C_chi
ax[0,0].plot([int(chi2_min-1), int(chi2_min+4)], [C_chi[C_sx], C_chi[C_sx]], color=line_c, ls='dashed')
ax[0,0].plot([int(chi2_min-1), int(chi2_min+4)], [C_chi[C_dx], C_chi[C_dx]], color=line_c, ls='dashed')
ax[0,0].set_xticks([int(chi2_min), int(chi2_min+1), int(chi2_min+4), int(chi2_min+6)])
ax[0,0].text(int(chi2_min+1), C_chi[np.argmin(prof_C)],
             r'{g:.2f}'.format(g=C_chi[np.argmin(prof_C)]), color='k', alpha=0.5, fontsize=9)
ax[0,0].text(int(chi2_min+2), C_chi[C_sx],
             r'{g:.2f}'.format(g=errCC), color='b', alpha=0.5, fontsize=9)
ax[0,0].text(int(chi2_min+2), C_chi[C_dx],
             r'{g:.2f}'.format(g=-errC), color='r', alpha=0.5, fontsize=9)

# Pannello in basso a destra: profilo di B
ax[1,1].plot(B_chi, prof_B)   #### CORREZIONE: B_chi contro prof_B
ax[1,1].plot([B_chi[B_sx], B_chi[B_sx]], [int(chi2_min-1), int(chi2_min+4)], color=line_c, ls='dashed')
ax[1,1].plot([B_chi[B_dx], B_chi[B_dx]], [int(chi2_min-1), int(chi2_min+4)], color=line_c, ls='dashed')
ax[1,1].text(B_chi[np.argmin(prof_B)], int(chi2_min+1),
             r'{g:.3e}'.format(g=B_chi[np.argmin(prof_B)]), color='k', alpha=0.5, fontsize=9)
ax[1,1].text(B_chi[B_sx], int(chi2_min+2),
             r'{g:.0e}'.format(g=errBB), color='b', alpha=0.5, fontsize=9)
ax[1,1].text(B_chi[B_dx], int(chi2_min+2),
             r'{g:.0e}'.format(g=-errB), color='r', alpha=0.5, fontsize=9)
ax[1,1].set_yticks([int(chi2_min), int(chi2_min+4), int(chi2_min+6)])

ax[1,0].set_axis_off()
ax[0,0].set_ylabel(r'$Q$-valore')
ax[1,1].set_xlabel(r'$\omega_0$ (Hz)', loc='center')
ax[0,0].set_xlim(int(chi2_min-1), int(chi2_min+4))
ax[1,1].set_ylim(int(chi2_min-1), int(chi2_min+4))

plt.savefig(file.replace('.txt','') + '_4.png',
            bbox_inches='tight', pad_inches=1, transparent=True,
            facecolor='w', edgecolor='w', orientation='Portrait', dpi=100)
plt.show()