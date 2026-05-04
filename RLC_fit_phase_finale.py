import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import mplhep as hep
from cycler import cycler
import time

# ============================================================
# PARAMETRI MODIFICABILI DALL'UTENTE
# ============================================================

# --- File di input ---
# 3 colonne: frequenza (kHz)   Dt (ns)   scala_Dt (ns)
file = 'RLC_Rphase.txt'

# --- Stime iniziali per il fit ---
# omega0 = 2*pi*f0  con f0 in kHz -> rad/s
B_init = 2.0 * np.pi * 18.0e3    # omega0 in rad/s  (esempio: f0 ~ 18 kHz)
C_init = 3.5                      # Q-valore

# --- Intervallo di frequenza per il fit (kHz) ---
frfit0 =  0.0      # limite inferiore (0 = tutto)
frfit1 = 1e6       # limite superiore

# --- Scansione del chi2 ---
n_sigma_scan = 2   # semiampiezza in unità di sigma del fit
step_scan    = 100 # punti per parametro (griglia step x step)

# --- Debug ---
DEB = False

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
    'figure.constrained_layout.use': True,
}
plt.rcParams.update(params)
plt.rcParams['axes.prop_cycle'] = cycler(color=['b','g','r','c','m','y','k'])

# ============================================================
# FUNZIONE DI FIT
# ============================================================
# Sfasamento ai capi di R nel circuito RLC serie:
#
#   Delta_phi_R(omega) = arctan[ Q * (omega/omega0 - omega0/omega) ]
#
# Parametri:
#   x  : frequenza in kHz
#   B  : omega0 = 1/sqrt(LC)  in rad/s
#   C  : Q-valore (adimensionale)
#
# NB: per Q -> inf la curva diventa uno scalino (±pi/2 -> 0 -> -pi/2),
#     per Q piccolo il passaggio da +pi/2 a -pi/2 è più graduale.

def fitf_phase_R(x, B, C):
    """
    Sfasamento Delta_phi_R in radianti.
    x  : frequenza in kHz
    B  : omega0 in rad/s
    C  : Q-valore
    """
    omega = 2.0 * np.pi * x * 1e3   # kHz -> rad/s
    return np.arctan(C * (omega / B - B / omega))

# ============================================================
# CARICAMENTO DATI
# ============================================================
data        = np.loadtxt(file)
fr_all      = data[:, 0]    # kHz
Dt_ns       = data[:, 1]    # ns
scala_Dt_ns = data[:, 2]    # ns  (fondoscala temporale)

# Sfasamento in radianti: phi = 2*pi*f[Hz]*Dt[s]
phi_all = 2.0 * np.pi * (fr_all * 1e3) * (Dt_ns * 1e-9)

# Incertezza su Dt (distribuzione triangolare, differenza di 2 letture)
# sigma_Dt = sqrt(2) * (scala/10) * 0.41   [ns]
eDt_ns  = np.sqrt(2) * (scala_Dt_ns / 10.0) * 0.41

# Propagazione su phi = 2*pi*f*Dt  (f noto con precisione trascurabile)
ephi_all = 2.0 * np.pi * (fr_all * 1e3) * (eDt_ns * 1e-9)

if DEB:
    print("Esempio Dt (ns):", Dt_ns[:5])
    print("Esempio eDt (ns):", eDt_ns[:5])
    print("Esempio phi (rad):", phi_all[:5])
    print("Esempio ephi (rad):", ephi_all[:5])

# ============================================================
# GRAFICO 1: dati completi con errori
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
ax.errorbar(fr_all, phi_all, yerr=ephi_all, fmt='o', ms=2, color='blue',
            label=r'$\Delta\phi_R(f)$')
ax.axhline( np.pi/2, color='gray', ls='dashed', lw=0.8)
ax.axhline(-np.pi/2, color='gray', ls='dashed', lw=0.8)
ax.axhline(0,        color='gray', ls='dotted', lw=0.8)
ax.set_xlabel(r'Frequenza (kHz)')
ax.set_ylabel(r'$\Delta\phi_R$ (rad)')
ax.legend(loc='best', prop={'size': 10})
ax.set_title(r'Sfasamento su R -- dati completi')

plt.savefig(file.replace('.txt', '') + '_1.png',
            bbox_inches='tight', transparent=True,
            facecolor='w', edgecolor='w', dpi=100)
plt.show()

# ============================================================
# MASCHERA DI FREQUENZA PER IL FIT
# ============================================================
mask = (fr_all >= frfit0) & (fr_all <= frfit1)
fr   = fr_all[mask]
phi  = phi_all[mask]
ephi = ephi_all[mask]
N    = len(fr)

# ============================================================
# FIT CON SCIPY
# ============================================================
popt, pcov = curve_fit(fitf_phase_R, fr, phi,
                       p0=[B_init, C_init],
                       method='lm', sigma=ephi, absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))

B_BF, C_BF = popt
eB_BF, eC_BF = perr

# Residui e chi2
residuA  = phi - fitf_phase_R(fr, *popt)
chisq    = np.sum((residuA / ephi)**2)
df_fit   = N - 2
chisq_rid = chisq / df_fit

f0_BF  = B_BF / (2.0 * np.pi * 1e3)   # kHz
ef0_BF = eB_BF / (2.0 * np.pi * 1e3)

print("============== BEST FIT with SciPy ====================")
print(f'omega0 = ({B_BF:.5e} +/- {eB_BF:.1e}) rad/s')
print(f'f0     = ({f0_BF:.4f} +/- {ef0_BF:.4f}) kHz')
print(f'Q      = ({C_BF:.4f} +/- {eC_BF:.4f})')
print(f'chi2   = {chisq:.2f}   chi2/ndf = {chisq_rid:.3f}  (ndf={df_fit})')
print("=======================================================")

# ============================================================
# GRAFICO 2: fit + residui (SciPy)
# ============================================================
x_fit = np.linspace(min(fr), max(fr), 1000)

fig, ax = plt.subplots(2, 1, figsize=(5, 4), sharex=True,
                       constrained_layout=True, height_ratios=[2, 1])

ax[0].plot(x_fit, fitf_phase_R(x_fit, *popt),
           label='Fit (SciPy)', ls='--', color='black')
ax[0].plot(x_fit, fitf_phase_R(x_fit, B_init, C_init),
           label='Guess iniziale', ls='dashed', color='green')
ax[0].errorbar(fr, phi, yerr=ephi, fmt='o', ms=2, color='blue',
               label=r'$\Delta\phi_R$')
ax[0].axhline( np.pi/4,  color='gray', ls='dotted', lw=0.8)
ax[0].axhline(-np.pi/4,  color='gray', ls='dotted', lw=0.8)
ax[0].axhline(0,         color='gray', ls='dotted', lw=0.8)
ax[0].set_ylabel(r'$\Delta\phi_R$ (rad)')
ax[0].legend(loc='best', prop={'size': 8})

ax[1].errorbar(fr, residuA, yerr=ephi, fmt='o', ms=2, color='blue',
               label='Residui')
ax[1].axhline(0, color='black', lw=0.8)
ax[1].set_ylabel(r'Residui (rad)')
ax[1].set_xlabel(r'Frequenza (kHz)')

plt.savefig(file.replace('.txt', '') + '_2.png',
            bbox_inches='tight', transparent=True,
            facecolor='w', edgecolor='w', dpi=100)
plt.show()

# ============================================================
# SCANSIONE VETTORIZZATA DEL CHI2  (griglia 2D: omega0 x Q)
# ============================================================
B0, B1 = B_BF - n_sigma_scan * eB_BF, B_BF + n_sigma_scan * eB_BF
C0, C1 = C_BF - n_sigma_scan * eC_BF, C_BF + n_sigma_scan * eC_BF

step  = step_scan
B_chi = np.linspace(B0, B1, step)   # omega0
C_chi = np.linspace(C0, C1, step)   # Q

print(f"\nScansione chi2 su griglia {step}x{step} = {step**2} punti ...")
t0 = time.time()

# Broadcasting puro: niente loop Python
# phi   : (N,)          -> (N, 1, 1)
# B_chi : (step,)       -> (1, step, 1)   asse = omega0
# C_chi : (step,)       -> (1, 1, step)   asse = Q
omega  = (2.0 * np.pi * fr * 1e3)[:, None, None]   # (N,1,1)
B4     = B_chi[None, :, None]                        # (1,step,1)
C4     = C_chi[None, None, :]                        # (1,1,step)
model  = np.arctan(C4 * (omega / B4 - B4 / omega))  # (N,step,step)

phi4   = phi[:, None, None]
ephi4  = ephi[:, None, None]
mappa  = np.sum(((phi4 - model) / ephi4)**2, axis=0)  # (step_B, step_C)

print(f"Scansione completata in {time.time()-t0:.2f} s")

chi2_min    = np.min(mappa)
idx_min     = np.unravel_index(np.argmin(mappa), mappa.shape)  # (iB, iC)
iB_min, iC_min = idx_min

# Verifica
omega_v   = 2.0 * np.pi * fr * 1e3
model_min = np.arctan(C_chi[iC_min] * (omega_v / B_chi[iB_min] - B_chi[iB_min] / omega_v))
chi2_check = np.sum(((phi - model_min) / ephi)**2)

print(f"chi2 minimo dalla mappa: {chi2_min:.4f}  (verifica: {chi2_check:.4f})")
print(f"omega0 (mappa) = {B_chi[iB_min]:.5e} rad/s  "
      f"f0 = {B_chi[iB_min]/(2*np.pi*1e3):.4f} kHz")
print(f"Q      (mappa) = {C_chi[iC_min]:.4f}")

# ============================================================
# GRAFICO 3: fit dal minimo chi2 + residui
# ============================================================
fig, ax = plt.subplots(2, 1, figsize=(5, 4), sharex=True,
                       constrained_layout=True, height_ratios=[2, 1])

ax[0].plot(x_fit,
           fitf_phase_R(x_fit, B_chi[iB_min], C_chi[iC_min]),
           label=r'Fit ($\chi^2$ min)', ls='--', color='blue')
ax[0].errorbar(fr, phi, yerr=ephi, fmt='o', ms=2, color='blue',
               label=r'$\Delta\phi_R$')
ax[0].axhline( np.pi/4,  color='gray', ls='dotted', lw=0.8)
ax[0].axhline(-np.pi/4,  color='gray', ls='dotted', lw=0.8)
ax[0].axhline(0,         color='gray', ls='dotted', lw=0.8)
ax[0].set_ylabel(r'$\Delta\phi_R$ (rad)')
ax[0].legend(loc='best', prop={'size': 8})

residui_chi2 = phi - model_min
ax[1].errorbar(fr, residui_chi2, yerr=ephi, fmt='o', ms=2, color='blue')
ax[1].axhline(0, color='black', lw=0.8)
ax[1].set_ylabel(r'Residui (rad)')
ax[1].set_xlabel(r'Frequenza (kHz)')

plt.savefig(file.replace('.txt', '') + '_3.png',
            bbox_inches='tight', transparent=True,
            facecolor='w', edgecolor='w', dpi=100)
plt.show()

# ============================================================
# PROFILI 1D DEL CHI2 E CALCOLO ERRORI
# ============================================================
prof_B = mappa.min(axis=1)   # profilo di omega0 (minimizzato su Q)
prof_C = mappa.min(axis=0)   # profilo di Q      (minimizzato su omega0)

lvl = chi2_min + 1.0

# Errori su omega0
diff_B = np.abs(prof_B - lvl)
B_dx   = np.argmin(diff_B[B_chi < B_BF])
B_sx   = np.argmin(diff_B[B_chi > B_BF]) + len(B_chi[B_chi < B_BF])
errB   = B_chi[iB_min] - B_chi[B_dx]
errBB  = B_chi[B_sx]   - B_chi[iB_min]

# Errori su Q
diff_C = np.abs(prof_C - lvl)
C_dx   = np.argmin(diff_C[C_chi < C_BF])
C_sx   = np.argmin(diff_C[C_chi > C_BF]) + len(C_chi[C_chi < C_BF])
errC   = C_chi[iC_min] - C_chi[C_dx]
errCC  = C_chi[C_sx]   - C_chi[iC_min]

ef0_dx = errB  / (2.0 * np.pi * 1e3)
ef0_sx = errBB / (2.0 * np.pi * 1e3)

print("============== BEST FIT with chi2 ====================")
print(f'omega0 = ({B_chi[iB_min]:.5e} - {errB:.1e} + {errBB:.1e}) rad/s')
print(f'f0     = ({B_chi[iB_min]/(2*np.pi*1e3):.4f} - {ef0_dx:.4f} + {ef0_sx:.4f}) kHz')
print(f'Q      = ({C_chi[iC_min]:.4f} - {errC:.4f} + {errCC:.4f})')
print(f'chi2   = {chi2_min:.2f}')
print("=======================================================")

# ============================================================
# GRAFICO 4: mappa chi2(omega0, Q) con profili 1D
# ============================================================
cmap  = mpl.colormaps['plasma'].reversed()
line_c = 'gray'

level = np.linspace(np.min(mappa), np.max(mappa), 100)

fig, ax = plt.subplots(2, 2, figsize=(5.5, 5), constrained_layout=True,
                       height_ratios=[3, 1], width_ratios=[1, 3],
                       sharex='col', sharey='row')
fig.suptitle(r'$\chi^2\left(\omega_0,\, Q\right)$')

# --- Pannello principale: contour 2D (omega0 su x, Q su y) ---
im = ax[0, 1].contourf(B_chi, C_chi, mappa.T, levels=level, cmap=cmap)
cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=ax[0, 1],
                    ticks=[int(chi2_min),
                           int(chi2_min + 2),
                           int(chi2_min + 4),
                           int(chi2_min + 6)])
cbar.set_label(r'$\chi^2$', rotation=360)

CS = ax[0, 1].contour(B_chi, C_chi, mappa.T,
                      levels=[chi2_min + 0.0001,
                               chi2_min + 1.0,
                               chi2_min + 2.3,
                               chi2_min + 3.8],
                      linewidths=1, colors='k', alpha=0.5, linestyles='dotted')
ax[0, 1].clabel(CS, inline=True, fontsize=9, fmt='%.1f')
ax[0, 1].text(B_chi[iB_min], C_chi[iC_min],
              fr'{chi2_min:.0f}', color='k', alpha=0.5, fontsize=9)

# Linee di errore
ax[0, 1].plot([B0, B1], [C_chi[C_sx], C_chi[C_sx]], color=line_c, ls='dashed')
ax[0, 1].plot([B0, B1], [C_chi[C_dx], C_chi[C_dx]], color=line_c, ls='dashed')
ax[0, 1].plot([B_chi[B_sx], B_chi[B_sx]], [C0, C1], color=line_c, ls='dashed')
ax[0, 1].plot([B_chi[B_dx], B_chi[B_dx]], [C0, C1], color=line_c, ls='dashed')

ax[0, 1].set_xlabel(r'$\omega_0$ (rad/s)')
ax[0, 1].xaxis.set_label_position('top')
ax[0, 1].xaxis.tick_top()

# --- Pannello sinistro: profilo di Q (asse y) ---
ax[0, 0].plot(prof_C, C_chi, ls='-')
ax[0, 0].plot([chi2_min - 1, chi2_min + 4],
              [C_chi[C_sx], C_chi[C_sx]], color=line_c, ls='dashed')
ax[0, 0].plot([chi2_min - 1, chi2_min + 4],
              [C_chi[C_dx], C_chi[C_dx]], color=line_c, ls='dashed')
ax[0, 0].set_xticks([int(chi2_min), int(chi2_min + 1), int(chi2_min + 4)])
ax[0, 0].set_xlim(chi2_min - 1, chi2_min + 4)
ax[0, 0].set_ylabel(r'$Q$')
ax[0, 0].text(chi2_min + 1.2, C_chi[iC_min],
              fr'{C_chi[iC_min]:.3f}', color='k', alpha=0.6, fontsize=8)
ax[0, 0].text(chi2_min + 1.2, C_chi[C_sx],
              fr'$+{errCC:.3f}$', color='b', alpha=0.6, fontsize=8)
ax[0, 0].text(chi2_min + 1.2, C_chi[C_dx],
              fr'$-{errC:.3f}$', color='r', alpha=0.6, fontsize=8)

# --- Pannello in basso: profilo di omega0 (asse x) ---
ax[1, 1].plot(B_chi, prof_B)
ax[1, 1].plot([B_chi[B_sx], B_chi[B_sx]], [chi2_min - 1, chi2_min + 4],
              color=line_c, ls='dashed')
ax[1, 1].plot([B_chi[B_dx], B_chi[B_dx]], [chi2_min - 1, chi2_min + 4],
              color=line_c, ls='dashed')
ax[1, 1].set_yticks([int(chi2_min), int(chi2_min + 1), int(chi2_min + 4)])
ax[1, 1].set_ylim(chi2_min - 1, chi2_min + 4)
ax[1, 1].set_xlabel(r'$\omega_0$ (rad/s)')
ax[1, 1].text(B_chi[iB_min], chi2_min + 1.5,
              fr'{B_chi[iB_min]:.3e}', color='k', alpha=0.6, fontsize=8)
ax[1, 1].text(B_chi[B_sx], chi2_min + 2.5,
              fr'$+{errBB:.2e}$', color='b', alpha=0.6, fontsize=8)
ax[1, 1].text(B_chi[B_dx], chi2_min + 2.5,
              fr'$-{errB:.2e}$', color='r', alpha=0.6, fontsize=8)

ax[1, 0].set_axis_off()

plt.savefig(file.replace('.txt', '') + '_4.png',
            bbox_inches='tight', transparent=True,
            facecolor='w', edgecolor='w', dpi=100)
plt.show()