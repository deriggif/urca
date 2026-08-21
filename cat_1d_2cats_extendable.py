#python3 -m venv ~/venvs/tensorflow
#source ~/venvs/tensorflow/bin/activate
import numpy as np
import tensorflow as tf
from scipy.interpolate import RegularGridInterpolator
import ROOT
import math
import matplotlib.pyplot as plt

N_categories = 3

def bernstein_basis(m, degree):
    m_min = tf.reduce_min(m)
    m_max = tf.reduce_max(m)

    x = (m - m_min) / (m_max - m_min + 1e-6)  # normalize to [0,1]
    x = tf.reshape(x, (-1, 1))  # [n_m,1]

    basis = []
    for k in range(degree + 1):
        coeff = math.comb(degree, k)
        term = coeff * (x**k) * ((1 - x)**(degree - k))
        basis.append(term)

    return tf.concat(basis, axis=1)  # [n_m, degree+1]

def project_th3_over_f2(th3, z_min=-np.inf, z_max=np.inf):

    h = th3.Clone()
    h.SetDirectory(0)

    # apply Y cut BEFORE projection
    zaxis = h.GetZaxis()
    zaxis.SetRangeUser(
        z_min,
        z_max
    )

    # project away Y (f1)
    proj = h.Project3D("yx")  # m vs f2

    x_centers = np.array([proj.GetXaxis().GetBinCenter(i+1)
                          for i in range(proj.GetNbinsX())])

    z_centers = np.array([proj.GetYaxis().GetBinCenter(j+1)
                          for j in range(proj.GetNbinsY())])

    values = np.zeros((len(x_centers), len(z_centers)))

    for i in range(len(x_centers)):
        for j in range(len(z_centers)):
            values[i, j] = proj.GetBinContent(i+1, j+1)

    return x_centers, z_centers, values



def fit_background_exponential(m, B_m_c, sb_mask_f, degree=2,
                                epsilon=1e-6, n_iter=15, tol=1e-8):
    """
    Fit di una forma esponenziale (polinomio esponenziale)
        mu(m) = exp( sum_k a_k * m^k ),  k = 0..degree
    al fondo nelle sideband, con pesi Poissoniani veri
    (IRLS / Fisher scoring, GLM Poisson a link canonico = log).

    Il link log e' canonico per Poisson: peso di Fisher w_i = mu_i,
    working response z_i = eta_i + (y_i - mu_i)/mu_i.

    Gestione bin a 0 eventi: con link identita' il floor era su
    var_i (per evitare pesi che esplodono quando mu->0). Con link
    log il rischio e' opposto: w_i = mu_i puo' collassare a 0 e
    sotto-pesare il bin. Per coerenza con la convenzione precedente
    (sigma_i = 1 per y_i = 0) si forza w_i = 1 in quei bin, ad ogni
    iterazione.
    """
    # basis: [1, m, m^2, ..., m^degree]
    X = tf.stack([tf.pow(m, k) for k in range(degree + 1)], axis=1)  # [n_m, K]
    n_m, N_cat = B_m_c.shape
    K = X.shape[1]
    dtype = X.dtype

    B_fit, B_err_fit = [], []

    for c in range(N_cat):
        y_c = B_m_c[:, c:c+1]  # [n_m, 1]
        zero_mask_c = tf.equal(y_c, 0.0)  # [n_m, 1]

        # inizializzazione: mu = max(y, eps), floor a 1 nei bin a zero
        # (cosi' log(mu) e' definito e la prima iterazione parte gia'
        # con il peso corretto)
        mu_c = tf.where(zero_mask_c, tf.ones_like(y_c), tf.maximum(y_c, epsilon))
        eta_c = tf.math.log(mu_c)

        XT_W_X = None
        for it in range(n_iter):
            # peso di Fisher: standard w = mu, floor a 1 sui bin a zero
            w_raw = tf.where(zero_mask_c, tf.ones_like(mu_c), mu_c)
            w_c = sb_mask_f[:, None] * w_raw  # [n_m, 1]
            w_c = w_c[:, 0]                   # [n_m]

            # working response z = eta + (y - mu)/mu_used
            # (mu_used = stesso floor usato nel peso, per coerenza)
            mu_used = tf.where(zero_mask_c, tf.ones_like(mu_c), mu_c)
            z_c = eta_c + (y_c - mu_c) / (mu_used + epsilon)

            Xw = X * tf.expand_dims(w_c, axis=1)
            Zw = z_c * tf.expand_dims(w_c, axis=1)

            XT = tf.transpose(X)
            XT_W_X = tf.matmul(XT, Xw) + 1e-6 * tf.eye(K, dtype=dtype)
            XT_W_Z = tf.matmul(XT, Zw)

            coeffs_c = tf.linalg.solve(XT_W_X, XT_W_Z)  # [K, 1]
            eta_new = tf.matmul(X, coeffs_c)
            eta_new = tf.clip_by_value(eta_new, -50.0, 50.0)  # evita overflow in exp
            mu_new = tf.exp(eta_new)
            mu_new = tf.maximum(mu_new, epsilon)

            if tf.reduce_max(tf.abs(mu_new - mu_c)) < tol:
                mu_c, eta_c = mu_new, eta_new
                break
            mu_c, eta_c = mu_new, eta_new

        B_fit.append(mu_c)  # [n_m, 1]

        # covarianza asintotica dei parametri (stessa W finale)
        XT_W_X_inv = tf.linalg.inv(XT_W_X)
        tmp = tf.matmul(X, XT_W_X_inv)
        # propagazione errore: mu = exp(eta) -> var(mu) ~= mu^2 * var(eta)
        var_eta = tf.reduce_sum(tmp * X, axis=1, keepdims=True)
        var_mu = tf.square(mu_c) * tf.maximum(var_eta, 0.0)
        sigma_y = tf.sqrt(var_mu)
        B_err_fit.append(sigma_y)

    B_fit = tf.concat(B_fit, axis=1)
    B_err_fit = tf.concat(B_err_fit, axis=1)
    return (B_fit, B_err_fit)


def main(th3_signal, th3_bkg):
    z_max = th3_signal.GetZaxis().GetXmax()
    z_min = 4
    m_centers, f1_centers, rho_signal_values = project_th3_over_f2( #mass centre value, f1 centre value, matrix of signal in 2D
        th3_signal, z_min=z_min, z_max=z_max
    )

    _, _, rho_bkg_values = project_th3_over_f2( 
        th3_background, z_min=z_min, z_max=z_max
    )

    rho_signal_tf = tf.convert_to_tensor(rho_signal_values, dtype=tf.float32)  # [n_m, n_f1]
    rho_bkg_tf    = tf.convert_to_tensor(rho_bkg_values, dtype=tf.float32)

    m_tf = tf.convert_to_tensor(m_centers, dtype=tf.float32)
    f1_tf = tf.convert_to_tensor(f1_centers.reshape(-1,1), dtype=tf.float32)
   
    f1_min = float(tf.reduce_min(f1_tf))
    print('f1_min')
    print(f1_min)
    f1_max = float(tf.reduce_max(f1_tf))
    print('f1_max')
    print(f1_max)

    cut_raw_1 = tf.Variable(0.6, dtype=tf.float32)
    cut_raw_2 = tf.Variable(0.9, dtype=tf.float32)

    trainable_vars = [cut_raw_1, cut_raw_2]

    #defining the window in which performing the fit

    m_low, m_high = 120.0, 130.0  # adjust if needed

    sr_mask = tf.logical_and(m_tf >= m_low, m_tf <= m_high)
    sb_mask = tf.logical_not(sr_mask)

    sr_mask_f = tf.cast(sr_mask, tf.float32)
    sb_mask_f = tf.cast(sb_mask, tf.float32)

    #defining the window in which performing the evaluation

    m_low_ev, m_high_ev = 123, 127.0  # adjust if needed

    sr_mask_ev = tf.logical_and(m_tf >= m_low, m_tf <= m_high)
    sb_mask_ev = tf.logical_not(sr_mask)

    sr_mask_f_ev = tf.cast(sr_mask, tf.float32)
    sb_mask_f_ev = tf.cast(sb_mask, tf.float32)

    # -----------------------------
    # Step 4: Training loop
    # -----------------------------
    n_epochs = 100

    loss_history = []
    metric_history = []

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)

    for epoch in range(n_epochs):
        with tf.GradientTape() as tape:

            # Project to (m, category)
            c_a = f1_min + (f1_max - f1_min) * tf.sigmoid(cut_raw_1)
            c_b = f1_min + (f1_max - f1_min) * tf.sigmoid(cut_raw_2)

            f1_cut_lo = tf.minimum(c_a, c_b)
            f1_cut_hi = tf.maximum(c_a, c_b)

            tau = 1e-3 * (f1_max - f1_min)

            s_lo = tf.sigmoid((f1_tf - f1_cut_lo) / tau)  # 0 sotto cut_lo, 1 sopra
            s_hi = tf.sigmoid((f1_tf - f1_cut_hi) / tau)  # 0 sotto cut_hi, 1 sopra

            p0 = 1.0 - s_lo          # categoria bassa purezza: f1 < cut_lo
            p1 = s_lo - s_hi         # categoria media: cut_lo <= f1 < cut_hi
            p2 = s_hi  
            p_f1 = tf.concat([p0, p1, p2], axis=-1)  # shape [n_f1, 3]

            rho_signal_tf = tf.cast(rho_signal_tf, tf.float32)
            rho_bkg_tf = tf.cast(rho_bkg_tf, tf.float32)
            p_f1 = tf.cast(p_f1, tf.float32)

            S_m_c = tf.matmul(rho_signal_tf, p_f1)
            B_m_c = tf.matmul(rho_bkg_tf, p_f1)
            # --- SAME AS BEFORE ---

            epsilon = 1e-15
            D_m_c = tf.identity(S_m_c)

            B_fit, B_err_fit = fit_background_exponential(m_tf, B_m_c, sb_mask_f)

            #D_sr = D_m_c * tf.expand_dims(sr_mask_f, axis=-1)
            D_sr = tf.boolean_mask(D_m_c, sr_mask_f, axis=0)

            #B_sr = B_fit * tf.expand_dims(sr_mask_f, axis=-1)
            B_sr = tf.boolean_mask(B_fit, sr_mask_f, axis=0)
            #B_side_bands_fit = B_fit * tf.expand_dims(sb_mask_f, axis=-1)
            B_side_bands_fit = tf.boolean_mask(B_fit, sb_mask_f, axis=0)
            #B_err_sr = B_err_fit * tf.expand_dims(sr_mask_f, axis=-1)
            B_err_sr = tf.boolean_mask(B_err_fit, sr_mask_f, axis=0)
            tf.print("B_err_sr", B_err_sr)
            #  np.sqrt(2 * ((signal + background) * np.log(1 + signal / background) - signal))
            #chi2_c = tf.reduce_sum((D_sr)**2 / (B_sr + B_err_sr*B_err_sr + epsilon), axis=0)
            #chi2_c = tf.reduce_sum((D_sr)**2 / (B_sr + epsilon), axis=0) #to add res-bkg error: add B_sr**2 * sigma_relative**2

            chi2_c = tf.reduce_sum(2 * (
                                    (D_sr + B_sr) * tf.math.log(
                                        ((D_sr + B_sr) * (B_sr + B_err_sr**2 )) /
                                        (B_sr**2 + (D_sr + B_sr) * B_err_sr**2)
                                    )
                                    - (B_sr**2 / (B_err_sr**2 + epsilon)) * tf.math.log(
                                        1 + D_sr * B_err_sr**2 / (B_sr * (B_sr + B_err_sr**2) + epsilon)
                                    )), axis=0)

            metric = tf.reduce_sum(chi2_c)
            loss = -metric

            # --- Balance penalty ---
            #frac_per_cat = tf.reduce_mean(p_f1, axis=0)
            #penalty = tf.reduce_sum(tf.nn.relu(0.05 - frac_per_cat))

            N_min = 10

            N_m_c = S_m_c + B_m_c
            N_m_c =  B_side_bands_fit
            N_per_cat = tf.reduce_sum(N_m_c, axis=0)

            penalty = tf.reduce_sum(tf.nn.relu( N_min - N_per_cat )) 

            loss += 0.1 * penalty

            S_counts = tf.reduce_sum(S_m_c * tf.expand_dims(sr_mask_f_ev, axis=-1), axis=0)
            B_counts = tf.reduce_sum(B_m_c * tf.expand_dims(sb_mask_f, axis=-1), axis=0)
            B_counts_sr = tf.reduce_sum(B_sr, axis=0)
            B_err_counts_sr = tf.reduce_sum(B_err_sr**2, axis=0)

            S_counts_np = S_counts.numpy()
            B_counts_np = B_counts.numpy()
            for c in range(N_categories):
                print(f"Epoch {epoch}: {m_low_ev}-{m_high_ev} GeV -  Category {c}: S={S_counts_np[c]:.7f}, B={B_counts_np[c]:.7f}  {chi2_c[c]:.7f} {B_counts_sr[c]:.7f}")
            print("metric", metric.numpy())

        # 4f. Apply gradients

        grads = tape.gradient(loss, trainable_vars)


        for g, v in zip(grads, trainable_vars):
            print(v.name, g)
        optimizer.apply_gradients(zip(grads, trainable_vars))
        print(f"[INFO] Learned f1 cuts ≈ {f1_cut_lo.numpy():.3f}, {f1_cut_hi.numpy():.3f}")

        print(f"Epoch {epoch}: loss = {loss.numpy():.4f}, metric = {metric.numpy():.4f}")
        loss_history.append(loss.numpy())
        metric_history.append(metric.numpy())


    # -----------------------------
    # After training: HARD category evaluation
    # -----------------------------

    plt.figure() 
    plt.plot(metric_history, label="Metric") 
    plt.xlabel("Epoch") 
    plt.ylabel("Value") 
    plt.legend() 
    plt.title("Training evolution") 
    plt.savefig("training.png")

    f1_cut_lo_val = f1_min + (f1_max - f1_min) * tf.sigmoid(cut_raw_1)
    f1_cut_hi_val = f1_min + (f1_max - f1_min) * tf.sigmoid(cut_raw_2)
    f1_cut_lo_val, f1_cut_hi_val = (tf.minimum(f1_cut_lo_val, f1_cut_hi_val),
                                    tf.maximum(f1_cut_lo_val, f1_cut_hi_val))

    hard_f1 = np.zeros_like(f1_centers, dtype=int)
    hard_f1[f1_centers >= f1_cut_hi_val.numpy()] = 2
    hard_f1[(f1_centers >= f1_cut_lo_val.numpy()) & (f1_centers < f1_cut_hi_val.numpy())] = 1
    # categoria 0 resta il default (f1 < cut_lo)

    # Signal region window

    S_mass = np.zeros((len(m_centers), N_categories))
    B_mass = np.zeros((len(m_centers), N_categories))

    S_hard = np.zeros(N_categories)
    B_hard = np.zeros(N_categories)

    for c in range(N_categories):
        mask = (hard_f1 == c).astype(np.float32)

        S_mass[:, c] = np.sum(rho_signal_values * mask, axis=1)
        B_mass[:, c] = np.sum(rho_bkg_values * mask, axis=1)

        # Integrate in evaluation window
        S_hard[c] = np.sum(S_mass[sr_mask_ev, c])
        B_hard[c] = np.sum(B_mass[sb_mask_ev, c])

    # Print counts per category
    print(f"Category-wise counts in {m_low_ev}-{m_high_ev} GeV (HARD cuts):")
    for c in range(N_categories):
        print(f"Category {c}: S = {S_hard[c]:.7f}, B = {B_hard[c]:.7f}")

    # Ensure B_fit_np is available
    B_fit_np = B_fit.numpy()          # shape (n_m, N_cat)
    B_err_fit_np = B_err_fit.numpy()  # shape (n_m, N_cat)

    m_low_sb, m_high_sb = 115, 135

    for c in range(N_categories):
        plt.figure(figsize=(6,5))

        # Masks
        inside_sr_mask = (m_centers >= m_low_sb) & (m_centers <= m_high_sb)
        outside_sr_mask = ~inside_sr_mask

        # Split sidebands into two continuous segments
        left_sb_mask = m_centers < m_low_sb
        right_sb_mask = m_centers > m_high_sb

        # 1️⃣ Signal + B_fit in SR
        plt.step(m_centers[inside_sr_mask],
                 S_mass[inside_sr_mask, c] + B_fit_np[inside_sr_mask, c],
                 where='mid', label='S + B_fit (SR)', color='C0')


        # 2️⃣ Real B only in sidebands (split)
        if np.any(left_sb_mask):
            plt.step(m_centers[left_sb_mask],
                     B_mass[left_sb_mask, c],
                     where='mid', color='C1', linestyle='--')
        if np.any(right_sb_mask):
            plt.step(m_centers[right_sb_mask],
                     B_mass[right_sb_mask, c],
                     where='mid', color='C1', linestyle='--', label='B (sidebands)')

        # 3️⃣ B_fit in the full range
        plt.step(m_centers[:],
                 B_fit_np[:, c],
                 where='mid', label='B_fit (full range)', color='C2')

        # 3️⃣bis banda d'errore ±1σ attorno a B_fit (fondo stimato dal fit)
        plt.fill_between(m_centers,
                          B_fit_np[:, c] - B_err_fit_np[:, c],
                          B_fit_np[:, c] + B_err_fit_np[:, c],
                          step='mid', color='C2', alpha=0.25,
                          label=r'B_fit $\pm 1\sigma$')

        # Highlight signal region
        plt.axvspan(m_low_sb, m_high_sb, alpha=0.2, color='gray', label='Signal Region')

        plt.xlabel("Mass [GeV]")
        plt.ylabel("Events")
        plt.title(f"Category {c} (HARD assignment)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"mass_spectra_hard_cat{c}.png")
        plt.close()


if __name__ == "__main__":
    # Paths
    bkg_path = "./th3_data.root"


    sig_path = "./th3_signal.root"

    # Open files
    f_bkg = ROOT.TFile.Open(bkg_path)
    f_sig = ROOT.TFile.Open(sig_path)

    if not f_bkg or f_bkg.IsZombie():
        raise RuntimeError(f"Cannot open background file: {bkg_path}")
    if not f_sig or f_sig.IsZombie():
        raise RuntimeError(f"Cannot open signal file: {sig_path}")

    # Retrieve TH3
    th3_background = f_bkg.Get("th3")
    th3_signal = f_sig.Get("th3")

    if not th3_background:
        raise RuntimeError("TH3 'th3' not found in background file")
    if not th3_signal:
        raise RuntimeError("TH3 'th3' not found in signal file")

    # Detach from file (important to avoid ROOT ownership issues)
    th3_background = th3_background.Clone("th3_background")
    th3_signal = th3_signal.Clone("th3_signal")

    th3_background.SetName("bkg_th3")
    th3_signal.SetName("sig_th3")

    th3_background.SetDirectory(0)
    th3_signal.SetDirectory(0)

    # -----------------------------
    # Rescale signal luminosity
    # -----------------------------
    lumi_data = 1.0
    lumi_signal = 1.0

    scale_factor = lumi_data / lumi_signal  # = 67
    th3_signal.Scale(scale_factor)

    print(f"[INFO] Signal scaled by factor {scale_factor}")

    # Pass filtered TH3 to main
    main(th3_signal, th3_background)