import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import base64
import tempfile
import os
import warnings
warnings.filterwarnings("ignore")
try:
    from scipy.stats import norm, chi2
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

st.set_page_config(page_title="Validation L-Dopa complète", layout="wide")
st.title("🔬 Validation bioanalytique L-Dopa — Application finale complète")

def norm_ppf(p):
    if SCIPY_AVAILABLE:
        return norm.ppf(p)
    return math.sqrt(2) * erfinv(2*p - 1)
def erfinv(y):
    a = 0.147
    sign = 1 if y >= 0 else -1
    ln = math.log(1 - y*y)
    first = 2/(math.pi*a) + ln/2
    second = ln/a
    return sign * math.sqrt(math.sqrt(first*first - second) - first)
def detect_rep_cols_strict(df):
    reps = []
    for c in df.columns:
        low = c.lower()
        if low.startswith("rep") or low.startswith("r") and any(ch.isdigit() for ch in c):
            if "serie" in low:
                continue
            reps.append(c)
    if len(reps) < 2:
        exclude = {c.lower() for c in df.columns if c.lower() in ("serie","sample","id","qc","concentration_nominale","concentration","cible","valeur_nominale")}
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in exclude]
        reps = numeric[:min(len(numeric), 8)]
    return reps
def pooled_within_sd(df_reps):
    run_ns = df_reps.notna().sum(axis=1).values
    run_sds = df_reps.std(axis=1, ddof=1).fillna(0).values
    numerator = ((run_ns - 1) * (run_sds**2)).sum()
    denom = run_ns.sum() - len(run_ns)
    if denom <= 0 or numerator < 0:
        return np.nan
    return np.sqrt(numerator / denom)
def intermediate_sd_from_components(df_reps, pooled_within):
    if df_reps.shape[0] < 2 or np.isnan(pooled_within):
        return np.nan
    means = df_reps.mean(axis=1, skipna=True).values
    n_rep_mean = df_reps.notna().sum(axis=1).mean()
    if n_rep_mean <= 0:
        return np.nan
    var_series_means = np.nanvar(means, ddof=1)
    between_var = max(0.0, var_series_means - (pooled_within**2) / n_rep_mean)
    return np.sqrt(between_var + pooled_within**2)
def safe_pdf_text(s):
    if s is None:
        return ""
    s = str(s)
    s = s.replace("λ", "lambda").replace("β", "beta").replace("γ", "gamma").replace("±", "+/-")
    for char in [";", ",", ":", "."]:
        s = s.replace(char, char + " ")
    return s
def df_to_csv_link(df, name="resultats.csv"):
    b = df.to_csv(index=False).encode('utf-8')
    b64 = base64.b64encode(b).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{name}">📥 Télécharger CSV</a>'

with st.sidebar:
    st.header("Paramètres")
    lambda_accept = st.number_input("± Limite d'acceptabilité λ (%)", value=15.0, step=0.5)
    risk_limit = st.number_input("Limite du risque (%)", value=5.0, step=0.5)
    beta_expect = st.number_input("β pour intervalle d'expectation", value=0.90, step=0.01, format="%.2f")
    beta_uncert = st.number_input("β pour intervalle d'incertitude", value=0.667, step=0.001, format="%.3f")
    gamma_uncert = st.number_input("γ pour intervalle d'incertitude", value=0.90, step=0.01, format="%.2f")
    st.markdown("---")
    st.subheader("Monte Carlo")
    mc_runs = st.number_input("Nombre de simulations Monte Carlo", value=2000, step=100)
    mc_show_hist = st.checkbox("Afficher histogrammes Monte Carlo", value=True)
    st.markdown("---")
    st.subheader("PDF")
    make_pdf = st.checkbox("Autoriser génération PDF (fpdf)", value=True)

tabs = st.tabs([
    "📊 Données",
    "✅ Résultats",
    "📈 Graphiques",
    "🤖 Prédiction",
    "🎲 Monte Carlo",
    "💬 Assistant",
    "🧾 Rapport PDF"
])

with tabs[0]:
    st.header("📊 Importer les données")
    uploaded = st.file_uploader("Uploader fichier Excel (.xlsx/.xls) ou CSV", type=["xlsx","xls","csv"])
    if uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
            else:
                df_raw = pd.read_excel(uploaded)
            st.success("Fichier chargé.")
            st.dataframe(df_raw.head(20))
            st.session_state["df_raw"] = df_raw
            rep_cols = detect_rep_cols_strict(df_raw)
            st.write("Colonnes répétitions détectées :", rep_cols)
        except Exception as e:
            st.error(f"Erreur lecture fichier : {e}")

with tabs[1]:
    st.header("✅ Calculs & Table Résultats")
    if "df_raw" not in st.session_state:
        st.info("Téléverse un fichier dans l'onglet Données pour commencer.")
    else:
        df_raw = st.session_state["df_raw"].copy()
        conc_candidates = [c for c in df_raw.columns if c.lower() in ("concentration_nominale","concentration","cible","conc","valeur_nominale","nominal")]
        default_conc = conc_candidates[0] if conc_candidates else df_raw.select_dtypes(include=[np.number]).columns[0]
        conc_col = st.selectbox("Colonne concentration nominale", options=list(df_raw.columns), index=list(df_raw.columns).index(default_conc))
        df_raw = df_raw.rename(columns={conc_col: "Concentration_nominale"})
        auto_reps = detect_rep_cols_strict(df_raw)
        chosen_reps = st.multiselect("Colonnes répétitions (strict)", options=list(df_raw.columns), default=auto_reps)
        if len(chosen_reps) < 2:
            st.warning("Au moins 2 colonnes répétitions (ex: Rep1, Rep2) requises.")
            st.stop()
        results = []
        for conc, g in df_raw.groupby("Concentration_nominale", sort=True):
            df_reps = g[chosen_reps].apply(pd.to_numeric, errors="coerce")
            reps_flat = df_reps.values.flatten()
            reps_flat = reps_flat[~np.isnan(reps_flat)]
            if len(reps_flat) == 0:
                continue
            mean = np.nanmean(reps_flat)
            bias = mean - conc
            bias_rel = (bias/conc*100) if conc != 0 else np.nan
            pooled_sd = pooled_within_sd(df_reps)
            repeat_cv = (100 * pooled_sd / conc) if (not np.isnan(pooled_sd) and conc !=0) else np.nan
            inter_sd = intermediate_sd_from_components(df_reps, pooled_sd) if not np.isnan(pooled_sd) else np.nan
            inter_cv = (100 * inter_sd / conc) if (not np.isnan(inter_sd) and conc !=0) else np.nan
            n_total = len(reps_flat)
            sample_sd = np.std(reps_flat, ddof=1) if n_total>1 else np.nan
            if n_total>1 and not np.isnan(sample_sd):
                z_beta = norm_ppf((1 + beta_expect)/2)
                tol_low = mean - z_beta * sample_sd
                tol_high = mean + z_beta * sample_sd
                tol_low_pct = (tol_low- conc)/conc*100
                tol_high_pct = (tol_high- conc)/conc*100
            else:
                tol_low_pct = tol_high_pct = np.nan
            if n_total>1 and not np.isnan(sample_sd):
                df_n = n_total - 1
                lower_prob = (1 - gamma_uncert)
                chi2_low = chi2.ppf(lower_prob, df_n) if 0 < lower_prob < 1 else chi2.ppf(0.1, df_n)
                if chi2_low > 0:
                    z_b = norm_ppf((1 + beta_uncert)/2)
                    k_factor = z_b * np.sqrt(df_n / chi2_low) * np.sqrt(1 + 1/n_total)
                    uncert_low = mean - k_factor * sample_sd
                    uncert_high = mean + k_factor * sample_sd
                    uncert_low_pct = (uncert_low- conc)/conc*100
                    uncert_high_pct = (uncert_high- conc)/conc*100
                else:
                    uncert_low_pct = uncert_high_pct = np.nan
            else:
                uncert_low_pct = uncert_high_pct = np.nan
            limit_low = conc*(1 - lambda_accept/100)
            limit_high = conc*(1 + lambda_accept/100)
            if (not np.isnan(inter_sd)) and inter_sd>0:
                if SCIPY_AVAILABLE:
                    p_inside = norm.cdf(limit_high, loc=mean, scale=inter_sd) - norm.cdf(limit_low, loc=mean, scale=inter_sd)
                    risk = 1 - p_inside
                else:
                    risk = np.sum(np.abs(reps_flat - conc)/conc*100 > lambda_accept)/len(reps_flat)
            else:
                risk = np.sum(np.abs(reps_flat - conc)/conc*100 > lambda_accept)/len(reps_flat)
            conclusion = "Conforme" if risk <= (risk_limit/100) else "Non conforme"
            results.append({
                "Concentration_nominale": conc,
                "Moyenne": mean,
                "Biais": bias,
                "Biais_rel (%)": bias_rel,
                "Recouvrement (%)": (mean/conc)*100 if conc!=0 else np.nan,
                "Répétabilité_SD": pooled_sd,
                "Répétabilité_CV (%)": repeat_cv,
                "Inter_SD": inter_sd,
                "Inter_CV (%)": inter_cv,
                "Tol_low (β) [%]": tol_low_pct,
                "Tol_high (β) [%]": tol_high_pct,
                "Uncert_low [%]": uncert_low_pct,
                "Uncert_high [%]": uncert_high_pct,
                "Risque (%)": risk*100,
                "Conclusion": conclusion
            })
        res_df = pd.DataFrame(results).sort_values("Concentration_nominale").reset_index(drop=True)
        if res_df.empty:
            st.error("Aucun résultat calculable — vérifie les colonnes Rep* sélectionnées.")
        else:
            display = res_df.copy()
            numcols = display.select_dtypes(include=[np.number]).columns
            display[numcols] = display[numcols].round(6)
            st.subheader("Résultats par niveau")
            st.dataframe(display)
            st.markdown(df_to_csv_link(display, "resultats_validation.csv"), unsafe_allow_html=True)
            st.session_state["res_df"] = res_df

with tabs[2]:
    st.header("📈 Graphiques")
    if "res_df" not in st.session_state:
        st.info("Calculer les résultats d'abord.")
    else:
        r = st.session_state["res_df"]

        # 1. Profil d'exactitude
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(r["Concentration_nominale"], r["Biais_rel (%)"]+100, 'o-', label="Recouvrement (%)")
        if not r["Tol_low (β) [%]"].isna().all():
            ax.plot(r["Concentration_nominale"], r["Tol_low (β) [%]"], 's--', label=f"Tol_low (β={beta_expect})")
            ax.plot(r["Concentration_nominale"], r["Tol_high (β) [%]"], 's--', label=f"Tol_high (β={beta_expect})")
        if not r["Uncert_low [%]"].isna().all():
            ax.fill_between(r["Concentration_nominale"], r["Uncert_low [%]"], r["Uncert_high [%]"], alpha=0.25, label=f"Inc. (β={beta_uncert}, γ={gamma_uncert})")
        ax.axhline(100 + lambda_accept, color='red', linestyle='--', label="Limite acceptabilité haute")
        ax.axhline(100 - lambda_accept, color='red', linestyle='--', label="Limite acceptabilité basse")
        ax.set_xscale('log')
        ax.set_ylim(80,120)
        ax.set_xlabel("Concentration nominale")
        ax.set_ylabel("%")
        ax.set_title("Profil d'exactitude")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # 2. Profil d'incertitude
        fig_inc, ax_inc = plt.subplots(figsize=(8,4))
        ax_inc.plot(r["Concentration_nominale"], r["Uncert_low [%]"], 'v--', color='green', label="Limite basse d'incertitude (%)")
        ax_inc.plot(r["Concentration_nominale"], r["Uncert_high [%]"], '^--', color='blue', label="Limite haute d'incertitude (%)")
        ax_inc.fill_between(r["Concentration_nominale"], r["Uncert_low [%]"], r["Uncert_high [%]"], alpha=0.25, color='lightgreen', label="Bande d'incertitude")
        ax_inc.axhline(-lambda_accept, color='red', linestyle='--', label="Limite basse d'acceptabilité")
        ax_inc.axhline(lambda_accept, color='red', linestyle='--', label="Limite haute d'acceptabilité")
        ax_inc.set_xlabel("Concentration nominale")
        ax_inc.set_ylabel("Incertitude relative (%)")
        ax_inc.set_title("Profil d'incertitude de la méthode")
        ax_inc.legend()
        ax_inc.grid(True)
        st.pyplot(fig_inc)

        # 3. Profil de précision (CV%)
        fig_cv, ax_cv = plt.subplots(figsize=(8,4))
        ax_cv.plot(r["Concentration_nominale"], r["Répétabilité_CV (%)"], 's-', label="Répétabilité CV%")
        ax_cv.plot(r["Concentration_nominale"], r["Inter_CV (%)"], 'o-', label="Inter CV%")
        ax_cv.set_xscale('log')
        ax_cv.set_xlabel("Concentration nominale")
        ax_cv.set_ylabel("CV (%)")
        ax_cv.set_title("Profil de précision")
        ax_cv.legend()
        ax_cv.grid(True)
        st.pyplot(fig_cv)


with tabs[3]:
    st.header("🤖 Module de prédiction (biais & CV%)")
    if "res_df" not in st.session_state or len(st.session_state["res_df"])<2:
        st.info("Générer d'abord les résultats.")
    else:
        train = st.session_state["res_df"].dropna(subset=["Biais_rel (%)","Inter_CV (%)"])
        X = train[["Concentration_nominale"]].values.reshape(-1,1)
        use_log = (X.max()/X.min() > 10) if X.min() > 0 else False
        Xtr = np.log(X) if use_log else X
        if SKLEARN_AVAILABLE:
            lr_bias = LinearRegression().fit(Xtr, train["Biais_rel (%)"].values)
            lr_cv = LinearRegression().fit(Xtr, train["Inter_CV (%)"].values)
            new_conc = st.number_input("Nouvelle concentration à prédire", value=float(train["Concentration_nominale"].median()))
            if st.button("Prédire biais & CV"):
                xpred = np.log([[new_conc]]) if use_log else [[new_conc]]
                pred_bias = lr_bias.predict(xpred)[0]
                pred_cv = lr_cv.predict(xpred)[0]
                st.success(f"Biais relatif prédit : {pred_bias:.3f} %")
                st.success(f"Inter CV prédit : {pred_cv:.3f} %")
        else:
            st.warning("scikit-learn non installé : prédiction désactivée.")

with tabs[4]:
    st.header("🎲 Monte Carlo (robustesse & probabilité de conformité)")
    if "res_df" not in st.session_state:
        st.info("Générer d'abord les résultats.")
    else:
        if st.button("Exécuter Monte Carlo"):
            res_df = st.session_state["res_df"]
            mc_out = {}
            for _, row in res_df.iterrows():
                conc = row["Concentration_nominale"]
                mean = row["Moyenne"]
                sigma = row["Inter_SD"] if (not np.isnan(row["Inter_SD"]) and row["Inter_SD"]>0) else row["Répétabilité_SD"]
                if np.isnan(sigma) or sigma <= 0:
                    mc_out[conc] = {"pct_in": np.nan, "pct_out": np.nan, "sims": None}
                    continue
                sims = np.random.normal(loc=mean, scale=sigma, size=int(mc_runs))
                pct_in = np.mean((sims >= conc*(1 - lambda_accept/100)) & (sims <= conc*(1 + lambda_accept/100))) * 100
                pct_out = 100 - pct_in
                mc_out[conc] = {"pct_in": pct_in, "pct_out": pct_out, "sims": sims}
            st.session_state["mc_out"] = mc_out
            st.success("Monte Carlo terminé.")
        if "mc_out" in st.session_state:
            table = []
            for conc, info in st.session_state["mc_out"].items():
                table.append({"Concentration_nominale": conc, "Pct_in_%": info["pct_in"], "Pct_out_%": info["pct_out"]})
            mc_df = pd.DataFrame(table).sort_values("Concentration_nominale")
            st.dataframe(mc_df.round(3))
            if mc_show_hist:
                for conc, info in st.session_state["mc_out"].items():
                    if info["sims"] is None:
                        continue
                    fig, ax = plt.subplots(figsize=(6,3))
                    ax.hist(info["sims"]/conc*100, bins=40)
                    ax.axvline(100 + lambda_accept, color='red', linestyle='--')
                    ax.axvline(100 - lambda_accept, color='red', linestyle='--')
                    ax.set_title(f"MC % du nominal (Conc={conc})")
                    ax.set_xlabel("% du nominal")
                    st.pyplot(fig)

with tabs[5]:
    st.header("💬 Assistant automatique")
    if "res_df" not in st.session_state:
        st.info("Générer d'abord les résultats.")
    else:
        df = st.session_state["res_df"]
        mean_bias_abs = df["Biais_rel (%)"].abs().mean()
        mean_inter_cv = df["Inter_CV (%)"].mean()
        mean_risk = df["Risque (%)"].mean()
        n_levels = len(df)
        n_conformes = (df["Conclusion"]=="Conforme").sum()
        lines = []
        lines.append(f"Analyse automatique sur {n_levels} niveau(x).")
        lines.append(f"- Biais relatif moyen (abs) : {mean_bias_abs:.2f} %")
        lines.append(f"- CV intermédiaire moyen : {mean_inter_cv:.2f} %")
        lines.append(f"- Risque moyen observé : {mean_risk:.2f} %")
        lines.append(f"- Niveaux conformes : {n_conformes}/{n_levels} (λ=±{lambda_accept}%)")
        non_conf = df[df["Conclusion"]!="Conforme"]
        if not non_conf.empty:
            lines.append("⚠️ Niveaux non conformes :")
            for _, r in non_conf.iterrows():
                lines.append(f"  - Conc={r['Concentration_nominale']}: Risque={r['Risque (%)']:.2f}%, Biais_rel={r['Biais_rel (%)']:.2f}%")
            lines.append("Recommandations : vérifier préparation, calibration, effet matrice, augmenter nombre de séries.")
        else:
            lines.append("✅ Tous les niveaux conformes selon les critères.")
            lines.append("Recommandations : documenter la validation et monitorer périodiquement.")
        st.text_area("Interprétation & recommandations", "\n".join(lines), height=300)

with tabs[6]:
    st.header("🧾 Génération du rapport PDF")
    if not FPDF_AVAILABLE:
        st.warning("Le package 'fpdf' n'est pas installé. Installez-le (`pip install fpdf`) pour activer l'export PDF.")
    if st.button("Générer le PDF"):
        if "res_df" not in st.session_state:
            st.error("Rien à mettre dans le PDF — calcule d'abord les résultats.")
        else:
            try:
                res_df = st.session_state["res_df"]
                tmp = tempfile.mkdtemp()
                figpath1 = os.path.join(tmp, "exactitude.png")
                figpath2 = os.path.join(tmp, "incertitude.png")
                res_df["Recouvrement (%)"] = (res_df["Moyenne"] / res_df["Concentration_nominale"]) * 100
                fig, ax = plt.subplots(figsize=(7,4))
                ax.plot(res_df["Concentration_nominale"], res_df["Recouvrement (%)"], 'ro-', label="Recouvrement (%)")
                ax.axhline(100 + lambda_accept, color='black', linestyle='--', label=f"Limite acceptabilité (+{lambda_accept}%)")
                ax.axhline(100 - lambda_accept, color='black', linestyle='--', label=f"Limite acceptabilité (-{lambda_accept}%)")
                ax.legend(); ax.grid(True); ax.set_title("Profil d'exactitude (recouvrement)")
                fig.savefig(figpath1, bbox_inches='tight', dpi=150)
                plt.close(fig)
                fig2, ax2 = plt.subplots(figsize=(7,4))
                ax2.plot(res_df["Concentration_nominale"], res_df["Uncert_low [%]"], 'v--', color='green', label="Basse incertitude")
                ax2.plot(res_df["Concentration_nominale"], res_df["Uncert_high [%]"], '^--', color='green', label="Haute incertitude")
                ax2.axhline(-lambda_accept, color='red', linestyle='--', label="Limite acceptabilité (-%)")
                ax2.axhline(lambda_accept, color='red', linestyle='--', label="Limite acceptabilité (+%)")
                ax2.legend(); ax2.grid(True); ax2.set_title("Profil d'incertitude de la méthode")
                fig2.savefig(figpath2, bbox_inches='tight', dpi=150); plt.close(fig2)
                class PDF(FPDF):
                    def header(self):
                        self.set_font("Arial", "B", 14)
                        self.cell(0, 10, "Rapport de validation bioanalytique - L-Dopa", ln=True, align="C")
                        self.ln(5)
                    def footer(self):
                        self.set_y(-12)
                        self.set_font("Arial", "I", 8)
                        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")
                pdf = PDF()
                pdf.set_auto_page_break(auto=True, margin=12)
                pdf.add_page()
                pdf.set_font("Arial", size=11)
                intro = ("Ce rapport présente le profil d'exactitude (recouvrement) et le profil d'incertitude de la méthode.")
                pdf.multi_cell(0, 7, safe_pdf_text(intro))
                pdf.ln(2)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 6, safe_pdf_text("Profil d'exactitude (recouvrement %)"), ln=True)
                pdf.image(figpath1, w=170)
                pdf.ln(4)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 6, safe_pdf_text("Profil d'incertitude de la méthode"), ln=True)
                pdf.image(figpath2, w=170)
                pdf.ln(8)
                pdf.set_font("Arial", size=10)
                pdf.multi_cell(0, 6, safe_pdf_text("Tableau synthétique des résultats :"))
                for _, r in res_df.iterrows():
                    line = (f"Conc={r['Concentration_nominale']} | Moy={r['Moyenne']:.2f} | Recouvrement={r['Recouvrement (%)']:.2f}% | "
                            f"InterCV={r['Inter_CV (%)']:.2f}% | Incert=[{r['Uncert_low [%]']:.2f};{r['Uncert_high [%]']:.2f}] | Risque={r['Risque (%)']:.2f}% | {r['Conclusion']}")
                    pdf.multi_cell(0, 5, safe_pdf_text(line))
                pdf_bytes = pdf.output(dest='S').encode('latin-1', errors='replace')
                b64 = base64.b64encode(pdf_bytes).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="rapport_validation_levodopa.pdf">📥 Télécharger PDF</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("PDF généré.")
            except Exception as e:
                st.error(f"Erreur lors de la génération du PDF : {e}")

st.write("✅ Application complète A-Z : import, calculs, profils, robustesse MC, interprétation, PDF.")



