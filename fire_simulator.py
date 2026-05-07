"""
Simulateur FIRE — Monte Carlo
==============================
Stratégie DCA multi-assets avec retrait Trinity Study.
Tous les montants sont en EUROS RÉELS (constants, après inflation).

Lancement local :
    pip install streamlit numpy pandas plotly
    streamlit run fire_simulator.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="FIRE Simulator", layout="wide", page_icon="🔥")

# =============================================================================
# Paramètres historiques par défaut (rendements RÉELS annuels long terme)
# Sources : Damodaran (NYU Stern), Crédit Suisse Global Investment Returns Yearbook
# Ajustés à la baisse vs historique pur pour refléter valuations actuelles élevées.
# =============================================================================
DEFAULT_ASSETS = {
    "S&P500":     {"mu": 0.070, "sigma": 0.16, "label": "S&P 500"},
    "Nasdaq100":  {"mu": 0.090, "sigma": 0.22, "label": "Nasdaq 100"},
    "STOXX600":   {"mu": 0.050, "sigma": 0.17, "label": "STOXX 600"},
    "Topix":      {"mu": 0.040, "sigma": 0.17, "label": "Topix Japon"},
    "Gold":       {"mu": 0.020, "sigma": 0.16, "label": "Or"},
    "BTC":        {"mu": 0.150, "sigma": 0.65, "label": "Bitcoin"},
}

CORR_MATRIX = pd.DataFrame(
    [
        [1.00, 0.90, 0.75, 0.55, 0.05, 0.25],
        [0.90, 1.00, 0.70, 0.50, 0.05, 0.30],
        [0.75, 0.70, 1.00, 0.55, 0.10, 0.20],
        [0.55, 0.50, 0.55, 1.00, 0.10, 0.15],
        [0.05, 0.05, 0.10, 0.10, 1.00, 0.10],
        [0.25, 0.30, 0.20, 0.15, 0.10, 1.00],
    ],
    columns=list(DEFAULT_ASSETS.keys()),
    index=list(DEFAULT_ASSETS.keys()),
)

PRESETS = {
    "Préréglage 1 (40 SP / 20 STOXX / 20 NDX / 10 Or / 10 BTC)": {
        "S&P500": 0.40, "STOXX600": 0.20, "Nasdaq100": 0.20,
        "Topix": 0.0, "Gold": 0.10, "BTC": 0.10,
    },
    "Préréglage 2 (30 SP / 20 NDX / 15 STOXX / 15 Topix / 10 Or / 10 BTC)": {
        "S&P500": 0.30, "Nasdaq100": 0.20, "STOXX600": 0.15,
        "Topix": 0.15, "Gold": 0.10, "BTC": 0.10,
    },
    "100% S&P500 (référence)": {
        "S&P500": 1.0, "Nasdaq100": 0.0, "STOXX600": 0.0,
        "Topix": 0.0, "Gold": 0.0, "BTC": 0.0,
    },
}

# =============================================================================
# Moteur Monte Carlo
# =============================================================================
@st.cache_data(show_spinner=False)
def simulate_accumulation(initial, monthly, weights_tuple, n_months, n_sims,
                          asset_params_tuple, seed=42):
    """Phase d'accumulation. Retourne array (n_sims, n_months+1)."""
    weights = dict(weights_tuple)
    asset_params = {a: dict(p) for a, p in asset_params_tuple}
    names = list(weights.keys())
    w = np.array([weights[a] for a in names])
    mu_a = np.array([asset_params[a]["mu"] for a in names])
    sig_a = np.array([asset_params[a]["sigma"] for a in names])
    mu_m = mu_a / 12.0
    sig_m = sig_a / np.sqrt(12.0)
    corr = CORR_MATRIX.loc[names, names].values
    cov_m = np.outer(sig_m, sig_m) * corr
    # Régularise la matrice si quasi-singulière
    cov_m = cov_m + np.eye(len(names)) * 1e-10
    L = np.linalg.cholesky(cov_m)
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(size=(n_sims, n_months, len(names)))
    returns_monthly = mu_m + eps @ L.T
    portfolio_ret = returns_monthly @ w
    values = np.zeros((n_sims, n_months + 1))
    values[:, 0] = initial
    for t in range(n_months):
        values[:, t+1] = (values[:, t] + monthly) * (1.0 + portfolio_ret[:, t])
    return values


@st.cache_data(show_spinner=False)
def simulate_withdrawal(initial, annual_w, weights_tuple, years, n_sims,
                        asset_params_tuple, seed=43):
    """Phase de retrait. Retire annual_w/12 chaque mois. Retourne (values, failed)."""
    weights = dict(weights_tuple)
    asset_params = {a: dict(p) for a, p in asset_params_tuple}
    n_months = years * 12
    names = list(weights.keys())
    w = np.array([weights[a] for a in names])
    mu_a = np.array([asset_params[a]["mu"] for a in names])
    sig_a = np.array([asset_params[a]["sigma"] for a in names])
    mu_m = mu_a / 12.0
    sig_m = sig_a / np.sqrt(12.0)
    corr = CORR_MATRIX.loc[names, names].values
    cov_m = np.outer(sig_m, sig_m) * corr
    cov_m = cov_m + np.eye(len(names)) * 1e-10
    L = np.linalg.cholesky(cov_m)
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(size=(n_sims, n_months, len(names)))
    returns_monthly = mu_m + eps @ L.T
    portfolio_ret = returns_monthly @ w
    monthly_w = annual_w / 12.0
    values = np.zeros((n_sims, n_months + 1))
    values[:, 0] = initial
    failed = np.zeros(n_sims, dtype=bool)
    for t in range(n_months):
        v_after = values[:, t] - monthly_w
        new_failed = v_after < 0
        failed = failed | new_failed
        v_after = np.maximum(v_after, 0.0)
        values[:, t+1] = v_after * (1.0 + portfolio_ret[:, t])
        values[failed, t+1] = 0.0
    return values, failed


def freeze_params(weights, asset_params):
    """Convertit les dicts en tuples hashables pour le cache Streamlit."""
    w_tuple = tuple(sorted(weights.items()))
    p_tuple = tuple((a, tuple(sorted(p.items()))) for a, p in sorted(asset_params.items()))
    return w_tuple, p_tuple


# =============================================================================
# UI
# =============================================================================
st.title("🔥 Simulateur FIRE — Monte Carlo")
st.caption(
    "Simulation de 10 000 trajectoires basée sur ton allocation, tes apports et ta cible. "
    "Tous les montants sont en **euros réels** (constants, après inflation)."
)

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Paramètres")

    st.subheader("Capital & apports")
    initial = st.number_input("Capital initial (€)", value=7500, step=500, min_value=0)
    monthly = st.number_input("Apport mensuel (€)", value=150, step=10, min_value=0)

    st.subheader("Objectif")
    target_mode = st.radio(
        "Définir la cible par...",
        ["Capital", "Rente mensuelle souhaitée"],
        horizontal=True,
    )
    if target_mode == "Capital":
        target = st.number_input("Capital cible (€)", value=600_000, step=50_000, min_value=10_000)
        rate_for_target = st.select_slider(
            "Taux de retrait sous-jacent", options=[0.03, 0.035, 0.04], value=0.035,
            format_func=lambda x: f"{x*100:.1f}%",
        )
        implied_income = target * rate_for_target / 12
        st.caption(f"→ équivaut à **{implied_income:,.0f} €/mois** au taux {rate_for_target*100:.1f}%")
    else:
        monthly_income = st.number_input("Rente mensuelle (€)", value=2000, step=100, min_value=100)
        rate_for_target = st.select_slider(
            "Taux de retrait", options=[0.03, 0.035, 0.04], value=0.035,
            format_func=lambda x: f"{x*100:.1f}% ({'safe' if x<=0.03 else 'équilibré' if x<=0.035 else 'agressif'})",
        )
        target = int(monthly_income * 12 / rate_for_target)
        st.caption(f"→ capital cible : **{target:,.0f} €**")

    st.subheader("Simulation")
    horizon = st.slider("Horizon max accumulation (années)", 5, 50, 30)
    n_sims = st.select_slider("Nombre de simulations",
                              options=[1000, 5000, 10000, 25000], value=10000)

    st.subheader("Allocation")
    preset_choice = st.selectbox("Préréglage", list(PRESETS.keys()) + ["Personnalisé"])

    if preset_choice == "Personnalisé":
        weights = {}
        st.markdown("*Ajuste les poids — total doit faire 100%*")
        for asset in DEFAULT_ASSETS.keys():
            weights[asset] = st.slider(
                DEFAULT_ASSETS[asset]["label"], 0.0, 1.0, 0.0, 0.05, key=f"w_{asset}"
            )
    else:
        weights = PRESETS[preset_choice].copy()

    total = sum(weights.values())
    if abs(total - 1.0) > 0.001:
        st.warning(f"⚠️ Allocation = {total*100:.0f}% (doit faire 100%)")
    else:
        st.success(f"✓ Allocation = 100%")

    with st.expander("🔬 Paramètres avancés (rendements & vol)"):
        st.caption("Rendements **réels** annuels (après inflation). Modifie pour stresser ton scénario.")
        asset_params = {}
        for a, p in DEFAULT_ASSETS.items():
            c1, c2 = st.columns(2)
            mu = c1.number_input(f"μ {p['label']} (%)",
                                 value=p["mu"]*100, step=0.5, key=f"mu_{a}") / 100
            sig = c2.number_input(f"σ {p['label']} (%)",
                                  value=p["sigma"]*100, step=1.0, key=f"sig_{a}") / 100
            asset_params[a] = {"mu": mu, "sigma": sig}

# Validation
if abs(sum(weights.values()) - 1.0) > 0.001:
    st.error("Corrige ton allocation à 100% dans la sidebar pour lancer la simulation.")
    st.stop()

w_tuple, p_tuple = freeze_params(weights, asset_params)

# =============================================================================
# Onglets
# =============================================================================
tab_accum, tab_trinity, tab_alloc, tab_help = st.tabs([
    "📈 Accumulation", "💸 Phase de retrait (Trinity)", "🥧 Allocation", "ℹ️ Méthodologie"
])

# -----------------------------------------------------------------------------
# TAB 1 : ACCUMULATION
# -----------------------------------------------------------------------------
with tab_accum:
    st.subheader("Combien de temps pour atteindre ta cible ?")

    n_months = horizon * 12
    with st.spinner(f"Simulation de {n_sims:,} trajectoires..."):
        values = simulate_accumulation(initial, monthly, w_tuple, n_months, n_sims, p_tuple)

    # Métriques principales
    final = values[:, -1]
    reached_mask = (values >= target).any(axis=1)
    reached_t = np.argmax(values >= target, axis=1).astype(float)
    reached_t[~reached_mask] = np.nan
    reached_yrs = reached_t / 12

    p_reach = reached_mask.mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"P(atteindre la cible en {horizon} ans)", f"{p_reach:.1f}%")
    if reached_mask.any():
        c2.metric("Temps médian (si atteint)", f"{np.nanmedian(reached_yrs):.1f} ans")
        c3.metric("P25 (rapide)", f"{np.nanpercentile(reached_yrs, 25):.1f} ans")
        c4.metric("P75 (lent)", f"{np.nanpercentile(reached_yrs, 75):.1f} ans")
    else:
        c2.metric("Temps médian", "Cible non atteinte")

    st.divider()

    # Fan chart
    st.markdown("**Évolution du capital — fan chart**")
    months_axis = np.arange(n_months + 1) / 12
    p10 = np.percentile(values, 10, axis=0)
    p25 = np.percentile(values, 25, axis=0)
    p50 = np.percentile(values, 50, axis=0)
    p75 = np.percentile(values, 75, axis=0)
    p90 = np.percentile(values, 90, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months_axis, y=p90, line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=months_axis, y=p10, fill="tonexty",
                             fillcolor="rgba(99,110,250,0.15)", line=dict(width=0),
                             name="P10–P90"))
    fig.add_trace(go.Scatter(x=months_axis, y=p75, line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=months_axis, y=p25, fill="tonexty",
                             fillcolor="rgba(99,110,250,0.30)", line=dict(width=0),
                             name="P25–P75"))
    fig.add_trace(go.Scatter(x=months_axis, y=p50,
                             line=dict(color="#636EFA", width=2.5),
                             name="Médiane (P50)"))
    fig.add_hline(y=target, line_dash="dash", line_color="red",
                  annotation_text=f"Cible {target:,.0f} €", annotation_position="right")

    # Cumul des apports comme référence
    cum_contrib = initial + monthly * np.arange(n_months + 1)
    fig.add_trace(go.Scatter(x=months_axis, y=cum_contrib,
                             line=dict(color="grey", width=1.5, dash="dot"),
                             name="Apports cumulés"))

    fig.update_layout(
        xaxis_title="Années", yaxis_title="Capital (€ réels)",
        height=450, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(tickformat=",.0f")
    st.plotly_chart(fig, width='stretch')

    # Probabilité d'atteindre la cible par année
    st.markdown("**Probabilité d'atteindre la cible par horizon**")
    years_grid = list(range(5, horizon + 1, 5))
    if horizon not in years_grid:
        years_grid.append(horizon)
    probs = []
    for y in years_grid:
        m = y * 12
        if m <= n_months:
            p = (values[:, :m+1] >= target).any(axis=1).mean() * 100
            probs.append({"Horizon (années)": y,
                          f"P(atteint {target:,.0f} €)": f"{p:.1f}%"})
    st.dataframe(pd.DataFrame(probs), hide_index=True, width='content')

    # Distribution du temps pour atteindre la cible
    if reached_mask.sum() >= 50:
        st.markdown("**Distribution du temps pour atteindre la cible**")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=reached_yrs[~np.isnan(reached_yrs)],
            nbinsx=40, marker_color="#636EFA", opacity=0.85,
        ))
        fig2.add_vline(x=np.nanmedian(reached_yrs), line_dash="dash",
                       line_color="red", annotation_text="Médiane")
        fig2.update_layout(
            xaxis_title="Années pour atteindre la cible",
            yaxis_title="Nombre de simulations",
            height=300, showlegend=False,
        )
        st.plotly_chart(fig2, width='stretch')

    # Distribution du capital final
    st.markdown(f"**Distribution du capital final à {horizon} ans**")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig3 = go.Figure()
        # Clip pour éviter les outliers BTC qui écrasent le graphique
        clip_max = np.percentile(final, 99)
        fig3.add_trace(go.Histogram(
            x=np.clip(final, 0, clip_max),
            nbinsx=60, marker_color="#00CC96", opacity=0.85,
        ))
        fig3.add_vline(x=target, line_dash="dash", line_color="red",
                       annotation_text=f"Cible")
        fig3.add_vline(x=np.median(final), line_dash="dot", line_color="blue",
                       annotation_text="Médiane")
        fig3.update_layout(
            xaxis_title="Capital final (€ réels)",
            yaxis_title="Nombre de simulations",
            height=350, showlegend=False,
        )
        fig3.update_xaxes(tickformat=",.0f")
        st.plotly_chart(fig3, width='stretch')
    with c2:
        st.markdown("**Statistiques finales**")
        stats = pd.DataFrame({
            "Percentile": ["P5", "P10", "P25", "Médiane", "P75", "P90", "P95"],
            "Capital (€)": [
                f"{np.percentile(final, p):,.0f}"
                for p in [5, 10, 25, 50, 75, 90, 95]
            ],
        })
        st.dataframe(stats, hide_index=True, width='stretch')
        total_invested = initial + monthly * 12 * horizon
        st.caption(f"**Apports totaux** : {total_invested:,.0f} €")
        st.caption(f"**Multiple médian** : ×{np.median(final)/total_invested:.1f}")


# -----------------------------------------------------------------------------
# TAB 2 : TRINITY (PHASE DE RETRAIT)
# -----------------------------------------------------------------------------
with tab_trinity:
    st.subheader("Une fois à la cible — combien de temps ça tient ?")
    st.caption(
        "Tu pars du capital cible et tu retires un montant **fixe en euros réels** chaque mois. "
        "Le simulateur compte le % de scénarios où le portefeuille survit jusqu'au bout."
    )

    c1, c2 = st.columns(2)
    with c1:
        trinity_capital = st.number_input(
            "Capital de départ Trinity (€)",
            value=int(target), step=50_000, min_value=10_000,
            help="Par défaut = ta cible accumulation. Tu peux tester d'autres niveaux.",
        )
    with c2:
        trinity_years = st.slider("Durée de retrait (années)", 20, 60, 30)

    rates = [0.025, 0.03, 0.035, 0.04, 0.045, 0.05]

    with st.spinner("Simulation des scénarios de retrait..."):
        results_trinity = []
        for r in rates:
            vals_w, failed = simulate_withdrawal(
                trinity_capital, trinity_capital * r, w_tuple,
                trinity_years, n_sims, p_tuple, seed=int(r*1000),
            )
            results_trinity.append({
                "rate": r,
                "annual_w": trinity_capital * r,
                "monthly_w": trinity_capital * r / 12,
                "success": (1 - failed.mean()) * 100,
                "median_end": np.median(vals_w[:, -1]),
                "p10_end": np.percentile(vals_w[:, -1], 10),
            })

    # Tableau
    df_trinity = pd.DataFrame([{
        "Taux retrait": f"{r['rate']*100:.1f}%",
        "Retrait annuel": f"{r['annual_w']:,.0f} €",
        "Retrait mensuel": f"{r['monthly_w']:,.0f} €",
        f"Succès sur {trinity_years} ans": f"{r['success']:.1f}%",
        "Capital médian final": f"{r['median_end']:,.0f} €",
        "Capital P10 final": f"{r['p10_end']:,.0f} €",
    } for r in results_trinity])
    st.dataframe(df_trinity, hide_index=True, width='stretch')

    # Graphique succès vs taux
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(
        x=[r["rate"]*100 for r in results_trinity],
        y=[r["success"] for r in results_trinity],
        mode="lines+markers", line=dict(color="#EF553B", width=3),
        marker=dict(size=12),
        name="Taux de succès",
    ))
    fig_t.add_hline(y=95, line_dash="dash", line_color="green",
                    annotation_text="Seuil 95% (très safe)")
    fig_t.add_hline(y=85, line_dash="dot", line_color="orange",
                    annotation_text="Seuil 85% (acceptable)")
    fig_t.update_layout(
        xaxis_title="Taux de retrait annuel (%)",
        yaxis_title=f"% scénarios où portefeuille tient {trinity_years} ans",
        height=400, yaxis_range=[0, 105],
    )
    st.plotly_chart(fig_t, width='stretch')

    st.info(
        "💡 **Lecture** : la règle des 4% du Trinity Study originale ciblait 95% de succès "
        "sur portefeuille 50/50 actions-obligations US sur 30 ans. Avec ton allocation "
        "(plus volatile, BTC inclus), le taux qui te donne 95% est probablement plus bas. "
        "Tes notes mentionnent 3.5% comme bon équilibre — la simulation te le confirme ou non."
    )

# -----------------------------------------------------------------------------
# TAB 3 : ALLOCATION
# -----------------------------------------------------------------------------
with tab_alloc:
    st.subheader("Vue de ton allocation actuelle")

    df_alloc = pd.DataFrame([
        {
            "Asset": DEFAULT_ASSETS[a]["label"],
            "Poids": w,
            "μ réel annuel": asset_params[a]["mu"],
            "σ annuelle": asset_params[a]["sigma"],
            "Contribution rendement": w * asset_params[a]["mu"],
        }
        for a, w in weights.items() if w > 0
    ])

    c1, c2 = st.columns([1, 1])
    with c1:
        fig_pie = go.Figure(data=[go.Pie(
            labels=df_alloc["Asset"], values=df_alloc["Poids"],
            hole=0.4, textinfo="label+percent",
        )])
        fig_pie.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_pie, width='stretch')

    with c2:
        # Calcul rendement et vol portefeuille
        names = list(weights.keys())
        w_arr = np.array([weights[a] for a in names])
        mu_arr = np.array([asset_params[a]["mu"] for a in names])
        sig_arr = np.array([asset_params[a]["sigma"] for a in names])
        corr = CORR_MATRIX.loc[names, names].values
        cov = np.outer(sig_arr, sig_arr) * corr
        port_mu = w_arr @ mu_arr
        port_sig = np.sqrt(w_arr @ cov @ w_arr)
        sharpe = port_mu / port_sig if port_sig > 0 else 0

        st.markdown("### Stats du portefeuille")
        st.metric("Rendement réel attendu", f"{port_mu*100:.2f}% / an")
        st.metric("Volatilité", f"{port_sig*100:.2f}% / an")
        st.metric("Sharpe (sans risque = 0)", f"{sharpe:.2f}")
        st.caption(
            f"**Règle de 70** : à ce rendement, ton capital doublerait en "
            f"~{70/(port_mu*100):.1f} ans (en réel)."
        )

    st.divider()
    st.markdown("**Détail par ligne**")
    df_display = df_alloc.copy()
    df_display["Poids"] = df_display["Poids"].apply(lambda x: f"{x*100:.0f}%")
    df_display["μ réel annuel"] = df_display["μ réel annuel"].apply(lambda x: f"{x*100:.1f}%")
    df_display["σ annuelle"] = df_display["σ annuelle"].apply(lambda x: f"{x*100:.1f}%")
    df_display["Contribution rendement"] = df_display["Contribution rendement"].apply(
        lambda x: f"{x*100:.2f}%")
    st.dataframe(df_display, hide_index=True, width='stretch')

# -----------------------------------------------------------------------------
# TAB 4 : MÉTHODOLOGIE
# -----------------------------------------------------------------------------
with tab_help:
    st.markdown("""
### Comment ça marche

**Modèle.** Chaque mois, les rendements des assets sont tirés d'une **loi normale multivariée**
avec moyennes (μ), volatilités (σ) et corrélations historiques. Les retours sont
corrélés via une décomposition de Cholesky. Le portefeuille est rebalancé mensuellement.

**Phase d'accumulation.**
1. On part de ton capital initial
2. Chaque mois, on ajoute ton apport DCA
3. Puis on applique le rendement aléatoire du portefeuille
4. On répète sur l'horizon choisi, pour 10 000 trajectoires en parallèle
5. On compte combien atteignent ta cible et quand

**Phase de retrait (Trinity).**
1. On part du capital cible
2. Chaque mois, on retire le montant fixe (en € réels)
3. Puis on applique le rendement aléatoire
4. Si le portefeuille atteint 0, le scénario est en échec
5. Le **taux de succès** = % de scénarios où le portefeuille survit jusqu'au bout

### Choix méthodologiques

- **Tout est en euros réels** (après inflation). Pas besoin de raisonner sur le pouvoir d'achat futur.
- **Rendements normaux** plutôt que log-normaux — approximation valide à fréquence mensuelle pour de faibles μ.
- **Pas de fat tails** — un crash type 2008 (-40%) reste possible mais sous-estimé. Pour stresser, baisse μ et augmente σ dans les paramètres avancés.
- **Pas de fiscalité** — modélise le portefeuille brut. Pour un simulateur fiscal PEA vs CTO, c'est un autre projet.
- **Hypothèses inflation 2%** déjà intégrées dans les μ réels par défaut.

### Limites à garder en tête

- BTC à 15% réel/an forward est une **hypothèse forte** — la performance historique est massivement plus élevée mais non-extrapolable. Stresse à 5-8% pour voir un scénario conservateur.
- Les corrélations sont **statiques** — en pratique elles montent en crise (toutes les actions chutent ensemble).
- Le **sequence of returns risk** est capturé en phase Trinity : c'est exactement ce que les scénarios ratés mettent en évidence.

### Sources

- Damodaran (NYU Stern) — données historiques actions/bonds
- Crédit Suisse Global Investment Returns Yearbook
- Trinity Study (Cooley, Hubbard, Walz, 1998) — règle des 4%
""")

st.divider()
st.caption(
    "⚠️ Ce simulateur est un **outil pédagogique**, pas un conseil en investissement. "
    "Les rendements futurs ne sont pas garantis. Les hypothèses peuvent être très éloignées "
    "de la réalité, surtout pour le Bitcoin."
)
