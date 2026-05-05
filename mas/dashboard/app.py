"""
PF-MAS Plotly Dash Dashboard.

Layout
------
Header  : live status bar (channel | decision | RUL | crit%)
Slider  : channel selector 1-45 (inspect any processed channel)
Row 1   : PF Overview plot (all channels, selected highlighted)  70%
          Status cards                                           30%
Row 2   : Raw force Fx/Fy for selected channel                  50%
          Wear signal vs predicted CI for selected channel       50%
Row 3   : Ft / Fn B-spline curves for selected channel          50%
          Particle histogram at selected channel                 50%
Row 4   : Image analysis panel (only when selected ch = TAKE_IMAGE)
            Annotated tool image  |  K-Means badge + metrics
Row 5   : Decision log table

PF Overview plot shows
  - coloured observation dots per channel (green=CONTINUE, blue=TAKE_IMAGE, red=REPLACE)
  - I-beam CI error bars (the CI that was predicted before each observation arrived)
  - gold star on the user-selected channel
  - dashed future trajectory + shaded 90% CI band from the LATEST PF state
  - red dashed critical threshold at 0.75
"""
from __future__ import annotations

import base64
import copy
import threading
from typing import Any

import dash
import plotly.graph_objs as go
from dash import dcc, html, Input, Output, State, dash_table

from mas.agents.particle_filter import CRITICAL_THRESHOLD

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

C = {
    "CONTINUE":     "#2ecc71",
    "TAKE_IMAGE":   "#3498db",
    "REPLACE":      "#e74c3c",
    "FACTORY_NEW":  "#27ae60",
    "MID_WORN":     "#f39c12",
    "CRITICAL":     "#e74c3c",
    "bg":           "#1a1a2e",
    "card":         "#16213e",
    "text":         "#eaeaea",
    "grid":         "#2c2c54",
    "threshold":    "#e74c3c",
    "ci_fill":      "rgba(52,152,219,0.18)",
    "future_line":  "#3498db",
    "particle":     "rgba(255,215,0,0.25)",
    "ft":           "#9b59b6",
    "fn":           "#1abc9c",
}

_CARD = dict(backgroundColor=C["card"], borderRadius="8px",
             padding="14px", marginBottom="10px", textAlign="center")
_DL   = dict(l=52, r=18, t=38, b=42)


def _dark_layout(title: str, xl: str = "Channel", yl: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(color=C["text"], size=13)),
        paper_bgcolor=C["bg"], plot_bgcolor=C["card"],
        font=dict(color=C["text"]), margin=_DL,
        xaxis=dict(title=xl, gridcolor=C["grid"], zerolinecolor=C["grid"]),
        yaxis=dict(title=yl, gridcolor=C["grid"], zerolinecolor=C["grid"]),
        legend=dict(font=dict(color=C["text"]), bgcolor="rgba(0,0,0,0)"),
    )


def _badge(colour: str) -> dict:
    return dict(backgroundColor=colour, color="#fff",
                borderRadius="4px", padding="2px 8px",
                fontSize="11px", display="inline-block")


def _stat_card(label: str, val_id: str, badge_id: str | None = None) -> html.Div:
    kids: list = [
        html.P(label, style=dict(color="#999", margin="0", fontSize="11px")),
        html.H4(id=val_id, style=dict(color=C["text"], margin="4px 0")),
    ]
    if badge_id:
        kids.append(html.Div(id=badge_id,
                             style=dict(borderRadius="4px", padding="2px 7px",
                                        fontSize="11px", display="inline-block")))
    return html.Div(kids, style=_CARD)


def _empty_fig(title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**_dark_layout(title))
    return fig


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_app(shared_state: dict, state_lock: threading.Lock) -> dash.Dash:

    app = dash.Dash(
        __name__,
        title="PF-MAS Tool Wear Monitor",
        suppress_callback_exceptions=True,
    )

    app.layout = html.Div(
        style=dict(backgroundColor=C["bg"], minHeight="100vh",
                   padding="20px", fontFamily="Segoe UI, sans-serif"),
        children=[
            dcc.Interval(id="interval", interval=700, n_intervals=0),

            # ---- Header ----
            html.Div([
                html.H2("Micro-Milling PF-MAS  |  Tool Wear Monitor",
                        style=dict(color=C["text"], margin="0", fontSize="20px")),
                html.Span(id="header-meta",
                          style=dict(color="#aaa", marginLeft="24px", fontSize="14px")),
            ], style=dict(display="flex", alignItems="center", marginBottom="14px")),

            # ---- Channel selector ----
            html.Div([
                html.P("Inspect Channel:", style=dict(color="#aaa", margin="0 10px 0 0",
                                                      fontSize="13px", display="inline")),
                html.Div(
                    dcc.Slider(
                        id="ch-slider", min=1, max=45, step=1, value=1,
                        marks={i: str(i) for i in range(1, 46, 5)},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    style=dict(flex="1"),
                ),
            ], style=dict(display="flex", alignItems="center",
                          marginBottom="16px", backgroundColor=C["card"],
                          borderRadius="8px", padding="10px 16px")),

            # ---- Row 1: PF overview + status cards ----
            html.Div([
                html.Div(dcc.Graph(id="pf-plot", style=dict(height="390px"),
                                   config=dict(displayModeBar=False)),
                         style=dict(width="70%", paddingRight="12px")),
                html.Div([
                    _stat_card("Tool State",      "state-val",    "state-badge"),
                    _stat_card("Leader Decision", "decision-val", "decision-badge"),
                    _stat_card("RUL (channels)",  "rul-val"),
                    _stat_card("Critical Prob.",  "crit-val"),
                    _stat_card("K-Means Result",  "kmeans-val",   "kmeans-badge"),
                ], style=dict(width="30%")),
            ], style=dict(display="flex", marginBottom="16px")),

            # ---- Row 2: Force | Signal vs CI ----
            html.Div([
                html.Div(dcc.Graph(id="force-plot", style=dict(height="270px"),
                                   config=dict(displayModeBar=False)),
                         style=dict(width="50%", paddingRight="10px")),
                html.Div(dcc.Graph(id="signal-ci-plot", style=dict(height="270px"),
                                   config=dict(displayModeBar=False)),
                         style=dict(width="50%")),
            ], style=dict(display="flex", marginBottom="16px")),

            # ---- Row 3: Ft/Fn spline | Particle histogram ----
            html.Div([
                html.Div(dcc.Graph(id="ftfn-plot", style=dict(height="260px"),
                                   config=dict(displayModeBar=False)),
                         style=dict(width="50%", paddingRight="10px")),
                html.Div(dcc.Graph(id="particle-hist", style=dict(height="260px"),
                                   config=dict(displayModeBar=False)),
                         style=dict(width="50%")),
            ], style=dict(display="flex", marginBottom="16px")),

            # ---- Row 4: Image analysis (conditional) ----
            html.Div(id="image-panel", children=[]),

            # ---- Row 5: Decision log ----
            html.Div([
                html.H5("Decision Log",
                        style=dict(color=C["text"], margin="0 0 8px 0")),
                dash_table.DataTable(
                    id="decision-log",
                    columns=[{"name": c, "id": c} for c in
                             ["Ch", "Signal", "CI", "Decision",
                              "State", "K-Means", "RUL", "Crit%"]],
                    data=[],
                    page_size=10,
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": C["card"], "color": C["text"],
                                  "fontWeight": "bold", "fontSize": "12px"},
                    style_data={"backgroundColor": C["bg"], "color": C["text"],
                                "fontSize": "12px"},
                    style_data_conditional=[
                        {"if": {"filter_query": '{Decision} = "REPLACE"'},
                         "backgroundColor": "#3d1515", "color": "#ff6b6b"},
                        {"if": {"filter_query": '{Decision} = "TAKE_IMAGE"'},
                         "backgroundColor": "#152030", "color": "#74b9ff"},
                    ],
                ),
            ], style=_CARD),
        ],
    )

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    @app.callback(
        Output("ch-slider", "value"),
        Input("interval", "n_intervals"),
        State("ch-slider", "value"),
    )
    def _auto_follow_slider(_, current_val: int) -> int:
        with state_lock:
            latest = shared_state.get("current_channel", 0)
        # Auto-follow only if slider is already at or 1 behind latest
        if current_val >= latest - 1:
            return max(1, latest)
        return current_val

    @app.callback(
        [
            Output("header-meta",    "children"),
            Output("pf-plot",        "figure"),
            Output("state-val",      "children"),
            Output("state-badge",    "children"),
            Output("state-badge",    "style"),
            Output("decision-val",   "children"),
            Output("decision-badge", "children"),
            Output("decision-badge", "style"),
            Output("rul-val",        "children"),
            Output("crit-val",       "children"),
            Output("kmeans-val",     "children"),
            Output("kmeans-badge",   "children"),
            Output("kmeans-badge",   "style"),
            Output("force-plot",     "figure"),
            Output("signal-ci-plot", "figure"),
            Output("ftfn-plot",      "figure"),
            Output("particle-hist",  "figure"),
            Output("image-panel",    "children"),
            Output("decision-log",   "data"),
        ],
        [
            Input("interval",   "n_intervals"),
            Input("ch-slider",  "value"),
        ],
    )
    def refresh(_n: int, selected_ch: int) -> tuple[Any, ...]:
        with state_lock:
            st = copy.deepcopy(shared_state)

        hist       = st.get("channel_history", {})
        latest_ch  = st.get("current_channel", 0)
        sel        = selected_ch or latest_ch or 1
        ch         = hist.get(sel)            # may be None if not yet processed

        # ---- Live status from latest PF state ----
        decision   = st.get("decision",   "CONTINUE")
        tool_state = st.get("tool_state", "FACTORY_NEW")
        rul        = st.get("rul",         45)
        crit_p     = st.get("critical_prob", 0.0)
        kmeans     = st.get("kmeans_result", "-")

        header = (f"Streaming: Ch {latest_ch}/45  |  "
                  f"RUL {rul} ch  |  Crit {crit_p*100:.0f}%  |  "
                  f"Inspecting Ch {sel}")

        # ==============================================================
        # PF Overview  (all processed channels)
        # ==============================================================
        pf_fig = go.Figure()

        pf_fig.add_hline(
            y=CRITICAL_THRESHOLD,
            line=dict(color=C["threshold"], dash="dash", width=1.5),
            annotation_text="Critical Threshold",
            annotation_font_color=C["threshold"],
            annotation_position="top right",
        )

        obs = st.get("observations", [])
        if obs:
            ch_all   = [o["channel"]  for o in obs]
            sig_all  = [o["signal"]   for o in obs]
            ci_lo    = [o["ci_low"]   for o in obs]
            ci_hi    = [o["ci_high"]  for o in obs]
            dec_all  = [o["decision"] for o in obs]
            dot_col  = [C.get(d, "#aaa") for d in dec_all]

            err_up   = [max(0.0, hi - s) for hi, s in zip(ci_hi, sig_all)]
            err_dn   = [max(0.0, s - lo) for s, lo in zip(sig_all, ci_lo)]

            pf_fig.add_trace(go.Scatter(
                x=ch_all, y=sig_all,
                mode="markers+lines",
                marker=dict(color=dot_col, size=8,
                            line=dict(color="#fff", width=0.8)),
                line=dict(color="rgba(180,180,180,0.3)", width=1),
                error_y=dict(type="data", symmetric=False,
                             array=err_up, arrayminus=err_dn,
                             color="rgba(150,150,150,0.45)",
                             thickness=1.2, width=5),
                name="Observed + CI",
                customdata=dec_all,
                hovertemplate="Ch %{x}<br>Signal %{y:.3f}<br>%{customdata}",
            ))

        # Selected channel highlight
        if ch is not None:
            pf_fig.add_trace(go.Scatter(
                x=[sel], y=[ch["signal"]],
                mode="markers",
                marker=dict(color="gold", size=16, symbol="star",
                            line=dict(color="white", width=1.5)),
                name=f"Selected Ch {sel}",
            ))

        # Particle cloud at current (latest) channel
        parts = st.get("particles", [])
        if parts and latest_ch:
            pf_fig.add_trace(go.Scatter(
                x=[latest_ch] * len(parts), y=parts,
                mode="markers",
                marker=dict(color=C["particle"], size=3),
                name="PF Particles (now)",
                hoverinfo="skip",
            ))

        # Future CI band + mean
        fut_chs = st.get("future_channels", [])
        fut_lo  = st.get("future_ci_low",   [])
        fut_hi  = st.get("future_ci_high",  [])
        fut_mn  = st.get("future_means",    [])
        if fut_chs and fut_lo and fut_hi:
            pf_fig.add_trace(go.Scatter(
                x=fut_chs + fut_chs[::-1],
                y=fut_hi + fut_lo[::-1],
                fill="toself", fillcolor=C["ci_fill"],
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip", name="90% Pred. CI",
            ))
            pf_fig.add_trace(go.Scatter(
                x=fut_chs, y=fut_mn,
                mode="lines",
                line=dict(color=C["future_line"], width=2, dash="dot"),
                name="PF Mean Trajectory",
            ))

        pf_fig.update_layout(
            **_dark_layout("PF Overview — All Channels  (gold star = selected)",
                           yl="Wear Signal [0-1]"),
        )
        pf_fig.update_yaxes(range=[-0.02, 1.05])

        # ==============================================================
        # Per-channel detail  (uses `ch` dict from channel_history)
        # ==============================================================

        # ---- Force Fx / Fy ----
        force_fig = _empty_fig(f"Force Signal — Ch {sel}" +
                               ("  (not yet processed)" if ch is None else ""))
        if ch:
            Fx_ch = ch.get("Fx", [])
            Fy_ch = ch.get("Fy", [])
            idx   = list(range(len(Fx_ch)))
            if Fx_ch:
                force_fig.add_trace(go.Scatter(
                    x=idx, y=Fx_ch, name="Fx",
                    mode="lines", line=dict(color=C["TAKE_IMAGE"], width=1)))
            if Fy_ch:
                force_fig.add_trace(go.Scatter(
                    x=idx, y=Fy_ch, name="Fy",
                    mode="lines", line=dict(color=C["REPLACE"], width=1)))
            # Annotate computed wear signal
            sig_v = ch["signal"]
            force_fig.add_annotation(
                xref="paper", yref="paper", x=0.01, y=0.97,
                text=f"Wear signal: {sig_v:.3f}",
                showarrow=False,
                font=dict(color="gold", size=12),
                bgcolor="rgba(0,0,0,0.4)", borderpad=4,
            )
            force_fig.update_layout(
                **_dark_layout(f"Force Signal — Ch {sel}",
                               xl="Sample (downsampled 1:20)", yl="Force (N)"),
            )

        # ---- Wear signal vs predicted CI ----
        sci_fig = _empty_fig(f"Wear Signal vs Predicted CI — Ch {sel}")
        if ch:
            sig_v = ch["signal"]
            lo_v  = ch["ci_low"]
            hi_v  = ch["ci_high"]
            inside = lo_v <= sig_v <= hi_v
            dot_colour = C["CONTINUE"] if inside else C["TAKE_IMAGE"]

            # CI band
            sci_fig.add_hrect(
                y0=lo_v, y1=hi_v,
                fillcolor="rgba(52,152,219,0.20)",
                line_width=0,
            )
            sci_fig.add_hline(y=lo_v, line=dict(color=C["future_line"],
                                                  dash="dot", width=1))
            sci_fig.add_hline(y=hi_v, line=dict(color=C["future_line"],
                                                  dash="dot", width=1),
                              annotation_text="Predicted CI",
                              annotation_font_color=C["future_line"],
                              annotation_position="top right")
            # Critical threshold
            sci_fig.add_hline(
                y=CRITICAL_THRESHOLD,
                line=dict(color=C["threshold"], dash="dash", width=1.5),
                annotation_text="Critical",
                annotation_font_color=C["threshold"],
            )
            # Observed signal dot
            label = "Inside CI" if inside else "Outside CI (trigger)"
            sci_fig.add_trace(go.Scatter(
                x=["Signal"], y=[sig_v],
                mode="markers+text",
                marker=dict(color=dot_colour, size=22,
                            line=dict(color="#fff", width=2)),
                text=[f"{sig_v:.3f}"],
                textposition="middle right",
                textfont=dict(color=C["text"], size=13),
                name=label,
            ))
            # CI bound labels
            sci_fig.add_annotation(
                x="Signal", y=lo_v, text=f"CI lo {lo_v:.3f}",
                showarrow=False, xanchor="left",
                font=dict(color=C["future_line"], size=11))
            sci_fig.add_annotation(
                x="Signal", y=hi_v, text=f"CI hi {hi_v:.3f}",
                showarrow=False, xanchor="left",
                font=dict(color=C["future_line"], size=11))

            sci_fig.update_layout(
                **_dark_layout(f"Wear Signal vs Predicted CI — Ch {sel}",
                               xl="", yl="Wear Level"),
            )
            sci_fig.update_yaxes(range=[0, 1.05])

        # ---- Ft / Fn spline ----
        ftfn_fig = _empty_fig(f"Ft / Fn Spline — Ch {sel}")
        if ch:
            ft = ch.get("ft_ctrl", [])
            fn = ch.get("fn_ctrl", [])
            if ft:
                xi = list(range(len(ft)))
                ftfn_fig.add_trace(go.Scatter(
                    x=xi, y=ft, name="Ft (tangential)",
                    mode="lines+markers",
                    line=dict(color=C["ft"], width=2),
                    marker=dict(size=5)))
            if fn:
                xi = list(range(len(fn)))
                ftfn_fig.add_trace(go.Scatter(
                    x=xi, y=fn, name="Fn (normal)",
                    mode="lines+markers",
                    line=dict(color=C["fn"], width=2),
                    marker=dict(size=5)))
            ps = ch.get("plateau_score", 1.0)
            ftfn_fig.add_annotation(
                xref="paper", yref="paper", x=0.01, y=0.97,
                text=f"Plateau score: {ps:.3f}",
                showarrow=False,
                font=dict(color="gold", size=11),
                bgcolor="rgba(0,0,0,0.4)", borderpad=4,
            )
            ftfn_fig.update_layout(
                **_dark_layout(f"Ft / Fn Spline — Ch {sel}",
                               xl="Control Point", yl="Force (N/mm·b)"),
            )

        # ---- Particle histogram ----
        hist_fig = _empty_fig(f"Particle Distribution — Ch {sel}")
        if ch:
            p_snap = ch.get("particles", [])
            if p_snap:
                hist_fig.add_trace(go.Histogram(
                    x=p_snap, nbinsx=30,
                    marker_color="#3498db", opacity=0.8,
                    name="Particles",
                ))
                hist_fig.add_vline(
                    x=CRITICAL_THRESHOLD,
                    line=dict(color=C["threshold"], dash="dash", width=1.5),
                    annotation_text="Critical",
                    annotation_font_color=C["threshold"],
                )
            hist_fig.update_layout(
                **_dark_layout(f"Particle Distribution — Ch {sel}",
                               xl="Wear Level", yl="Count"),
                bargap=0.05,
            )
            hist_fig.update_xaxes(range=[0, 1])

        # ==============================================================
        # Image analysis panel
        # ==============================================================
        image_panel: list = []
        if ch and ch.get("decision") == "TAKE_IMAGE" and ch.get("img_b64"):
            enc       = ch["img_b64"]
            analysis  = ch.get("img_analysis") or {}
            km_lbl    = ch.get("kmeans_label", "-")
            km_col    = C.get(km_lbl, "#aaa")
            edge_r    = analysis.get("edge_radius_px", float("nan"))
            tool_len  = analysis.get("tool_length_px", float("nan"))
            worn_area = analysis.get("worn_area_px",   float("nan"))

            image_panel = [html.Div([
                # Title + K-Means badge
                html.Div([
                    html.H5(f"Image Analysis — Ch {sel}",
                            style=dict(color=C["text"], display="inline",
                                       margin="0 14px 0 0")),
                    html.Span(f"K-Means: {km_lbl}",
                              style={**_badge(km_col),
                                     "fontSize": "13px", "padding": "4px 12px"}),
                ], style=dict(marginBottom="12px")),

                # Image + metrics side by side
                html.Div([
                    # Annotated image
                    html.Div([
                        html.Img(
                            src=f"data:image/jpeg;base64,{enc}",
                            style=dict(width="100%", maxWidth="420px",
                                       borderRadius="6px",
                                       border=f"2px solid {km_col}"),
                        ),
                    ], style=dict(flex="0 0 auto", paddingRight="20px")),

                    # Metrics
                    html.Div([
                        _metric_row("Edge Radius",
                                    f"{edge_r:.1f} px" if not _nan(edge_r) else "N/A"),
                        _metric_row("Tool Length",
                                    f"{tool_len:.0f} px" if not _nan(tool_len) else "N/A"),
                        _metric_row("Worn Area",
                                    f"{worn_area:.0f} px" if not _nan(worn_area) else "N/A"),
                        _metric_row("Plateau Score",
                                    f"{ch.get('plateau_score', 0.0):.3f}"),
                        _metric_row("PF Mean (at this ch)",
                                    f"{ch.get('pf_mean', 0.0):.3f}"),
                        _metric_row("Critical Prob",
                                    f"{ch.get('critical_prob', 0.0)*100:.1f}%"),
                        _metric_row("RUL after image",
                                    f"{ch.get('rul', '-')} channels"),
                    ], style=dict(flex="1")),
                ], style=dict(display="flex", alignItems="flex-start")),
            ], style=_CARD)]

        # ==============================================================
        # Decision log
        # ==============================================================
        log      = st.get("decision_log", [])
        log_data = list(reversed(log[-20:]))

        # ==============================================================
        # Status card values (always from latest state)
        # ==============================================================
        s_col  = C.get(tool_state, "#aaa")
        d_col  = C.get(decision,   "#aaa")
        km_col = C.get(kmeans, C.get(tool_state, "#aaa"))

        return (
            header,
            pf_fig,
            tool_state,  tool_state,  _badge(s_col),
            decision,    decision,    _badge(d_col),
            str(rul),
            f"{crit_p*100:.1f}%",
            kmeans,      kmeans,      _badge(km_col),
            force_fig,
            sci_fig,
            ftfn_fig,
            hist_fig,
            image_panel,
            log_data,
        )

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nan(v: Any) -> bool:
    try:
        import math
        return math.isnan(float(v))
    except Exception:
        return True


def _metric_row(label: str, value: str) -> html.Div:
    return html.Div([
        html.Span(label + ":", style=dict(color="#aaa", fontSize="12px",
                                          marginRight="8px", display="inline-block",
                                          width="140px")),
        html.Span(value, style=dict(color="#eaeaea", fontSize="13px",
                                    fontWeight="600")),
    ], style=dict(marginBottom="8px"))
