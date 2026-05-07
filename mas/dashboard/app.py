"""
PF-MAS Plotly Dash Dashboard.

Layout
------
Header  : live status bar (channel | decision | RUL | crit%)
Slider  : channel selector 1-45 (inspect any processed channel)
Row 1   : PF Overview plot (all channels, future CI anchored to last obs)  70%
          Status cards                                                      30%
Row 2   : Raw force Fx/Fy for selected channel                             50%
          Ft/Fn Force History + Predictions (all channels)                 50%
Row 3   : Ft / Fn B-spline curves for selected channel                     50%
          Particle histogram at selected channel                            50%
Row 4   : Image analysis panel (TAKE_IMAGE or REPLACE decisions with image)
Row 5   : Decision log table

PF Overview:
  - coloured observation dots (green=CONTINUE, blue=TAKE_IMAGE, red=REPLACE)
  - I-beam CI error bars
  - gold star on the user-selected channel
  - future trajectory + 90% CI band ANCHORED at the last observed point
  - red dashed critical threshold at 0.75

Ft/Fn Prediction chart:
  - Historical Ft_max and Fn_max dots per channel
  - From the last processed channel: predicted Ft/Fn means + 90% CI bands
    anchored at the last observed value so continuity is preserved
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
    "ft_fill":      "rgba(155,89,182,0.15)",
    "fn_fill":      "rgba(26,188,156,0.15)",
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
                html.Div(dcc.Graph(id="pf-plot", style=dict(height="440px"),
                                   config=dict(displayModeBar=False)),
                         style=dict(width="72%", paddingRight="12px")),
                html.Div([
                    _stat_card("Tool State",      "state-val",    "state-badge"),
                    _stat_card("Leader Decision", "decision-val", "decision-badge"),
                    _stat_card("RUL (channels)",  "rul-val"),
                    _stat_card("Critical Prob.",  "crit-val"),
                    _stat_card("K-Means Result",  "kmeans-val",   "kmeans-badge"),
                ], style=dict(width="28%")),
            ], style=dict(display="flex", marginBottom="16px")),

            # ---- Row 2: Tangential / Normal force chart ----
            html.Div(
                dcc.Graph(id="force-chart", style=dict(height="300px"),
                          config=dict(displayModeBar=False)),
                style=dict(marginBottom="16px"),
            ),

            # ---- Row 3: Image panel ----
            html.Div(id="image-panel", children=[]),

            # ---- Row 3: Particle histogram ----
            html.Div(
                dcc.Graph(id="particle-hist", style=dict(height="280px"),
                          config=dict(displayModeBar=False)),
                style=dict(marginBottom="16px"),
            ),

            # ---- Row 4: Decision log ----
            html.Div([
                html.H5("Decision Log",
                        style=dict(color=C["text"], margin="0 0 8px 0")),
                dash_table.DataTable(
                    id="decision-log",
                    columns=[{"name": c, "id": c} for c in
                             ["Ch", "Signal", "Ft", "Fn", "CI", "Decision",
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
        if latest == 0:
            return current_val
        # Follow if near the latest (within 5 ch lag) OR at the initial default (1)
        near_end     = current_val >= latest - 5
        at_default   = current_val <= 1
        if near_end or at_default:
            return max(1, latest)
        return current_val

    @app.callback(
        [
            Output("header-meta",      "children"),
            Output("pf-plot",          "figure"),
            Output("particle-hist",    "figure"),
            Output("force-chart",      "figure"),
            Output("state-val",        "children"),
            Output("state-badge",      "children"),
            Output("state-badge",      "style"),
            Output("decision-val",     "children"),
            Output("decision-badge",   "children"),
            Output("decision-badge",   "style"),
            Output("rul-val",          "children"),
            Output("crit-val",         "children"),
            Output("kmeans-val",       "children"),
            Output("kmeans-badge",     "children"),
            Output("kmeans-badge",     "style"),
            Output("image-panel",      "children"),
            Output("decision-log",     "data"),
        ],
        [
            Input("interval",  "n_intervals"),
            Input("ch-slider", "value"),
        ],
    )
    def refresh(_n: int, selected_ch: int) -> tuple[Any, ...]:
        with state_lock:
            st = copy.deepcopy(shared_state)

        hist       = st.get("channel_history", {})
        latest_ch  = st.get("current_channel", 0)
        sel        = selected_ch or latest_ch or 1
        ch         = hist.get(sel)

        decision   = st.get("decision",   "CONTINUE")
        tool_state = st.get("tool_state", "FACTORY_NEW")
        rul        = st.get("rul",         45)
        crit_p     = st.get("critical_prob", 0.0)
        kmeans     = st.get("kmeans_result", "-")

        header = (f"Streaming: Ch {latest_ch}/45  |  "
                  f"RUL {rul} ch  |  Crit {crit_p*100:.0f}%  |  "
                  f"Inspecting Ch {sel}")

        # ==============================================================
        # PF Overview
        # Shows observed data for channels 1…sel (gold star at sel),
        # then PF CI from sel forward until the mean crosses threshold.
        # No observed dots beyond the star.
        # ==============================================================
        pf_fig = go.Figure()

        pf_fig.add_hline(
            y=CRITICAL_THRESHOLD,
            line=dict(color=C["threshold"], dash="dash", width=1.5),
            annotation_text=f"Critical Threshold ({CRITICAL_THRESHOLD})",
            annotation_font_color=C["threshold"],
            annotation_position="top right",
        )

        obs     = st.get("observations", [])
        obs_sel = [o for o in obs if o["channel"] <= sel]  # only up to selected

        if obs_sel:
            ch_all  = [o["channel"]  for o in obs_sel]
            sig_all = [o["signal"]   for o in obs_sel]
            ci_lo   = [o["ci_low"]   for o in obs_sel]
            ci_hi   = [o["ci_high"]  for o in obs_sel]
            dec_all = [o["decision"] for o in obs_sel]
            dot_col = [C.get(d, "#aaa") for d in dec_all]

            err_up = [max(0.0, hi - s) for hi, s in zip(ci_hi, sig_all)]
            err_dn = [max(0.0, s - lo) for s, lo in zip(sig_all, ci_lo)]

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
                name="Observed",
                customdata=dec_all,
                hovertemplate="Ch %{x}<br>Signal %{y:.3f}<br>%{customdata}",
            ))

        # Gold star at selected channel (last observed point)
        if ch is not None:
            pf_fig.add_trace(go.Scatter(
                x=[sel], y=[ch["signal"]],
                mode="markers",
                marker=dict(color="gold", size=18, symbol="star",
                            line=dict(color="white", width=1.5)),
                name=f"Ch {sel} (selected)",
            ))

            # --- CI indicator: predicted mean + in/out status ---
            pred_mean = ch.get("ci_mean",
                               (ch["ci_low"] + ch["ci_high"]) / 2.0)
            in_ci = ch["ci_low"] <= ch["signal"] <= ch["ci_high"]
            ci_color  = C["CONTINUE"] if in_ci else C["REPLACE"]
            ci_label  = "In CI" if in_ci else "Out of CI"

            # Hollow blue circle at predicted PF mean
            pf_fig.add_trace(go.Scatter(
                x=[sel], y=[pred_mean],
                mode="markers",
                marker=dict(
                    color="rgba(0,0,0,0)", size=14, symbol="circle",
                    line=dict(color=C["future_line"], width=2.5),
                ),
                name=f"Pred. Mean ({pred_mean:.3f})",
                hovertemplate=f"Ch %{{x}}<br>PF pred. mean {pred_mean:.3f}",
            ))

            # Annotation arrow on the actual observation
            pf_fig.add_annotation(
                x=sel, y=ch["signal"],
                text=ci_label,
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowcolor=ci_color,
                font=dict(color=ci_color, size=11),
                bgcolor=C["card"],
                bordercolor=ci_color,
                borderwidth=1,
                borderpad=3,
                ax=44, ay=-38,
            )

        # CI band anchored at selected channel's stored PF trajectory
        if ch is not None and ch.get("future_channels"):
            c_fut_chs = ch["future_channels"]
            c_fut_lo  = ch["future_ci_low"]
            c_fut_hi  = ch["future_ci_high"]
            c_fut_mn  = ch["future_means"]
            anchor_sig = ch["signal"]
            anchor_ch  = ch["channel"]
        elif obs_sel:
            # Fallback: use global trajectory anchored at last visible observation
            c_fut_chs = st.get("future_channels", [])
            c_fut_lo  = st.get("future_ci_low",   [])
            c_fut_hi  = st.get("future_ci_high",  [])
            c_fut_mn  = st.get("future_means",    [])
            anchor_sig = obs_sel[-1]["signal"]
            anchor_ch  = obs_sel[-1]["channel"]
        else:
            c_fut_chs = []
            c_fut_lo = c_fut_hi = c_fut_mn = []
            anchor_sig = anchor_ch = None

        if c_fut_chs and anchor_ch is not None:
            # Trim at first step where mean crosses threshold
            trim = len(c_fut_mn)
            for _i, _m in enumerate(c_fut_mn):
                if _m >= CRITICAL_THRESHOLD:
                    trim = _i + 1
                    break

            anch_chs = [anchor_ch] + list(c_fut_chs)[:trim]
            anch_hi  = [anchor_sig] + list(c_fut_hi)[:trim]
            anch_lo  = [anchor_sig] + list(c_fut_lo)[:trim]
            anch_mn  = [anchor_sig] + list(c_fut_mn)[:trim]

            pf_fig.add_trace(go.Scatter(
                x=anch_chs + list(reversed(anch_chs)),
                y=anch_hi  + list(reversed(anch_lo)),
                fill="toself", fillcolor=C["ci_fill"],
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip", name="90% Pred. CI",
            ))
            pf_fig.add_trace(go.Scatter(
                x=anch_chs, y=anch_mn,
                mode="lines",
                line=dict(color=C["future_line"], width=2, dash="dot"),
                name="PF Mean Projection",
            ))

        pf_fig.update_layout(
            **_dark_layout(
                "PF Wear Overview — observed up to ★, CI projection until threshold",
                yl="Wear Signal [0-1]"),
        )
        pf_fig.update_yaxes(range=[-0.02, 1.05])

        # ==============================================================
        # Particle histogram for selected channel
        # ==============================================================
        hist_particles = (ch.get("particles") if ch else None) or st.get("particles", [])
        hist_pf_mean   = (ch.get("pf_mean",       0.0)  if ch else st.get("pf_mean",       0.0))
        hist_crit_p    = (ch.get("critical_prob",  0.0)  if ch else st.get("critical_prob", 0.0))

        hist_fig = go.Figure()
        if hist_particles:
            hist_fig.add_trace(go.Histogram(
                x=hist_particles,
                nbinsx=30,
                marker_color="rgba(255,215,0,0.55)",
                marker_line=dict(color="rgba(255,215,0,0.9)", width=0.6),
                name="Particles",
                hovertemplate="Wear %{x:.3f}<br>Count %{y}",
            ))

        hist_fig.add_vline(
            x=CRITICAL_THRESHOLD,
            line=dict(color=C["threshold"], dash="dash", width=2),
            annotation_text=f"Critical ({CRITICAL_THRESHOLD})",
            annotation_font_color=C["threshold"],
            annotation_position="top right",
        )
        hist_fig.add_vline(
            x=hist_pf_mean,
            line=dict(color=C["CONTINUE"], dash="dot", width=1.8),
            annotation_text=f"Mean {hist_pf_mean:.3f}",
            annotation_font_color=C["CONTINUE"],
            annotation_position="top left",
        )
        hist_fig.update_layout(
            **_dark_layout(
                f"Particle Distribution — Ch {sel}  |  Crit {hist_crit_p*100:.0f}%",
                xl="Wear Level [0–1]", yl="Count",
            )
        )
        hist_fig.update_xaxes(range=[0.0, 1.0])

        # ==============================================================
        # Tangential (Ft) / Normal (Fn) force history + predictions
        # ==============================================================
        ft_hist = st.get("ft_history", [])
        force_fig = go.Figure()

        if ft_hist:
            ch_hist  = [e["channel"] for e in ft_hist]
            ft_vals  = [e["ft_max"]  for e in ft_hist]
            fn_vals  = [e["fn_max"]  for e in ft_hist]

            # --- Ft history ---
            force_fig.add_trace(go.Scatter(
                x=ch_hist, y=ft_vals,
                mode="lines+markers",
                name="Ft (tangential)",
                line=dict(color=C["ft"], width=1.8),
                marker=dict(color=C["ft"], size=5),
                hovertemplate="Ch %{x}<br>Ft %{y:.3f} N",
            ))
            # --- Fn history ---
            force_fig.add_trace(go.Scatter(
                x=ch_hist, y=fn_vals,
                mode="lines+markers",
                name="Fn (normal)",
                line=dict(color=C["fn"], width=1.8),
                marker=dict(color=C["fn"], size=5),
                hovertemplate="Ch %{x}<br>Fn %{y:.3f} N",
            ))

            # Highlight selected channel
            sel_entry = next((e for e in ft_hist if e["channel"] == sel), None)
            if sel_entry:
                force_fig.add_trace(go.Scatter(
                    x=[sel], y=[sel_entry["ft_max"]],
                    mode="markers",
                    marker=dict(color="gold", size=14, symbol="star",
                                line=dict(color="white", width=1.2)),
                    name=f"Ft Ch {sel} (selected)", showlegend=False,
                ))
                force_fig.add_trace(go.Scatter(
                    x=[sel], y=[sel_entry["fn_max"]],
                    mode="markers",
                    marker=dict(color="gold", size=14, symbol="star",
                                line=dict(color="white", width=1.2)),
                    name=f"Fn Ch {sel} (selected)", showlegend=False,
                ))

        # --- Ft predicted CI band anchored at last observed ---
        ft_m  = st.get("future_ft_means",   [])
        ft_lo = st.get("future_ft_ci_low",  [])
        ft_hi = st.get("future_ft_ci_high", [])
        fn_m  = st.get("future_fn_means",   [])
        fn_lo = st.get("future_fn_ci_low",  [])
        fn_hi = st.get("future_fn_ci_high", [])

        if ft_hist and ft_m:
            last_ch  = ft_hist[-1]["channel"]
            fut_chs  = list(range(last_ch, last_ch + len(ft_m) + 1))
            last_ft  = ft_hist[-1]["ft_max"]
            last_fn  = ft_hist[-1]["fn_max"]

            anch_ft_m  = [last_ft]  + list(ft_m)
            anch_ft_lo = [last_ft]  + list(ft_lo)
            anch_ft_hi = [last_ft]  + list(ft_hi)
            anch_fn_m  = [last_fn]  + list(fn_m)
            anch_fn_lo = [last_fn]  + list(fn_lo)
            anch_fn_hi = [last_fn]  + list(fn_hi)

            force_fig.add_trace(go.Scatter(
                x=fut_chs + list(reversed(fut_chs)),
                y=anch_ft_hi + list(reversed(anch_ft_lo)),
                fill="toself", fillcolor=C["ft_fill"],
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip", name="Ft 90% CI",
            ))
            force_fig.add_trace(go.Scatter(
                x=fut_chs, y=anch_ft_m,
                mode="lines",
                line=dict(color=C["ft"], width=1.5, dash="dot"),
                name="Ft predicted",
            ))
            force_fig.add_trace(go.Scatter(
                x=fut_chs + list(reversed(fut_chs)),
                y=anch_fn_hi + list(reversed(anch_fn_lo)),
                fill="toself", fillcolor=C["fn_fill"],
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip", name="Fn 90% CI",
            ))
            force_fig.add_trace(go.Scatter(
                x=fut_chs, y=anch_fn_m,
                mode="lines",
                line=dict(color=C["fn"], width=1.5, dash="dot"),
                name="Fn predicted",
            ))

        force_fig.update_layout(
            **_dark_layout("Tangential (Ft) & Normal (Fn) Force History + Prediction",
                           yl="Force [N]")
        )

        # ==============================================================
        # Image panel — shown for any channel that has an image
        # ==============================================================
        image_panel: list = []
        shows_image = ch is not None and bool(ch.get("img_b64"))
        if shows_image:
            enc      = ch["img_b64"]
            analysis = ch.get("img_analysis") or {}
            km_lbl   = ch.get("kmeans_label", "-")
            km_col   = C.get(km_lbl, "#888")

            # Primary KPIs requested: channel, edge radius, tool length, worn area
            edge_r     = analysis.get("edge_radius_px", float("nan"))
            tool_len   = analysis.get("tool_length_px", float("nan"))
            worn_area  = analysis.get("worn_area_px",   float("nan"))
            tool_rad   = analysis.get("tool_radius_px", float("nan"))
            corner_rad = analysis.get("corner_radius_px", float("nan"))

            dec_label = ch.get("decision", "CONTINUE")
            dec_col   = C.get(dec_label, "#aaa")

            # K-Means badge only when an actual classification happened
            km_badge_items: list = []
            if km_lbl != "-":
                km_badge_items = [
                    html.Span(f"K-Means: {km_lbl}",
                              style={**_badge(km_col),
                                     "fontSize": "13px", "padding": "4px 12px"}),
                ]

            image_panel = [html.Div([
                # Title row
                html.Div([
                    html.H5(f"Tool Image — Ch {sel}",
                            style=dict(color=C["text"], display="inline",
                                       margin="0 14px 0 0")),
                    html.Span(dec_label,
                              style={**_badge(dec_col),
                                     "fontSize": "12px", "padding": "3px 10px",
                                     "marginRight": "8px"}),
                    *km_badge_items,
                ], style=dict(marginBottom="12px")),

                # Image + metrics
                html.Div([
                    html.Div([
                        html.Img(
                            src=f"data:image/jpeg;base64,{enc}",
                            style=dict(width="100%", maxWidth="500px",
                                       borderRadius="6px",
                                       border=f"2px solid {km_col}"),
                        ),
                    ], style=dict(flex="0 0 auto", paddingRight="20px")),

                    html.Div([
                        # Primary KPIs first
                        _metric_row("Channel",      str(sel)),
                        _metric_row("Worn Area",
                                    f"{worn_area:.0f} px" if not _nan(worn_area) else "N/A"),
                        _metric_row("Edge Radius",
                                    f"{edge_r:.1f} px"    if not _nan(edge_r)    else "N/A"),
                        _metric_row("Tool Length",
                                    f"{tool_len:.0f} px"  if not _nan(tool_len)  else "N/A"),
                        # Secondary KPIs
                        _metric_row("Tool Radius",
                                    f"{tool_rad:.0f} px"  if not _nan(tool_rad)  else "N/A"),
                        _metric_row("Corner Radius",
                                    f"{corner_rad:.1f} px" if not _nan(corner_rad) else "N/A"),
                        _metric_row("Plateau Score",
                                    f"{ch.get('plateau_score', 0.0):.3f}"),
                        _metric_row("Ft (tangential)",
                                    f"{ch.get('ft_max', 0.0):.3f} N"),
                        _metric_row("Fn (normal)",
                                    f"{ch.get('fn_max', 0.0):.3f} N"),
                        _metric_row("PF Mean",
                                    f"{ch.get('pf_mean', 0.0):.3f}"),
                        _metric_row("Critical Prob",
                                    f"{ch.get('critical_prob', 0.0)*100:.1f}%"),
                        _metric_row("RUL",
                                    f"{ch.get('rul', '-')} channels"),
                    ], style=dict(flex="1")),
                ], style=dict(display="flex", alignItems="flex-start")),
            ], style=_CARD)]

        # ==============================================================
        # Decision log
        # ==============================================================
        log      = st.get("decision_log", [])
        log_data = list(reversed(log[-45:]))

        # ==============================================================
        # Status card values
        # ==============================================================
        s_col  = C.get(tool_state, "#aaa")
        d_col  = C.get(decision,   "#aaa")
        km_col = C.get(kmeans, C.get(tool_state, "#aaa"))

        return (
            header,
            pf_fig,
            hist_fig,
            force_fig,
            tool_state,  tool_state,  _badge(s_col),
            decision,    decision,    _badge(d_col),
            str(rul),
            f"{crit_p*100:.1f}%",
            kmeans,      kmeans,      _badge(km_col),
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
                                          width="160px")),
        html.Span(value, style=dict(color="#eaeaea", fontSize="13px",
                                    fontWeight="600")),
    ], style=dict(marginBottom="8px"))
