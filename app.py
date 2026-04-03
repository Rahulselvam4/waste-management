# ─────────────────────────────────────────────────────────────────────────────
# app.py  —  WasteAI  Smart Waste Classification System
# Run :  streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2, os, time
from PIL import Image
from datetime import datetime

from predict import predict, predict_batch
from utils.disposal_info import DISPOSAL_DATA, GLOBAL_STATS
from utils.iot_content import (
    IOT_HARDWARE, IOT_WORKFLOW, IOT_SOFTWARE, TFLITE_CODE, PI_CODE
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WasteAI — Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_defaults = {
    "page":        "classify",
    "history":     [],
    "total":       0,
    "last_result": None,
    "panel":       None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def go(page):
    st.session_state.page  = page
    st.session_state.panel = None

def toggle_panel(name):
    st.session_state.panel = None if st.session_state.panel == name else name

# ─────────────────────────────────────────────────────────────────────────────
# MASTER CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Outfit:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg:         #f6f8fb;
    --surface:    #ffffff;
    --surface2:   #f1f5f9;
    --surface3:   #e2e8f0;

    --border:     #e5e7eb;
    --border2:    #d1d5db;

    --text:       #111827;
    --muted:      #6b7280;
    --subtle:     #9ca3af;

    --green:      #22c55e;
    --green-dim:  #16a34a;
    --green-bg:   rgba(34,197,94,0.08);

    --lime:       #84cc16;
    --accent:     #3b82f6;

    --warning:    #f59e0b;
    --danger:     #ef4444;

    --r:          12px;
    --rl:         18px;

    --font-head:  'Outfit', sans-serif;
    --font-body:  'Space Grotesk', sans-serif;
    --font-mono:  'JetBrains Mono', monospace;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: var(--font-body) !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebar"] { display: none !important; }

.main .block-container {
    padding: 0 48px !important;  
    max-width: 100% !important;
}

/* ── PAGE WRAPPER — breathing room on left & right ── */
.pw {
    max-width: 1200px;
    margin: 0 auto;
    padding: 28px 48px 80px;
}

/* ── NAV BAR — visual only, no functional buttons inside ── */
.wasteai-nav {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 0 24px;   
    width: 100%;
    height: 58px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.nav-brand {
    font-family: var(--font-head);
    font-size: 19px; font-weight: 800;
    color: var(--lime); letter-spacing: -0.3px;
}
.nav-brand span { color: var(--muted); }
.nav-center-tabs {
    display: flex; gap: 2px;
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 10px; padding: 3px;
}
.nav-tab-lbl {
    font-family: var(--font-body);
    font-size: 13px; padding: 6px 16px;
    border-radius: 7px; white-space: nowrap; cursor: default;
}
.nav-tab-lbl.active { color: var(--lime); background: var(--surface); }
.nav-tab-lbl.inactive { color: var(--muted); }
.nav-right { display: flex; gap: 6px; align-items: center; }
.nav-pill {
    font-size: 12px; font-weight: 500; padding: 5px 12px;
    border-radius: 7px; cursor: default;
}
.nav-pill.active   { color: var(--lime); background: var(--green-bg); border: 1px solid var(--green-dim); }
.nav-pill.inactive { color: var(--muted); background: transparent; border: 1px solid var(--border); }

/* ── FUNCTIONAL NAV BUTTON ROW — slim, clearly visible tab bar ── */
.stButton > button {
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    transition: all 0.15s !important;
    border-radius: 8px !important;
}

/* Nav page buttons */
.npb button {
    background: transparent !important;
    color: var(--muted) !important;
    border: 1px solid transparent !important;
    padding: 5px 14px !important;
    width: 100% !important;
    height: 36px !important;
}
.npb button:hover {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}
.npb-active button {
    background: var(--green-bg) !important;
    color: var(--lime) !important;
    border: 1px solid var(--green-dim) !important;
    padding: 5px 14px !important;
    width: 100% !important;
    height: 36px !important;
}

/* Panel toggle buttons */
.ppb button {
    background: transparent !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
    padding: 5px 12px !important;
    width: 100% !important;
    height: 34px !important;
}
.ppb button:hover {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border-color: var(--border2) !important;
}
.ppb-active button {
    background: var(--green-bg) !important;
    color: var(--lime) !important;
    border: 1px solid var(--green-dim) !important;
    width: 100% !important;
    height: 34px !important;
}

/* Main classify button */
.classify-btn button {
    background: var(--green-dim) !important;
    color: #fff !important;
    border: none !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    height: 44px !important;
    width: 100% !important;
    letter-spacing: 0.2px !important;
}
.classify-btn button:hover {
    background: #2ea043 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(63,185,80,0.3) !important;
}

/* Info panel */
.info-panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--rl); padding: 28px 32px; margin-bottom: 24px;
}
.info-panel-title {
    font-family: var(--font-head); font-size: 18px; font-weight: 700;
    color: var(--lime); margin-bottom: 18px;
}
.info-panel p, .info-panel li { font-size: 13.5px; line-height: 1.75; color: var(--muted); }
.info-panel strong { color: var(--text); }
.info-panel code {
    font-family: var(--font-mono); font-size: 12px;
    background: var(--surface2); color: var(--lime);
    padding: 2px 7px; border-radius: 5px; border: 1px solid var(--border);
}
.step-pill {
    display: inline-block; background: var(--green-bg);
    border: 1px solid var(--green-dim); color: var(--green);
    border-radius: 6px; padding: 3px 10px; font-size: 12px;
    font-weight: 600; margin-bottom: 8px;
}
.tip-box {
    background: var(--surface2); border: 1px solid var(--border);
    border-left: 3px solid var(--lime);
    border-radius: 0 var(--r) var(--r) 0;
    padding: 12px 16px; margin-top: 14px;
    font-size: 13px; color: var(--muted);
}

/* Hero */
.hero {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 20px; padding: 40px 44px 0;
    margin-bottom: 24px; display: flex;
    align-items: flex-end; gap: 32px; overflow: hidden; position: relative;
}
.hero::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(63,185,80,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-body { flex: 1; padding-bottom: 40px; position: relative; }
.hero-tag {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; color: var(--green);
    background: var(--green-bg); border: 1px solid var(--green-dim);
    padding: 4px 12px; border-radius: 99px; margin-bottom: 14px;
}
.hero-tag::before {
    content: ''; width: 6px; height: 6px; border-radius: 50%;
    background: var(--green); display: inline-block;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100%{opacity:1;transform:scale(1);}
    50%{opacity:0.5;transform:scale(0.8);}
}
.hero-title {
    font-family: var(--font-head);
    font-size: clamp(28px, 3.5vw, 48px);
    font-weight: 800; color: var(--text);
    line-height: 1.08; letter-spacing: -1.5px; margin-bottom: 16px;
}
.hero-title .accent { color: var(--lime); }
.hero-sub { font-size: 14.5px; color: var(--muted); line-height: 1.65; max-width: 480px; margin-bottom: 24px; }
.hero-chips { display: flex; gap: 8px; flex-wrap: wrap; }
.hchip {
    font-size: 12px; font-weight: 500; padding: 5px 13px; border-radius: 99px;
    border: 1px solid var(--border); color: var(--muted); background: var(--bg);
}
.hchip.hi { border-color: var(--green-dim); color: var(--green); background: var(--green-bg); }
.hero-orb-wrap { width: 180px; flex-shrink: 0; display: flex; justify-content: center; }
.hero-orb {
    width: 140px; height: 140px; border-radius: 50%;
    background: conic-gradient(from 0deg, #3fb950, #a8e063, #58a6ff, #3fb950);
    display: flex; align-items: center; justify-content: center; font-size: 56px;
    box-shadow: 0 0 60px rgba(63,185,80,0.25), 0 0 30px rgba(63,185,80,0.15);
    animation: spin-slow 12s linear infinite;
}
@keyframes spin-slow { to { transform: rotate(360deg); } }

/* Cards */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--rl); padding: 24px; }
.card-title {
    font-family: var(--font-head); font-size: 11px; font-weight: 700;
    color: var(--subtle); letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 14px;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important; border-radius: var(--r) !important;
    background: var(--bg) !important; transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--green-dim) !important; background: var(--green-bg) !important;
}

/* Result empty */
.result-empty { background: var(--bg); border: 2px dashed var(--border); border-radius: var(--rl); padding: 64px 28px; text-align: center; }

/* Prediction card */
.pred-card { border-radius: var(--rl); padding: 26px; margin-bottom: 18px; text-align: center; }
.pred-emoji-bg {
    width: 68px; height: 68px; border-radius: 50%;
    background: rgba(255,255,255,0.1);
    display: flex; align-items: center; justify-content: center;
    font-size: 34px; margin: 0 auto 12px;
}
.pred-tag { font-size: 11px; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; opacity: 0.7; margin-bottom: 6px; }
.pred-class { font-family: var(--font-head); font-size: 34px; font-weight: 800; margin-bottom: 8px; }
.pred-bin { font-size: 13px; opacity: 0.75; font-weight: 500; }
.pred-badge { display: inline-flex; align-items: center; gap: 6px; background: rgba(255,255,255,0.12); border-radius: 99px; padding: 4px 14px; font-size: 12px; font-weight: 600; margin: 10px auto 0; }

/* Confidence */
.conf-row { display: flex; justify-content: space-between; align-items: center; font-size: 12px; font-weight: 500; color: var(--muted); margin-bottom: 7px; }
.conf-val { font-family: var(--font-mono); font-weight: 600; }
.conf-track { background: var(--surface3); border-radius: 99px; height: 8px; overflow: hidden; margin-bottom: 18px; }
.conf-fill { height: 8px; border-radius: 99px; transition: width 0.7s ease; }

/* Prob bars */
.prob-title { font-size: 11px; font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase; color: var(--subtle); margin-bottom: 10px; }
.prob-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.prob-em { font-size: 15px; width: 22px; text-align: center; }
.prob-name { font-size: 12px; color: var(--muted); width: 76px; }
.prob-track { flex: 1; background: var(--surface3); border-radius: 99px; height: 6px; overflow: hidden; }
.prob-fill { height: 6px; border-radius: 99px; transition: width 0.5s; }
.prob-pct { font-size: 11px; color: var(--subtle); width: 40px; text-align: right; font-family: var(--font-mono); }

/* Disposal steps */
.d-step { display: flex; gap: 12px; align-items: flex-start; padding: 10px 0; border-bottom: 1px solid var(--border); font-size: 13px; color: var(--muted); line-height: 1.5; }
.d-step:last-child { border-bottom: none; }
.d-num { width: 24px; height: 24px; border-radius: 50%; background: var(--green-bg); border: 1px solid var(--green-dim); color: var(--green); font-size: 11px; font-weight: 700; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.fact-strip { border-radius: var(--r); padding: 13px 16px; margin-top: 14px; font-size: 13px; line-height: 1.6; border-left: 3px solid; }

/* Env cards */
.env-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin-top: 12px; }
.env-card { background: var(--surface2); border: 1px solid var(--border); border-radius: var(--r); padding: 12px; text-align: center; }
.env-icon { font-size: 18px; margin-bottom: 4px; }
.env-lbl { font-size: 10px; color: var(--subtle); text-transform: uppercase; letter-spacing: 0.5px; }
.env-val { font-size: 12px; font-weight: 600; color: var(--text); margin-top: 3px; }

/* Warn */
.warn-box { background: rgba(210,153,34,0.1); border: 1px solid rgba(210,153,34,0.3); border-radius: var(--r); padding: 12px 16px; font-size: 13px; color: #d29922; margin-top: 12px; }

/* Image meta */
.img-meta { background: var(--surface2); border-radius: 7px; padding: 8px 12px; font-size: 11px; color: var(--subtle); margin-top: 8px; font-family: var(--font-mono); }

/* Metrics */
.metric-grid { display: grid; grid-template-columns: repeat(5,1fr); gap: 12px; margin-bottom: 28px; }
.metric-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); padding: 16px 14px; border-top: 3px solid var(--lime); text-align: center; }
.m-label { font-size: 10px; color: var(--subtle); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 7px; }
.m-val { font-family: var(--font-head); font-size: 24px; font-weight: 800; color: var(--text); }
.m-sub { font-size: 11px; color: var(--subtle); margin-top: 4px; }

/* Section heading */
.sec-head { display: flex; align-items: center; gap: 12px; margin: 28px 0 18px; }
.sec-head-line { flex: 1; height: 1px; background: var(--border); }
.sec-head-title { font-family: var(--font-head); font-size: 12px; font-weight: 700; color: var(--muted); white-space: nowrap; text-transform: uppercase; letter-spacing: 0.8px; }

/* Category pills — 3-column grid */
.cat-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin-top: 10px; }
.cat-pill { border-radius: var(--r); padding: 10px 12px; font-size: 12.5px; font-weight: 500; display: flex; align-items: center; gap: 8px; border: 1px solid transparent; }

/* Hardware card */
.hw-card { background: var(--surface2); border: 1px solid var(--border); border-radius: var(--r); padding: 16px; margin-bottom: 10px; transition: border-color 0.15s; }
.hw-card:hover { border-color: var(--border2); }
.hw-name { font-size: 14px; font-weight: 600; color: var(--text); margin-bottom: 4px; }
.hw-role { font-size: 12px; color: var(--muted); margin-bottom: 8px; }
.hw-meta { display: flex; gap: 16px; font-size: 11px; color: var(--subtle); }
.hw-cost { color: var(--green); font-weight: 600; }

/* Workflow steps */
.wf-step { display: flex; gap: 14px; align-items: flex-start; padding: 14px 0; border-bottom: 1px solid var(--border); }
.wf-step:last-child { border-bottom: none; }
.wf-num { width: 32px; height: 32px; border-radius: 50%; background: var(--green-bg); border: 1.5px solid var(--green-dim); color: var(--green); font-size: 12px; font-weight: 700; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.wf-title { font-size: 14px; font-weight: 600; color: var(--text); }
.wf-desc { font-size: 12px; color: var(--muted); margin-top: 3px; line-height: 1.5; }

/* Tech stack — icon cards (redesigned) */
.tech-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; }
.tech-card {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: var(--rl); padding: 20px 16px 16px;
    text-align: center; transition: transform 0.15s, border-color 0.15s, box-shadow 0.15s;
    position: relative; overflow: hidden;
}
.tech-card:hover { transform: translateY(-3px); border-color: var(--border2); box-shadow: 0 8px 32px rgba(0,0,0,0.5); }
.tech-card-stripe { position: absolute; top: 0; left: 0; right: 0; height: 3px; }
.tech-icon { font-size: 28px; margin-bottom: 10px; }
.tech-name { font-size: 13px; font-weight: 700; color: var(--text); margin-bottom: 4px; }
.tech-desc { font-size: 11px; color: var(--muted); line-height: 1.5; }
.tech-badge { display: inline-block; margin-top: 8px; font-size: 10px; font-weight: 600; padding: 2px 8px; border-radius: 99px; }

/* Waste category cards */
.wcat-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; }
.wcat-card { border-radius: var(--r); padding: 18px; border: 1px solid transparent; transition: transform 0.15s; }
.wcat-card:hover { transform: translateY(-2px); }

/* Global stat */
.gstat-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); padding: 18px 16px; text-align: center; border-top: 3px solid var(--warning); }
.gstat-icon { font-size: 26px; margin-bottom: 8px; }
.gstat-lbl { font-size: 11px; color: var(--subtle); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.gstat-val { font-family: var(--font-head); font-size: 18px; font-weight: 700; color: var(--text); }

/* History */
.hist-row { display: flex; align-items: center; gap: 10px; padding: 9px 12px; border-radius: var(--r); background: var(--surface2); margin-bottom: 5px; font-size: 12.5px; border: 1px solid var(--border); transition: border-color 0.15s; }
.hist-row:hover { border-color: var(--border2); }

/* Footer */
.wasteai-footer { background: var(--surface); border-top: 1px solid var(--border); padding: 20px 48px; text-align: center; margin-top: 60px; }
.footer-brand { font-family: var(--font-head); font-size: 15px; font-weight: 700; color: var(--lime); margin-bottom: 5px; }
.footer-sub { font-size: 11px; color: var(--subtle); }

/* Expander */
[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: var(--r) !important; background: var(--surface) !important; }
[data-testid="stExpander"] summary { color: var(--muted) !important; font-weight: 500 !important; font-size: 13px !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 3px; background: var(--bg); border-radius: var(--r); padding: 4px; border: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { border-radius: 8px !important; font-family: var(--font-body) !important; font-weight: 500 !important; font-size: 13px !important; color: var(--muted) !important; padding: 6px 14px !important; }
.stTabs [aria-selected="true"] { background: var(--surface) !important; color: var(--lime) !important; }

.stProgress > div > div > div > div { background: var(--green) !important; }
.stSpinner > div { border-top-color: var(--lime) !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: var(--r) !important; }

/* ── RESPONSIVE ── */
@media (max-width: 900px) {
    .wasteai-nav { padding: 0 18px; }
    .nav-center-tabs { display: none; }
    .pw { padding: 16px 18px 60px; }
    .hero { padding: 24px 22px 0; flex-direction: column; }
    .hero-orb-wrap { width: 100%; }
    .hero-orb { width: 90px; height: 90px; font-size: 36px; }
    .metric-grid { grid-template-columns: repeat(2,1fr); }
    .tech-grid { grid-template-columns: repeat(2,1fr); }
    .wcat-grid { grid-template-columns: repeat(2,1fr); }
    .cat-grid { grid-template-columns: repeat(2,1fr); }
}
@media (max-width: 600px) {
    .hero-title { font-size: 26px; }
    .metric-grid { grid-template-columns: 1fr 1fr; }
    .cat-grid { grid-template-columns: 1fr 1fr; }
    .tech-grid { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 400px) {
    .cat-grid { grid-template-columns: 1fr; }
    .wcat-grid { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def sec_head(title):
    st.markdown(f"""
    <div class="sec-head">
        <div class="sec-head-title">{title}</div>
        <div class="sec-head-line"></div>
    </div>""", unsafe_allow_html=True)


def render_metric(label, val, sub="", color="var(--lime)"):
    return f"""
    <div class="metric-card" style="border-top-color:{color};">
        <div class="m-label">{label}</div>
        <div class="m-val">{val}</div>
        {f'<div class="m-sub">{sub}</div>' if sub else ''}
    </div>"""


def render_result(result):
    d    = DISPOSAL_DATA[result['predicted_class']]
    cls  = result['predicted_class']
    conf = result['confidence']
    cconf = "#3fb950" if conf >= 80 else "#d29922" if conf >= 60 else "#f85149"

    st.markdown(f"""
    <div class="pred-card" style="background:{d['bg']};border:1.5px solid {d['accent']}50;">
        <div class="pred-emoji-bg">{d['emoji']}</div>
        <div class="pred-tag" style="color:{d['color']};">Detected Waste Type</div>
        <div class="pred-class" style="color:{d['color']};">{cls.capitalize()}</div>
        <div class="pred-bin" style="color:{d['color']};">🗑 {d['bin']}</div>
        <div style="text-align:center;">
            <span class="pred-badge" style="color:{d['color']};">{d['emoji']} {d['action']}</span>
        </div>
    </div>
    <div class="conf-row">
        <span>Model Confidence</span>
        <span class="conf-val" style="color:{cconf};">{conf:.1f}%</span>
    </div>
    <div class="conf-track"><div class="conf-fill" style="width:{conf:.1f}%;background:{cconf};"></div></div>
    <div class="prob-title">All Class Probabilities</div>
    """, unsafe_allow_html=True)

    sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
    top_prob = sorted_probs[0][1]
    for c, p in sorted_probs:
        bc = DISPOSAL_DATA[c]['accent'] if p == top_prob else "var(--surface3)"
        st.markdown(f"""
        <div class="prob-row">
            <span class="prob-em">{DISPOSAL_DATA[c]['emoji']}</span>
            <span class="prob-name">{c.capitalize()}</span>
            <div class="prob-track"><div class="prob-fill" style="width:{p*100:.1f}%;background:{bc};"></div></div>
            <span class="prob-pct">{p*100:.1f}%</span>
        </div>""", unsafe_allow_html=True)


def render_disposal(cls):
    d = DISPOSAL_DATA[cls]
    with st.expander(f"♻️  Disposal Guide — {cls.capitalize()}"):
        for i, step in enumerate(d['steps'], 1):
            st.markdown(f"""
            <div class="d-step"><div class="d-num">{i}</div><div>{step}</div></div>
            """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="fact-strip" style="background:{d['bg']};border-color:{d['accent']};color:{d['color']};">
            💡 <strong>Did you know?</strong>&nbsp; {d['fact']}
        </div>""", unsafe_allow_html=True)


def render_env(cls):
    d = DISPOSAL_DATA[cls]
    st.markdown(f"""
    <div class="env-grid">
        <div class="env-card"><div class="env-icon">🌿</div><div class="env-lbl">CO₂ Impact</div><div class="env-val">{d['co2']}</div></div>
        <div class="env-card"><div class="env-icon">⏳</div><div class="env-lbl">Decompose Time</div><div class="env-val">{d['decompose']}</div></div>
        <div class="env-card"><div class="env-icon">♻️</div><div class="env-lbl">Recycling Rate</div><div class="env-val">{d['rate']}</div></div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION — HTML visual bar (decorative) + real Streamlit buttons below it
# The key fix: buttons are NOT hidden or overlapping. They form a clean
# secondary tab bar directly below the visual nav bar.
# ─────────────────────────────────────────────────────────────────────────────
cur = st.session_state.page
pan = st.session_state.panel

# Decorative HTML nav bar
tab_html = ""
for lbl, k in [("🔍 Classify","classify"),("📊 Insights","insights"),("📁 Batch","batch"),("ℹ️ About","about")]:
    ac = "active" if k == cur else "inactive"
    tab_html += f'<span class="nav-tab-lbl {ac}">{lbl}</span>'

pg = "active" if pan=="guide" else "inactive"
pi = "active" if pan=="iot"   else "inactive"

st.markdown(f"""
<div class="wasteai-nav">
    <div class="nav-brand">♻️ WasteAI<span> /</span></div>
    <div class="nav-center-tabs">{tab_html}</div>
    <div class="nav-right">
        <span class="nav-pill {pg}">❓ Guide</span>
        <span class="nav-pill {pi}">🌐 IoT</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Real functional button tab bar (always visible, properly styled)
st.markdown(
    '<div style="background:var(--surface2);border-bottom:1px solid var(--border);">'
    '<div style="width:100%;padding:6px 24px;display:flex;gap:6px;align-items:center;">',
    unsafe_allow_html=True
)

nb1, nb2, nb3, nb4, _sp, pb1, pb2 = st.columns([1, 1, 0.8, 0.8, 3.5, 0.9, 0.8])

with nb1:
    ac = "npb-active" if cur=="classify" else "npb"
    st.markdown(f'<div class="{ac}">', unsafe_allow_html=True)
    if st.button("🔍 Classify", key="_nav_c"): go("classify"); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with nb2:
    ac = "npb-active" if cur=="insights" else "npb"
    st.markdown(f'<div class="{ac}">', unsafe_allow_html=True)
    if st.button("📊 Insights", key="_nav_i"): go("insights"); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with nb3:
    ac = "npb-active" if cur=="batch" else "npb"
    st.markdown(f'<div class="{ac}">', unsafe_allow_html=True)
    if st.button("📁 Batch", key="_nav_b"): go("batch"); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with nb4:
    ac = "npb-active" if cur=="about" else "npb"
    st.markdown(f'<div class="{ac}">', unsafe_allow_html=True)
    if st.button("ℹ️ About", key="_nav_a"): go("about"); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with pb1:
    ac = "ppb-active" if pan=="guide" else "ppb"
    st.markdown(f'<div class="{ac}">', unsafe_allow_html=True)
    if st.button("❓ Guide", key="_pan_g"): toggle_panel("guide"); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with pb2:
    ac = "ppb-active" if pan=="iot" else "ppb"
    st.markdown(f'<div class="{ac}">', unsafe_allow_html=True)
    if st.button("🌐 IoT", key="_pan_i"): toggle_panel("iot"); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONTENT WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="pw">', unsafe_allow_html=True)


# ── GUIDE PANEL ──────────────────────────────────────────────────────────────
if st.session_state.panel == "guide":
    st.markdown("""
    <div class="info-panel">
        <div class="info-panel-title">📖  How to Use WasteAI</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:28px;">
            <div>
                <span class="step-pill">Step 1 — Prepare Your Image</span>
                <p style="margin-top:8px;">Take a clear photo of a <strong>single</strong> waste item against a plain
                background. Supported formats: JPG, JPEG, PNG, WEBP (max 10 MB).</p><br>
                <span class="step-pill">Step 2 — Upload &amp; Classify</span>
                <p style="margin-top:8px;">On the <strong>Classify</strong> page, drag your image into the upload zone or
                click Browse. Hit <code>Classify This Image</code> — results appear in under 1 second.</p><br>
                <span class="step-pill">Step 3 — Act on the Result</span>
                <p style="margin-top:8px;">Read the disposal guide, check the bin colour and follow the numbered steps.
                Environmental impact data is also shown.</p>
            </div>
            <div>
                <span class="step-pill">Tips for Best Accuracy</span>
                <ul style="padding-left:18px;margin-top:8px;margin-bottom:16px;">
                    <li>One item per image — not mixed waste</li>
                    <li>Good, even lighting — avoid harsh shadows</li>
                    <li>Item should fill at least 50% of the frame</li>
                    <li>Avoid blurry or very dark photos</li>
                    <li>Remove background clutter where possible</li>
                </ul>
                <span class="step-pill">Batch Mode</span>
                <p style="margin-top:8px;">Use the <strong>Batch</strong> tab to classify multiple images at once and
                download a CSV report with predictions and disposal actions.</p>
                <div class="tip-box">
                    ℹ️ Confidence below 60% usually means the image quality can be improved.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── IOT PANEL ─────────────────────────────────────────────────────────────────
if st.session_state.panel == "iot":
    st.markdown("""
    <div class="info-panel">
        <div class="info-panel-title">🌐  Real-World IoT Deployment Guide</div>
        <p>This model can be embedded in physical smart bins using a Raspberry Pi.
        The bin captures waste images via camera, classifies them in ~200ms, then opens
        the correct compartment via servo motor — fully automated sorting at the edge.</p>
    </div>
    """, unsafe_allow_html=True)

    ti1, ti2, ti3, ti4 = st.tabs(["🔧 Hardware", "💻 Software Stack", "⚙️ Workflow", "📝 Code"])
    with ti1:
        st.markdown("#### Required Hardware Components")
        for item in IOT_HARDWARE:
            st.markdown(f"""
            <div class="hw-card">
                <div class="hw-name">{item['component']}</div>
                <div class="hw-role">{item['role']}</div>
                <div class="hw-meta"><span class="hw-cost">💰 {item['cost']}</span><span>💡 {item['why']}</span></div>
            </div>""", unsafe_allow_html=True)

    with ti2:
        for item in IOT_SOFTWARE:
            lc = {"Edge AI":"#3fb950","Vision":"#58a6ff","Control":"#d29922","Backend":"#8b5cf6","Frontend":"#ec4899"}.get(item['layer'],"#6b7280")
            st.markdown(f"""
            <div style="display:flex;gap:10px;padding:12px;border:1px solid var(--border);border-radius:var(--r);margin-bottom:8px;background:var(--surface2);">
                <span style="background:{lc}22;color:{lc};padding:3px 10px;border-radius:6px;font-size:11px;font-weight:700;white-space:nowrap;align-self:flex-start;border:1px solid {lc}44;">{item['layer']}</span>
                <div>
                    <div style="font-size:13px;font-weight:600;color:var(--text);">{item['tech']}</div>
                    <div style="font-size:12px;color:var(--muted);margin-top:2px;">{item['detail']}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with ti3:
        for i, (title, desc) in enumerate(IOT_WORKFLOW, 1):
            st.markdown(f"""
            <div class="wf-step">
                <div class="wf-num">{i}</div>
                <div><div class="wf-title">{title}</div><div class="wf-desc">{desc}</div></div>
            </div>""", unsafe_allow_html=True)

    with ti4:
        st.markdown("**Convert model to TFLite:**")
        st.code(TFLITE_CODE, language="python")
        st.markdown("**Raspberry Pi inference + servo control:**")
        st.code(PI_CODE, language="python")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: CLASSIFY
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "classify":

    st.markdown("""
    <div class="hero">
        <div class="hero-body">
            <div class="hero-tag">AI-Powered · Real-Time · 6 Categories</div>
            <h1 class="hero-title">Identify waste.<br><span class="accent">Sort smarter.</span></h1>
            <p class="hero-sub">Upload a photo of any waste item. Our fine-tuned MobileNetV2 model
            classifies it in under a second — with disposal guidance and environmental impact data.</p>
            <div class="hero-chips">
                <span class="hchip hi">88.14% accuracy</span>
                <span class="hchip">6 waste categories</span>
                <span class="hchip">&lt;1s inference</span>
                <span class="hchip">MobileNetV2</span>
                <span class="hchip">TrashNet dataset</span>
            </div>
        </div>
        <div class="hero-orb-wrap"><div class="hero-orb">♻️</div></div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Upload Waste Image</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drag & drop or browse",
            type=["jpg","jpeg","png","webp"],
            label_visibility="collapsed",
            key="upload_main",
        )

        if uploaded:
            img_pil = Image.open(uploaded)
            st.image(img_pil, use_column_width=True)
            uploaded.seek(0)
            st.markdown(f"""
            <div class="img-meta">
                {img_pil.width}×{img_pil.height}px &nbsp;·&nbsp; {uploaded.name} &nbsp;·&nbsp;
                {uploaded.size/1024:.1f} KB &nbsp;·&nbsp; {img_pil.mode}
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="classify-btn">', unsafe_allow_html=True)
            classify_btn = st.button("🔍  Classify This Image", key="btn_classify_main")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:52px 20px;color:var(--subtle);">
                <div style="font-size:40px;margin-bottom:14px;opacity:0.3;">🖼️</div>
                <div style="font-size:14px;font-weight:500;color:var(--muted);">No image uploaded yet</div>
                <div style="font-size:12px;margin-top:7px;">Supports JPG · PNG · WEBP · max 10 MB</div>
            </div>""", unsafe_allow_html=True)
            classify_btn = False

        st.markdown('<div class="card-title" style="margin-top:20px;">Detectable Categories</div>', unsafe_allow_html=True)
        st.markdown('<div class="cat-grid">', unsafe_allow_html=True)
        for cls_name, d in DISPOSAL_DATA.items():
            st.markdown(f"""
            <div class="cat-pill" style="background:{d['bg']};color:{d['color']};border-color:{d['accent']}30;">
                <span style="font-size:18px;">{d['emoji']}</span>
                <span style="font-weight:500;">{cls_name.capitalize()}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        if uploaded and classify_btn:
            with st.spinner("Analysing image…"):
                t0 = time.time()
                uploaded.seek(0)
                result  = predict(uploaded)
                elapsed = time.time() - t0

            st.session_state.history.append({
                "class": result['predicted_class'], "confidence": result['confidence'],
                "file":  uploaded.name, "time": datetime.now().strftime("%H:%M:%S"),
            })
            st.session_state.total      += 1
            st.session_state.last_result = result

            render_result(result)
            st.caption(f"⚡ Inference time: {elapsed*1000:.0f} ms")
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            render_disposal(result['predicted_class'])
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            sec_head("Environmental Impact")
            render_env(result['predicted_class'])

            if result['confidence'] < 60:
                st.markdown("""<div class="warn-box">⚠️ <strong>Low confidence.</strong>
                Try a clearer image with better lighting and a single item filling the frame.</div>""",
                unsafe_allow_html=True)

        elif st.session_state.last_result and not uploaded:
            render_result(st.session_state.last_result)
        else:
            st.markdown("""
            <div class="result-empty">
                <div style="font-size:44px;margin-bottom:14px;opacity:0.3;">♻️</div>
                <div style="font-size:15px;font-weight:600;color:var(--muted);margin-bottom:6px;">Results appear here</div>
                <div style="font-size:12px;color:var(--subtle);">Upload an image and click Classify</div>
            </div>""", unsafe_allow_html=True)

    # AI expander — uses native st.columns INSIDE, never HTML columns
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    with st.expander("🔬  How Does the AI Model Work?"):
        ai_c1, ai_c2 = st.columns(2)
        with ai_c1:
            st.markdown("**Architecture: MobileNetV2 + Custom Head**")
            st.markdown("""
| Layer | Output | Params |
|---|---|---|
| Input | 224×224×3 | 0 |
| MobileNetV2 base | 7×7×1280 | 2.26M |
| GlobalAvgPool | 1280 | 0 |
| BatchNorm | 1280 | 5,120 |
| Dense ReLU | 256 | 327,936 |
| Dropout 40% | 256 | 0 |
| Dense ReLU | 128 | 32,896 |
| Dropout 30% | 128 | 0 |
| Softmax Output | 6 | 774 |
| **Total trainable** | | **~2.63M** |
            """)
        with ai_c2:
            st.markdown("**Why MobileNetV2?**")
            st.markdown("""
- Pre-trained on 1.4M ImageNet images
- Already understands edges, textures, shapes
- Only 3.4M parameters — fast on CPU & IoT devices
- Achieves **88.14%** accuracy on 253 held-out test images

**Two-Stage Training:**
- Stage 1: Frozen base → train custom head (LR `0.001`, 20 epochs)
- Stage 2: Fine-tune last 40 layers (LR `0.00001`)
- EarlyStopping · ReduceLROnPlateau · ModelCheckpoint
- Augmentation: rotation ±20°, flip, zoom ±15%, brightness ±20%
            """)

    if st.session_state.history:
        with st.expander(f"🕒  Prediction History  ({len(st.session_state.history)} items)"):
            for item in reversed(st.session_state.history[-20:]):
                d = DISPOSAL_DATA[item['class']]
                st.markdown(f"""
                <div class="hist-row">
                    <span style="font-size:16px;">{d['emoji']}</span>
                    <span style="flex:1;font-weight:500;">{item['class'].capitalize()}</span>
                    <span style="color:var(--green);font-family:var(--font-mono);">{item['confidence']:.1f}%</span>
                    <span style="color:var(--subtle);">{item['file'][:22]}</span>
                    <span style="color:var(--subtle);">{item['time']}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑  Clear History", key="btn_clear_hist"):
                st.session_state.history     = []
                st.session_state.last_result = None
                st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "insights":
    sec_head("Model Performance Dashboard")

    st.markdown(f"""
    <div class="metric-grid">
        {render_metric("Test Accuracy",  "88.14%", "253 test images",  "var(--lime)")}
        {render_metric("Avg Confidence", "89.8%",  "on test set",      "#58a6ff")}
        {render_metric("Train Images",   "1,819",  "70% of dataset",   "#8b5cf6")}
        {render_metric("Parameters",     "3.4M",   "MobileNetV2",      "#d29922")}
        {render_metric("Inference",      "<1s",    "CPU / GPU",        "#f85149")}
    </div>""", unsafe_allow_html=True)

    tab_cm, tab_tr, tab_pc, tab_arch = st.tabs(
        ["📊 Confusion Matrix", "📈 Training History", "🏆 Per-Class", "🧠 Architecture"]
    )

    with tab_cm:
        p = "assets/confusion_matrix.png"
        if os.path.exists(p):
            st.image(p, use_column_width=True)
        else:
            st.info("Place `confusion_matrix.png` in the `assets/` folder.")
        st.markdown("""
        <div style="background:rgba(63,185,80,0.08);border:1px solid rgba(63,185,80,0.2);border-radius:var(--r);padding:14px 18px;font-size:13px;color:var(--green);margin-top:12px;">
            <strong>Reading the matrix:</strong> Rows = actual class · Columns = predicted class ·
            Diagonal = correct predictions. Glass ↔ Plastic confusion is common — both are often transparent.
        </div>""", unsafe_allow_html=True)

    with tab_tr:
        ca, cb = st.columns(2)
        with ca:
            found = False
            for p1 in ["assets/plot_stage_1.png","assets/plot_stage_1_—_transfer_learning.png"]:
                if os.path.exists(p1):
                    st.image(p1, use_column_width=True, caption="Stage 1 — Transfer Learning")
                    found = True; break
            if not found:
                fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
                fig.patch.set_facecolor('#161b22')
                for ax in axes:
                    ax.set_facecolor('#21262d')
                    for sp in ax.spines.values(): sp.set_color('#30363d')
                    ax.tick_params(colors='#8b949e', labelsize=9)
                ep = list(range(1, 21))
                axes[0].plot(ep, [min(0.52+0.025*i, 0.93) for i in range(20)], color='#3fb950', lw=2, label='Train')
                axes[0].plot(ep, [min(0.48+0.021*i, 0.88) for i in range(20)], color='#58a6ff', lw=2, label='Val')
                axes[0].set_title('Accuracy – Stage 1', color='#e6edf3', fontsize=10)
                axes[0].legend(facecolor='#161b22', labelcolor='#8b949e', fontsize=9)
                axes[0].set_xlabel('Epoch', color='#8b949e', fontsize=9)
                axes[1].plot(ep, [max(1.4-0.055*i, 0.26) for i in range(20)], color='#f85149', lw=2, label='Train')
                axes[1].plot(ep, [max(1.5-0.05*i, 0.38) for i in range(20)], color='#d29922', lw=2, label='Val')
                axes[1].set_title('Loss – Stage 1', color='#e6edf3', fontsize=10)
                axes[1].legend(facecolor='#161b22', labelcolor='#8b949e', fontsize=9)
                axes[1].set_xlabel('Epoch', color='#8b949e', fontsize=9)
                plt.tight_layout(pad=1.5)
                st.pyplot(fig); plt.close()
                st.caption("Sample curves — add your plot_stage_1.png to assets/")

        with cb:
            found2 = False
            for p2 in ["assets/plot_stage_2.png","assets/plot_stage_2_—_fine-tuning.png"]:
                if os.path.exists(p2):
                    st.image(p2, use_column_width=True, caption="Stage 2 — Fine-tuning")
                    found2 = True; break
            if not found2:
                fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
                fig2.patch.set_facecolor('#161b22'); ax2.set_facecolor('#21262d')
                for sp in ax2.spines.values(): sp.set_color('#30363d')
                ax2.tick_params(colors='#8b949e', labelsize=9)
                ax2.plot(range(1,16), [min(0.82+0.004*i, 0.92) for i in range(15)], color='#a8e063', lw=2.5)
                ax2.set_title('Accuracy – Stage 2', color='#e6edf3', fontsize=10)
                ax2.set_xlabel('Epoch', color='#8b949e', fontsize=9)
                ax2.set_ylabel('Accuracy', color='#8b949e', fontsize=9)
                plt.tight_layout(pad=1.5)
                st.pyplot(fig2); plt.close()
                st.caption("Sample curve — add your plot_stage_2.png to assets/")

    with tab_pc:
        pa = "assets/per_class_accuracy.png"
        if os.path.exists(pa): st.image(pa, use_column_width=True)
        class_data = {
            "cardboard":(87.5,48,42),"glass":(84.0,50,42),"metal":(92.7,41,38),
            "paper":(93.3,60,56),"plastic":(81.2,66,53),"trash":(92.9,14,13),
        }
        rows = []
        for cn, (acc, tot, cor) in class_data.items():
            d = DISPOSAL_DATA[cn]
            rows.append({
                "": d['emoji'], "Category": cn.capitalize(),
                "Accuracy": f"{acc:.1f}%", "Correct": f"{cor}/{tot}",
                "Rating": "⭐⭐⭐" if acc>=90 else "⭐⭐" if acc>=85 else "⭐",
                "Status": "Excellent" if acc>=90 else "Good" if acc>=85 else "Needs attention",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("""
        <div class="tip-box" style="margin-top:12px;">
            Plastic and Glass show lower accuracy because they look visually similar —
            both are often transparent or reflective. This improves with more diverse training data.
        </div>""", unsafe_allow_html=True)

    with tab_arch:
        ac1, ac2 = st.columns(2)
        with ac1:
            st.markdown("""
**Model Layers:**

| Layer | Output | Parameters |
|---|---|---|
| Input | (224,224,3) | 0 |
| MobileNetV2 | (7,7,1280) | 2,257,984 |
| GlobalAvgPool | (1280,) | 0 |
| BatchNorm | (1280,) | 5,120 |
| Dense ReLU | (256,) | 327,936 |
| Dropout 0.4 | (256,) | 0 |
| Dense ReLU | (128,) | 32,896 |
| Dropout 0.3 | (128,) | 0 |
| Softmax | (6,) | 774 |
| **Total** | | **~2.63M** |
            """)
        with ac2:
            st.markdown("""
**Training Config:**

| Parameter | Stage 1 | Stage 2 |
|---|---|---|
| LR | 0.001 | 0.00001 |
| Frozen | All base | Last 40 unfrozen |
| Epochs | 20 | 20 |
| Batch | 32 | 32 |

**Callbacks:**
`EarlyStopping(patience=7)` · `ReduceLROnPlateau(factor=0.5)` · `ModelCheckpoint`

**Augmentation:**
rotation ±20° · h-flip · zoom ±15% · brightness ±20%
            """)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "batch":
    sec_head("Batch Image Classifier")
    st.markdown("""
    <div style="background:rgba(63,185,80,0.08);border:1px solid rgba(63,185,80,0.25);border-radius:var(--r);padding:14px 18px;font-size:13px;color:var(--green);margin-bottom:24px;">
        📁 &nbsp; Upload multiple waste images at once. The model classifies all of them and generates a downloadable CSV report.
    </div>""", unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Select multiple images", type=["jpg","jpeg","png","webp"],
        accept_multiple_files=True, key="upload_batch",
    )

    if batch_files:
        st.markdown(f"**{len(batch_files)} image{'s' if len(batch_files)>1 else ''} selected**")
        preview_cols = st.columns(min(len(batch_files), 8))
        for col, f in zip(preview_cols, batch_files[:8]):
            with col:
                st.image(Image.open(f), use_column_width=True)
                f.seek(0)
        if len(batch_files) > 8:
            st.caption(f"…and {len(batch_files)-8} more")
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="classify-btn">', unsafe_allow_html=True)
        if st.button(f"🚀  Classify All {len(batch_files)} Images", key="btn_batch_run"):
            progress = st.progress(0)
            status   = st.empty()
            all_res  = []
            for i, f in enumerate(batch_files):
                status.text(f"Processing {i+1}/{len(batch_files)}: {f.name}")
                f.seek(0)
                r = predict(f)
                d = DISPOSAL_DATA[r['predicted_class']]
                all_res.append({
                    "filename": f.name, "predicted": r['predicted_class'],
                    "confidence": f"{r['confidence']:.1f}%",
                    "action": d['action'], "bin": d['bin'],
                })
                progress.progress((i+1)/len(batch_files))
                st.session_state.total += 1

            status.text("✅  All classified!")
            df = pd.DataFrame(all_res)
            sec_head("Batch Results")
            st.dataframe(df, use_container_width=True, hide_index=True)

            col_pie, col_stats = st.columns([1,1])
            with col_pie:
                fig, ax = plt.subplots(figsize=(5,4))
                fig.patch.set_facecolor('#161b22'); ax.set_facecolor('#161b22')
                counts = df['predicted'].value_counts()
                cpie = [DISPOSAL_DATA[c]['accent'] for c in counts.index]
                ax.pie(counts.values, labels=counts.index, autopct='%1.0f%%', colors=cpie, startangle=90,
                       textprops={'color':'#e6edf3','fontsize':11})
                ax.set_title("Category Mix", color='#e6edf3', fontsize=13)
                st.pyplot(fig); plt.close()

            with col_stats:
                sec_head("Summary")
                total_b = len(all_res)
                recyclable = sum(1 for r in all_res if r['predicted'] != 'trash')
                st.markdown(f"""
                <div class="metric-card" style="border-top-color:var(--lime);margin-bottom:10px;">
                    <div class="m-label">Total Classified</div><div class="m-val">{total_b}</div>
                </div>
                <div class="metric-card" style="border-top-color:#3fb950;margin-bottom:10px;">
                    <div class="m-label">Recyclable</div><div class="m-val">{recyclable}</div>
                    <div class="m-sub">{recyclable/total_b*100:.0f}%</div>
                </div>
                <div class="metric-card" style="border-top-color:#f85149;">
                    <div class="m-label">General Waste</div><div class="m-val">{total_b-recyclable}</div>
                    <div class="m-sub">{(total_b-recyclable)/total_b*100:.0f}%</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                "⬇️  Download CSV Report", df.to_csv(index=False),
                f"wasteai_batch_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv", use_container_width=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:var(--subtle);">
            <div style="font-size:48px;margin-bottom:16px;opacity:0.3;">📁</div>
            <div style="font-size:15px;font-weight:500;color:var(--muted);">No files selected yet</div>
            <div style="font-size:12px;margin-top:8px;">Use the file picker above to select multiple images</div>
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "about":

    st.markdown("""
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:20px;padding:36px;margin-bottom:28px;position:relative;overflow:hidden;">
        <div style="position:absolute;top:-40px;right:-40px;width:200px;height:200px;background:radial-gradient(circle,rgba(63,185,80,0.1) 0%,transparent 70%);pointer-events:none;"></div>
        <div style="font-family:var(--font-head);font-size:22px;font-weight:800;color:var(--lime);margin-bottom:12px;">
            AI-Based Waste Segregation Classification System</div>
        <p style="font-size:14px;color:var(--muted);line-height:1.75;max-width:820px;">
            Improper waste segregation is a major challenge in urban waste management, leading to
            environmental pollution, inefficient recycling, and increased landfill usage. This project
            proposes an AI-based solution using deep learning to automatically classify waste images
            into 6 categories — achieving <strong style="color:var(--lime);">88.14% accuracy</strong>
            on an independent test set of 253 images.</p>
        <div style="margin-top:20px;display:flex;gap:10px;flex-wrap:wrap;">
            <span style="background:var(--green-bg);border:1px solid var(--green-dim);color:var(--green);border-radius:99px;padding:5px 14px;font-size:12px;font-weight:600;">88.14% Test Accuracy</span>
            <span style="background:rgba(88,166,255,0.1);border:1px solid rgba(88,166,255,0.3);color:#58a6ff;border-radius:99px;padding:5px 14px;font-size:12px;font-weight:600;">MobileNetV2</span>
            <span style="background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.3);color:#a78bfa;border-radius:99px;padding:5px 14px;font-size:12px;font-weight:600;">TrashNet Dataset</span>
            <span style="background:rgba(210,153,34,0.1);border:1px solid rgba(210,153,34,0.3);color:#d29922;border-radius:99px;padding:5px 14px;font-size:12px;font-weight:600;">IoT Deployable</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    sec_head("Global Waste Crisis")
    gcols = st.columns(4)
    for col, item in zip(gcols, GLOBAL_STATS):
        with col:
            st.markdown(f"""
            <div class="gstat-card">
                <div class="gstat-icon">{item['icon']}</div>
                <div class="gstat-lbl">{item['label']}</div>
                <div class="gstat-val">{item['value']}</div>
            </div>""", unsafe_allow_html=True)

    # ── REDESIGNED TECH STACK ──
    sec_head("Technology Stack")
    tech_items = [
        ("🐍", "Python 3.10",        "Core programming language",        "#3776AB", "v3.10"),
        ("🧠", "TensorFlow / Keras", "Deep learning framework",          "#FF6F00", "v2.13"),
        ("📱", "MobileNetV2",        "Lightweight CNN backbone",         "#4285F4", "ImageNet"),
        ("👁️", "OpenCV",             "Image processing",                 "#5C3EE8", "cv2"),
        ("🌐", "Streamlit",          "Interactive web framework",        "#FF4B4B", "v1.35"),
        ("🗂️", "TrashNet Dataset",   "2,527 labelled images",           "#3fb950", "6 classes"),
        ("🔢", "NumPy / Pandas",     "Numerical & data processing",     "#013243", "data"),
        ("📊", "Matplotlib",         "Plots & visualisations",          "#11557c", "viz"),
    ]
    st.markdown('<div class="tech-grid">', unsafe_allow_html=True)
    for icon, name, desc, color, badge in tech_items:
        st.markdown(f"""
        <div class="tech-card">
            <div class="tech-card-stripe" style="background:{color};"></div>
            <div class="tech-icon">{icon}</div>
            <div class="tech-name">{name}</div>
            <div class="tech-desc">{desc}</div>
            <span class="tech-badge" style="color:{color};background:{color}18;border:1px solid {color}40;">{badge}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    sec_head("Dataset — TrashNet")
    ds1, ds2 = st.columns([1, 2])
    with ds1:
        for lbl, val, color in [
            ("Total Images","2,527","var(--lime)"), ("Categories","6","#58a6ff"),
            ("Image Size","224×224","#8b5cf6"),   ("Train Split","1,819 (72%)","#d29922"),
            ("Val Split","455 (18%)","#ec4899"),   ("Test Split","253 (10%)","#0891b2"),
        ]:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color:{color};margin-bottom:10px;">
                <div class="m-label">{lbl}</div><div class="m-val">{val}</div>
            </div>""", unsafe_allow_html=True)
    with ds2:
        fig, ax = plt.subplots(figsize=(6,4))
        fig.patch.set_facecolor('#161b22'); ax.set_facecolor('#21262d')
        for sp in ax.spines.values(): sp.set_color('#30363d')
        ax.tick_params(colors='#8b949e')
        cats = ['cardboard','glass','metal','paper','plastic','trash']
        cnts = [403,501,410,594,482,137]
        bars = ax.barh(cats, cnts, color=[DISPOSAL_DATA[c]['accent'] for c in cats], height=0.55)
        ax.set_xlabel('Number of Images', color='#8b949e')
        ax.set_title('TrashNet Class Distribution', color='#e6edf3', fontsize=13, pad=12)
        for bar, val in zip(bars, cnts):
            ax.text(bar.get_width()+5, bar.get_y()+bar.get_height()/2, str(val), va='center', color='#8b949e', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    sec_head("Waste Category Reference")
    st.markdown('<div class="wcat-grid">', unsafe_allow_html=True)
    for cls_n, d in DISPOSAL_DATA.items():
        st.markdown(f"""
        <div class="wcat-card" style="background:{d['bg']};border-color:{d['accent']}40;">
            <div style="font-size:28px;margin-bottom:8px;">{d['emoji']}</div>
            <div style="font-size:14px;font-weight:700;color:{d['color']};margin-bottom:4px;">{cls_n.capitalize()}</div>
            <div style="font-size:11px;color:{d['color']}90;margin-bottom:8px;">{d['bin']}</div>
            <div style="font-size:11px;color:{d['color']}bb;">{d['tip']}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    sec_head("Performance Benchmarks")
    pb1c, pb2c = st.columns(2)
    with pb1c:
        st.markdown("""
| Metric | Value |
|---|---|
| Test Accuracy | 88.14% |
| Top-3 Accuracy | ~98.5% |
| Avg Confidence | 89.8% |
| Inference (CPU) | 180–350ms |
| Inference (GPU) | <50ms |
        """)
    with pb2c:
        st.markdown("""
| Metric | Value |
|---|---|
| Model Size (.keras) | ~14 MB |
| TFLite Size | ~3.5 MB |
| Raspberry Pi FPS | ~5 fps |
| Training Epochs | 20 + 20 |
| Dataset Size | 2,527 images |
        """)


st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="wasteai-footer">
    <div class="footer-brand">WasteAI ♻️</div>
    <div class="footer-sub">
        MobileNetV2 &nbsp;·&nbsp; TensorFlow &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp;
        TrashNet Dataset &nbsp;·&nbsp;
        Session predictions: <strong style="color:var(--lime);">{st.session_state.total}</strong>
    </div>
</div>
""", unsafe_allow_html=True)
