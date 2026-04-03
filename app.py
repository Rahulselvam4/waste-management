# ─────────────────────────────────────────────────────────────────────────────
# app.py  —  WasteAI  Smart Waste Classification System
# Run :  streamlit run app.py
# Lines: 1200+  |  Author: WasteAI Team  |  Model: MobileNetV2  |  Acc: 88.14%
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2, os, io, time, base64
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
# SESSION STATE  — initialise every key once
# ─────────────────────────────────────────────────────────────────────────────
_defaults = {
    "page":         "classify",
    "history":      [],
    "total":        0,
    "last_result":  None,
    "panel":        None,   # "guide" | "iot" | "dev" | None
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION HELPERS  — must be defined before any widget render
# ─────────────────────────────────────────────────────────────────────────────

def go(page: str):
    """Switch main page and clear any open panel."""
    st.session_state.page  = page
    st.session_state.panel = None

def toggle_panel(name: str):
    """Toggle an info panel open / closed."""
    st.session_state.panel = None if st.session_state.panel == name else name

# ─────────────────────────────────────────────────────────────────────────────
# MASTER CSS  — full design system
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ──────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* ── CSS Variables ───────────────────────────────────────────────────────── */
:root {
    --bg:          #0d1117;
    --surface:     #161b22;
    --surface2:    #21262d;
    --surface3:    #30363d;
    --border:      #30363d;
    --border2:     #484f58;
    --green:       #3fb950;
    --green-dim:   #238636;
    --green-glow:  rgba(63,185,80,0.15);
    --green-bg:    rgba(63,185,80,0.08);
    --lime:        #a8e063;
    --text:        #e6edf3;
    --text-muted:  #8b949e;
    --text-subtle: #484f58;
    --accent:      #58a6ff;
    --warning:     #d29922;
    --danger:      #f85149;
    --radius-sm:   8px;
    --radius:      12px;
    --radius-lg:   18px;
    --radius-xl:   24px;
    --shadow:      0 8px 32px rgba(0,0,0,0.4);
    --shadow-sm:   0 2px 8px rgba(0,0,0,0.3);
    --font-head:   'Outfit', sans-serif;
    --font-body:   'Space Grotesk', sans-serif;
    --font-mono:   'JetBrains Mono', monospace;
}

/* ── Reset ───────────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: var(--font-body) !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ───────────────────────────────────────────────── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebar"]            { display: none !important; }

.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── TOP NAV ─────────────────────────────────────────────────────────────── */
.wasteai-nav {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 0 28px;
    height: 58px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 1000;
}

.nav-brand {
    font-family: var(--font-head);
    font-size: 19px;
    font-weight: 800;
    color: var(--lime);
    letter-spacing: -0.3px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.nav-brand .dot { color: var(--text-muted); }

.nav-center {
    display: flex;
    gap: 2px;
    background: var(--bg);
    padding: 4px;
    border-radius: 10px;
    border: 1px solid var(--border);
}

.nav-tab {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-muted);
    padding: 6px 16px;
    border-radius: 7px;
    cursor: pointer;
    transition: all 0.15s ease;
    white-space: nowrap;
    border: none;
    background: transparent;
    text-decoration: none;
    display: inline-block;
}
.nav-tab:hover  { color: var(--text); background: var(--surface2); }
.nav-tab.active { color: var(--lime); background: var(--surface); box-shadow: var(--shadow-sm); }

.nav-right {
    display: flex;
    gap: 6px;
    align-items: center;
}

.pill-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-muted);
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 5px 12px;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
}
.pill-btn:hover        { color: var(--text); border-color: var(--border2); background: var(--surface2); }
.pill-btn.pill-active  { color: var(--lime); border-color: var(--green-dim); background: var(--green-bg); }

/* ── PAGE WRAP ───────────────────────────────────────────────────────────── */
.pw {
    max-width: 1280px;
    margin: 0 auto;
    padding: 32px 24px 80px;
}

/* ── INFO PANEL ──────────────────────────────────────────────────────────── */
.info-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 28px 32px;
    margin-bottom: 28px;
    position: relative;
}
.info-panel-title {
    font-family: var(--font-head);
    font-size: 18px;
    font-weight: 700;
    color: var(--lime);
    margin-bottom: 20px;
}
.info-panel p, .info-panel li {
    font-size: 13.5px;
    line-height: 1.75;
    color: var(--text-muted);
}
.info-panel strong { color: var(--text); }
.info-panel code {
    font-family: var(--font-mono);
    font-size: 12px;
    background: var(--surface2);
    color: var(--lime);
    padding: 2px 7px;
    border-radius: 5px;
    border: 1px solid var(--border);
}
.step-highlight {
    display: inline-block;
    background: var(--green-bg);
    border: 1px solid var(--green-dim);
    color: var(--green);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 6px;
}
.tip-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--lime);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 12px 16px;
    margin-top: 14px;
    font-size: 13px;
    color: var(--text-muted);
}

/* ── HERO ────────────────────────────────────────────────────────────────── */
.hero {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 44px 44px 0;
    margin-bottom: 28px;
    display: flex;
    align-items: flex-end;
    gap: 32px;
    overflow: hidden;
    position: relative;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(63,185,80,0.12) 0%, transparent 70%);
    pointer-events: none;
}

.hero-body { flex: 1; padding-bottom: 44px; position: relative; }

.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--green);
    background: var(--green-bg);
    border: 1px solid var(--green-dim);
    padding: 4px 12px;
    border-radius: 99px;
    margin-bottom: 16px;
}
.hero-tag::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--green);
    display: inline-block;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.8); }
}

.hero-title {
    font-family: var(--font-head);
    font-size: clamp(30px, 3.8vw, 52px);
    font-weight: 800;
    color: var(--text);
    line-height: 1.08;
    letter-spacing: -1.5px;
    margin-bottom: 18px;
}
.hero-title .accent { color: var(--lime); }

.hero-sub {
    font-size: 15px;
    color: var(--text-muted);
    line-height: 1.65;
    max-width: 480px;
    margin-bottom: 28px;
}

.hero-chips { display: flex; gap: 8px; flex-wrap: wrap; }
.hchip {
    font-size: 12px; font-weight: 500;
    padding: 5px 13px;
    border-radius: 99px;
    border: 1px solid var(--border);
    color: var(--text-muted);
    background: var(--bg);
}
.hchip.hi { border-color: var(--green-dim); color: var(--green); background: var(--green-bg); }

.hero-orb-wrap {
    width: 200px;
    flex-shrink: 0;
    display: flex;
    justify-content: center;
    padding-bottom: 0;
}
.hero-orb {
    width: 148px; height: 148px;
    border-radius: 50%;
    background: conic-gradient(from 0deg, #3fb950, #a8e063, #58a6ff, #3fb950);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 60px;
    box-shadow: 0 0 80px rgba(63,185,80,0.25), 0 0 30px rgba(63,185,80,0.15);
    animation: spin-slow 12s linear infinite;
    position: relative;
}
.hero-orb::after {
    content: 'RECYCLING AI';
    font-family: var(--font-head);
    font-size: 7px;
    font-weight: 700;
    letter-spacing: 2px;
    color: rgba(255,255,255,0.6);
    position: absolute;
    bottom: 22px;
    text-align: center;
    animation: spin-slow 12s linear infinite reverse;
    white-space: nowrap;
}
@keyframes spin-slow { to { transform: rotate(360deg); } }

/* ── CARD SYSTEM ─────────────────────────────────────────────────────────── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px;
}
.card-title {
    font-family: var(--font-head);
    font-size: 12px;
    font-weight: 700;
    color: var(--text-muted);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 16px;
}

/* ── UPLOAD ZONE ─────────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--bg) !important;
    transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--green-dim) !important;
    background: var(--green-bg) !important;
}
[data-testid="stFileUploader"] label { color: var(--text-muted) !important; }

/* ── RESULT STATES ───────────────────────────────────────────────────────── */
.result-empty {
    background: var(--bg);
    border: 2px dashed var(--border);
    border-radius: var(--radius-lg);
    padding: 64px 28px;
    text-align: center;
}
.result-empty-icon  { font-size: 48px; margin-bottom: 14px; opacity: 0.4; }
.result-empty-title { font-size: 15px; font-weight: 600; color: var(--text-muted); margin-bottom: 6px; }
.result-empty-sub   { font-size: 13px; color: var(--text-subtle); }

/* ── PREDICTION CARD ─────────────────────────────────────────────────────── */
.pred-card {
    border-radius: var(--radius-lg);
    padding: 26px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}
.pred-emoji-wrap {
    width: 68px; height: 68px;
    border-radius: 50%;
    background: rgba(255,255,255,0.1);
    display: flex; align-items: center; justify-content: center;
    font-size: 34px;
    margin: 0 auto 14px;
}
.pred-tag    { font-size: 11px; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; opacity: 0.7; text-align: center; margin-bottom: 6px; }
.pred-class  { font-family: var(--font-head); font-size: 36px; font-weight: 800; text-align: center; margin-bottom: 8px; }
.pred-bin    { font-size: 13px; opacity: 0.75; text-align: center; font-weight: 500; }
.pred-badge  {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.12);
    border-radius: 99px;
    padding: 4px 14px;
    font-size: 12px; font-weight: 600;
    margin: 10px auto 0;
}

/* ── CONFIDENCE BAR ──────────────────────────────────────────────────────── */
.conf-row {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 12px; font-weight: 500;
    color: var(--text-muted);
    margin-bottom: 7px;
}
.conf-val { font-family: var(--font-mono); font-weight: 600; }
.conf-track {
    background: var(--surface3);
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
    margin-bottom: 18px;
}
.conf-fill {
    height: 8px;
    border-radius: 99px;
    transition: width 0.7s ease;
}

/* ── PROB BARS ───────────────────────────────────────────────────────────── */
.prob-section-title {
    font-size: 11px; font-weight: 600; letter-spacing: 0.8px;
    text-transform: uppercase; color: var(--text-subtle);
    margin-bottom: 10px;
}
.prob-row {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 8px;
}
.prob-em    { font-size: 15px; width: 22px; text-align: center; }
.prob-name  { font-size: 12px; color: var(--text-muted); width: 76px; }
.prob-track { flex: 1; background: var(--surface3); border-radius: 99px; height: 6px; overflow: hidden; }
.prob-fill  { height: 6px; border-radius: 99px; transition: width 0.5s ease; }
.prob-pct   {
    font-size: 11px; color: var(--text-subtle);
    width: 40px; text-align: right;
    font-family: var(--font-mono);
}

/* ── DISPOSAL STEPS ──────────────────────────────────────────────────────── */
.d-step {
    display: flex; gap: 12px; align-items: flex-start;
    padding: 10px 0; border-bottom: 1px solid var(--border);
    font-size: 13px; color: var(--text-muted); line-height: 1.5;
}
.d-step:last-child { border-bottom: none; }
.d-num {
    width: 24px; height: 24px; border-radius: 50%;
    background: var(--green-bg); border: 1px solid var(--green-dim);
    color: var(--green);
    font-size: 11px; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.fact-strip {
    border-radius: var(--radius-sm);
    padding: 13px 16px;
    margin-top: 14px;
    font-size: 13px; line-height: 1.6;
    border-left: 3px solid;
}

/* ── ENV CARDS ───────────────────────────────────────────────────────────── */
.env-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin-top: 12px; }
.env-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 12px;
    text-align: center;
}
.env-icon  { font-size: 18px; margin-bottom: 4px; }
.env-lbl   { font-size: 10px; color: var(--text-subtle); text-transform: uppercase; letter-spacing: 0.5px; }
.env-val   { font-size: 12px; font-weight: 600; color: var(--text); margin-top: 3px; }

/* ── WARN BOX ────────────────────────────────────────────────────────────── */
.warn-box {
    background: rgba(210,153,34,0.1);
    border: 1px solid rgba(210,153,34,0.3);
    border-radius: var(--radius-sm);
    padding: 12px 16px;
    font-size: 13px;
    color: #d29922;
    margin-top: 12px;
}

/* ── METRICS GRID ────────────────────────────────────────────────────────── */
.metric-grid { display: grid; grid-template-columns: repeat(5,1fr); gap: 12px; margin-bottom: 28px; }
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 14px;
    border-top: 3px solid var(--lime);
    text-align: center;
}
.m-label { font-size: 10px; color: var(--text-subtle); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 7px; }
.m-val   { font-family: var(--font-head); font-size: 24px; font-weight: 800; color: var(--text); }
.m-sub   { font-size: 11px; color: var(--text-subtle); margin-top: 4px; }

/* ── SECTION HEADING ─────────────────────────────────────────────────────── */
.sec-head {
    display: flex; align-items: center; gap: 12px;
    margin: 32px 0 20px;
}
.sec-head-line { flex: 1; height: 1px; background: var(--border); }
.sec-head-title {
    font-family: var(--font-head);
    font-size: 13px; font-weight: 700;
    color: var(--text-muted);
    white-space: nowrap;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── CATEGORY PILLS ──────────────────────────────────────────────────────── */
.cat-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin-top: 12px; }
.cat-pill {
    border-radius: var(--radius-sm);
    padding: 11px 14px;
    font-size: 12.5px; font-weight: 500;
    display: flex; align-items: center; gap: 8px;
    border: 1px solid transparent;
}

/* ── IMAGE META ──────────────────────────────────────────────────────────── */
.img-meta {
    background: var(--surface2);
    border-radius: 7px;
    padding: 8px 12px;
    font-size: 11px;
    color: var(--text-subtle);
    margin-top: 8px;
    font-family: var(--font-mono);
}

/* ── IOT HARDWARE CARD ───────────────────────────────────────────────────── */
.hw-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    margin-bottom: 10px;
    transition: border-color 0.15s;
}
.hw-card:hover { border-color: var(--border2); }
.hw-name  { font-size: 14px; font-weight: 600; color: var(--text); margin-bottom: 4px; }
.hw-role  { font-size: 12px; color: var(--text-muted); margin-bottom: 8px; }
.hw-meta  { display: flex; gap: 16px; font-size: 11px; color: var(--text-subtle); }
.hw-cost  { color: var(--green); font-weight: 600; }

/* ── WORKFLOW STEP ───────────────────────────────────────────────────────── */
.wf-step {
    display: flex; gap: 14px; align-items: flex-start;
    padding: 14px 0; border-bottom: 1px solid var(--border);
}
.wf-step:last-child { border-bottom: none; }
.wf-num {
    width: 32px; height: 32px; border-radius: 50%;
    background: var(--green-bg); border: 1.5px solid var(--green-dim);
    color: var(--green);
    font-size: 12px; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.wf-title { font-size: 14px; font-weight: 600; color: var(--text); }
.wf-desc  { font-size: 12.5px; color: var(--text-muted); margin-top: 3px; line-height: 1.5; }

/* ── TECH STACK ──────────────────────────────────────────────────────────── */
.tech-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; }
.tech-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 16px;
    text-align: center;
    transition: transform 0.15s, border-color 0.15s, box-shadow 0.15s;
    border-top: 3px solid var(--border);
}
.tech-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow);
    border-color: var(--border2);
}
.tech-name { font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 4px; }
.tech-desc { font-size: 11px; color: var(--text-muted); }

/* ── GLOBAL STAT CARDS ───────────────────────────────────────────────────── */
.gstat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 16px;
    text-align: center;
    border-top: 3px solid var(--warning);
}
.gstat-icon  { font-size: 26px; margin-bottom: 8px; }
.gstat-lbl   { font-size: 11px; color: var(--text-subtle); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.gstat-val   { font-family: var(--font-head); font-size: 18px; font-weight: 700; color: var(--text); }

/* ── HISTORY ROW ─────────────────────────────────────────────────────────── */
.hist-row {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 12px;
    border-radius: var(--radius-sm);
    background: var(--surface2);
    margin-bottom: 5px;
    font-size: 12.5px;
    border: 1px solid var(--border);
    transition: border-color 0.15s;
}
.hist-row:hover { border-color: var(--border2); }

/* ── FOOTER ──────────────────────────────────────────────────────────────── */
.wasteai-footer {
    background: var(--surface);
    border-top: 1px solid var(--border);
    padding: 20px 32px;
    text-align: center;
    margin-top: 60px;
}
.footer-brand {
    font-family: var(--font-head);
    font-size: 15px; font-weight: 700;
    color: var(--lime);
    margin-bottom: 5px;
}
.footer-sub { font-size: 11px; color: var(--text-subtle); }

/* ── STREAMLIT OVERRIDES ─────────────────────────────────────────────────── */
.stButton > button {
    background: var(--green-dim) !important;
    color: #ffffff !important;
    border: 1.5px solid var(--green-dim) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 22px !important;
    transition: all 0.15s !important;
    width: 100% !important;
    letter-spacing: 0.2px !important;
}
.stButton > button:hover {
    background: #2ea043 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(63,185,80,0.3) !important;
}
.stButton > button:active { transform: scale(0.98) !important; }

[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface) !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 3px;
    background: var(--bg);
    border-radius: var(--radius-sm);
    padding: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    color: var(--text-muted) !important;
    padding: 6px 14px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--surface) !important;
    color: var(--lime) !important;
}

.stProgress > div > div > div > div {
    background: var(--green) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
.stDataFrame tr { background: var(--surface) !important; }
.stDataFrame td, .stDataFrame th { color: var(--text) !important; border-color: var(--border) !important; }

.stCodeBlock { border-radius: var(--radius-sm) !important; }
.stSpinner > div { border-top-color: var(--lime) !important; }

/* ── RESPONSIVE ──────────────────────────────────────────────────────────── */
@media (max-width: 900px) {
    .wasteai-nav   { padding: 0 16px; }
    .nav-center    { display: none; }
    .pw            { padding: 16px 14px 80px; }
    .hero          { padding: 28px 24px 0; flex-direction: column; align-items: flex-start; }
    .hero-orb-wrap { width: 100%; justify-content: flex-start; }
    .hero-orb      { width: 100px; height: 100px; font-size: 36px; }
    .metric-grid   { grid-template-columns: repeat(2,1fr); }
    .tech-grid     { grid-template-columns: repeat(2,1fr); }
    .env-grid      { grid-template-columns: repeat(3,1fr); }
    .cat-grid      { grid-template-columns: repeat(2,1fr); }
    .hero-title    { font-size: 30px; }
    .info-panel > div[style*="grid"] { display: block !important; }
}

@media (max-width: 600px) {
    .hero-title       { font-size: 24px; }
    .metric-grid      { grid-template-columns: 1fr 1fr; }
    .tech-grid        { grid-template-columns: 1fr 1fr; }
    .hero-chips       { flex-wrap: wrap; }
    .pred-class       { font-size: 28px; }
    .pill-btn span    { display: none; }
}

@media (max-width: 400px) {
    .cat-grid         { grid-template-columns: 1fr; }
}

/* ── MOBILE BOTTOM NAV ───────────────────────────────────────────────────── */
.mobile-nav {
    display: none;
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: var(--surface);
    border-top: 1px solid var(--border);
    z-index: 9999;
    padding: 6px 0 10px;
    gap: 0;
}
@media (max-width: 900px) { .mobile-nav { display: flex; justify-content: space-around; } }
.mobile-tab {
    display: flex; flex-direction: column; align-items: center; gap: 2px;
    font-size: 10px; font-weight: 500; color: var(--text-muted);
    flex: 1; padding: 4px 0; cursor: pointer;
}
.mobile-tab.active { color: var(--lime); }
.mobile-tab .mt-icon { font-size: 20px; }

/* Hide the functional nav button row visually but keep it interactive */
div[data-testid="stHorizontalBlock"]:first-of-type {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    height: 58px !important;
    z-index: 1001 !important;
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 0 !important;
    background: transparent !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0 !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type > div {
    flex: 1 !important;
    height: 58px !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type button {
    opacity: 0 !important;
    height: 58px !important;
    border-radius: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    transform: none !important;
    pointer-events: auto !important;
    cursor: pointer !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def sec_head(title: str):
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
    """Full prediction result card with confidence and probability bars."""
    d    = DISPOSAL_DATA[result['predicted_class']]
    cls  = result['predicted_class']
    conf = result['confidence']

    # Confidence colour
    if conf >= 80:   cconf = "#3fb950"
    elif conf >= 60: cconf = "#d29922"
    else:            cconf = "#f85149"

    # Main prediction card
    st.markdown(f"""
    <div class="pred-card" style="background:{d['bg']};border:1.5px solid {d['accent']}50;">
        <div class="pred-emoji-wrap">{d['emoji']}</div>
        <div class="pred-tag" style="color:{d['color']};">Detected Waste Type</div>
        <div class="pred-class" style="color:{d['color']};">{cls.capitalize()}</div>
        <div class="pred-bin" style="color:{d['color']};">🗑 {d['bin']}</div>
        <div style="text-align:center;">
            <span class="pred-badge" style="color:{d['color']};">
                {d['emoji']} {d['action']}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence bar
    st.markdown(f"""
    <div class="conf-row">
        <span>Model Confidence</span>
        <span class="conf-val" style="color:{cconf};">{conf:.1f}%</span>
    </div>
    <div class="conf-track">
        <div class="conf-fill" style="width:{conf:.1f}%;background:{cconf};"></div>
    </div>
    """, unsafe_allow_html=True)

    # Probability bars
    st.markdown('<div class="prob-section-title">All Class Probabilities</div>', unsafe_allow_html=True)
    sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
    top_prob = sorted_probs[0][1]
    for c, p in sorted_probs:
        bar_col = DISPOSAL_DATA[c]['accent'] if p == top_prob else "var(--surface3)"
        st.markdown(f"""
        <div class="prob-row">
            <span class="prob-em">{DISPOSAL_DATA[c]['emoji']}</span>
            <span class="prob-name">{c.capitalize()}</span>
            <div class="prob-track">
                <div class="prob-fill" style="width:{p*100:.1f}%;background:{bar_col};"></div>
            </div>
            <span class="prob-pct">{p*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)


def render_disposal(cls: str):
    d = DISPOSAL_DATA[cls]
    with st.expander(f"♻️  Disposal Guide — {cls.capitalize()}", expanded=False):
        for i, step in enumerate(d['steps'], 1):
            st.markdown(f"""
            <div class="d-step">
                <div class="d-num">{i}</div>
                <div>{step}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="fact-strip"
             style="background:{d['bg']};border-color:{d['accent']};color:{d['color']};">
            💡 <strong>Did you know?</strong>&nbsp; {d['fact']}
        </div>
        """, unsafe_allow_html=True)


def render_env(cls: str):
    d = DISPOSAL_DATA[cls]
    st.markdown(f"""
    <div class="env-grid">
        <div class="env-card">
            <div class="env-icon">🌿</div>
            <div class="env-lbl">CO₂ Impact</div>
            <div class="env-val">{d['co2']}</div>
        </div>
        <div class="env-card">
            <div class="env-icon">⏳</div>
            <div class="env-lbl">Decompose Time</div>
            <div class="env-val">{d['decompose']}</div>
        </div>
        <div class="env-card">
            <div class="env-icon">♻️</div>
            <div class="env-lbl">Recycling Rate</div>
            <div class="env-val">{d['rate']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TOP NAVIGATION  — HTML visual layer
# ─────────────────────────────────────────────────────────────────────────────
PAGE_LIST = [
    ("🔍 Classify",  "classify"),
    ("📊 Insights",  "insights"),
    ("📁 Batch",     "batch"),
    ("ℹ️  About",    "about"),
]

_cur = st.session_state.page
_pan = st.session_state.panel

nav_tabs = "".join(
    f'<span class="nav-tab {"active" if k == _cur else ""}">{lbl}</span>'
    for lbl, k in PAGE_LIST
)
guide_cls = "pill-btn pill-active" if _pan == "guide" else "pill-btn"
iot_cls   = "pill-btn pill-active" if _pan == "iot"   else "pill-btn"
dev_cls   = "pill-btn pill-active" if _pan == "dev"   else "pill-btn"

st.markdown(f"""
<div class="wasteai-nav">
    <div class="nav-brand">♻️ WasteAI<span class="dot"> /</span></div>
    <div class="nav-center">{nav_tabs}</div>
    <div class="nav-right">
        <span class="{guide_cls}">❓ <span>Guide</span></span>
        <span class="{iot_cls}">🌐 <span>IoT</span></span>
        <span class="{dev_cls}">⚙️ <span>Docs</span></span>
    </div>
</div>
""", unsafe_allow_html=True)

# Mobile bottom nav bar (purely decorative — functional buttons below)
mob_tabs_html = "".join(
    f'<div class="mobile-tab {"active" if k == _cur else ""}"><span class="mt-icon">{lbl.split()[0]}</span><span>{lbl.split(None,1)[1] if len(lbl.split(None,1))>1 else ""}</span></div>'
    for lbl, k in PAGE_LIST
)
st.markdown(f'<div class="mobile-nav">{mob_tabs_html}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FUNCTIONAL NAVIGATION BUTTONS
# These are rendered as INVISIBLE Streamlit buttons overlapping the nav bar.
# The CSS above positions the first stHorizontalBlock fixed at the top.
# ─────────────────────────────────────────────────────────────────────────────
_n1, _n2, _n3, _n4, _sp, _p1, _p2, _p3 = st.columns([1.1, 1.1, 0.9, 0.9, 2.8, 0.8, 0.8, 0.8])
with _n1:
    if st.button("🔍 Classify",  key="_nav_classify"):  go("classify"); st.rerun()
with _n2:
    if st.button("📊 Insights",  key="_nav_insights"):  go("insights"); st.rerun()
with _n3:
    if st.button("📁 Batch",     key="_nav_batch"):     go("batch");    st.rerun()
with _n4:
    if st.button("ℹ️ About",     key="_nav_about"):     go("about");    st.rerun()
with _p1:
    if st.button("❓ Guide",     key="_pan_guide"):     toggle_panel("guide"); st.rerun()
with _p2:
    if st.button("🌐 IoT",       key="_pan_iot"):       toggle_panel("iot");   st.rerun()
with _p3:
    if st.button("⚙️ Docs",      key="_pan_dev"):       toggle_panel("dev");   st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE WRAPPER START
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="pw">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# GUIDE PANEL
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.panel == "guide":
    st.markdown("""
    <div class="info-panel">
        <div class="info-panel-title">📖  How to Use WasteAI</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
            <div>
                <span class="step-highlight">Step 1 — Prepare Your Image</span>
                <p style="margin-top:8px;">Take a clear photo of a <strong>single</strong> waste item against a plain or neutral
                background. Good lighting is important. Supported formats: JPG, JPEG, PNG, WEBP
                (max 10 MB per file).</p>
                <br>
                <span class="step-highlight">Step 2 — Upload &amp; Classify</span>
                <p style="margin-top:8px;">On the <strong>Classify</strong> page, drag your image into the upload zone or click
                Browse Files. Hit <code>Classify This Image</code> — results appear in under
                one second.</p>
                <br>
                <span class="step-highlight">Step 3 — Act on the Result</span>
                <p style="margin-top:8px;">Read the disposal guide, check the bin colour and follow the numbered steps. Each
                result also shows environmental impact data so you understand why proper sorting
                matters.</p>
            </div>
            <div>
                <span class="step-highlight">Tips for Best Accuracy</span>
                <ul style="padding-left:18px;margin-top:8px;margin-bottom:16px;">
                    <li>One item per image — not mixed waste</li>
                    <li>Good, even lighting — avoid harsh shadows</li>
                    <li>Item should fill at least 50% of the frame</li>
                    <li>Avoid blurry or very dark photos</li>
                    <li>Remove any background clutter</li>
                    <li>Close-up shots work better than wide-angle</li>
                </ul>
                <span class="step-highlight">Batch Classification</span>
                <p style="margin-top:8px;">Use the <strong>Batch</strong> tab to classify multiple images at once and
                download a CSV report. Ideal for processing collections of waste images or
                testing the model on a dataset.</p>
                <div class="tip-box">
                    ℹ️ Confidence below 60% usually means the image quality can be improved.
                    Try re-taking the photo with better lighting or a cleaner background.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# IOT PANEL
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.panel == "iot":
    st.markdown("""
    <div class="info-panel">
        <div class="info-panel-title">🌐  Real-World IoT Deployment Guide</div>
        <p>This model can be embedded in physical smart bins using a Raspberry Pi.
        The bin captures waste images via camera, classifies them in ~200ms, then opens the correct
        compartment via servo motor — fully automated sorting at the edge with no cloud required.</p>
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
                <div class="hw-meta">
                    <span class="hw-cost">💰 {item['cost']}</span>
                    <span>💡 {item['why']}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    with ti2:
        st.markdown("#### Software & Firmware Stack")
        for item in IOT_SOFTWARE:
            layer_color = {"Edge AI": "#3fb950", "Vision": "#58a6ff", "Control": "#d29922",
                          "Backend": "#8b5cf6", "Frontend": "#ec4899"}.get(item['layer'], "#6b7280")
            st.markdown(f"""
            <div style="display:flex;gap:10px;padding:12px;border:1px solid var(--border);
                        border-radius:var(--radius);margin-bottom:8px;background:var(--surface2);">
                <span style="background:{layer_color}22;color:{layer_color};padding:3px 10px;
                             border-radius:6px;font-size:11px;font-weight:700;
                             white-space:nowrap;align-self:flex-start;margin-top:2px;
                             border:1px solid {layer_color}44;">
                    {item['layer']}</span>
                <div>
                    <div style="font-size:13px;font-weight:600;color:var(--text);">{item['tech']}</div>
                    <div style="font-size:12px;color:var(--text-muted);margin-top:2px;">{item['detail']}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with ti3:
        st.markdown("#### Automated Sorting Workflow")
        for i, (title, desc) in enumerate(IOT_WORKFLOW, 1):
            st.markdown(f"""
            <div class="wf-step">
                <div class="wf-num">{i}</div>
                <div>
                    <div class="wf-title">{title}</div>
                    <div class="wf-desc">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with ti4:
        st.markdown("**Step 1 — Convert model to TFLite for edge deployment:**")
        st.code(TFLITE_CODE, language="python")
        st.markdown("**Step 2 — Inference + servo control on Raspberry Pi:**")
        st.code(PI_CODE, language="python")

# ─────────────────────────────────────────────────────────────────────────────
# DEV DOCS PANEL
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.panel == "dev":
    st.markdown("""
    <div class="info-panel">
        <div class="info-panel-title">⚙️  Developer Documentation</div>
        <p>The <code>predict.py</code> module exposes a clean Python API for single-image
        and batch inference. You can also wrap it in a Flask or FastAPI REST service for
        integration with other systems.</p>
    </div>
    """, unsafe_allow_html=True)

    dd1, dd2, dd3, dd4 = st.tabs(["📦 predict.py API", "🔁 Batch API", "🌐 Flask REST", "🚀 FastAPI"])

    with dd1:
        st.markdown("""
**Single image prediction — accepts file path, PIL Image, numpy array, or Streamlit UploadedFile:**
```python
from predict import predict

result = predict("path/to/image.jpg")   # or PIL Image / numpy / uploaded file

print(result['predicted_class'])     # 'plastic'
print(result['confidence'])          # 94.32  (0–100 float)
print(result['all_probabilities'])   # {'cardboard': 0.01, 'glass': 0.03, ...}
print(result['top3'])                # [('plastic', 0.94), ('glass', 0.03), ...]
```
**Command-line usage:**
```bash
python predict.py --image path/to/waste.jpg
```
        """)

    with dd2:
        st.markdown("""
**Batch classification — efficient single forward pass for multiple images:**
```python
from predict import predict_batch

files   = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = predict_batch(files)

for r in results:
    print(r['predicted_class'], r['confidence'])
```
**Performance:** ~50–80ms per image when batched on GPU; ~150–300ms on CPU.
Invalid images are skipped and flagged with `{"error": "Could not process image"}`.
        """)

    with dd3:
        st.markdown("""
```python
from flask import Flask, request, jsonify
from predict import predict
from PIL import Image
import io

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify():
    img_bytes = request.files['image'].read()
    img       = Image.open(io.BytesIO(img_bytes))
    result    = predict(img)
    return jsonify(result)

@app.route('/batch', methods=['POST'])
def batch():
    files   = request.files.getlist('images')
    results = []
    for f in files:
        img    = Image.open(io.BytesIO(f.read()))
        results.append(predict(img))
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```
**Test with curl:**
```bash
curl -X POST http://localhost:5000/classify -F "image=@waste.jpg"
```
        """)

    with dd4:
        st.markdown("""
```python
from fastapi import FastAPI, UploadFile, File
from predict import predict
from PIL import Image
import io, uvicorn

app = FastAPI(title="WasteAI API", version="1.0")

@app.post("/classify")
async def classify(image: UploadFile = File(...)):
    contents = await image.read()
    img      = Image.open(io.BytesIO(contents))
    result   = predict(img)
    return result

@app.get("/health")
def health():
    return {"status": "ok", "model": "MobileNetV2", "classes": 6}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
**Run:** `uvicorn main:app --reload`
**Swagger UI:** http://localhost:8000/docs
        """)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: CLASSIFY
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "classify":

    # ── Hero section ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-body">
            <div class="hero-tag">AI-Powered · Real-Time · 6 Categories</div>
            <h1 class="hero-title">
                Identify waste.<br>
                <span class="accent">Sort smarter.</span>
            </h1>
            <p class="hero-sub">
                Upload a photo of any waste item. Our fine-tuned MobileNetV2 deep
                learning model classifies it into one of 6 categories in under a second
                — with disposal guidance and environmental impact data.
            </p>
            <div class="hero-chips">
                <span class="hchip hi">88.14% accuracy</span>
                <span class="hchip">6 waste categories</span>
                <span class="hchip">&lt;1s inference</span>
                <span class="hchip">MobileNetV2</span>
                <span class="hchip">TrashNet dataset</span>
            </div>
        </div>
        <div class="hero-orb-wrap">
            <div class="hero-orb">♻️</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    # ── Left column: upload + categories ──────────────────────────────────────
    with col_l:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Upload Waste Image</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drag & drop or browse",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="upload_main",
        )

        if uploaded:
            img_pil = Image.open(uploaded)
            st.image(img_pil, use_column_width=True, caption=None)
            uploaded.seek(0)
            st.markdown(f"""
            <div class="img-meta">
                {img_pil.width} × {img_pil.height} px &nbsp;·&nbsp;
                {uploaded.name} &nbsp;·&nbsp;
                {uploaded.size / 1024:.1f} KB &nbsp;·&nbsp;
                {img_pil.mode}
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            classify_btn = st.button("🔍  Classify This Image", key="btn_classify_main")
        else:
            st.markdown("""
            <div style="text-align:center;padding:52px 20px;color:var(--text-subtle);">
                <div style="font-size:44px;margin-bottom:14px;opacity:0.35;">🖼️</div>
                <div style="font-size:14px;font-weight:500;color:var(--text-muted);">
                    No image uploaded yet
                </div>
                <div style="font-size:12px;margin-top:7px;">
                    Supports JPG · PNG · WEBP · max 10 MB
                </div>
            </div>""", unsafe_allow_html=True)
            classify_btn = False

        # Category reference grid
        st.markdown('<div class="card-title" style="margin-top:22px;">Detectable Categories</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="cat-grid">', unsafe_allow_html=True)
        for cls_name, d in DISPOSAL_DATA.items():
            st.markdown(f"""
            <div class="cat-pill"
                 style="background:{d['bg']};color:{d['color']};border-color:{d['accent']}30;">
                <span style="font-size:18px;">{d['emoji']}</span>
                <span style="font-weight:500;">{cls_name.capitalize()}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)  # close .card

    # ── Right column: results ─────────────────────────────────────────────────
    with col_r:
        if uploaded and classify_btn:
            with st.spinner("Analysing image…"):
                t0 = time.time()
                uploaded.seek(0)
                result  = predict(uploaded)
                elapsed = time.time() - t0

            # Save to session
            st.session_state.history.append({
                "class":      result['predicted_class'],
                "confidence": result['confidence'],
                "file":       uploaded.name,
                "time":       datetime.now().strftime("%H:%M:%S"),
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
                st.markdown("""
                <div class="warn-box">
                    ⚠️ <strong>Low confidence.</strong>
                    Try a clearer image with better lighting and a single item filling the frame.
                </div>""", unsafe_allow_html=True)

        elif st.session_state.last_result and not uploaded:
            render_result(st.session_state.last_result)
        else:
            st.markdown("""
            <div class="result-empty">
                <div class="result-empty-icon">♻️</div>
                <div class="result-empty-title">Results appear here</div>
                <div class="result-empty-sub">Upload an image and click Classify</div>
            </div>""", unsafe_allow_html=True)

    # ── Expanders: AI info + history ──────────────────────────────────────────
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    with st.expander("🔬  How Does the AI Model Work?"):
        c1e, c2e = st.columns([1, 1])
        with c1e:
            st.markdown("""
**Architecture: MobileNetV2 + Custom Classification Head**

| Layer | Output Shape | Parameters |
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
        with c2e:
            st.markdown("""
**Why MobileNetV2?**
- Pre-trained on 1.4M ImageNet images — already understands edges, textures, shapes
- Only 3.4M parameters — fast on CPU and deployable to IoT devices
- Inverted residuals + linear bottlenecks → efficient memory usage
- Achieves **88.14%** accuracy on 253 held-out test images

**Two-Stage Training Pipeline:**
- **Stage 1:** Frozen MobileNetV2 base → train only the custom head
  - LR: `0.001` | Epochs: up to 20 | EarlyStopping(patience=7)
- **Stage 2:** Fine-tune last 40 layers of base with lower LR
  - LR: `0.00001` | ReduceLROnPlateau callback
- **Augmentation:** rotation ±20°, horizontal flip, zoom 15%, brightness ±20%
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
                    <span style="color:var(--text-subtle);">{item['file'][:22]}</span>
                    <span style="color:var(--text-subtle);">{item['time']}</span>
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

    # Metrics row
    st.markdown(f"""
    <div class="metric-grid">
        {render_metric("Test Accuracy",  "88.14%",  "253 test images",  "var(--lime)")}
        {render_metric("Avg Confidence", "89.8%",   "on test set",      "#58a6ff")}
        {render_metric("Train Images",   "1,819",   "70% of dataset",   "#8b5cf6")}
        {render_metric("Parameters",     "3.4M",    "MobileNetV2",      "#d29922")}
        {render_metric("Inference",      "<1s",     "CPU / GPU",        "#f85149")}
    </div>""", unsafe_allow_html=True)

    tab_cm, tab_tr, tab_pc, tab_arch = st.tabs(
        ["📊 Confusion Matrix", "📈 Training History", "🏆 Per-Class", "🧠 Architecture"]
    )

    with tab_cm:
        p = "assets/confusion_matrix.png"
        if os.path.exists(p):
            st.image(p, use_column_width=True)
        else:
            st.info("Place `confusion_matrix.png` in the `assets/` folder to view it here.")
        st.markdown("""
        <div style="background:rgba(63,185,80,0.08);border:1px solid rgba(63,185,80,0.2);
                    border-radius:var(--radius);padding:14px 18px;font-size:13px;color:var(--green);">
            <strong>Reading the confusion matrix:</strong> Rows = actual class &nbsp;·&nbsp;
            Columns = predicted class &nbsp;·&nbsp; Diagonal (bright values) = correct predictions.
            Glass ↔ Plastic confusion is common — both are often transparent/reflective.
        </div>""", unsafe_allow_html=True)

    with tab_tr:
        ca, cb = st.columns(2)
        with ca:
            p1 = "assets/plot_stage_1.png"
            if os.path.exists(p1):
                st.image(p1, use_column_width=True, caption="Stage 1 — Transfer Learning")
            else:
                # Generated training curve placeholder
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                fig.patch.set_facecolor('#161b22')
                for ax in axes:
                    ax.set_facecolor('#21262d')
                    for sp in ax.spines.values(): sp.set_color('#30363d')
                    ax.tick_params(colors='#8b949e')
                epochs    = list(range(1, 21))
                tr_acc    = [0.52 + 0.03*i - 0.0008*i*i for i in range(20)]
                vl_acc    = [0.48 + 0.026*i - 0.0007*i*i for i in range(20)]
                axes[0].plot(epochs, tr_acc, color='#3fb950', lw=2, label='Train')
                axes[0].plot(epochs, vl_acc, color='#58a6ff', lw=2, label='Val')
                axes[0].set_title('Accuracy – Stage 1', color='#e6edf3', fontsize=11)
                axes[0].legend(facecolor='#161b22', labelcolor='#8b949e')
                axes[0].set_xlabel('Epoch', color='#8b949e')
                tr_loss   = [1.4 - 0.06*i + 0.002*i*i for i in range(20)]
                vl_loss   = [1.5 - 0.055*i + 0.0019*i*i for i in range(20)]
                axes[1].plot(epochs, tr_loss, color='#f85149', lw=2, label='Train')
                axes[1].plot(epochs, vl_loss, color='#d29922', lw=2, label='Val')
                axes[1].set_title('Loss – Stage 1', color='#e6edf3', fontsize=11)
                axes[1].legend(facecolor='#161b22', labelcolor='#8b949e')
                axes[1].set_xlabel('Epoch', color='#8b949e')
                plt.tight_layout()
                st.pyplot(fig); plt.close()
                st.caption("Example curve — add plot_stage_1.png to assets/ for real data")

        with cb:
            p2 = "assets/plot_stage_2.png"
            if os.path.exists(p2):
                st.image(p2, use_column_width=True, caption="Stage 2 — Fine-tuning")
            else:
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                fig2.patch.set_facecolor('#161b22')
                ax2.set_facecolor('#21262d')
                for sp in ax2.spines.values(): sp.set_color('#30363d')
                ax2.tick_params(colors='#8b949e')
                ep2  = list(range(1, 16))
                acc2 = [0.80 + 0.005*i for i in range(15)]
                ax2.plot(ep2, acc2, color='#a8e063', lw=2.5)
                ax2.set_title('Accuracy – Stage 2 Fine-tuning', color='#e6edf3', fontsize=11)
                ax2.set_xlabel('Epoch', color='#8b949e'); ax2.set_ylabel('Accuracy', color='#8b949e')
                plt.tight_layout()
                st.pyplot(fig2); plt.close()
                st.caption("Example curve — add plot_stage_2.png to assets/ for real data")

    with tab_pc:
        pa = "assets/per_class_accuracy.png"
        if os.path.exists(pa):
            st.image(pa, use_column_width=True)

        class_data = {
            "cardboard": (87.5, 48, 42),
            "glass":     (84.0, 50, 42),
            "metal":     (92.7, 41, 38),
            "paper":     (93.3, 60, 56),
            "plastic":   (81.2, 66, 53),
            "trash":     (92.9, 14, 13),
        }
        rows = []
        for cls_n, (acc, total, correct) in class_data.items():
            d = DISPOSAL_DATA[cls_n]
            rows.append({
                "": d['emoji'],
                "Category":  cls_n.capitalize(),
                "Accuracy":  f"{acc:.1f}%",
                "Correct":   f"{correct}/{total}",
                "Rating":    "⭐⭐⭐" if acc >= 90 else "⭐⭐" if acc >= 85 else "⭐",
                "Status":    "Excellent" if acc >= 90 else "Good" if acc >= 85 else "Needs attention",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("""
        <div class="tip-box">
            Plastic and Glass show lower accuracy because they look visually similar —
            both are often transparent or reflective. This is a known challenge in waste
            classification and can be improved with more diverse training data.
        </div>""", unsafe_allow_html=True)

    with tab_arch:
        st.markdown("""
| Layer | Output Shape | Parameters |
|---|---|---|
| Input Layer | (224, 224, 3) | 0 |
| MobileNetV2 (base) | (7, 7, 1280) | 2,257,984 |
| GlobalAveragePooling2D | (1280,) | 0 |
| BatchNormalization | (1280,) | 5,120 |
| Dense — ReLU | (256,) | 327,936 |
| Dropout (0.4) | (256,) | 0 |
| Dense — ReLU | (128,) | 32,896 |
| Dropout (0.3) | (128,) | 0 |
| Dense — Softmax | (6,) | 774 |
| **Total Trainable** | | **~2.63M** |

**Training Configuration:**

| Parameter | Stage 1 | Stage 2 |
|---|---|---|
| Learning Rate | 0.001 | 0.00001 |
| Frozen Layers | All MobileNetV2 | Last 40 unfrozen |
| Epochs | 20 | 30 |
| Batch Size | 32 | 32 |
| Loss | categorical_crossentropy | categorical_crossentropy |
| Optimiser | Adam | Adam |

**Callbacks:**
- `EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)`
- `ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7)`
- `ModelCheckpoint(save_best_only=True)`

**Augmentation (training only):**
rotation ±20° · horizontal flip · zoom ±15% · brightness ±20% · width/height shift ±10%
        """)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "batch":
    sec_head("Batch Image Classifier")

    st.markdown("""
    <div style="background:rgba(63,185,80,0.08);border:1px solid rgba(63,185,80,0.25);
                border-radius:var(--radius);padding:14px 18px;font-size:13px;
                color:var(--green);margin-bottom:24px;">
        📁 &nbsp; Upload multiple waste images at once. The model classifies all of them and
        generates a downloadable CSV report with predictions, confidence scores and
        disposal actions.
    </div>""", unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Select multiple images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="upload_batch",
    )

    if batch_files:
        st.markdown(f"**{len(batch_files)} image{'s' if len(batch_files) > 1 else ''} selected**")

        # Preview grid
        max_preview = 8
        preview_cols = st.columns(min(len(batch_files), max_preview))
        for col, f in zip(preview_cols, batch_files[:max_preview]):
            with col:
                st.image(Image.open(f), use_column_width=True)
                f.seek(0)
        if len(batch_files) > max_preview:
            st.caption(f"…and {len(batch_files) - max_preview} more images")

        st.markdown("<br>", unsafe_allow_html=True)

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
                    "filename":   f.name,
                    "predicted":  r['predicted_class'],
                    "confidence": f"{r['confidence']:.1f}%",
                    "action":     d['action'],
                    "bin":        d['bin'],
                    "co2_saved":  d['co2'],
                })
                progress.progress((i + 1) / len(batch_files))
                st.session_state.total += 1

            status.text("✅  All images classified!")
            df = pd.DataFrame(all_res)

            st.markdown("<br>", unsafe_allow_html=True)
            sec_head("Batch Results")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Pie chart + summary stats
            col_pie, col_stats = st.columns([1, 1])
            with col_pie:
                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_facecolor('#161b22')
                ax.set_facecolor('#161b22')
                counts = df['predicted'].value_counts()
                colors = [DISPOSAL_DATA[c]['accent'] for c in counts.index]
                wedges, texts, autotexts = ax.pie(
                    counts.values, labels=counts.index, autopct='%1.0f%%',
                    colors=colors, startangle=90,
                    textprops={'color': '#e6edf3', 'fontsize': 11},
                )
                for at in autotexts:
                    at.set_color('#e6edf3'); at.set_fontsize(10)
                ax.set_title("Category Mix", color='#e6edf3', fontsize=13)
                st.pyplot(fig); plt.close()

            with col_stats:
                sec_head("Summary Stats")
                total_b    = len(all_res)
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
                    <div class="m-label">General Waste</div><div class="m-val">{total_b - recyclable}</div>
                    <div class="m-sub">{(total_b-recyclable)/total_b*100:.0f}%</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            csv_data = df.to_csv(index=False)
            st.download_button(
                "⬇️  Download Full CSV Report", csv_data,
                f"wasteai_batch_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv", use_container_width=True, key="btn_download_csv",
            )
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:var(--text-subtle);">
            <div style="font-size:48px;margin-bottom:16px;opacity:0.3;">📁</div>
            <div style="font-size:15px;font-weight:500;color:var(--text-muted);">No files selected yet</div>
            <div style="font-size:12px;margin-top:8px;">Use the file picker above to select multiple images</div>
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "about":

    st.markdown("""
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-xl);
                padding:36px;margin-bottom:28px;position:relative;overflow:hidden;">
        <div style="position:absolute;top:-40px;right:-40px;width:200px;height:200px;
                    background:radial-gradient(circle,rgba(63,185,80,0.1) 0%,transparent 70%);
                    pointer-events:none;"></div>
        <div style="font-family:var(--font-head);font-size:24px;font-weight:800;
                    color:var(--lime);margin-bottom:14px;">
            AI-Based Waste Segregation Classification System
        </div>
        <p style="font-size:14px;color:var(--text-muted);line-height:1.75;max-width:820px;">
            Improper waste segregation is a major challenge in urban waste management, leading to
            environmental pollution, inefficient recycling, and increased landfill usage. This project
            proposes an AI-based solution using deep learning to automatically classify waste images
            into 6 categories — achieving <strong style="color:var(--lime);">88.14% accuracy</strong>
            on an independent test set of 253 images from the TrashNet dataset.
        </p>
        <div style="margin-top:20px;display:flex;gap:10px;flex-wrap:wrap;">
            <span style="background:var(--green-bg);border:1px solid var(--green-dim);
                         color:var(--green);border-radius:99px;padding:5px 14px;font-size:12px;font-weight:600;">
                88.14% Test Accuracy</span>
            <span style="background:rgba(88,166,255,0.1);border:1px solid rgba(88,166,255,0.3);
                         color:#58a6ff;border-radius:99px;padding:5px 14px;font-size:12px;font-weight:600;">
                MobileNetV2 Architecture</span>
            <span style="background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.3);
                         color:#a78bfa;border-radius:99px;padding:5px 14px;font-size:12px;font-weight:600;">
                TrashNet Dataset</span>
            <span style="background:rgba(210,153,34,0.1);border:1px solid rgba(210,153,34,0.3);
                         color:#d29922;border-radius:99px;padding:5px 14px;font-size:12px;font-weight:600;">
                IoT Deployable</span>
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

    sec_head("Technology Stack")
    tech_items = [
        ("Python 3.10",          "Core language",            "#3776AB"),
        ("TensorFlow / Keras",   "Deep learning framework",  "#FF6F00"),
        ("MobileNetV2",          "CNN backbone",             "#4285F4"),
        ("OpenCV",               "Image preprocessing",      "#5C3EE8"),
        ("Streamlit",            "Web app framework",        "#FF4B4B"),
        ("TrashNet Dataset",     "2,527 labelled images",    "#3fb950"),
        ("NumPy / Pandas",       "Data processing",          "#013243"),
        ("Matplotlib",           "Visualisation",            "#11557c"),
    ]
    st.markdown('<div class="tech-grid">', unsafe_allow_html=True)
    for name, desc, color in tech_items:
        st.markdown(f"""
        <div class="tech-card" style="border-top-color:{color};">
            <div class="tech-name" style="color:{color};">{name}</div>
            <div class="tech-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    sec_head("Dataset — TrashNet")
    ds_col1, ds_col2 = st.columns([1, 2])
    with ds_col1:
        ds_items = [
            ("Total Images", "2,527", "var(--lime)"),
            ("Categories",   "6",     "#58a6ff"),
            ("Image Size",   "224×224","#8b5cf6"),
            ("Train Split",  "1,819 (72%)", "#d29922"),
            ("Val Split",    "455 (18%)",   "#ec4899"),
            ("Test Split",   "253 (10%)",   "#0891b2"),
        ]
        for label, val, color in ds_items:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color:{color};margin-bottom:10px;">
                <div class="m-label">{label}</div>
                <div class="m-val">{val}</div>
            </div>""", unsafe_allow_html=True)
    with ds_col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#161b22')
        ax.set_facecolor('#21262d')
        for sp in ax.spines.values(): sp.set_color('#30363d')
        ax.tick_params(colors='#8b949e')
        categories  = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        counts_dist = [403, 501, 410, 594, 482, 137]
        colors_dist = [DISPOSAL_DATA[c]['accent'] for c in categories]
        bars = ax.barh(categories, counts_dist, color=colors_dist, height=0.55)
        ax.set_xlabel('Number of Images', color='#8b949e')
        ax.set_title('TrashNet Class Distribution', color='#e6edf3', fontsize=13, pad=12)
        for bar, val in zip(bars, counts_dist):
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                    str(val), va='center', color='#8b949e', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    sec_head("Waste Categories — Disposal Reference")
    about_cats = st.columns(3)
    for idx, (cls_n, d) in enumerate(DISPOSAL_DATA.items()):
        with about_cats[idx % 3]:
            st.markdown(f"""
            <div style="background:{d['bg']};border:1px solid {d['accent']}40;
                        border-radius:var(--radius);padding:16px;margin-bottom:12px;">
                <div style="font-size:28px;margin-bottom:8px;">{d['emoji']}</div>
                <div style="font-size:14px;font-weight:700;color:{d['color']};margin-bottom:4px;">
                    {cls_n.capitalize()}</div>
                <div style="font-size:11px;color:{d['color']}99;margin-bottom:8px;">{d['bin']}</div>
                <div style="font-size:11px;color:{d['color']}bb;">{d['tip']}</div>
            </div>""", unsafe_allow_html=True)

    sec_head("Project Objectives & Performance")
    obj_cols = st.columns(2)
    with obj_cols[0]:
        st.markdown("""
**Primary Objectives:**
- Automate waste classification to reduce human error in sorting
- Provide clear, actionable disposal guidance for each waste type
- Demonstrate real-time AI classification in a responsive web interface
- Design an architecture deployable to IoT edge devices

**Secondary Objectives:**
- Visualise model performance and training metrics
- Support batch classification for high-throughput scenarios
- Generate downloadable CSV reports for waste tracking
- Provide developer API documentation for system integration
        """)
    with obj_cols[1]:
        st.markdown("""
**Performance Benchmarks:**

| Metric | Value |
|---|---|
| Test Accuracy | 88.14% |
| Top-3 Accuracy | ~98.5% |
| Average Confidence | 89.8% |
| Inference (CPU) | 180–350ms |
| Inference (GPU) | <50ms |
| Model Size (.h5) | ~14 MB |
| TFLite Size | ~3.5 MB |
| Raspberry Pi FPS | ~5 fps |
        """)

st.markdown('</div>', unsafe_allow_html=True)  # close .pw

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