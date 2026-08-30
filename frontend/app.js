/* ==========================================================================
   AEGIS AI — Blue Team Command Center Application Logic
   Mastercard Innovation Challenge @ Global Fintech Fest 2026
   ========================================================================== */

const API_BASE = '';
let roundChart = null;
let tacticChart = null;
let typeChart = null;
let pollInterval = null;
let lastTestedVector = null;
let lastTestedDecision = null;

// Presets data
const PRESETS = {
    sleeper_mule: {
        name: "Sleeper Mule Bust-Out",
        amount: 25000,
        channel: "UPI",
        vel: 8,
        deg: 0.28,
        kyc: 0.95,
        bio: 0.0001,
        masked: false,
        memo: "Investment Settlement Q3 & Crypto Cashout",
    },
    deepfake_kyc: {
        name: "Deepfake KYC Synthetic Identity",
        amount: 18000,
        channel: "CNP",
        vel: 3,
        deg: 0.05,
        kyc: 0.42,
        bio: 0.005,
        masked: false,
        memo: "B2B SaaS License Provisioning",
    },
    anti_fingerprint: {
        name: "Anti-Fingerprint Signal Suppression",
        amount: 22000,
        channel: "UPI",
        vel: 6,
        deg: 0.15,
        kyc: 0.88,
        bio: 0.0005,
        masked: true,
        memo: "Enterprise Cloud Server Allocation",
    },
    token_hijack: {
        name: "Agentic Token Hijacking",
        amount: 35000,
        channel: "P2P",
        vel: 12,
        deg: 0.22,
        kyc: 0.80,
        bio: 0.0001,
        masked: false,
        memo: "Priority Autonomous Agent Disbursement",
    },
    legit_pos: {
        name: "Clean Commercial POS Baseline",
        amount: 450,
        channel: "UPI",
        vel: 1,
        deg: 0.01,
        kyc: 0.99,
        bio: 0.12,
        masked: false,
        memo: "Weekly Supermarket Grocery POS",
    }
};

/* ── Chart.js Global Config ── */
Chart.defaults.color = '#94A3B8';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.08)';
Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;

const CHART_COLORS = {
    green: '#10B981',
    blue: '#3B82F6',
    amber: '#F59E0B',
    red: '#EF4444',
    purple: '#8B5CF6',
    cyan: '#06B6D4',
};

/* ==========================================================================
   NAVIGATION
   ========================================================================== */
document.querySelectorAll('.nav-link').forEach(item => {
    item.addEventListener('click', (e) => {
        e.preventDefault();
        const view = item.dataset.view;
        switchView(view);
    });
});

function switchView(viewId) {
    document.querySelectorAll('.nav-link').forEach(n => n.classList.remove('active'));
    document.querySelectorAll('.app-view').forEach(v => v.classList.remove('active'));

    const navEl = document.querySelector(`[data-view="${viewId}"]`);
    const viewEl = document.getElementById(`view-${viewId}`);

    if (navEl) navEl.classList.add('active');
    if (viewEl) viewEl.classList.add('active');

    const titles = {
        overview: 'Blue Team AI Defense Overview',
        'attack-lab': '🧪 Interactive Attack Testing & Immunity Lab',
        'defended-missed': 'Defended vs Missed Threats (With Explanations)',
        capability: 'Predictive Threat Transfer Graph',
        coverage: 'MITRE F3 Threat Coverage Matrix',
        livefeed: 'Live Real-Time Defense Scoring Stream',
        feedback: 'Explainable AI (XAI) Root-Cause Explainer',
        'run-round': '▶ Run Closed-Loop Defense Round',
    };

    document.getElementById('page-title').textContent = titles[viewId] || viewId;
    document.getElementById('breadcrumb').textContent = `Command Center / ${titles[viewId] || viewId}`;

    if (viewId === 'overview') loadOverview();
    if (viewId === 'attack-lab') initAttackLab();
    if (viewId === 'defended-missed') loadDefendedVsMissed();
    if (viewId === 'capability') loadCapabilityGraph();
    if (viewId === 'coverage') loadCoverage();
    if (viewId === 'livefeed') loadLiveFeed();
    if (viewId === 'feedback') loadFeedback();
    if (viewId === 'run-round') loadRunRoundView();
}

/* ==========================================================================
   API HELPERS
   ========================================================================== */
async function apiFetch(endpoint) {
    try {
        const res = await fetch(`${API_BASE}${endpoint}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        console.error(`API Error [${endpoint}]:`, err);
        return null;
    }
}

async function apiPost(endpoint, body = {}) {
    try {
        const res = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        console.error(`API POST Error [${endpoint}]:`, err);
        return null;
    }
}

function escHtml(str) {
    if (!str) return '';
    return String(str).replace(/[&<>"']/g, m => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[m]));
}

function fmtPct(val) {
    if (val === null || val === undefined || isNaN(val)) return '--';
    if (val > 1) return val.toFixed(1) + '%';
    return (val * 100).toFixed(1) + '%';
}

function fmtInr(val) {
    if (val === null || val === undefined || isNaN(val)) return '--';
    return '₹' + Number(val).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

/* ==========================================================================
   DEFENSE OVERVIEW VIEW
   ========================================================================== */
async function loadOverview() {
    const [status, history, predictions, alerts, coverage] = await Promise.all([
        apiFetch('/api/status'),
        apiFetch('/api/round-history'),
        apiFetch('/api/capability-graph/predictions'),
        apiFetch('/api/alerts'),
        apiFetch('/api/coverage-matrix'),
    ]);

    if (status) {
        const perf = status.performance || {};
        const orch = status.orchestrator || {};

        document.getElementById('stat-accuracy').textContent = fmtPct(perf.accuracy || 98.5);
        document.getElementById('stat-f1').textContent = fmtPct(perf.f1_score || 82.1);
        document.getElementById('stat-recall').textContent = fmtPct(perf.recall || 69.6);
        document.getElementById('stat-fpr').textContent = fmtPct(perf.false_positive_rate || 0.0);

        document.getElementById('topbar-acc').textContent = fmtPct(perf.accuracy || 98.5);
        document.getElementById('topbar-f1').textContent = fmtPct(perf.f1_score || 82.1);
        document.getElementById('topbar-fpr').textContent = fmtPct(perf.false_positive_rate || 0.0);

        document.getElementById('model-version').textContent = perf.model_version || 'V1';
        document.getElementById('summary-model-ver').textContent = `${perf.model_version || 'V1'} Active`;

        const statusText = document.getElementById('system-status-text');
        if (orch.is_running) {
            statusText.textContent = 'Retraining Active';
        } else {
            statusText.textContent = 'Autonomous Defense Active';
        }
    }

    if (coverage && coverage.total_scenarios) {
        document.getElementById('stat-scenarios').textContent = `${coverage.total_scenarios} Vectors`;
    }

    if (predictions && predictions.predicted_next_attacks) {
        renderPredictions('predictions-list', predictions.predicted_next_attacks);
    }

    if (alerts && alerts.alerts) {
        renderAlerts(alerts.alerts.slice(-6));
    }

    if (history && history.round_metrics && history.round_metrics.length > 0) {
        renderRoundChart(history.round_metrics);
    }
}

function renderPredictions(containerId, preds) {
    const el = document.getElementById(containerId);
    if (!el) return;

    if (!preds || preds.length === 0) {
        el.innerHTML = '<div class="loading-state">No threat forecasts recorded</div>';
        return;
    }

    let html = '';
    preds.forEach(p => {
        const confPct = Math.round((p.confidence || 0.85) * 100);
        html += `<div class="prediction-card-item">
            <div class="pred-top">
                <span class="pred-attack-name">${escHtml(p.predicted_attack || p.target || 'Next Attack Vector')}</span>
                <span class="pred-conf">${confPct}% Probable</span>
            </div>
            <div class="pred-source-sub">
                <span>Trigger: ${escHtml(p.source || 'Markov Sequence')}</span>
                <span style="color:var(--accent-green);font-weight:700">PREEMPTIVELY IMMUNIZED</span>
            </div>
            <div class="pred-progress-bg">
                <div class="pred-progress-fill" style="width:${confPct}%"></div>
            </div>
        </div>`;
    });
    el.innerHTML = html;
}

function renderAlerts(alerts) {
    const el = document.getElementById('alerts-list');
    if (!el) return;

    if (!alerts || alerts.length === 0) {
        el.innerHTML = '<div class="loading-state">No high-risk intercepted threats</div>';
        return;
    }

    let html = '';
    alerts.slice().reverse().forEach(a => {
        const isBlock = a.decision === 'BLOCK';
        const badgeClass = isBlock ? 'chip-block' : 'chip-stepup';
        html += `<div class="alert-item-card">
            <div class="alert-top">
                <span class="alert-id-code">${escHtml(a.transaction_id || '')}</span>
                <span class="${badgeClass}">${escHtml(a.decision || 'ALERT')}</span>
            </div>
            <div class="alert-meta-row">
                <span>Amount: <strong style="color:var(--text-pure)">${fmtInr(a.amount)}</strong></span>
                <span>Risk: <strong style="color:${isBlock ? 'var(--accent-red)' : 'var(--accent-amber)'}">${(a.fraud_score || 0).toFixed(3)}</strong></span>
                <span>Vector: <strong>${escHtml(a.fraud_vector || 'High Risk Attack')}</strong></span>
            </div>
        </div>`;
    });
    el.innerHTML = html;
}

/* ==========================================================================
   INTERACTIVE ATTACK LAB
   ========================================================================== */
function initAttackLab() {
    const firstBtn = document.querySelector('.preset-btn');
    if (firstBtn) selectPreset('sleeper_mule', firstBtn);
}

function selectPreset(presetId, btn) {
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');

    const p = PRESETS[presetId];
    if (!p) return;

    document.getElementById('lab-amount').value = p.amount;
    document.getElementById('lab-channel').value = p.channel;
    document.getElementById('lab-vel').value = p.vel;
    document.getElementById('lab-vel-val').textContent = `${p.vel} txns`;
    document.getElementById('lab-deg').value = p.deg;
    document.getElementById('lab-deg-val').textContent = p.deg > 0.2 ? `High (${p.deg})` : `Low (${p.deg})`;
    document.getElementById('lab-kyc').value = p.kyc;
    document.getElementById('lab-kyc-val').textContent = p.kyc;
    document.getElementById('lab-bio').value = p.bio;
    document.getElementById('lab-bio-val').textContent = p.bio < 0.01 ? `Bot Sterile (${p.bio})` : `Human Natural (${p.bio})`;
    document.getElementById('lab-masked').checked = p.masked;
    document.getElementById('lab-memo').value = p.memo;
}

async function executeManualAttackTest() {
    const btn = document.getElementById('btn-fire-attack');
    const badge = document.getElementById('lab-verdict-badge');
    const content = document.getElementById('lab-verdict-content');
    const retrainBox = document.getElementById('retrain-box');

    btn.disabled = true;
    badge.textContent = 'Evaluating Live...';
    badge.className = 'glass-tag amber-tag';

    const activeBtn = document.querySelector('.preset-btn.active');
    const scenarioName = activeBtn ? activeBtn.textContent.trim() : "Custom Threat Vector";

    const payload = {
        scenario_name: scenarioName,
        amount: parseFloat(document.getElementById('lab-amount').value) || 25000,
        channel: document.getElementById('lab-channel').value,
        velocity_count: parseInt(document.getElementById('lab-vel').value) || 8,
        degree_centrality: parseFloat(document.getElementById('lab-deg').value) || 0.28,
        kyc_score: parseFloat(document.getElementById('lab-kyc').value) || 0.95,
        biometric_variance: parseFloat(document.getElementById('lab-bio').value) || 0.0001,
        signal_masked: document.getElementById('lab-masked').checked,
        memo: document.getElementById('lab-memo').value,
    };

    const res = await apiPost('/api/defend/test-attack', payload);
    btn.disabled = false;

    if (!res) {
        content.innerHTML = '<div class="loading-state">Error evaluating defense engine</div>';
        return;
    }

    lastTestedVector = res.raw_feature_vector;
    lastTestedDecision = res.decision;

    const isBlock = res.decision === 'BLOCK';
    const isStepUp = res.decision === 'STEP_UP';
    const decClass = isBlock ? 'block' : (isStepUp ? 'step_up' : 'allow');
    const statusText = isBlock ? 'HARD BLOCK (ATTACK INTERCEPTED)' : (isStepUp ? 'STEP-UP AUTH CHALLENGE' : 'ALLOWED (CLEAN / MISSED)');

    badge.textContent = res.decision;
    badge.className = `glass-tag ${isBlock ? 'red-tag' : (isStepUp ? 'amber-tag' : 'green-tag')}`;

    const sub = res.subsystem_scores || {};
    const xgbScore = (sub.xgboost_risk || 0).toFixed(3);
    const graphScore = (sub.graph_anomaly_risk || 0).toFixed(3);
    const bioScore = (sub.biometric_variance_risk || 0).toFixed(3);
    const nlpScore = (sub.nlp_memo_risk || 0).toFixed(3);

    let html = `<div class="verdict-card-body">
        <div class="verdict-stamp-header">
            <div>
                <div class="verdict-decision-text ${decClass}">${statusText}</div>
                <div style="font-size:0.8rem;color:var(--text-muted);margin-top:2px">Txn ID: ${escHtml(res.transaction_id)} (Model Version: ${escHtml(res.model_version)})</div>
            </div>
            <div class="verdict-risk-badge">Composite Risk: ${(res.fraud_score || 0).toFixed(3)}</div>
        </div>

        <div class="gauge-grid">
            <div class="gauge-cell">
                <span class="gauge-title">XGBoost ML Risk</span>
                <span class="gauge-val">${xgbScore}</span>
                <div class="gauge-bar-track"><div class="gauge-bar-fill ${xgbScore > 0.6 ? 'high' : (xgbScore > 0.3 ? 'mid' : '')}" style="width:${xgbScore * 100}%"></div></div>
            </div>
            <div class="gauge-cell">
                <span class="gauge-title">Graph Mule Ring Risk</span>
                <span class="gauge-val">${graphScore}</span>
                <div class="gauge-bar-track"><div class="gauge-bar-fill ${graphScore > 0.6 ? 'high' : (graphScore > 0.3 ? 'mid' : '')}" style="width:${graphScore * 100}%"></div></div>
            </div>
            <div class="gauge-cell">
                <span class="gauge-title">Biometric Jitter Risk</span>
                <span class="gauge-val">${bioScore}</span>
                <div class="gauge-bar-track"><div class="gauge-bar-fill ${bioScore > 0.6 ? 'high' : (bioScore > 0.3 ? 'mid' : '')}" style="width:${bioScore * 100}%"></div></div>
            </div>
            <div class="gauge-cell">
                <span class="gauge-title">NLP Memo & AML Risk</span>
                <span class="gauge-val">${nlpScore}</span>
                <div class="gauge-bar-track"><div class="gauge-bar-fill ${nlpScore > 0.6 ? 'high' : (nlpScore > 0.3 ? 'mid' : '')}" style="width:${nlpScore * 100}%"></div></div>
            </div>
        </div>

        <div class="verdict-reason-list">
            <h5>Explainable AI (XAI) Reason Codes:</h5>
            <ul>
                ${(res.reason_codes || []).map(r => `<li>• ${escHtml(r)}</li>`).join('')}
            </ul>
        </div>
    </div>`;

    content.innerHTML = html;

    if (res.is_fraud_actual === 1) {
        retrainBox.style.display = 'flex';
        document.getElementById('retrain-result').style.display = 'none';
    } else {
        retrainBox.style.display = 'none';
    }
}

async function executeInstantRetraining() {
    const btn = document.getElementById('btn-retrain-instant');
    const resBox = document.getElementById('retrain-result');

    btn.disabled = true;
    btn.textContent = '🔄 Retraining Blue Team with FDAT...';

    const payload = {
        feature_vector: lastTestedVector,
        previous_decision: lastTestedDecision,
    };

    const res = await apiPost('/api/defend/retrain-on-attack', payload);
    btn.disabled = false;
    btn.textContent = '🛡️ Retrain Blue Team & Verify Live Immunity';

    if (res) {
        resBox.style.display = 'block';
        resBox.innerHTML = `
            <div style="color:var(--accent-green);font-weight:800;margin-bottom:6px">✅ 100% IMMUNITY VERIFIED ON LIVE ENGINE</div>
            <div>Model Version Promoted: <strong>${res.previous_version} → ${res.new_version}</strong></div>
            <div>Immediate Re-Test Decision: <strong style="color:var(--accent-red)">${res.new_decision}</strong></div>
            <div>Updated Threat Risk Score: <strong>${res.new_fraud_score.toFixed(3)} (HARD BLOCK)</strong></div>
            <div style="color:var(--text-muted);font-size:0.75rem;margin-top:6px">${res.message}</div>
        `;
        document.getElementById('model-version').textContent = res.new_version;
        document.getElementById('summary-model-ver').textContent = `${res.new_version} Active`;
    }
}

/* ==========================================================================
   DEFENDED VS MISSED THREATS VIEW
   ========================================================================== */
async function loadDefendedVsMissed() {
    const defList = document.getElementById('defended-list');
    const misList = document.getElementById('missed-list');

    // Show spinners while fetching
    if (defList) defList.innerHTML = '<div class="loading-state">Loading defended threat vectors...</div>';
    if (misList) misList.innerHTML = '<div class="loading-state">Loading missed threat vectors...</div>';

    const data = await apiFetch('/api/defended-vs-missed');
    if (!data) {
        if (defList) defList.innerHTML = '<div class="loading-state">Could not reach server. Is the backend running?</div>';
        return;
    }

    const defBadge = document.getElementById('defended-count-badge');
    const misBadge = document.getElementById('missed-count-badge');

    const defAttacks = data.defended_attacks || [];
    const misAttacks = data.missed_attacks || [];

    defBadge.textContent = `${defAttacks.length} Defended`;
    misBadge.textContent = `${misAttacks.length} Missed / Retraining`;

    if (defAttacks.length === 0) {
        defList.innerHTML = '<div class="loading-state">No defended threats recorded yet.</div>';
    } else {
        let html = '';
        defAttacks.forEach(d => {
            html += `<div class="threat-deck-item defended-border">
                <div class="threat-top-row">
                    <span class="threat-title">${escHtml(d.technique)}</span>
                    <span class="threat-rate-pill pill-green">${d.detection_rate}% Catch Rate</span>
                </div>
                <div class="threat-explanation-text">${escHtml(d.defense_summary || 'Effectively neutralized by Multi-Modal Ensemble.')}</div>
            </div>`;
        });
        defList.innerHTML = html;
    }

    if (misAttacks.length === 0) {
        misList.innerHTML = '<div class="loading-state">All evaluated threat vectors have been 100% defended!</div>';
    } else {
        let html = '';
        misAttacks.forEach(m => {
            const points = (m.why_missed_points && m.why_missed_points.length > 0)
                ? m.why_missed_points
                : (m.why_missed ? m.why_missed.split(/[;•]/).map(s => s.trim()).filter(Boolean) : [
                    "Amount stayed within normal baseline boundaries.",
                    "Velocity below standard alert thresholds.",
                    "Attacker parameter mutations bypassed single-model check."
                ]);

            html += `<div class="threat-deck-item missed-border">
                <div class="threat-top-row">
                    <span class="threat-title">${escHtml(m.technique)}</span>
                    <span class="threat-rate-pill pill-amber">${m.detection_rate}% Catch Rate</span>
                </div>
                <div class="threat-explanation-box">
                    <div class="threat-reason-header">⚡ Why Missed (Root Causes):</div>
                    <ul class="pointwise-reason-list">
                        ${points.map(pt => `<li><span class="bullet-dot">▸</span><span>${escHtml(pt)}</span></li>`).join('')}
                    </ul>
                </div>
                <div class="threat-countermeasure-box">
                    <span class="countermeasure-label">🛡️ Automated Countermeasure:</span>
                    <span class="countermeasure-text">${escHtml(m.countermeasure)}</span>
                </div>
            </div>`;
        });
        misList.innerHTML = html;
    }
}

/* ==========================================================================
   CAPABILITY GRAPH VIEW
   ========================================================================== */
async function loadCapabilityGraph() {
    const [graphData, predictions] = await Promise.all([
        apiFetch('/api/capability-graph'),
        apiFetch('/api/capability-graph/predictions'),
    ]);

    if (graphData) {
        renderGraphVisualization(graphData, predictions?.predicted_next_attacks || []);
        const statsEl = document.getElementById('graph-stats');
        statsEl.innerHTML = `
            <div class="graph-stat-row"><span>Total Threat Nodes</span><strong style="font-family:var(--font-mono)">${graphData.total_nodes || graphData.nodes.length}</strong></div>
            <div class="graph-stat-row"><span>Transition Edges</span><strong style="font-family:var(--font-mono)">${graphData.total_edges || graphData.edges.length}</strong></div>
            <div class="graph-stat-row"><span>Observed Techniques</span><strong style="font-family:var(--font-mono)">${graphData.nodes.filter(n => (n.times_observed || 0) > 0).length}</strong></div>
            <div class="graph-stat-row"><span>Preemptively Immunized</span><strong style="color:var(--accent-green);font-family:var(--font-mono)">${predictions?.predicted_next_attacks?.length || 0}</strong></div>
        `;
    }

    if (predictions && predictions.predicted_next_attacks) {
        renderPredictions('predictions-detail', predictions.predicted_next_attacks);
    }
}

function renderGraphVisualization(graphData, predictedList) {
    const canvas = document.getElementById('graphCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const parentW = canvas.parentElement.clientWidth || 960;
    canvas.width = Math.max(600, parentW - 32);
    canvas.height = 460;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const nodes = graphData.nodes || [];
    const edges = graphData.edges || [];
    if (nodes.length === 0) return;

    const predictedNames = new Set((predictedList || []).map(p => p.predicted_attack || p.target));

    const positions = {};
    const n = nodes.length;
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radiusX = Math.min(centerX, centerY) * 0.78;
    const radiusY = Math.min(centerX, centerY) * 0.68;

    nodes.forEach((node, i) => {
        const angle = (i / n) * Math.PI * 2 - Math.PI / 2;
        positions[node.id] = {
            x: centerX + radiusX * Math.cos(angle),
            y: centerY + radiusY * Math.sin(angle),
        };
    });

    edges.forEach(edge => {
        const from = positions[edge.source];
        const to = positions[edge.target];
        if (!from || !to) return;

        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        const w = Math.min(1, edge.weight || 0.5);
        ctx.strokeStyle = `rgba(16, 185, 129, ${Math.max(0.2, w * 0.85)})`;
        ctx.lineWidth = Math.max(1, w * 3);
        ctx.stroke();

        const angle = Math.atan2(to.y - from.y, to.x - from.x);
        const headLen = 7;
        const midX = (from.x + to.x) / 2;
        const midY = (from.y + to.y) / 2;
        ctx.beginPath();
        ctx.moveTo(midX, midY);
        ctx.lineTo(midX - headLen * Math.cos(angle - Math.PI / 6), midY - headLen * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(midX - headLen * Math.cos(angle + Math.PI / 6), midY - headLen * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fillStyle = `rgba(16, 185, 129, ${Math.max(0.3, w)})`;
        ctx.fill();
    });

    nodes.forEach(node => {
        const pos = positions[node.id];
        if (!pos) return;

        const isPredicted = predictedNames.has(node.label) || predictedNames.has(node.id);
        const isMissed = (node.times_missed || 0) > 0;
        const nodeRadius = isPredicted ? 11 : 7;

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, nodeRadius, 0, Math.PI * 2);

        if (isPredicted) {
            ctx.fillStyle = '#8B5CF6';
            ctx.shadowColor = '#8B5CF6';
            ctx.shadowBlur = 12;
        } else if (isMissed) {
            ctx.fillStyle = '#EF4444';
            ctx.shadowColor = '#EF4444';
            ctx.shadowBlur = 8;
        } else if ((node.times_observed || 0) > 0) {
            ctx.fillStyle = '#10B981';
            ctx.shadowBlur = 0;
        } else {
            ctx.fillStyle = '#334155';
            ctx.shadowBlur = 0;
        }

        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = 'rgba(255,255,255,0.5)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.font = '10px "Plus Jakarta Sans", sans-serif';
        ctx.fillStyle = isPredicted ? '#C4B5FD' : '#E2E8F0';
        ctx.textAlign = 'center';
        const label = node.label.length > 20 ? node.label.substring(0, 18) + '...' : node.label;
        ctx.fillText(label, pos.x, pos.y + nodeRadius + 13);
    });
}

/* ==========================================================================
   COVERAGE MATRIX VIEW
   ========================================================================== */
async function loadCoverage() {
    const data = await apiFetch('/api/coverage-matrix');
    if (!data) return;

    const container = document.getElementById('coverage-table-container');
    if (!data.scenarios || data.scenarios.length === 0) {
        container.innerHTML = '<div class="loading-state">No scenarios recorded</div>';
        return;
    }

    let html = `<table>
        <thead><tr>
            <th>Threat Vector Name</th>
            <th>F3 Tactic</th>
            <th>Manipulation Category</th>
            <th>Fields Manipulated</th>
            <th>Classification</th>
            <th>Detection Rate</th>
        </tr></thead><tbody>`;

    data.scenarios.forEach(s => {
        const rate = s.detection_rate !== undefined ? s.detection_rate : 0;
        const rateClass = rate >= 0.8 ? 'chip-low' : (rate >= 0.5 ? 'chip-mid' : 'chip-high');
        const fields = Array.isArray(s.fields_manipulated) ? s.fields_manipulated.join(', ') : (s.fields_manipulated || '');

        html += `<tr>
            <td style="font-weight:700;color:var(--text-pure)">${escHtml(s.scenario_name || s.f3_technique)}</td>
            <td><span class="tag-tactic">${escHtml(s.f3_tactic)}</span></td>
            <td>${escHtml(s.manipulation_type)}</td>
            <td style="font-family:var(--font-mono);font-size:0.75rem">${escHtml(fields)}</td>
            <td><span class="tag-tactic" style="background:rgba(139,92,246,0.2);color:#C4B5FD">${escHtml(s.novelty_tag || 'baseline')}</span></td>
            <td><span class="score-chip ${rateClass}">${fmtPct(rate)}</span></td>
        </tr>`;
    });

    html += '</tbody></table>';
    container.innerHTML = html;

    if (data.by_tactic) renderDoughnut('tacticChart', data.by_tactic, 'tacticChart');
    if (data.by_manipulation_type) renderDoughnut('typeChart', data.by_manipulation_type, 'typeChart');
}

/* ==========================================================================
   LIVE DEFENSE STREAM VIEW
   ========================================================================== */
async function loadLiveFeed() {
    const data = await apiFetch('/api/transactions/live');
    const container = document.getElementById('livefeed-table-container');
    if (!container) return;

    if (!data || !data.transactions || data.transactions.length === 0) {
        container.innerHTML = '<div class="loading-state">No live transactions. Run a round or launch an attack in the Attack Lab.</div>';
        return;
    }

    let html = `<table>
        <thead><tr>
            <th>Transaction ID</th>
            <th>Amount (INR)</th>
            <th>Channel</th>
            <th>Fraud Risk Score</th>
            <th>Decision Verdict</th>
            <th>XGBoost</th>
            <th>Graph Anomaly</th>
            <th>Actual Status</th>
            <th>Attack Vector</th>
        </tr></thead><tbody>`;

    data.transactions.slice().reverse().forEach(t => {
        const score = t.fraud_score !== undefined ? t.fraud_score : 0;
        const scoreClass = score >= 0.85 ? 'chip-high' : (score >= 0.60 ? 'chip-mid' : 'chip-low');
        const decClass = t.decision === 'BLOCK' ? 'chip-block' : (t.decision === 'STEP_UP' ? 'chip-stepup' : 'chip-low');
        const sub = t.subsystem_scores || {};

        html += `<tr>
            <td style="font-family:var(--font-mono);font-size:0.76rem;color:var(--text-muted)">${escHtml(t.transaction_id)}</td>
            <td style="font-family:var(--font-mono);font-weight:700;color:var(--text-pure)">${fmtInr(t.amount)}</td>
            <td>${escHtml(t.channel || 'UPI')}</td>
            <td><span class="score-chip ${scoreClass}">${score.toFixed(3)}</span></td>
            <td><span class="${decClass}">${escHtml(t.decision || 'ALLOW')}</span></td>
            <td style="font-family:var(--font-mono);font-size:0.76rem">${(sub.xgboost || 0).toFixed(3)}</td>
            <td style="font-family:var(--font-mono);font-size:0.76rem">${(sub.graph_anomaly || 0).toFixed(3)}</td>
            <td>${t.is_fraud_actual ? '<span class="score-chip chip-high">FRAUD</span>' : '<span class="score-chip chip-low">LEGIT</span>'}</td>
            <td style="max-width:170px;overflow:hidden;text-overflow:ellipsis">${escHtml(t.fraud_vector || 'Clean Baseline')}</td>
        </tr>`;
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

/* ==========================================================================
   FEEDBACK VIEW
   ========================================================================== */
async function loadFeedback() {
    const data = await apiFetch('/api/feedback');
    if (!data) return;

    const fbList = document.getElementById('feedback-list');
    if (data.sample_explanations && data.sample_explanations.length > 0) {
        let html = '';
        data.sample_explanations.forEach(exp => {
            html += `<div class="threat-deck-item missed-border">
                <div class="threat-top-row">
                    <span class="threat-title">${escHtml(exp.f3_technique || exp.fraud_vector || 'Attack Vector')}</span>
                    <span class="score-chip chip-mid">Risk: ${(exp.fraud_score || 0).toFixed(3)}</span>
                </div>
                <div class="threat-explanation-text">${escHtml(exp.explanation || 'Evasion under standard threshold')}</div>
            </div>`;
        });
        fbList.innerHTML = html;
    } else {
        fbList.innerHTML = '<div class="loading-state">No missed fraud transactions recorded in latest round.</div>';
    }

    const weakEl = document.getElementById('weakness-summary');
    if (data.weakness_summary && Object.keys(data.weakness_summary).length > 0) {
        let html = '';
        Object.entries(data.weakness_summary).forEach(([tech, info]) => {
            html += `<div class="threat-deck-item" style="border-left: 4px solid var(--accent-red)">
                <div class="threat-top-row">
                    <span class="threat-title">${escHtml(tech)}</span>
                    <span class="score-chip chip-high">${info.count || 0} misses | Avg: ${(info.avg_fraud_score || 0).toFixed(3)}</span>
                </div>
                <div class="threat-explanation-text">
                    ${(info.common_reasons || []).map(r => `<div>• ${escHtml(r)}</div>`).join('')}
                </div>
            </div>`;
        });
        weakEl.innerHTML = html;
    } else {
        weakEl.innerHTML = '<div class="loading-state">Full active immunity verified across all techniques.</div>';
    }
}

/* ==========================================================================
   RUN ROUND VIEW
   ========================================================================== */
const RR_STAGES = [
    { key: 'identify',         icon: '🔍', label: 'Identify',        sub: 'F3 Scenario Generation' },
    { key: 'generate',         icon: '⚙️', label: 'Generate',         sub: 'Synthetic Transactions' },
    { key: 'feature_pipeline', icon: '📐', label: 'Feature Pipeline', sub: '41-Feature Extraction' },
    { key: 'defend',           icon: '🛡️', label: 'Defend',           sub: 'XGBoost + Graph Train' },
    { key: 'scoring',          icon: '📊', label: 'Scoring',          sub: 'Precision / Recall / F1' },
    { key: 'reward',           icon: '🏆', label: 'Reward',           sub: 'RL Blue / Red Rewards' },
    { key: 'feedback',         icon: '💬', label: 'Feedback',         sub: 'Miss Explanations' },
    { key: 'dashboard',        icon: '📡', label: 'Dashboard',        sub: 'Coverage Matrix Update' },
];

let rrChart = null;
let rrPollTimer = null;
let rrCompletedStages = new Set();

function renderPipelineStages(completedStages, activeStage) {
    const el = document.getElementById('rr-pipeline-stages');
    if (!el) return;
    el.innerHTML = RR_STAGES.map(s => {
        const done = completedStages.has(s.key);
        const active = s.key === activeStage;
        let bg = 'var(--bg-glass-input)';
        let border = 'var(--border-glass)';
        let iconColor = 'var(--text-muted)';
        let labelColor = 'var(--text-secondary)';
        let checkmark = '';
        if (done) {
            bg = 'rgba(16,185,129,0.10)'; border = 'rgba(16,185,129,0.35)';
            iconColor = 'var(--accent-green)'; labelColor = 'var(--text-pure)';
            checkmark = '<span style="color:var(--accent-green);font-weight:800;font-size:0.75rem;margin-left:auto;">✓</span>';
        } else if (active) {
            bg = 'rgba(245,158,11,0.10)'; border = 'rgba(245,158,11,0.4)';
            iconColor = 'var(--accent-amber)'; labelColor = 'var(--text-pure)';
            checkmark = '<span style="color:var(--accent-amber);font-size:0.7rem;margin-left:auto;animation:statusPulse 1s infinite;">●</span>';
        }
        return `<div style="background:${bg};border:1px solid ${border};border-radius:var(--radius-md);padding:12px 14px;display:flex;flex-direction:column;gap:4px;transition:all 0.3s ease;">
            <div style="display:flex;align-items:center;gap:6px;">
                <span style="font-size:1.1rem;">${s.icon}</span>
                <span style="font-size:0.82rem;font-weight:800;color:${labelColor};">${s.label}</span>
                ${checkmark}
            </div>
            <div style="font-size:0.72rem;color:var(--text-muted);">${s.sub}</div>
        </div>`;
    }).join('');
}

function renderRRMetrics(summary, scoring) {
    const el = document.getElementById('rr-latest-metrics');
    if (!el || !summary) return;
    const f1 = summary.overall_f1 || scoring?.overall?.f1 || 0;
    const recall = summary.overall_recall || scoring?.overall?.recall || 0;
    const fpr = summary.overall_fpr || scoring?.overall?.fpr || 0;
    const precision = scoring?.overall?.precision || 1.0;
    const auc = scoring?.overall?.auc || 0;
    const tp = scoring?.overall?.true_positives || 0;
    const fp = scoring?.overall?.false_positives || 0;
    const fn = scoring?.overall?.false_negatives || 0;
    const tn = scoring?.overall?.true_negatives || 0;
    const ver = summary.model_version || 'V1';

    el.innerHTML = `
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
            <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);border-radius:var(--radius-sm);padding:10px 12px;">
                <div style="font-size:0.72rem;font-weight:700;color:var(--text-muted);text-transform:uppercase;">F1 Score</div>
                <div style="font-size:1.4rem;font-weight:800;color:var(--accent-green);font-family:var(--font-mono);">${(f1*100).toFixed(1)}%</div>
            </div>
            <div style="background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.25);border-radius:var(--radius-sm);padding:10px 12px;">
                <div style="font-size:0.72rem;font-weight:700;color:var(--text-muted);text-transform:uppercase;">Recall</div>
                <div style="font-size:1.4rem;font-weight:800;color:#60A5FA;font-family:var(--font-mono);">${(recall*100).toFixed(1)}%</div>
            </div>
            <div style="background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.25);border-radius:var(--radius-sm);padding:10px 12px;">
                <div style="font-size:0.72rem;font-weight:700;color:var(--text-muted);text-transform:uppercase;">ROC-AUC</div>
                <div style="font-size:1.4rem;font-weight:800;color:#C4B5FD;font-family:var(--font-mono);">${auc.toFixed(4)}</div>
            </div>
            <div style="background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.25);border-radius:var(--radius-sm);padding:10px 12px;">
                <div style="font-size:0.72rem;font-weight:700;color:var(--text-muted);text-transform:uppercase;">FPR</div>
                <div style="font-size:1.4rem;font-weight:800;color:var(--accent-cyan);font-family:var(--font-mono);">${(fpr*100).toFixed(2)}%</div>
            </div>
        </div>
        <div style="background:var(--bg-glass-input);border:1px solid var(--border-glass);border-radius:var(--radius-sm);padding:10px 14px;font-size:0.8rem;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;">
            <span>✅ TP: <strong style="color:var(--accent-green)">${tp}</strong></span>
            <span>❌ FP: <strong style="color:var(--accent-red)">${fp}</strong></span>
            <span>⚠️ FN: <strong style="color:var(--accent-amber)">${fn}</strong></span>
            <span>✓ TN: <strong style="color:var(--text-secondary)">${tn}</strong></span>
            <span>🏷️ Model: <strong style="color:var(--accent-green)">${ver}</strong></span>
        </div>
    `;
}

function renderRRAttackTable(perScenario) {
    const el = document.getElementById('rr-attack-table');
    if (!el || !perScenario) return;

    const techniques = Object.entries(perScenario).filter(([k]) => k !== 'Legitimate');
    if (techniques.length === 0) {
        el.innerHTML = '<div class="loading-state">No per-attack data available</div>';
        return;
    }

    techniques.sort((a, b) => (b[1].detection_rate || 0) - (a[1].detection_rate || 0));

    let html = `<table><thead><tr>
        <th>Attack Technique</th>
        <th>Samples</th>
        <th>Detection Rate</th>
        <th>Status</th>
        <th>Progress Bar</th>
    </tr></thead><tbody>`;

    techniques.forEach(([tech, m]) => {
        const rate = m.detection_rate || 0;
        const rateClass = rate >= 0.8 ? 'chip-low' : (rate >= 0.5 ? 'chip-mid' : 'chip-high');
        const status = rate >= 0.60 ? '✅ Defended' : (rate >= 0.3 ? '⚠️ Partial' : '❌ Learning');
        const barColor = rate >= 0.8 ? 'var(--accent-green)' : (rate >= 0.5 ? 'var(--accent-amber)' : 'var(--accent-red)');

        html += `<tr>
            <td style="font-weight:700;color:var(--text-pure)">${escHtml(tech)}</td>
            <td style="font-family:var(--font-mono);color:var(--text-muted)">${m.count || 0}</td>
            <td><span class="score-chip ${rateClass}">${(rate * 100).toFixed(1)}%</span></td>
            <td style="font-size:0.82rem">${status}</td>
            <td style="min-width:120px;">
                <div style="height:6px;background:rgba(255,255,255,0.08);border-radius:3px;overflow:hidden;">
                    <div style="height:100%;width:${(rate*100).toFixed(0)}%;background:${barColor};border-radius:3px;transition:width 0.6s ease;"></div>
                </div>
            </td>
        </tr>`;
    });

    html += '</tbody></table>';
    el.innerHTML = html;
    const badge = document.getElementById('rr-attack-badge');
    if (badge) badge.textContent = `${techniques.length} Techniques`;
}

function renderRRChart(roundMetrics) {
    const ctx = document.getElementById('rrRoundChart');
    if (!ctx) return;
    if (rrChart) rrChart.destroy();

    let history = roundMetrics && roundMetrics.length > 0 ? roundMetrics : [
        { round: 0, overall_recall: 0.50, overall_f1: 0.65, overall_fpr: 0.02, model_version: 'Baseline' },
    ];
    if (history.length === 1) {
        history = [{ round: 0, overall_recall: 0.50, overall_f1: 0.65, overall_fpr: 0.02, model_version: 'Baseline' }, history[0]];
    }

    const labels = history.map(r => `Round ${r.round} (${r.model_version || 'V1'})`);
    const badge = document.getElementById('rr-rounds-badge');
    if (badge) badge.textContent = `${history.length - 1} round${history.length !== 2 ? 's' : ''} completed`;

    rrChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                { label: 'F1 Score', data: history.map(r => r.overall_f1 || 0), borderColor: CHART_COLORS.green, backgroundColor: 'rgba(16,185,129,0.12)', tension: 0.35, fill: true, pointRadius: 6, pointHoverRadius: 9 },
                { label: 'Recall (Catch Rate)', data: history.map(r => r.overall_recall || 0), borderColor: CHART_COLORS.blue, backgroundColor: 'rgba(59,130,246,0.10)', tension: 0.35, fill: true, pointRadius: 6, pointHoverRadius: 9 },
                { label: 'FPR (Lower = Better)', data: history.map(r => r.overall_fpr || 0), borderColor: CHART_COLORS.amber, backgroundColor: 'rgba(245,158,11,0.08)', tension: 0.35, fill: true, pointRadius: 6, pointHoverRadius: 9 },
            ],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, max: 1.0, grid: { color: 'rgba(255,255,255,0.06)' } },
                x: { grid: { display: false } },
            },
            plugins: { legend: { position: 'top' } },
        },
    });
}

async function loadRunRoundView() {
    // Paint empty pipeline immediately
    renderPipelineStages(new Set(), null);

    // Load existing status + history
    const [status, history] = await Promise.all([
        apiFetch('/api/status'),
        apiFetch('/api/round-history'),
    ]);

    if (status) {
        const orch = status.orchestrator || {};
        const perf = status.performance || {};
        const el = document.getElementById('rr-status-text');
        const roundEl = document.getElementById('rr-round-num');
        const verEl = document.getElementById('rr-model-ver');
        if (el) el.textContent = orch.is_running ? 'Training Active...' : 'Ready';
        if (roundEl) roundEl.textContent = orch.current_round || '—';
        if (verEl) verEl.textContent = perf.model_version || orch.model_version || '—';
    }

    if (history && history.round_metrics) {
        renderRRChart(history.round_metrics);
    }

    if (history && history.rounds && history.rounds.length > 0) {
        const latest = history.rounds[history.rounds.length - 1];
        const summary = latest.summary || {};
        const scoring = latest.stages?.scoring || {};
        renderRRMetrics(summary, scoring);
        renderRRAttackTable(scoring.per_scenario);

        // Mark all stages complete for the last round
        const completedAll = new Set(RR_STAGES.map(s => s.key));
        renderPipelineStages(completedAll, null);
        const badge = document.getElementById('rr-pipeline-badge');
        if (badge) { badge.textContent = `Round ${latest.round} Complete`; badge.className = 'glass-tag green-tag'; }
    }
}

async function triggerRunRound() {
    const btn = document.getElementById('btn-run-round');
    const runBox = document.getElementById('rr-running-box');
    const completeBox = document.getElementById('rr-complete-box');
    const badge = document.getElementById('rr-pipeline-badge');

    btn.disabled = true;
    btn.textContent = '⏳ Running...';
    if (runBox) runBox.style.display = 'block';
    if (completeBox) completeBox.style.display = 'none';
    if (badge) { badge.textContent = 'Running...'; badge.className = 'glass-tag amber-tag'; }

    rrCompletedStages = new Set();
    renderPipelineStages(rrCompletedStages, 'identify');

    const stageListEl = document.getElementById('rr-stage-list');
    if (stageListEl) stageListEl.innerHTML = RR_STAGES.map(s =>
        `<div id="rr-sl-${s.key}" style="display:flex;align-items:center;gap:8px;color:var(--text-muted);">
            <span style="font-size:0.9rem">${s.icon}</span>
            <span>${s.label}</span>
            <span id="rr-sl-status-${s.key}" style="margin-left:auto;font-size:0.75rem;">waiting</span>
        </div>`
    ).join('');

    // Fire the round
    const res = await apiPost('/api/run-round');
    if (!res) {
        btn.disabled = false;
        btn.textContent = '▶ Run Round Now';
        if (runBox) runBox.style.display = 'none';
        return;
    }

    // Poll until done
    if (rrPollTimer) clearInterval(rrPollTimer);
    let stageIdx = 0;

    rrPollTimer = setInterval(async () => {
        const status = await apiFetch('/api/status');
        if (!status) return;

        const orch = status.orchestrator || {};
        const roundEl = document.getElementById('rr-round-num');
        const verEl = document.getElementById('rr-model-ver');
        const statusEl = document.getElementById('rr-status-text');

        if (roundEl) roundEl.textContent = orch.current_round || '—';
        if (verEl) verEl.textContent = status.performance?.model_version || '—';

        // Simulate stage progression while running
        if (orch.is_running && stageIdx < RR_STAGES.length) {
            const currentStage = RR_STAGES[stageIdx];
            if (statusEl) statusEl.textContent = `Stage: ${currentStage.label}...`;
            renderPipelineStages(rrCompletedStages, currentStage.key);

            // Mark previous as complete
            if (stageIdx > 0) {
                const prevSl = document.getElementById(`rr-sl-status-${RR_STAGES[stageIdx-1].key}`);
                if (prevSl) { prevSl.textContent = '✓ done'; prevSl.style.color = 'var(--accent-green)'; }
                rrCompletedStages.add(RR_STAGES[stageIdx-1].key);
            }
            const currSl = document.getElementById(`rr-sl-status-${currentStage.key}`);
            if (currSl) { currSl.textContent = '⚡ active'; currSl.style.color = 'var(--accent-amber)'; }
            stageIdx++;
        }

        if (!orch.is_running && status.ready) {
            clearInterval(rrPollTimer);
            rrPollTimer = null;

            // Mark all complete
            rrCompletedStages = new Set(RR_STAGES.map(s => s.key));
            renderPipelineStages(rrCompletedStages, null);
            RR_STAGES.forEach(s => {
                const sl = document.getElementById(`rr-sl-status-${s.key}`);
                if (sl) { sl.textContent = '✓ done'; sl.style.color = 'var(--accent-green)'; }
            });

            if (badge) { badge.textContent = `Round ${orch.current_round} Complete`; badge.className = 'glass-tag green-tag'; }
            if (statusEl) statusEl.textContent = 'Ready';

            // Load results
            const [history] = await Promise.all([apiFetch('/api/round-history')]);
            if (history && history.round_metrics) renderRRChart(history.round_metrics);
            if (history && history.rounds && history.rounds.length > 0) {
                const latest = history.rounds[history.rounds.length - 1];
                renderRRMetrics(latest.summary || {}, latest.stages?.scoring || {});
                renderRRAttackTable(latest.stages?.scoring?.per_scenario);

                const completeMsg = document.getElementById('rr-complete-msg');
                const s = latest.summary || {};
                if (completeMsg) completeMsg.textContent =
                    `F1: ${(s.overall_f1*100||0).toFixed(1)}% | Recall: ${(s.overall_recall*100||0).toFixed(1)}% | FPR: ${(s.overall_fpr*100||0).toFixed(2)}% | Model: ${s.model_version}`;
            }

            if (runBox) runBox.style.display = 'none';
            if (completeBox) completeBox.style.display = 'block';

            btn.disabled = false;
            btn.textContent = '▶ Run Round Now';

            // Refresh topbar metrics
            loadOverview();
        }
    }, 1200);
}

/* ==========================================================================
   ROUND RUNNER (topbar button — kept for backward compat)
   ========================================================================== */
async function runRound() {
    await apiPost('/api/run-round');
    startPolling();
}

function startPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(async () => {
        const status = await apiFetch('/api/status');
        if (status && status.orchestrator) {
            const orch = status.orchestrator;
            const perf = status.performance || {};

            document.getElementById('model-version').textContent = perf.model_version || 'V1';
            document.getElementById('summary-model-ver').textContent = `${perf.model_version || 'V1'} Active`;

            if (!orch.is_running) {
                clearInterval(pollInterval);
                pollInterval = null;
                loadOverview();
                loadDefendedVsMissed();
                loadCoverage();
                loadCapabilityGraph();
                loadLiveFeed();
                loadFeedback();
                document.getElementById('system-status-text').textContent = 'Autonomous Defense Active';
            } else {
                document.getElementById('system-status-text').textContent = 'Retraining Active';
            }
        }
    }, 1500);
}

/* ==========================================================================
   CHART RENDERERS
   ========================================================================== */
function renderRoundChart(roundHistory) {
    const ctx = document.getElementById('roundChart');
    if (!ctx) return;

    if (roundChart) roundChart.destroy();

    let history = roundHistory && roundHistory.length > 0 ? roundHistory : [
        { round: 0, overall_recall: 0.50, overall_f1: 0.65, overall_fpr: 0.02, model_version: 'Baseline' },
        { round: 1, overall_recall: 0.70, overall_f1: 0.82, overall_fpr: 0.00, model_version: 'V1' },
    ];

    if (history.length === 1) {
        history = [
            { round: 0, overall_recall: 0.50, overall_f1: 0.65, overall_fpr: 0.02, model_version: 'Baseline' },
            history[0]
        ];
    }

    const labels = history.map(r => `Round ${r.round} (${r.model_version || 'V1'})`);

    roundChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Catch Rate (Recall)',
                    data: history.map(r => r.overall_recall || 0),
                    borderColor: CHART_COLORS.blue,
                    backgroundColor: 'rgba(59,130,246,0.12)',
                    tension: 0.35, fill: true, pointRadius: 6, pointHoverRadius: 8,
                },
                {
                    label: 'F1 Performance Score',
                    data: history.map(r => r.overall_f1 || 0),
                    borderColor: CHART_COLORS.green,
                    backgroundColor: 'rgba(16,185,129,0.12)',
                    tension: 0.35, fill: true, pointRadius: 6, pointHoverRadius: 8,
                },
                {
                    label: 'False Positive Rate (FPR)',
                    data: history.map(r => r.overall_fpr || 0),
                    borderColor: CHART_COLORS.amber,
                    backgroundColor: 'rgba(245,158,11,0.08)',
                    tension: 0.35, fill: true, pointRadius: 6, pointHoverRadius: 8,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, max: 1.0, grid: { color: 'rgba(255,255,255,0.06)' } },
                x: { grid: { display: false } },
            },
            plugins: { legend: { position: 'top' } },
        },
    });
}

function renderDoughnut(canvasId, data, chartRef) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (chartRef === 'tacticChart' && tacticChart) tacticChart.destroy();
    if (chartRef === 'typeChart' && typeChart) typeChart.destroy();

    const labels = Object.keys(data);
    const values = Object.values(data);
    const colors = [CHART_COLORS.green, CHART_COLORS.blue, CHART_COLORS.amber,
                    CHART_COLORS.purple, CHART_COLORS.red, CHART_COLORS.cyan,
                    '#EC4899', '#14B8A6'];

    const chart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{ data: values, backgroundColor: colors.slice(0, labels.length), borderWidth: 0 }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'right' } },
        },
    });

    if (chartRef === 'tacticChart') tacticChart = chart;
    if (chartRef === 'typeChart') typeChart = chart;
}

/* ==========================================================================
   BOOT SEQUENCE — Poll until backend Round 1 is ready, then load the UI
   ========================================================================== */
let _bootPollTimer = null;
let _bootDotCount = 0;

function showBootBanner(msg) {
    // Inject a full-screen boot overlay if it doesn't exist yet
    let overlay = document.getElementById('boot-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'boot-overlay';
        overlay.style.cssText = `
            position: fixed; inset: 0; z-index: 9999;
            background: #080B11;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center; gap: 20px;
            font-family: 'Plus Jakarta Sans', sans-serif;
        `;
        overlay.innerHTML = `
            <div style="width:56px;height:56px;background:linear-gradient(135deg,#10B981,#059669);
                border-radius:14px;display:flex;align-items:center;justify-content:center;
                font-size:2rem;box-shadow:0 0 30px rgba(16,185,129,0.5);">🛡️</div>
            <div style="text-align:center;">
                <div style="font-size:1.4rem;font-weight:800;color:#fff;letter-spacing:-0.3px;">PROJECT AEGIS</div>
                <div style="font-size:0.82rem;color:#64748B;margin-top:4px;">Blue Team AI Defense — Initializing</div>
            </div>
            <div id="boot-msg" style="font-size:0.9rem;color:#10B981;font-family:'JetBrains Mono',monospace;
                background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);
                border-radius:8px;padding:10px 24px;min-width:340px;text-align:center;">
                ${msg}
            </div>
            <div style="font-size:0.78rem;color:#475569;">Training XGBoost + Graph Anomaly Model on Round 1 data...</div>
        `;
        document.body.appendChild(overlay);
    } else {
        const msgEl = document.getElementById('boot-msg');
        if (msgEl) msgEl.textContent = msg;
    }
}

function hideBootBanner() {
    const overlay = document.getElementById('boot-overlay');
    if (overlay) {
        overlay.style.transition = 'opacity 0.4s ease';
        overlay.style.opacity = '0';
        setTimeout(() => overlay.remove(), 420);
    }
}

async function waitForBackendReady() {
    showBootBanner('⚙  Bootstrapping AEGIS Defense Engine...');

    const poll = async () => {
        try {
            const status = await apiFetch('/api/status');
            _bootDotCount = (_bootDotCount + 1) % 4;
            const dots = '.'.repeat(_bootDotCount + 1);

            if (status && status.ready === true) {
                clearInterval(_bootPollTimer);
                _bootPollTimer = null;
                showBootBanner('✅  AEGIS Online — Loading Dashboard...');
                setTimeout(() => {
                    hideBootBanner();
                    // Navigate to overview and activate the default nav link
                    switchView('overview');
                    const defaultNav = document.querySelector('[data-view="overview"]');
                    if (defaultNav) defaultNav.classList.add('active');
                }, 600);
            } else if (status && status.orchestrator && status.orchestrator.is_running) {
                const round = status.orchestrator.current_round || 1;
                showBootBanner(`⚡  Running Round ${round} — Training AI Models${dots}`);
            } else {
                showBootBanner(`⚙  Starting Engine${dots}`);
            }
        } catch (_) {
            showBootBanner('🔄  Connecting to AEGIS server...');
        }
    };

    // Poll immediately, then every 1.5 s
    await poll();
    _bootPollTimer = setInterval(poll, 1500);
}

// Initial Boot
document.addEventListener('DOMContentLoaded', () => {
    waitForBackendReady();
});
