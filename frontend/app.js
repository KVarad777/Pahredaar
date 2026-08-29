/* ==========================================================================
   PROJECT AEGIS — Dashboard Application Logic
   ========================================================================== */

const API_BASE = '';
let roundChart = null;
let tacticChart = null;
let typeChart = null;
let rewardChart = null;
let perTechniqueChart = null;
let pollInterval = null;

/* ── Chart.js Global Config ── */
Chart.defaults.color = '#A1A1A1';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;

const CHART_COLORS = {
    green: '#3ECF8E',
    blue: '#3B82F6',
    amber: '#F59E0B',
    red: '#EF4444',
    purple: '#8B5CF6',
    cyan: '#06B6D4',
};

/* ==========================================================================
   NAVIGATION
   ========================================================================== */
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
        e.preventDefault();
        const view = item.dataset.view;
        switchView(view);
    });
});

function switchView(viewId) {
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));

    const navEl = document.querySelector(`[data-view="${viewId}"]`);
    const viewEl = document.getElementById(`view-${viewId}`);

    if (navEl) navEl.classList.add('active');
    if (viewEl) viewEl.classList.add('active');

    const titles = {
        overview: 'Overview',
        coverage: 'Coverage Matrix',
        metrics: 'Detection Metrics',
        capability: 'Capability Graph',
        livefeed: 'Live Feed',
        rounds: 'Round Control',
        feedback: 'Feedback & Explain',
    };

    document.getElementById('page-title').textContent = titles[viewId] || viewId;
    document.getElementById('breadcrumb').textContent = `Dashboard / ${titles[viewId] || viewId}`;

    // Load data for the view
    if (viewId === 'coverage') loadCoverage();
    if (viewId === 'metrics') loadMetrics();
    if (viewId === 'capability') loadCapabilityGraph();
    if (viewId === 'livefeed') loadLiveFeed();
    if (viewId === 'rounds') loadRoundHistory();
    if (viewId === 'feedback') loadFeedback();
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

async function apiPost(endpoint) {
    try {
        const res = await fetch(`${API_BASE}${endpoint}`, { method: 'POST' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        console.error(`API POST Error [${endpoint}]:`, err);
        return null;
    }
}

/* ==========================================================================
   OVERVIEW
   ========================================================================== */
async function loadOverview() {
    const [status, metrics, predictions, alerts] = await Promise.all([
        apiFetch('/api/status'),
        apiFetch('/api/metrics/latest'),
        apiFetch('/api/capability-graph/predictions'),
        apiFetch('/api/alerts'),
    ]);

    // Status
    if (status) {
        const orch = status.orchestrator || {};
        document.getElementById('model-version').textContent = orch.model_version || 'V1';
        document.getElementById('current-round').textContent = `Round ${orch.current_round || 0}`;

        const dot = document.getElementById('system-status-dot');
        const statusText = document.getElementById('system-status-text');
        if (orch.is_running) {
            dot.className = 'status-dot running';
            statusText.textContent = 'Running';
        } else if (orch.total_rounds_completed > 0) {
            dot.className = 'status-dot online';
            statusText.textContent = 'Online';
        } else {
            dot.className = 'status-dot';
            statusText.textContent = 'Ready';
        }
    }

    // Stats
    if (metrics && metrics.summary) {
        const s = metrics.summary;
        document.getElementById('stat-recall').textContent = fmtPct(s.overall_recall);
        document.getElementById('stat-f1').textContent = fmtPct(s.overall_f1);
        document.getElementById('stat-fpr').textContent = fmtPct(s.overall_fpr);
    }

    // Scenario count from coverage
    const coverage = await apiFetch('/api/coverage-matrix');
    if (coverage) {
        document.getElementById('stat-scenarios').textContent = coverage.total_scenarios || 0;
    }

    // Predictions
    if (predictions && predictions.predicted_next_attacks) {
        renderPredictions('predictions-list', predictions.predicted_next_attacks);
    }

    // Alerts
    if (alerts && alerts.alerts && alerts.alerts.length > 0) {
        renderAlerts('alerts-list', alerts.alerts.slice(-8).reverse());
    }

    // Round chart
    const roundData = await apiFetch('/api/metrics');
    if (roundData && roundData.round_history && roundData.round_history.length > 0) {
        renderRoundChart(roundData.round_history);
    }
}

/* ==========================================================================
   COVERAGE MATRIX
   ========================================================================== */
async function loadCoverage() {
    const data = await apiFetch('/api/coverage-matrix');
    if (!data || !data.scenarios) return;

    const container = document.getElementById('coverage-table-container');
    const scenarios = data.scenarios;

    let html = `<table>
        <thead><tr>
            <th>Scenario</th>
            <th>F3 Tactic</th>
            <th>F3 Technique</th>
            <th>Manipulation Type</th>
            <th>Fields</th>
            <th>Novelty</th>
            <th>Detection Rate</th>
            <th>Round</th>
        </tr></thead><tbody>`;

    scenarios.forEach(s => {
        const noveltyClass = s.novelty_tag === 'ai_specific' ? 'ai_specific' :
                             s.novelty_tag === 'adversarial_variant' ? 'adversarial_variant' : 'baseline';
        const drClass = s.detection_rate >= 0.7 ? 'low' : s.detection_rate >= 0.3 ? 'medium' : 'high';

        html += `<tr>
            <td><strong>${escHtml(s.scenario_name)}</strong></td>
            <td>${escHtml(s.f3_tactic)}</td>
            <td><span class="technique-tag">${escHtml(s.f3_technique)}</span></td>
            <td>${escHtml(s.manipulation_type)}</td>
            <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis">${Array.isArray(s.fields_manipulated) ? s.fields_manipulated.join(', ') : s.fields_manipulated}</td>
            <td><span class="novelty-tag ${noveltyClass}">${escHtml(s.novelty_tag)}</span></td>
            <td><span class="score-badge ${drClass}">${fmtPct(s.detection_rate)}</span></td>
            <td style="font-family:var(--font-mono)">${s.round_introduced}</td>
        </tr>`;
    });

    html += '</tbody></table>';
    container.innerHTML = html;

    // Tactic chart
    if (data.by_tactic) renderDoughnut('tacticChart', data.by_tactic, 'tacticChart');
    if (data.by_manipulation_type) renderDoughnut('typeChart', data.by_manipulation_type, 'typeChart');
}

/* ==========================================================================
   DETECTION METRICS
   ========================================================================== */
async function loadMetrics() {
    const data = await apiFetch('/api/metrics');
    if (!data || !data.round_history || data.round_history.length === 0) return;

    const latest = data.round_history[data.round_history.length - 1];
    const perScenario = latest.per_scenario || {};

    const container = document.getElementById('metrics-table-container');
    let html = `<table>
        <thead><tr>
            <th>Technique</th>
            <th>Count</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>Detection Rate</th>
            <th>FPR</th>
        </tr></thead><tbody>`;

    Object.entries(perScenario).forEach(([tech, m]) => {
        if (tech === 'Legitimate') return;
        const drClass = (m.detection_rate || 0) >= 0.7 ? 'low' : (m.detection_rate || 0) >= 0.3 ? 'medium' : 'high';
        html += `<tr>
            <td><span class="technique-tag">${escHtml(tech)}</span></td>
            <td style="font-family:var(--font-mono)">${m.count || 0}</td>
            <td style="font-family:var(--font-mono)">${fmtPct(m.precision)}</td>
            <td style="font-family:var(--font-mono)">${fmtPct(m.recall)}</td>
            <td style="font-family:var(--font-mono)">${fmtPct(m.f1)}</td>
            <td><span class="score-badge ${drClass}">${fmtPct(m.detection_rate)}</span></td>
            <td style="font-family:var(--font-mono)">${fmtPct(m.fpr)}</td>
        </tr>`;
    });

    html += '</tbody></table>';
    container.innerHTML = html;

    // Per-technique chart over rounds
    renderPerTechniqueChart(data.round_history);
}

/* ==========================================================================
   CAPABILITY GRAPH
   ========================================================================== */
async function loadCapabilityGraph() {
    const [graphData, predictions] = await Promise.all([
        apiFetch('/api/capability-graph'),
        apiFetch('/api/capability-graph/predictions'),
    ]);

    if (graphData) {
        renderGraphVisualization(graphData);

        // Stats
        const statsEl = document.getElementById('graph-stats');
        statsEl.innerHTML = `
            <div class="graph-stat-item"><span class="gs-label">Total Nodes</span><span class="gs-value">${graphData.total_nodes}</span></div>
            <div class="graph-stat-item"><span class="gs-label">Total Edges</span><span class="gs-value">${graphData.total_edges}</span></div>
            <div class="graph-stat-item"><span class="gs-label">Attack Techniques</span><span class="gs-value">${graphData.nodes.filter(n => n.times_observed > 0).length}</span></div>
            <div class="graph-stat-item"><span class="gs-label">Missed Techniques</span><span class="gs-value">${graphData.nodes.filter(n => n.times_missed > 0).length}</span></div>
        `;
    }

    if (predictions && predictions.predicted_next_attacks) {
        renderPredictions('predictions-detail', predictions.predicted_next_attacks);
    }
}

function renderGraphVisualization(graphData) {
    const canvas = document.getElementById('graphCanvas');
    const ctx = canvas.getContext('2d');

    canvas.width = canvas.parentElement.clientWidth - 32;
    canvas.height = 480;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const nodes = graphData.nodes || [];
    const edges = graphData.edges || [];
    if (nodes.length === 0) return;

    // Layout: force-directed approximation
    const positions = {};
    const cols = Math.ceil(Math.sqrt(nodes.length));
    const cellW = canvas.width / (cols + 1);
    const cellH = canvas.height / (Math.ceil(nodes.length / cols) + 1);

    nodes.forEach((node, i) => {
        const row = Math.floor(i / cols);
        const col = i % cols;
        positions[node.id] = {
            x: cellW * (col + 0.5) + (Math.random() - 0.5) * 30,
            y: cellH * (row + 0.5) + (Math.random() - 0.5) * 20,
        };
    });

    // Draw edges
    edges.forEach(edge => {
        const from = positions[edge.source];
        const to = positions[edge.target];
        if (!from || !to) return;

        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.strokeStyle = `rgba(62, 207, 142, ${Math.min(1, edge.weight)})`;
        ctx.lineWidth = Math.max(0.5, edge.weight * 2.5);
        ctx.stroke();

        // Arrowhead
        const angle = Math.atan2(to.y - from.y, to.x - from.x);
        const headLen = 8;
        const midX = (from.x + to.x) / 2 + (to.x - from.x) * 0.15;
        const midY = (from.y + to.y) / 2 + (to.y - from.y) * 0.15;
        ctx.beginPath();
        ctx.moveTo(midX, midY);
        ctx.lineTo(midX - headLen * Math.cos(angle - Math.PI / 6), midY - headLen * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(midX - headLen * Math.cos(angle + Math.PI / 6), midY - headLen * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fillStyle = `rgba(62, 207, 142, ${Math.min(1, edge.weight)})`;
        ctx.fill();
    });

    // Draw nodes
    nodes.forEach(node => {
        const pos = positions[node.id];
        if (!pos) return;

        const radius = 6 + Math.min(8, (node.times_observed || 0) * 2);
        const isMissed = (node.times_missed || 0) > 0;

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = isMissed ? '#EF4444' : (node.times_observed > 0 ? '#3ECF8E' : '#555');
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.2)';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Label
        ctx.font = '10px Inter, sans-serif';
        ctx.fillStyle = '#A1A1A1';
        ctx.textAlign = 'center';
        const label = node.label.length > 22 ? node.label.substring(0, 20) + '...' : node.label;
        ctx.fillText(label, pos.x, pos.y + radius + 12);
    });
}

/* ==========================================================================
   LIVE FEED
   ========================================================================== */
async function loadLiveFeed() {
    const data = await apiFetch('/api/transactions/live');
    if (!data || !data.transactions || data.transactions.length === 0) return;

    const container = document.getElementById('livefeed-table-container');
    let html = `<table>
        <thead><tr>
            <th>Transaction ID</th>
            <th>Amount</th>
            <th>Channel</th>
            <th>Fraud Score</th>
            <th>Decision</th>
            <th>GBM</th>
            <th>GNN</th>
            <th>LSTM</th>
            <th>Actual</th>
            <th>Vector</th>
        </tr></thead><tbody>`;

    data.transactions.slice().reverse().forEach(t => {
        const scoreClass = t.fraud_score >= 0.85 ? 'high' : t.fraud_score >= 0.6 ? 'medium' : 'low';
        const decClass = t.decision === 'BLOCK' ? 'block' : t.decision === 'STEP_UP' ? 'step-up' : 'allow';
        const sub = t.subsystem_scores || {};

        html += `<tr>
            <td style="font-family:var(--font-mono);font-size:0.72rem">${escHtml(t.transaction_id)}</td>
            <td style="font-family:var(--font-mono)">${t.amount ? t.amount.toFixed(2) : '--'}</td>
            <td>${escHtml(t.channel || '')}</td>
            <td><span class="score-badge ${scoreClass}">${(t.fraud_score || 0).toFixed(3)}</span></td>
            <td><span class="decision-badge ${decClass}">${escHtml(t.decision || '')}</span></td>
            <td style="font-family:var(--font-mono);font-size:0.72rem">${(sub.tabular_gbm || 0).toFixed(3)}</td>
            <td style="font-family:var(--font-mono);font-size:0.72rem">${(sub.graph_gnn || 0).toFixed(3)}</td>
            <td style="font-family:var(--font-mono);font-size:0.72rem">${(sub.sequence_lstm || 0).toFixed(3)}</td>
            <td>${t.is_fraud_actual ? '<span class="score-badge high">FRAUD</span>' : '<span class="score-badge low">LEGIT</span>'}</td>
            <td style="max-width:140px;overflow:hidden;text-overflow:ellipsis">${escHtml(t.fraud_vector || '')}</td>
        </tr>`;
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

/* ==========================================================================
   ROUND CONTROL
   ========================================================================== */
async function runRound() {
    const btn = document.getElementById('btn-run-round');
    const statusEl = document.getElementById('round-status');
    btn.disabled = true;
    statusEl.textContent = 'Starting round...';

    const result = await apiPost('/api/run-round');
    if (result) {
        statusEl.textContent = `${result.message || 'Round started'}`;
        startPolling();
    } else {
        statusEl.textContent = 'Error starting round';
        btn.disabled = false;
    }
}

async function runMultipleRounds(n) {
    const statusEl = document.getElementById('round-status');
    statusEl.textContent = `Starting ${n} rounds...`;

    const result = await apiPost(`/api/run-multiple-rounds?n=${n}`);
    if (result) {
        statusEl.textContent = result.message || 'Rounds started';
        startPolling();
    } else {
        statusEl.textContent = 'Error';
    }
}

function startPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(async () => {
        const status = await apiFetch('/api/status');
        if (status && status.orchestrator) {
            const orch = status.orchestrator;
            const statusEl = document.getElementById('round-status');
            const btn = document.getElementById('btn-run-round');

            statusEl.textContent = `Status: ${orch.status} | Round: ${orch.current_round}`;
            document.getElementById('current-round').textContent = `Round ${orch.current_round}`;
            document.getElementById('model-version').textContent = orch.model_version || 'V1';

            if (!orch.is_running) {
                clearInterval(pollInterval);
                pollInterval = null;
                btn.disabled = false;
                statusEl.textContent = `Completed Round ${orch.current_round}`;

                // Refresh all data
                loadOverview();
                loadRoundHistory();

                const dot = document.getElementById('system-status-dot');
                dot.className = 'status-dot online';
                document.getElementById('system-status-text').textContent = 'Online';
            } else {
                const dot = document.getElementById('system-status-dot');
                dot.className = 'status-dot running';
                document.getElementById('system-status-text').textContent = 'Running';
            }
        }
    }, 2000);
}

async function loadRoundHistory() {
    const data = await apiFetch('/api/round-history');
    if (!data || !data.round_metrics) return;

    const container = document.getElementById('round-history-list');
    if (data.round_metrics.length === 0) {
        container.innerHTML = '<div class="empty-state">No rounds completed</div>';
        return;
    }

    let html = '';
    data.round_metrics.forEach(r => {
        html += `<div class="round-history-item">
            <span class="rh-round">Round ${r.round}</span>
            <div class="rh-metrics">
                <span class="rh-metric">F1: <span style="color:var(--accent-green)">${fmtPct(r.overall_f1)}</span></span>
                <span class="rh-metric">Recall: <span style="color:var(--accent-blue)">${fmtPct(r.overall_recall)}</span></span>
                <span class="rh-metric">FPR: <span style="color:var(--accent-amber)">${fmtPct(r.overall_fpr)}</span></span>
                <span class="rh-metric">Blue: <span style="color:var(--accent-cyan)">${r.blue_reward?.toFixed(3) || '--'}</span></span>
            </div>
        </div>`;
    });

    container.innerHTML = html;

    // Reward chart
    renderRewardChart(data.round_metrics);
}

/* ==========================================================================
   FEEDBACK
   ========================================================================== */
async function loadFeedback() {
    const data = await apiFetch('/api/feedback');
    if (!data) return;

    // Miss explanations
    const fbList = document.getElementById('feedback-list');
    if (data.sample_explanations && data.sample_explanations.length > 0) {
        let html = '';
        data.sample_explanations.forEach(exp => {
            html += `<div class="feedback-item">
                <div class="fb-header">
                    <span class="fb-technique">${escHtml(exp.f3_technique || 'Unknown')}</span>
                    <span class="fb-score">Score: ${(exp.fraud_score || 0).toFixed(3)}</span>
                </div>
                <div class="fb-explanation">${escHtml(exp.explanation || '')}</div>
            </div>`;
        });
        fbList.innerHTML = html;
    }

    // Weakness summary
    const weakEl = document.getElementById('weakness-summary');
    if (data.weakness_summary && Object.keys(data.weakness_summary).length > 0) {
        let html = '';
        Object.entries(data.weakness_summary).forEach(([tech, info]) => {
            html += `<div class="weakness-item">
                <div class="wi-header">
                    <span class="wi-technique">${escHtml(tech)}</span>
                    <span class="wi-count">${info.count || 0} misses | Avg score: ${(info.avg_fraud_score || 0).toFixed(3)}</span>
                </div>
                ${(info.common_reasons || []).map(r => `<div class="wi-reason">${escHtml(r)}</div>`).join('')}
            </div>`;
        });
        weakEl.innerHTML = html;
    }
}

/* ==========================================================================
   CHART RENDERERS
   ========================================================================== */
function renderRoundChart(roundHistory) {
    const ctx = document.getElementById('roundChart');
    if (!ctx) return;

    if (roundChart) roundChart.destroy();

    const labels = roundHistory.map(r => `Round ${r.round}`);

    roundChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Recall',
                    data: roundHistory.map(r => r.overall_recall || 0),
                    borderColor: CHART_COLORS.blue,
                    backgroundColor: 'rgba(59,130,246,0.1)',
                    tension: 0.3, fill: true, pointRadius: 4,
                },
                {
                    label: 'F1 Score',
                    data: roundHistory.map(r => r.overall_f1 || 0),
                    borderColor: CHART_COLORS.green,
                    backgroundColor: 'rgba(62,207,142,0.1)',
                    tension: 0.3, fill: true, pointRadius: 4,
                },
                {
                    label: 'FPR',
                    data: roundHistory.map(r => r.overall_fpr || 0),
                    borderColor: CHART_COLORS.amber,
                    backgroundColor: 'rgba(245,158,11,0.1)',
                    tension: 0.3, fill: true, pointRadius: 4,
                },
            ],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: { y: { beginAtZero: true, max: 1, grid: { color: 'rgba(255,255,255,0.04)' } }, x: { grid: { display: false } } },
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
            responsive: true, maintainAspectRatio: false,
            cutout: '65%',
            plugins: { legend: { position: 'bottom', labels: { padding: 12, font: { size: 10 } } } },
        },
    });

    if (chartRef === 'tacticChart') tacticChart = chart;
    if (chartRef === 'typeChart') typeChart = chart;
}

function renderRewardChart(roundMetrics) {
    const ctx = document.getElementById('rewardChart');
    if (!ctx) return;

    if (rewardChart) rewardChart.destroy();

    rewardChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: roundMetrics.map(r => `Round ${r.round}`),
            datasets: [{
                label: 'Blue Reward',
                data: roundMetrics.map(r => r.blue_reward || 0),
                backgroundColor: roundMetrics.map(r => (r.blue_reward || 0) >= 0.5 ? 'rgba(62,207,142,0.6)' : 'rgba(245,158,11,0.6)'),
                borderRadius: 4,
            }],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: { y: { grid: { color: 'rgba(255,255,255,0.04)' } }, x: { grid: { display: false } } },
            plugins: { legend: { display: false } },
        },
    });
}

function renderPerTechniqueChart(roundHistory) {
    const ctx = document.getElementById('perTechniqueChart');
    if (!ctx) return;

    if (perTechniqueChart) perTechniqueChart.destroy();

    // Collect all techniques across rounds
    const allTechs = new Set();
    roundHistory.forEach(r => {
        Object.keys(r.per_scenario || {}).forEach(t => {
            if (t !== 'Legitimate' && t !== '') allTechs.add(t);
        });
    });

    const techArray = Array.from(allTechs).slice(0, 8);
    const colorArr = [CHART_COLORS.green, CHART_COLORS.blue, CHART_COLORS.amber,
                      CHART_COLORS.red, CHART_COLORS.purple, CHART_COLORS.cyan, '#EC4899', '#14B8A6'];

    const datasets = techArray.map((tech, i) => ({
        label: tech.length > 25 ? tech.substring(0, 23) + '...' : tech,
        data: roundHistory.map(r => {
            const m = (r.per_scenario || {})[tech];
            return m ? (m.detection_rate || m.recall || 0) : null;
        }),
        borderColor: colorArr[i % colorArr.length],
        tension: 0.3,
        pointRadius: 3,
        spanGaps: true,
    }));

    perTechniqueChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: roundHistory.map(r => `Round ${r.round}`),
            datasets,
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: { y: { beginAtZero: true, max: 1, grid: { color: 'rgba(255,255,255,0.04)' } }, x: { grid: { display: false } } },
            plugins: { legend: { position: 'bottom', labels: { font: { size: 10 } } } },
        },
    });
}

/* ==========================================================================
   RENDER HELPERS
   ========================================================================== */
function renderPredictions(containerId, predictions) {
    const el = document.getElementById(containerId);
    if (!predictions || predictions.length === 0) {
        el.innerHTML = '<div class="empty-state">No predictions available</div>';
        return;
    }

    let html = '';
    predictions.forEach(p => {
        html += `<div class="prediction-item">
            <div class="pred-header">
                <span class="pred-name">${escHtml(p.predicted_attack || '')}</span>
                <span class="pred-confidence">${(p.confidence * 100).toFixed(0)}%</span>
            </div>
            <span class="pred-source">From: ${escHtml(p.source_technique || '')} | Evidence: ${p.evidence_count || 0}</span>
        </div>`;
    });
    el.innerHTML = html;
}

function renderAlerts(containerId, alerts) {
    const el = document.getElementById(containerId);
    if (!alerts || alerts.length === 0) return;

    let html = '';
    alerts.forEach(a => {
        const scoreColor = a.fraud_score >= 0.85 ? 'var(--accent-red)' : 'var(--accent-amber)';
        html += `<div class="alert-item">
            <span class="alert-score" style="color:${scoreColor}">${(a.fraud_score || 0).toFixed(3)}</span>
            <div class="alert-details">
                <div class="alert-id">${escHtml(a.transaction_id || '')}</div>
                <div class="alert-vector">${escHtml(a.fraud_vector || a.channel || '')}</div>
            </div>
            <span class="decision-badge ${a.decision === 'BLOCK' ? 'block' : 'step-up'}">${escHtml(a.decision || '')}</span>
        </div>`;
    });
    el.innerHTML = html;
}

/* ── Utilities ── */
function fmtPct(val) {
    if (val === undefined || val === null) return '--';
    return (val * 100).toFixed(1) + '%';
}

function escHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

/* ==========================================================================
   INITIALIZATION
   ========================================================================== */
document.addEventListener('DOMContentLoaded', () => {
    loadOverview();
    // Auto-refresh every 10 seconds
    setInterval(loadOverview, 10000);
});
