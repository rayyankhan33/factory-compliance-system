/**
 * Factory Compliance Dashboard — Client-side Logic
 *
 * Handles: tab switching, video source selection, MJPEG stream display,
 * SSE live violations, history table with filters, export, and alert banners.
 */

// ── State ───────────────────────────────────────────────────────
let sseConnection = null;
let statusPollInterval = null;
let historyRefreshInterval = null;
let liveViolations = [];
let alertBannerTimeout = null;
const MAX_LIVE_VIOLATIONS = 150;

// ── DOM Helpers ─────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ── Tab Switching ───────────────────────────────────────────────
function initTabs() {
    $$('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.tab;
            $$('.tab-btn').forEach(b => b.classList.remove('active'));
            $$('.tab-content').forEach(t => t.classList.remove('active'));
            btn.classList.add('active');
            $(`#tab-${target}`).classList.add('active');

            // Load data for the activated tab
            if (target === 'timeline') loadTimeline();
            if (target === 'history') loadHistory();
        });
    });
}

// ── Video Source Loading ────────────────────────────────────────
async function loadVideoList() {
    try {
        const res = await fetch('/api/videos');
        const data = await res.json();
        const select = $('#video-select');
        select.innerHTML = '<option value="">— Select a video clip —</option>';

        // Group by category
        const categories = {};
        data.videos.forEach(v => {
            if (!categories[v.category]) categories[v.category] = [];
            categories[v.category].push(v);
        });

        Object.keys(categories).sort().forEach(cat => {
            const group = document.createElement('optgroup');
            group.label = cat;
            categories[cat].forEach(v => {
                const opt = document.createElement('option');
                opt.value = v.path;
                opt.textContent = `${v.filename} (${v.size_mb} MB)`;
                group.appendChild(opt);
            });
            select.appendChild(group);
        });

        $('#total-clips').textContent = data.total;
    } catch (e) {
        console.error('Failed to load video list:', e);
    }
}

// ── Stream Control ──────────────────────────────────────────────
async function startStream() {
    const selectSource = $('#video-select').value;
    const customSource = $('#custom-source').value.trim();
    const source = customSource || selectSource;

    if (!source) {
        showNotification('Please select a video clip or enter a source URL.', 'warning');
        return;
    }

    // Disable controls
    $('#btn-start').disabled = true;
    $('#btn-stop').disabled = false;

    try {
        const res = await fetch(`/api/stream/start?source=${encodeURIComponent(source)}`);
        const data = await res.json();

        // Show the MJPEG stream
        const feedImg = $('#video-feed');
        const placeholder = $('#video-placeholder');
        feedImg.src = '/api/stream/feed?' + Date.now();  // cache bust
        feedImg.classList.remove('hidden');
        placeholder.classList.add('hidden');

        // Update status indicator
        updateStatusDot('active');

        // Start SSE for live violations
        connectSSE();

        // Start polling status
        startStatusPolling();

        // Clear previous violations
        liveViolations = [];
        renderLiveViolations();

    } catch (e) {
        console.error('Failed to start stream:', e);
        showNotification('Failed to start stream: ' + e.message, 'error');
        $('#btn-start').disabled = false;
        $('#btn-stop').disabled = true;
    }
}

async function stopStream() {
    try {
        await fetch('/api/stream/stop');
    } catch (e) {
        console.error('Failed to stop stream:', e);
    }

    // Disconnect SSE
    if (sseConnection) {
        sseConnection.close();
        sseConnection = null;
    }

    // Stop status polling
    if (statusPollInterval) {
        clearInterval(statusPollInterval);
        statusPollInterval = null;
    }

    // Hide stream
    $('#video-feed').classList.add('hidden');
    $('#video-placeholder').classList.remove('hidden');

    // Update controls
    $('#btn-start').disabled = false;
    $('#btn-stop').disabled = true;
    updateStatusDot('idle');
    updateStats({ fps: 0, frames_processed: 0, violations_detected: 0, status: 'idle' });
}

// ── SSE Connection ──────────────────────────────────────────────
function connectSSE() {
    if (sseConnection) {
        sseConnection.close();
    }

    sseConnection = new EventSource('/api/violations/live');

    sseConnection.onmessage = (event) => {
        try {
            const violation = JSON.parse(event.data);
            handleNewViolation(violation);
        } catch (e) {
            console.warn('Failed to parse SSE event:', e);
        }
    };

    sseConnection.onerror = () => {
        console.warn('SSE connection error, will auto-reconnect...');
    };
}

function handleNewViolation(violation) {
    // Add to the top of live violations list
    liveViolations.unshift(violation);
    if (liveViolations.length > MAX_LIVE_VIOLATIONS) {
        liveViolations = liveViolations.slice(0, MAX_LIVE_VIOLATIONS);
    }

    renderLiveViolations();
    updateViolationCount();

    // Show alert banner for HIGH/CRITICAL
    if (violation.severity === 'HIGH' || violation.severity === 'CRITICAL') {
        showAlertBanner(violation);
    }
}

// ── Live Violations Rendering ───────────────────────────────────
function renderLiveViolations() {
    const container = $('#violations-list');
    const empty = $('#violations-empty');
    const countEl = $('#live-violation-count');

    countEl.textContent = liveViolations.length;

    if (liveViolations.length === 0) {
        container.innerHTML = '';
        empty.classList.remove('hidden');
        return;
    }

    empty.classList.add('hidden');

    // Only render the first 80 for performance
    const toRender = liveViolations.slice(0, 80);
    container.innerHTML = toRender.map(v => `
        <div class="violation-card sev-${v.severity}">
            <div class="v-header">
                <span class="v-behavior">${escapeHtml(v.behavior_class || 'Unknown')}</span>
                <span class="v-time">${formatTimestamp(v.timestamp)}</span>
            </div>
            <div class="v-desc">${escapeHtml(v.event_description || '')}</div>
            <div class="v-meta">
                <span class="severity-badge sev-${v.severity}">${v.severity}</span>
                <span class="zone-badge">${escapeHtml(v.zone || 'N/A')}</span>
                <span class="meta-tag">${escapeHtml(v.policy_rule_ref || '')}</span>
            </div>
        </div>
    `).join('');
}

// ── Alert Banner ────────────────────────────────────────────────
function showAlertBanner(violation) {
    // Remove existing banner
    const existing = $('.alert-banner');
    if (existing) existing.remove();

    const banner = document.createElement('div');
    banner.className = `alert-banner sev-${violation.severity}`;
    banner.innerHTML = `
        <span>⚠️ <strong>${violation.severity}</strong> — ${escapeHtml(violation.behavior_class || 'Violation')}
        detected in ${escapeHtml(violation.zone || 'Unknown Zone')}
        (${escapeHtml(violation.clip_id || '')})</span>
        <button class="dismiss-btn" onclick="this.parentElement.remove()">Dismiss</button>
    `;
    document.body.prepend(banner);

    // Auto-dismiss after 8 seconds
    if (alertBannerTimeout) clearTimeout(alertBannerTimeout);
    alertBannerTimeout = setTimeout(() => {
        if (banner.parentElement) banner.remove();
    }, 8000);
}

// ── Status Polling ──────────────────────────────────────────────
function startStatusPolling() {
    if (statusPollInterval) clearInterval(statusPollInterval);
    statusPollInterval = setInterval(async () => {
        try {
            const res = await fetch('/api/stream/status');
            const stats = await res.json();
            updateStats(stats);

            if (stats.status === 'finished' || stats.status === 'idle') {
                updateStatusDot('idle');
            } else if (stats.status === 'error') {
                updateStatusDot('error');
            }
        } catch (e) {
            console.warn('Status poll failed:', e);
        }
    }, 1000);
}

function updateStats(stats) {
    $('#stat-fps').textContent = (stats.fps || 0).toFixed(1);
    $('#stat-frames').textContent = stats.frames_processed || 0;
    $('#stat-violations').textContent = stats.violations_detected || 0;
    $('#stat-status').textContent = stats.status || 'idle';
}

function updateStatusDot(state) {
    const dot = $('#status-dot');
    dot.classList.remove('active', 'error');
    if (state === 'active') dot.classList.add('active');
    if (state === 'error') dot.classList.add('error');
    $('#status-text').textContent = state === 'active' ? 'Processing' : state === 'error' ? 'Error' : 'Idle';
}

function updateViolationCount() {
    const badge = $('#alert-count-badge');
    const count = liveViolations.filter(v => v.severity === 'HIGH' || v.severity === 'CRITICAL').length;
    if (count > 0) {
        badge.textContent = count;
        badge.classList.remove('hidden');
    } else {
        badge.classList.add('hidden');
    }
}

// ── Timeline (Tab B) ────────────────────────────────────────────
async function loadTimeline() {
    try {
        const res = await fetch('/api/violations/history?limit=200');
        const data = await res.json();
        const container = $('#timeline-list');

        if (!data.events || data.events.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">📭</div>
                    <p>No events logged yet. Start processing a video clip to see events appear here.</p>
                </div>`;
            return;
        }

        container.innerHTML = data.events.map(e => `
            <div class="timeline-card sev-${e.severity}">
                <div class="tc-header">
                    <span class="tc-title">${escapeHtml(e.behavior_class)}</span>
                    <span class="tc-time">${formatTimestamp(e.timestamp)}</span>
                </div>
                <div class="tc-body">${escapeHtml(e.event_description)}</div>
                <div class="tc-meta">
                    <span class="severity-badge sev-${e.severity}">${e.severity}</span>
                    <span class="meta-tag">📹 ${escapeHtml(e.clip_id)}</span>
                    <span class="meta-tag">📍 ${escapeHtml(e.zone)}</span>
                    <span class="meta-tag">📋 ${escapeHtml(e.policy_rule_ref)}</span>
                    <span class="meta-tag">${escapeHtml(e.escalation_action)}</span>
                </div>
            </div>
        `).join('');
    } catch (e) {
        console.error('Failed to load timeline:', e);
    }
}

let timelineAutoRefresh = false;
function toggleTimelineRefresh() {
    timelineAutoRefresh = !timelineAutoRefresh;
    const btn = $('#btn-timeline-refresh');
    if (timelineAutoRefresh) {
        btn.textContent = '⏸ Stop Auto-refresh';
        btn.classList.add('btn-danger');
        btn.classList.remove('btn-secondary');
        historyRefreshInterval = setInterval(loadTimeline, 5000);
    } else {
        btn.textContent = '🔄 Auto-refresh (5s)';
        btn.classList.remove('btn-danger');
        btn.classList.add('btn-secondary');
        if (historyRefreshInterval) {
            clearInterval(historyRefreshInterval);
            historyRefreshInterval = null;
        }
    }
}

// ── History Table (Tab C) ───────────────────────────────────────
async function loadHistory() {
    const severity = $('#filter-severity').value;
    const classId = $('#filter-class').value;
    const startDate = $('#filter-start-date').value;
    const endDate = $('#filter-end-date').value;

    const params = new URLSearchParams();
    if (severity) params.set('severity', severity);
    if (classId) params.set('class_id', classId);
    if (startDate) params.set('start_date', startDate + 'T00:00:00Z');
    if (endDate) params.set('end_date', endDate + 'T23:59:59Z');
    params.set('limit', '1000');

    try {
        const res = await fetch(`/api/violations/history?${params}`);
        const data = await res.json();
        renderHistoryTable(data.events);
        $('#history-count').textContent = `${data.total} records`;
    } catch (e) {
        console.error('Failed to load history:', e);
    }
}

function renderHistoryTable(events) {
    const tbody = $('#history-tbody');
    const empty = $('#history-empty');

    if (!events || events.length === 0) {
        tbody.innerHTML = '';
        empty.classList.remove('hidden');
        return;
    }

    empty.classList.add('hidden');
    tbody.innerHTML = events.map(e => `
        <tr>
            <td>${formatTimestamp(e.timestamp)}</td>
            <td class="truncate">${escapeHtml(e.clip_id)}</td>
            <td>${escapeHtml(e.zone)}</td>
            <td>${escapeHtml(e.behavior_class)}</td>
            <td>${escapeHtml(e.policy_rule_ref)}</td>
            <td><span class="severity-badge sev-${e.severity}">${e.severity}</span></td>
            <td class="truncate">${escapeHtml(e.event_description)}</td>
            <td>${escapeHtml(e.escalation_action)}</td>
        </tr>
    `).join('');
}

// ── Export ───────────────────────────────────────────────────────
function exportData(format) {
    const severity = $('#filter-severity').value;
    const classId = $('#filter-class').value;
    const startDate = $('#filter-start-date').value;
    const endDate = $('#filter-end-date').value;

    const params = new URLSearchParams();
    if (severity) params.set('severity', severity);
    if (classId) params.set('class_id', classId);
    if (startDate) params.set('start_date', startDate + 'T00:00:00Z');
    if (endDate) params.set('end_date', endDate + 'T23:59:59Z');

    window.location.href = `/api/export/${format}?${params}`;
}

// ── Notification ────────────────────────────────────────────────
function showNotification(message, type = 'info') {
    const existing = $('.notification-toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = `notification-toast notification-${type}`;
    toast.style.cssText = `
        position: fixed; bottom: 24px; right: 24px; z-index: 300;
        padding: 12px 20px; border-radius: 10px; font-size: 0.88rem;
        font-weight: 500; color: white; animation: slideIn 0.3s ease;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    `;

    if (type === 'warning') toast.style.background = 'linear-gradient(135deg, #f59e0b, #d97706)';
    else if (type === 'error') toast.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
    else toast.style.background = 'linear-gradient(135deg, #4f8aff, #3b6de0)';

    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// ── Utilities ───────────────────────────────────────────────────
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatTimestamp(ts) {
    if (!ts) return 'N/A';
    try {
        const d = new Date(ts);
        return d.toLocaleString('en-US', {
            month: 'short', day: 'numeric',
            hour: '2-digit', minute: '2-digit', second: '2-digit'
        });
    } catch {
        return ts;
    }
}

// ── Initialization ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    loadVideoList();

    // Wire up buttons
    $('#btn-start').addEventListener('click', startStream);
    $('#btn-stop').addEventListener('click', stopStream);
    $('#btn-timeline-refresh').addEventListener('click', toggleTimelineRefresh);
    $('#btn-apply-filters').addEventListener('click', loadHistory);
    $('#btn-export-csv').addEventListener('click', () => exportData('csv'));
    $('#btn-export-json').addEventListener('click', () => exportData('json'));

    // Load initial history
    loadHistory();
});
