/**
 * 统计面板页面
 */
import { api } from '../api/client.js';
import { esc } from '../utils/escape.js';
import { toast } from '../components/toast.js';

export async function loadDashboard() {
  const grid = document.getElementById('stats-grid');
  grid.innerHTML = '<div class="stat-card"><div class="spinner"></div></div>';

  const res = await api.get('/dashboard/stats');
  if (!res || res.status !== 'ok') { grid.innerHTML = '<div class="stat-card">加载失败</div>'; return; }

  const d = res.data;
  const sys = d.system || {};
  const mem = d.memories || {};
  const kg = d.knowledge_graph || {};
  const health = d.health || {};

  grid.innerHTML = `
    <div class="stat-card"><div class="stat-value">${esc(mem.total ?? 0)}</div><div class="stat-label">记忆总数</div></div>
    <div class="stat-card"><div class="stat-value">${esc(mem.by_layer?.working ?? 0)}</div><div class="stat-label">工作记忆</div></div>
    <div class="stat-card"><div class="stat-value">${esc(mem.by_layer?.episodic ?? 0)}</div><div class="stat-label">情景记忆</div></div>
    <div class="stat-card"><div class="stat-value">${esc(mem.by_layer?.semantic ?? 0)}</div><div class="stat-label">语义记忆</div></div>
    <div class="stat-card"><div class="stat-value">${esc(sys.total_sessions ?? 0)}</div><div class="stat-label">会话数</div></div>
    <div class="stat-card"><div class="stat-value">${esc(sys.total_personas ?? 0)}</div><div class="stat-label">用户画像</div></div>
    <div class="stat-card"><div class="stat-value">${esc(kg.nodes ?? 0)}</div><div class="stat-label">图谱节点</div></div>
    <div class="stat-card"><div class="stat-value">${esc(kg.edges ?? 0)}</div><div class="stat-label">图谱边</div></div>
  `;

  // 记忆类型分布
  const distEl = document.getElementById('type-distribution');
  if (distEl && mem.by_type) {
    const total = Object.values(mem.by_type).reduce((a, b) => a + b, 0) || 1;
    const labels = { fact: '事实', emotion: '情感', relationship: '关系', interaction: '交互', inferred: '推断' };
    const colors = { fact: 'var(--accent)', emotion: 'var(--danger)', relationship: 'var(--success)', interaction: 'var(--warning)', inferred: '#9d6cf0' };
    distEl.innerHTML = Object.entries(mem.by_type).map(([k, v]) =>
      `<div class="stat-card" style="flex:1;min-width:100px;border-left:3px solid ${colors[k] || 'var(--accent)'}">
        <div class="stat-value" style="font-size:20px;">${v}</div>
        <div class="stat-label">${esc(labels[k] || k)} (${Math.round(v / total * 100)}%)</div>
      </div>`
    ).join('');
  }

  loadTrend();
}

export async function loadTrend() {
  const days = Number(document.getElementById('trend-days')?.value || 30);
  const res = await api.get('/dashboard/trend', { days });
  if (!res || res.status !== 'ok') return;

  const items = res.data || [];
  if (!items.length) return;

  const maxVal = Math.max(...items.map(i => i.count), 1);
  const barsEl = document.getElementById('trend-bars');
  const labelsEl = document.getElementById('trend-labels');

  barsEl.innerHTML = items.map(i => {
    const h = Math.max(2, (i.count / maxVal) * 100);
    return `<div class="chart-bar" style="height:${h}%"><div class="tooltip">${esc(i.date)}: ${i.count}</div></div>`;
  }).join('');

  labelsEl.innerHTML = items.map((i, idx) => {
    // 只显示部分日期标签避免重叠
    const show = items.length <= 14 || idx % Math.ceil(items.length / 10) === 0;
    return `<span>${show ? esc(i.date?.slice(5) || '') : ''}</span>`;
  }).join('');
}
