/**
 * LLM 监控页面
 */
import { api } from '../api/client.js';
import { esc } from '../utils/escape.js';
import { fmtTime } from '../utils/format.js';
import { toast } from '../components/toast.js';

export async function loadLlm() {
  const container = el('llm-container');
  container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

  const [sumRes, aggRes] = await Promise.all([
    api.get('/llm/summary'),
    api.get('/llm/aggregated'),
  ]);

  if (!sumRes || sumRes.status !== 'ok') {
    container.innerHTML = '<div class="card"><div style="text-align:center;color:var(--text2);padding:20px">LLM 统计不可用</div></div>';
    return;
  }

  const s = sumRes.data || {};
  const agg = aggRes?.data || {};

  container.innerHTML = `
    <div class="stats-grid">
      <div class="stat-card"><div class="stat-value">${s.total_calls ?? 0}</div><div class="stat-label">总调用次数</div></div>
      <div class="stat-card"><div class="stat-value">${s.total_tokens ?? 0}</div><div class="stat-label">总 Token 消耗</div></div>
      <div class="stat-card"><div class="stat-value">${s.success_rate != null ? (s.success_rate * 100).toFixed(1) + '%' : '-'}</div><div class="stat-label">成功率</div></div>
      <div class="stat-card"><div class="stat-value">${s.avg_duration_ms != null ? s.avg_duration_ms.toFixed(0) + 'ms' : '-'}</div><div class="stat-label">平均耗时</div></div>
    </div>
    ${renderProviderStats(agg)}
    ${renderSourceStats(agg)}
    <div class="card">
      <div class="card-title">最近调用</div>
      <div id="llm-recent-container"><div class="loading"><div class="spinner"></div></div></div>
    </div>`;

  loadRecent();
}

function renderProviderStats(agg) {
  const providers = agg.by_provider || agg.calls_by_provider;
  if (!providers || !Object.keys(providers).length) return '';
  return `<div class="card"><div class="card-title">按 Provider 统计</div>
    <div class="table-wrap"><table>
      <thead><tr><th>Provider</th><th>调用次数</th><th>Token</th></tr></thead>
      <tbody>${Object.entries(providers).map(([k, v]) => {
        const tokens = agg.tokens_by_provider?.[k] ?? '-';
        const calls = typeof v === 'object' ? v.calls : v;
        return `<tr><td>${esc(k)}</td><td>${calls}</td><td>${tokens}</td></tr>`;
      }).join('')}</tbody>
    </table></div></div>`;
}

function renderSourceStats(agg) {
  const sources = agg.by_source || agg.calls_by_source;
  if (!sources || !Object.keys(sources).length) return '';
  return `<div class="card"><div class="card-title">按调用来源统计</div>
    <div style="display:flex;flex-wrap:wrap;gap:8px">${Object.entries(sources).map(([k, v]) => {
      const count = typeof v === 'object' ? v.calls : v;
      return `<div class="stat-card" style="min-width:120px"><div class="stat-value" style="font-size:18px">${count}</div><div class="stat-label">${esc(k)}</div></div>`;
    }).join('')}</div></div>`;
}

async function loadRecent() {
  const res = await api.get('/llm/recent', { limit: 30 });
  const container = el('llm-recent-container');
  if (!res || res.status !== 'ok') { container.innerHTML = '加载失败'; return; }

  const items = res.data || [];
  if (!items.length) { container.innerHTML = '<div style="color:var(--text2)">暂无调用记录</div>'; return; }

  container.innerHTML = `<div class="table-wrap"><table>
    <thead><tr><th>时间</th><th>来源</th><th>Provider</th><th>状态</th><th>Token</th><th>耗时</th></tr></thead>
    <tbody>${items.map(i => `<tr>
      <td>${esc(fmtTime(i.timestamp || i.created_at))}</td>
      <td>${esc(i.source || '-')}</td>
      <td>${esc(i.provider_id || '-')}</td>
      <td>${i.success ? '<span style="color:var(--success)">成功</span>' : '<span style="color:var(--danger)">失败</span>'}</td>
      <td>${i.tokens_used ?? '-'}</td>
      <td>${i.duration_ms != null ? i.duration_ms + 'ms' : '-'}</td>
    </tr>`).join('')}</tbody>
  </table></div>`;
}

// ── 辅助 ──
function el(id) { return document.getElementById(id); }
