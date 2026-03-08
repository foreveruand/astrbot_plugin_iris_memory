/**
 * 系统监控页面
 */
import { api } from '../api/client.js';
import { esc } from '../utils/escape.js';
import { fmtDuration } from '../utils/format.js';

export async function loadSystem() {
  const container = el('system-container');
  container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

  const res = await api.get('/system/overview');
  if (!res || res.status !== 'ok') {
    // fallback: 分别请求
    const [healthRes, storageRes] = await Promise.all([
      api.get('/system/health'),
      api.get('/system/storage'),
    ]);
    renderSystem(healthRes?.data, storageRes?.data);
    return;
  }

  renderSystem(res.data?.health || res.data, res.data?.storage);
}

function renderSystem(health, storage) {
  const container = el('system-container');
  health = health || {};
  storage = storage || {};

  const statusColor = health.status === 'healthy' ? 'var(--success)' :
    health.status === 'degraded' ? 'var(--warning)' : 'var(--danger)';
  const statusText = health.status === 'healthy' ? '健康' :
    health.status === 'degraded' ? '降级' : health.status || '未知';

  const chroma = storage.chroma || storage.memories || {};
  const kg = storage.kg || {};

  container.innerHTML = `
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-value" style="color:${statusColor}">
          <span class="health-dot ${health.status || 'healthy'}" style="margin-right:6px"></span>${esc(statusText)}
        </div>
        <div class="stat-label">系统状态</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${health.uptime_seconds ? fmtDuration(health.uptime_seconds) : '-'}</div>
        <div class="stat-label">运行时间</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" style="color:${health.initialized ? 'var(--success)' : 'var(--danger)'}">${health.initialized ? '是' : '否'}</div>
        <div class="stat-label">已初始化</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">存储概况</div>
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-value">${chroma.total ?? (chroma.working + chroma.episodic + chroma.semantic) ?? '-'}</div>
          <div class="stat-label">记忆总数</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${chroma.working ?? '-'}</div>
          <div class="stat-label">工作记忆</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${chroma.episodic ?? '-'}</div>
          <div class="stat-label">情景记忆</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${chroma.semantic ?? '-'}</div>
          <div class="stat-label">语义记忆</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${kg.nodes ?? '-'}</div>
          <div class="stat-label">KG 节点</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${kg.edges ?? '-'}</div>
          <div class="stat-label">KG 边</div>
        </div>
      </div>
    </div>

    ${renderComponentHealth(health.components)}`;
}

function renderComponentHealth(components) {
  if (!components || !Object.keys(components).length) return '';
  return `<div class="card"><div class="card-title">组件状态</div>
    <div class="table-wrap"><table>
      <thead><tr><th>组件</th><th>状态</th><th>详情</th></tr></thead>
      <tbody>${Object.entries(components).map(([k, v]) => {
        const ok = v.status === 'ok' || v.status === 'healthy';
        return `<tr>
          <td>${esc(k)}</td>
          <td><span class="health-dot ${ok ? 'healthy' : 'unhealthy'}"></span> ${esc(v.status)}</td>
          <td>${esc(v.message || '-')}</td>
        </tr>`;
      }).join('')}</tbody>
    </table></div></div>`;
}

// ── 辅助 ──
function el(id) { return document.getElementById(id); }
