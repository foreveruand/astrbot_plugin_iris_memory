/**
 * 群冷却管理页面
 */
import { api } from '../api/client.js';
import { esc } from '../utils/escape.js';
import { fmtDuration } from '../utils/format.js';
import { toast } from '../components/toast.js';
import { showConfirm } from '../components/modal.js';

export async function loadCooldown() {
  const container = el('cooldown-container');
  container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

  const res = await api.get('/cooldown');
  if (!res || res.status !== 'ok') { container.innerHTML = '<div class="card">加载失败</div>'; return; }

  const d = res.data;
  if (!d.available) {
    container.innerHTML = '<div class="card"><div style="text-align:center;color:var(--text2);padding:20px">冷却模块未启用</div></div>';
    return;
  }

  const items = d.items || [];

  container.innerHTML = `
    <div class="stats-grid" style="margin-bottom:16px">
      <div class="stat-card"><div class="stat-value">${d.active_count ?? 0}</div><div class="stat-label">活跃冷却</div></div>
      <div class="stat-card"><div class="stat-value">${d.default_duration ?? '-'}</div><div class="stat-label">默认时长(分)</div></div>
    </div>
    <div class="card">
      <div class="card-title">手动激活冷却</div>
      <div class="filter-bar">
        <input type="text" id="cd-group-id" placeholder="群组 ID">
        <input type="number" id="cd-duration" placeholder="时长(分)" value="30" min="1" style="width:100px">
        <input type="text" id="cd-reason" placeholder="原因(可选)">
        <button class="btn btn-warning" onclick="window.__cd.activate()">激活冷却</button>
      </div>
    </div>
    <div class="card">
      <div class="card-title">查询冷却状态</div>
      <div class="filter-bar">
        <input type="text" id="cd-check-group" placeholder="群组 ID" oninput="window.__cd.checkStatus()">
        <span id="cd-check-result" style="font-size:13px;min-width:200px"></span>
      </div>
    </div>
    ${items.length ? `<div class="card"><div class="card-title">当前冷却列表</div>
      <div class="table-wrap"><table>
        <thead><tr><th>群组</th><th>状态</th><th>原因</th><th>剩余时间</th><th>操作</th></tr></thead>
        <tbody>${items.map(i => `<tr>
          <td>${esc(i.group_id)}</td>
          <td><span class="health-dot ${i.active ? 'unhealthy' : 'healthy'}"></span> ${i.active ? '冷却中' : '正常'}</td>
          <td>${esc(i.reason || '-')}</td>
          <td>${i.remaining_seconds ? fmtDuration(i.remaining_seconds) : '-'}</td>
          <td>${i.active ? `<button class="btn btn-outline btn-sm" onclick="window.__cd.deactivate('${esc(i.group_id)}')">取消冷却</button>` : '-'}</td>
        </tr>`).join('')}</tbody>
      </table></div></div>` : ''}`;
}

export async function activate() {
  const gid = val('cd-group-id');
  if (!gid) { toast.err('请输入群组 ID'); return; }
  const duration = Number(val('cd-duration')) || 30;
  const reason = val('cd-reason');

  const res = await api.post(`/cooldown/${encodeURIComponent(gid)}/activate`, {
    duration_minutes: duration, reason: reason || undefined,
  });
  if (res?.status === 'ok') { toast.ok('冷却已激活'); loadCooldown(); }
  else toast.err(res?.message || '激活失败');
}

export async function deactivate(gid) {
  const res = await api.post(`/cooldown/${encodeURIComponent(gid)}/deactivate`);
  if (res?.status === 'ok') { toast.ok('冷却已取消'); loadCooldown(); }
  else toast.err(res?.message || '取消失败');
}

let checkTimer = null;
export async function checkStatus() {
  clearTimeout(checkTimer);
  checkTimer = setTimeout(async () => {
    const gid = val('cd-check-group');
    const resultEl = el('cd-check-result');
    if (!gid) { resultEl.innerHTML = ''; return; }

    const res = await api.get(`/cooldown/${encodeURIComponent(gid)}`);
    if (res?.status === 'ok') {
      const d = res.data;
      resultEl.innerHTML = d.active
        ? `<span style="color:var(--danger)">🔴 冷却中 — 剩余 ${fmtDuration(d.remaining_seconds || 0)}</span>`
        : `<span style="color:var(--success)">🟢 正常</span>`;
    }
  }, 300);
}

// ── 辅助 ──
function el(id) { return document.getElementById(id); }
function val(id) { return (el(id)?.value ?? '').trim(); }

window.__cd = { activate, deactivate, checkStatus };
