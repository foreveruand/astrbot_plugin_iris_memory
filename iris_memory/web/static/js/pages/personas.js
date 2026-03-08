/**
 * 用户画像页面
 */
import { api } from '../api/client.js';
import { esc } from '../utils/escape.js';
import { fmtTime } from '../utils/format.js';
import { toast } from '../components/toast.js';
import { showConfirm, closeModal, showDetailModal } from '../components/modal.js';
import { renderPagination } from '../components/pagination.js';

const state = { page: 1, pageSize: 12, total: 0, loaded: false };

export function getState() { return state; }

export async function searchPersonas() {
  state.loaded = true;
  const q = val('persona-query');

  const res = await api.get('/personas', { query: q, page: state.page, page_size: state.pageSize });
  if (!res || res.status !== 'ok') return;

  const d = res.data;
  state.total = d.total || 0;
  el('persona-total-info').textContent = `共 ${state.total} 位用户`;

  const items = d.items || [];
  const container = el('personas-container');

  if (!items.length) {
    container.innerHTML = '<div style="text-align:center;color:var(--text2);padding:40px">暂无用户画像数据</div>';
    el('persona-pagination').innerHTML = '';
    return;
  }

  container.innerHTML = items.map(p => {
    const traits = [];
    if (p.interests) {
      const entries = typeof p.interests === 'object' ? Object.keys(p.interests).slice(0, 3) : [];
      traits.push(...entries);
    }
    if (p.work_style) traits.push(p.work_style);
    if (p.lifestyle) traits.push(p.lifestyle);

    return `<div class="persona-card" onclick="window.__persona.showDetail('${esc(p.user_id)}')">
      <div class="persona-header">
        <span class="persona-uid">${esc(p.user_id)}</span>
        <span class="persona-meta">v${p.version ?? 1} | 更新 ${p.update_count ?? 0} 次</span>
      </div>
      ${renderMiniPersonality(p.personality)}
      <div class="persona-traits">${traits.map(t => `<span class="persona-trait">${esc(t)}</span>`).join('')}</div>
    </div>`;
  }).join('');

  renderPagination({
    page: state.page, pageSize: state.pageSize, total: state.total,
    onChange: p => { state.page = p; searchPersonas(); },
    container: el('persona-pagination'),
  });
}

export function loadPersonas() { searchPersonas(); }

export function resetPersonaFilters() {
  el('persona-query').value = '';
  state.page = 1;
  searchPersonas();
}

export function changePageSize(v) { state.pageSize = Number(v); state.page = 1; searchPersonas(); }

export async function showDetail(userId) {
  const res = await api.get(`/personas/${encodeURIComponent(userId)}`);
  if (!res || res.status !== 'ok') { toast.err('无法加载画像详情'); return; }
  const p = res.data;

  const html = `
    <h3>◉ ${esc(userId)} 的用户画像</h3>
    ${renderPersonalitySection(p.personality)}
    ${renderCommunicationSection(p.communication_style)}
    ${renderInterestsSection(p.interests)}
    ${renderRelationshipSection(p.relationship)}
    ${renderWorkLifeSection(p)}
    ${renderEmotionSection(p.emotion)}
    ${renderMetaSection(p)}
    <div class="modal-actions">
      <button class="btn btn-outline" onclick="window.__persona.closeDetail()">关闭</button>
      <button class="btn btn-danger" onclick="window.__persona.deletePersona('${esc(userId)}')">删除画像</button>
    </div>`;
  showDetailModal('persona-detail-modal', html);
}

export function closeDetail() { closeModal('persona-detail-modal'); }

export function deletePersona(uid) {
  showConfirm('删除画像', `确定要删除 ${uid} 的用户画像吗？`, async () => {
    const res = await api.del(`/personas/${encodeURIComponent(uid)}`);
    if (res?.status === 'ok') { toast.ok('已删除'); closeDetail(); searchPersonas(); }
    else toast.err(res?.message || '删除失败');
  });
}

// ── 渲染组件 ──

function renderMiniPersonality(p) {
  if (!p) return '';
  const bars = [
    ['O', p.openness], ['C', p.conscientiousness], ['E', p.extraversion],
    ['A', p.agreeableness], ['N', p.neuroticism],
  ].filter(([, v]) => v != null);
  if (!bars.length) return '';
  return `<div style="display:flex;gap:4px;margin:6px 0">${bars.map(([l, v]) =>
    `<div style="flex:1;text-align:center"><div style="font-size:10px;color:var(--text2)">${l}</div>` +
    `<div class="persona-bar"><div class="persona-bar-fill" style="width:${(v * 100).toFixed(0)}%;background:var(--accent)"></div></div></div>`
  ).join('')}</div>`;
}

function renderPersonalitySection(p) {
  if (!p) return '';
  const items = [
    ['开放性', p.openness], ['尽责性', p.conscientiousness], ['外向性', p.extraversion],
    ['宜人性', p.agreeableness], ['神经质', p.neuroticism],
  ];
  return `<div class="card"><div class="card-title">🧠 大五人格</div>${items.map(([label, val]) =>
    val != null ? renderPersonalityBar(label, val) : ''
  ).join('')}</div>`;
}

function renderPersonalityBar(label, value) {
  const pct = (value * 100).toFixed(0);
  const color = value > 0.7 ? 'var(--success)' : value < 0.3 ? 'var(--danger)' : 'var(--accent)';
  return `<div style="margin-bottom:8px"><div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:2px">
    <span>${esc(label)}</span><span style="color:var(--text2)">${pct}%</span></div>
    <div class="persona-bar"><div class="persona-bar-fill" style="width:${pct}%;background:${color}"></div></div></div>`;
}

function renderCommunicationSection(cs) {
  if (!cs) return '';
  const items = [['正式度', cs.formality], ['直接性', cs.directness], ['幽默感', cs.humor], ['同理心', cs.empathy]];
  return `<div class="card"><div class="card-title">💬 沟通风格</div>${items.map(([l, v]) =>
    v != null ? renderPersonalityBar(l, v) : ''
  ).join('')}</div>`;
}

function renderInterestsSection(interests) {
  if (!interests || !Object.keys(interests).length) return '';
  return `<div class="card"><div class="card-title">🎯 兴趣</div>
    <div style="display:flex;flex-wrap:wrap;gap:6px">${Object.entries(interests).map(([k, v]) =>
      `<span class="badge" style="padding:4px 10px">${esc(k)}${v ? ` (${esc(v)})` : ''}</span>`
    ).join('')}</div></div>`;
}

function renderRelationshipSection(r) {
  if (!r) return '';
  const items = [];
  if (r.trust != null) items.push(['信任度', r.trust]);
  if (r.intimacy != null) items.push(['亲密度', r.intimacy]);
  if (r.social_style) items.push(['社交风格', null, r.social_style]);
  if (r.emotional_baseline) items.push(['情绪基线', null, r.emotional_baseline]);
  if (!items.length) return '';
  return `<div class="card"><div class="card-title">🤝 关系</div>${items.map(([l, v, txt]) =>
    v != null ? renderPersonalityBar(l, v) : `<div style="margin-bottom:6px"><span style="font-size:12px;color:var(--text2)">${esc(l)}: </span>${esc(txt)}</div>`
  ).join('')}</div>`;
}

function renderWorkLifeSection(p) {
  const items = [];
  if (p.work_style) items.push(`<div><strong>工作风格: </strong>${esc(p.work_style)}</div>`);
  if (p.lifestyle) items.push(`<div><strong>生活方式: </strong>${esc(p.lifestyle)}</div>`);
  if (p.goals?.length) items.push(`<div><strong>目标: </strong>${p.goals.map(g => esc(g)).join(', ')}</div>`);
  if (p.habits?.length) items.push(`<div><strong>习惯: </strong>${p.habits.map(h => esc(h)).join(', ')}</div>`);
  if (!items.length) return '';
  return `<div class="card"><div class="card-title">🏠 工作与生活</div>${items.join('')}</div>`;
}

function renderEmotionSection(emotion) {
  if (!emotion) return '';
  return `<div class="emotion-card">
    <div class="card-title">🎭 情绪状态</div>
    <div class="emotion-primary">${esc(emotion.primary_emotion || '-')}</div>
    <div style="font-size:13px;color:var(--text2)">强度: ${emotion.intensity != null ? (emotion.intensity * 100).toFixed(0) + '%' : '-'}</div>
    <div class="emotion-intensity"><div class="emotion-intensity-fill" style="width:${(emotion.intensity || 0) * 100}%;background:var(--accent)"></div></div>
    ${emotion.trajectory ? `<div style="font-size:12px;color:var(--text2)">趋势: ${esc(emotion.trajectory)}</div>` : ''}
    ${emotion.volatility != null ? `<div style="font-size:12px;color:var(--text2)">波动性: ${(emotion.volatility * 100).toFixed(0)}%</div>` : ''}
  </div>`;
}

function renderMetaSection(p) {
  return `<div class="card"><div class="card-title">📋 元数据</div>
    <div class="detail-grid">
      <div class="detail-item"><div class="detail-label">版本</div><div class="detail-value">${p.version ?? '-'}</div></div>
      <div class="detail-item"><div class="detail-label">更新次数</div><div class="detail-value">${p.update_count ?? '-'}</div></div>
      <div class="detail-item"><div class="detail-label">最后更新</div><div class="detail-value">${esc(fmtTime(p.last_updated))}</div></div>
      <div class="detail-item"><div class="detail-label">主动回复偏好</div><div class="detail-value">${p.proactive_preference ?? '-'}</div></div>
    </div></div>`;
}

// ── 辅助 ──
function el(id) { return document.getElementById(id); }
function val(id) { return (el(id)?.value ?? '').trim(); }

window.__persona = { showDetail, closeDetail, deletePersona };
