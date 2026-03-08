/**
 * 分页渲染器
 * @param {Object} opts
 * @param {number} opts.page       当前页
 * @param {number} opts.pageSize   每页数量
 * @param {number} opts.total      总条数
 * @param {Function} opts.onChange  (page) => void
 * @param {HTMLElement} opts.container
 */
export function renderPagination({ page, pageSize, total, onChange, container }) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  if (totalPages <= 1) { container.innerHTML = ''; return; }

  let html = '';

  html += `<button class="btn btn-outline btn-sm" ${page <= 1 ? 'disabled' : ''} data-p="${page - 1}">‹</button>`;

  const range = buildRange(page, totalPages);
  for (const p of range) {
    if (p === '...') {
      html += `<span style="color:var(--text2)">…</span>`;
    } else {
      html += `<button class="btn ${p === page ? 'btn-primary' : 'btn-outline'} btn-sm" data-p="${p}">${p}</button>`;
    }
  }

  html += `<button class="btn btn-outline btn-sm" ${page >= totalPages ? 'disabled' : ''} data-p="${page + 1}">›</button>`;

  container.innerHTML = html;
  container.querySelectorAll('button[data-p]').forEach(btn => {
    btn.addEventListener('click', () => {
      const p = Number(btn.dataset.p);
      if (p >= 1 && p <= totalPages) onChange(p);
    });
  });
}

function buildRange(current, total) {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1);
  const pages = [1];
  let start = Math.max(2, current - 1);
  let end = Math.min(total - 1, current + 1);
  if (current <= 3) { start = 2; end = 4; }
  if (current >= total - 2) { start = total - 3; end = total - 1; }
  if (start > 2) pages.push('...');
  for (let i = start; i <= end; i++) pages.push(i);
  if (end < total - 1) pages.push('...');
  pages.push(total);
  return pages;
}
