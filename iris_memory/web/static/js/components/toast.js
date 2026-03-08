/**
 * Toast 通知组件
 */

let containerEl = null;

function ensureContainer() {
  if (!containerEl) {
    containerEl = document.getElementById('toast-container');
    if (!containerEl) {
      containerEl = document.createElement('div');
      containerEl.id = 'toast-container';
      containerEl.className = 'toast-container';
      document.body.appendChild(containerEl);
    }
  }
  return containerEl;
}

/**
 * 显示 Toast
 * @param {string} msg
 * @param {'success'|'error'|'info'} type
 * @param {number} duration  毫秒
 */
export function toast(msg, type = 'info', duration = 3000) {
  const c = ensureContainer();
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  c.appendChild(el);
  setTimeout(() => { el.remove(); }, duration);
}

toast.ok = (msg) => toast(msg, 'success');
toast.err = (msg) => toast(msg, 'error');
toast.info = (msg) => toast(msg, 'info');
