/**
 * XSS 防护工具
 */

const ESCAPE_MAP = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#x27;',
};

const ESCAPE_RE = /[&<>"']/g;

/**
 * HTML 转义
 * @param {*} s
 * @returns {string}
 */
export function esc(s) {
  if (s == null) return '';
  return String(s).replace(ESCAPE_RE, ch => ESCAPE_MAP[ch]);
}
