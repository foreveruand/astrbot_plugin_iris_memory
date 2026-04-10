/**
 * API 客户端 — 封装所有后端 HTTP 交互
 */

const API_BASE = '/api/v1';
const TOKEN_KEY = 'iris_token';

class ApiClient {
  #token = '';
  authRequired = false;

  constructor() {
    try { this.#token = localStorage.getItem(TOKEN_KEY) || ''; } catch { /* noop */ }
  }

  get hasToken() { return !!this.#token; }

  saveToken(t) {
    this.#token = t || '';
    try { localStorage.setItem(TOKEN_KEY, this.#token); } catch { /* noop */ }
  }

  clearToken() {
    this.#token = '';
    try { localStorage.removeItem(TOKEN_KEY); } catch { /* noop */ }
  }

  /** 检查服务端是否要求认证 */
  async checkAuth() {
    try {
      const r = await fetch(`${API_BASE}/auth/check`);
      const d = await r.json();
      this.authRequired = d.data?.auth_required ?? false;
      return { authRequired: this.authRequired, authenticated: d.data?.authenticated ?? false };
    } catch {
      return { authRequired: false, authenticated: false };
    }
  }

  /** 登录 @returns {{ok:boolean, msg:string}} */
  async login(accessKey) {
    try {
      const r = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ access_key: accessKey }),
      });
      const d = await r.json();
      if (d.status === 'ok' && d.data?.token) {
        this.saveToken(d.data.token);
        return { ok: true, msg: '登录成功' };
      }
      return { ok: false, msg: d.message || '登录失败' };
    } catch (e) {
      return { ok: false, msg: e.message };
    }
  }

  /**
   * 通用请求
   * @param {string} path  - 相对于 /api/v1 的路径
   * @param {RequestInit} opts
   * @returns {Promise<Object|null>}
   */
  async request(path, opts = {}) {
    const headers = { 'Content-Type': 'application/json', ...opts.headers };
    if (this.#token) headers['Authorization'] = `Bearer ${this.#token}`;

    let resp;
    try {
      resp = await fetch(API_BASE + path, { ...opts, headers });
    } catch (e) {
      window.dispatchEvent(new CustomEvent('api:error', { detail: { error: e, path } }));
      return null;
    }

    if (resp.status === 401) {
      this.clearToken();
      window.dispatchEvent(new CustomEvent('auth:required'));
      return null;
    }

    const ct = resp.headers.get('content-type') || '';
    if (ct.includes('application/json')) return resp.json();
    // 非 JSON（文件下载等）直接返回 response
    return resp;
  }

  /** GET with query params */
  get(path, params = {}) {
    const qs = Object.entries(params)
      .filter(([, v]) => v !== undefined && v !== null && v !== '')
      .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
      .join('&');
    return this.request(path + (qs ? `?${qs}` : ''));
  }

  /** POST JSON */
  post(path, body = {}) {
    return this.request(path, { method: 'POST', body: JSON.stringify(body) });
  }

  /** PUT JSON */
  put(path, body = {}) {
    return this.request(path, { method: 'PUT', body: JSON.stringify(body) });
  }

  /** PATCH JSON */
  patch(path, body = {}) {
    return this.request(path, { method: 'PATCH', body: JSON.stringify(body) });
  }

  /** DELETE */
  del(path) {
    return this.request(path, { method: 'DELETE' });
  }

  /**
   * POST raw (for file upload)
   * @param {string} path
   * @param {BodyInit} body
   * @param {string} contentType
   */
  postRaw(path, body, contentType) {
    return this.request(path, {
      method: 'POST',
      headers: { 'Content-Type': contentType },
      body,
    });
  }

  /**
   * GET 并返回 Response 对象（用于文件下载）
   */
  async download(path, params = {}) {
    const qs = Object.entries(params)
      .filter(([, v]) => v !== undefined && v !== null && v !== '')
      .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
      .join('&');
    const url = API_BASE + path + (qs ? `?${qs}` : '');
    const headers = {};
    if (this.#token) headers['Authorization'] = `Bearer ${this.#token}`;
    return fetch(url, { headers });
  }
}

export const api = new ApiClient();
