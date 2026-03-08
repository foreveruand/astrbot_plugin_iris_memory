/**
 * 简单响应式状态管理
 */
export class Store {
  /** @type {Map<string, Set<Function>>} */
  #listeners = new Map();
  /** @type {Object} */
  #data;

  /** @param {Object} initial */
  constructor(initial = {}) {
    this.#data = { ...initial };
  }

  /** @param {string} key */
  get(key) { return this.#data[key]; }

  /** @param {string} key @param {*} value */
  set(key, value) {
    const old = this.#data[key];
    if (old === value) return;
    this.#data[key] = value;
    this.#notify(key, value, old);
  }

  /** 批量更新 @param {Object} obj */
  merge(obj) {
    for (const [k, v] of Object.entries(obj)) this.set(k, v);
  }

  /**
   * 订阅变化
   * @param {string} key  具体 key 或 '*' 监听全部
   * @param {Function} fn
   * @returns {Function} 取消订阅
   */
  on(key, fn) {
    if (!this.#listeners.has(key)) this.#listeners.set(key, new Set());
    this.#listeners.get(key).add(fn);
    return () => this.#listeners.get(key)?.delete(fn);
  }

  #notify(key, val, old) {
    this.#listeners.get(key)?.forEach(fn => fn(val, old));
    this.#listeners.get('*')?.forEach(fn => fn(key, val, old));
  }
}

export const store = new Store({
  currentPage: 'dashboard',
});
