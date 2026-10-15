// Load once, after the stock document; all provider credentials stay server-side.
(() => {
  const preview = document.getElementById('news-preview');
  if (!preview) return;
  const content = preview.querySelector('[data-news-content]');
  const fullLink = preview.querySelector('[data-news-link]');
  const query = new URLSearchParams({ ticker: preview.dataset.ticker });
  const selection = new URLSearchParams(window.location.search).get('listing');
  if (selection) query.set('listing', selection);
  fullLink.href = '/news?' + query;
  const node = (tag, text, classes) => {
    const el = document.createElement(tag);
    if (text) el.textContent = text;
    if (classes) el.className = classes;
    return el;
  };
  const showMessage = text => content.replaceChildren(node('p', text, 'py-3 text-sm text-zinc-500 dark:text-zinc-400'));
  fetch('/api/news?' + query, { credentials: 'same-origin' })
    .then(response => response.json())
    .then(feed => {
      const articles = Array.isArray(feed.articles) ? feed.articles.slice(0, 3) : [];
      if (!articles.length) {
        showMessage(feed.message || feed.detail || 'No recent headlines available.');
        return;
      }
      content.replaceChildren();
      if (feed.message) content.append(node('p', feed.message, 'py-2 text-sm text-zinc-500 dark:text-zinc-400'));
      const table = preview.dataset.excel === 'true';
      const list = node(table ? 'table' : 'ol', '', 'w-full text-sm');
      const body = table ? node('tbody') : list;
      if (table) list.append(body);
      articles.forEach(article => {
        let url;
        try { url = new URL(article.url); } catch (_) { return; }
        if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password) return;
        const row = node(table ? 'tr' : 'li', '', 'border-b border-zinc-200 dark:border-zinc-800 last:border-b-0');
        const cell = node(table ? 'td' : 'div', '', 'py-3');
        const meta = node('div', '', 'text-xs text-zinc-500 dark:text-zinc-400 mb-1 flex flex-wrap gap-x-2');
        meta.append(node('span', article.source));
        const time = node('time', article.published_at.replace('T', ' ').replace('Z', '') + ' UTC', 'font-mono');
        time.dateTime = article.published_at;
        meta.append(time);
        const link = node('a', article.title, 'font-semibold hover:text-amber-600 hover:underline break-words');
        link.href = url.href;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        cell.append(meta, link);
        row.append(cell);
        body.append(row);
      });
      content.append(list);
      if (feed.fetched_at) content.append(node('p',
        (feed.stale ? 'Saved headlines · ' : '') + 'Last updated ' + feed.fetched_at + ' (UTC)',
        'pt-3 text-xs text-zinc-500 dark:text-zinc-400'));
    })
    .catch(() => showMessage('News could not be loaded. Open the News page to try again.'));
})();
