/* Touch behaviour shared by every page. Dependency-free; loaded `defer` from base.html.
   Contract: markup hooks only (data-tip*, .has-more, #desktop-banner) -- no page-specific code here.
   Limits: popovers are only wired when the primary input cannot hover; desktop keeps CSS hover/focus. */
(function () {
  'use strict';
  var touch = window.matchMedia('(hover: none)');

  /* 1. Tap-to-open definitions. The popover is position:fixed so scroll wrappers
        (overflow-x:auto tables) can't clip it; it closes on scroll so it never detaches from its icon. */
  function resetTip(tip) {
    var pop = tip.querySelector('[role=tooltip]');
    if (pop) pop.style.cssText = '';
    var b = tip.querySelector('[data-tip-toggle]');
    if (b) b.setAttribute('aria-expanded', 'false');
  }
  function closeTips(except) {
    document.querySelectorAll('[data-tip][data-open]').forEach(function (t) {
      if (t === except) return;
      t.removeAttribute('data-open');
      resetTip(t);
    });
  }
  function placeTip(tip) {
    var pop = tip.querySelector('[role=tooltip]');
    var btn = tip.querySelector('[data-tip-toggle]');
    if (!pop || !btn) return;
    var a = btn.getBoundingClientRect();
    var vw = document.documentElement.clientWidth;
    pop.style.position = 'fixed';
    pop.style.maxWidth = (vw - 16) + 'px';
    var w = pop.offsetWidth, h = pop.offsetHeight;
    pop.style.left = Math.min(Math.max(8, a.left), vw - 8 - w) + 'px';
    var below = a.bottom + 6;
    pop.style.top = (below + h > window.innerHeight - 8 ? Math.max(8, a.top - 6 - h) : below) + 'px';
    pop.style.margin = '0';
  }
  document.addEventListener('click', function (e) {
    if (!touch.matches) return;
    var toggle = e.target.closest('[data-tip-toggle]');
    if (toggle) {
      e.preventDefault();                       // the metric() icon is an <a href=/glossary#key>
      var tip = toggle.closest('[data-tip]');
      var open = !tip.hasAttribute('data-open');
      closeTips(open ? tip : null);
      if (open) {
        tip.setAttribute('data-open', '');
        toggle.setAttribute('aria-expanded', 'true');
        placeTip(tip);
      } else {
        tip.removeAttribute('data-open');
        resetTip(tip);
      }
      return;
    }
    if (!e.target.closest('[data-tip] [role=tooltip]')) closeTips(null);
  });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeTips(null); });
  window.addEventListener('scroll', function () { closeTips(null); }, { passive: true, capture: true });

  /* 2. Edge-fade hint on horizontally scrollable tables/tabs. */
  function hint(el) {
    el.classList.toggle('has-more', el.scrollWidth - el.clientWidth - el.scrollLeft > 4);
  }
  var wired = new WeakSet();
  function scan() {
    document.querySelectorAll('.overflow-x-auto, .responsive-table').forEach(function (el) {
      if (!wired.has(el)) {
        wired.add(el);
        el.addEventListener('scroll', function () { hint(el); }, { passive: true });
      }
      hint(el);
    });
  }
  var timer;
  function scanSoon() { clearTimeout(timer); timer = setTimeout(scan, 200); }
  scan();
  window.addEventListener('resize', scanSoon);
  // Pages inject tables after load (options chain, strategies): rescan on DOM changes.
  new MutationObserver(scanSoon).observe(document.body, { childList: true, subtree: true });

  /* 3. Keep the browser UI colour in step with the active theme. */
  var meta = document.querySelector('meta[name=theme-color]');
  function syncTheme() { if (meta) meta.content = getComputedStyle(document.body).backgroundColor; }
  setTimeout(syncTheme, 250);                   // body has a 200ms colour transition
  new MutationObserver(function () { setTimeout(syncTheme, 250); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });

  /* 4. Dismissible "best on desktop" banner (Tier 3 pages). */
  var banner = document.getElementById('desktop-banner');
  if (banner) {
    if (localStorage.getItem('desktop_banner_dismissed')) banner.remove();
    else {
      var x = banner.querySelector('[data-dismiss]');
      if (x) x.addEventListener('click', function () {
        localStorage.setItem('desktop_banner_dismissed', '1');
        banner.remove();
      });
    }
  }
})();
