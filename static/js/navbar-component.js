(function () {
  'use strict';

  const path = window.location.pathname;
  const links = [
    { href: '/',                  label: 'Home',     icon: '' },
    { href: '/crop-predictor',    label: 'Crop Recommendation', icon: '' },
    { href: '/disease-predictor', label: 'Disease',  icon: '' },
    { href: '/about',             label: 'About',    icon: ''  },
  ];

  const nav = document.createElement('nav');
  nav.setAttribute('role', 'navigation');
  nav.setAttribute('aria-label', 'Main navigation');
  nav.innerHTML = `
    <a class="nav-brand" href="/" aria-label="Kissan Connect home">
      <div class="brand-icon" aria-hidden="true">🌱</div>
      <span>Kissan Connect</span>
    </a>
    <ul class="nav-links" id="navLinks" role="menubar">
      ${links.map(l => `
        <li role="none">
          <a href="${l.href}" role="menuitem"
             class="${path === l.href ? 'active' : ''}"
             ${path === l.href ? 'aria-current="page"' : ''}>
            <span aria-hidden="true">${l.icon}</span>${l.label}
          </a>
        </li>`).join('')}
    </ul>
    <button class="nav-hamburger" id="navHamburger"
            aria-label="Toggle navigation menu"
            aria-expanded="false" aria-controls="navLinks">
      <span></span><span></span><span></span>
    </button>`;

  document.body.insertBefore(nav, document.body.firstChild);

  const hamburger = document.getElementById('navHamburger');
  const navLinks  = document.getElementById('navLinks');

  hamburger.addEventListener('click', () => {
    const open = navLinks.classList.toggle('open');
    hamburger.classList.toggle('open', open);
    hamburger.setAttribute('aria-expanded', String(open));
  });

  // Close on outside click
  document.addEventListener('click', e => {
    if (!nav.contains(e.target)) {
      navLinks.classList.remove('open');
      hamburger.classList.remove('open');
      hamburger.setAttribute('aria-expanded', 'false');
    }
  });

  // Close on Escape
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      navLinks.classList.remove('open');
      hamburger.classList.remove('open');
      hamburger.setAttribute('aria-expanded', 'false');
    }
  });
})();