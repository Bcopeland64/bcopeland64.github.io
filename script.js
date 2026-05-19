/* ============================================================
   CHAT WIDGET
============================================================ */
const chatFab   = document.getElementById('chat-fab');
const chatPanel = document.getElementById('chat-panel');
const chatClose = document.getElementById('chat-close');
const chatIcon  = document.getElementById('chat-fab-icon');

function openChat() {
  chatPanel.classList.remove('chat-hidden');
  chatFab.classList.add('active');
  chatIcon.className = 'fas fa-times';
}
function closeChat() {
  chatPanel.classList.add('chat-hidden');
  chatFab.classList.remove('active');
  chatIcon.className = 'fas fa-comment-dots';
}

chatFab.addEventListener('click', () =>
  chatPanel.classList.contains('chat-hidden') ? openChat() : closeChat()
);
chatClose.addEventListener('click', closeChat);

/* ============================================================
   NEURAL NETWORK CANVAS ANIMATION
============================================================ */
const canvas = document.getElementById('neural-canvas');
const ctx    = canvas.getContext('2d');

let nodes = [];
const NODE_COUNT = 55;
const LINK_DIST  = 145;
const SPEED      = 0.45;

function resize() {
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
}

class Node {
  constructor() { this.init(); }
  init() {
    this.x  = Math.random() * canvas.width;
    this.y  = Math.random() * canvas.height;
    this.vx = (Math.random() - 0.5) * SPEED;
    this.vy = (Math.random() - 0.5) * SPEED;
    this.r  = Math.random() * 1.8 + 0.8;
    this.a  = Math.random() * 0.45 + 0.15;
  }
  update() {
    this.x += this.vx;
    this.y += this.vy;
    if (this.x < 0 || this.x > canvas.width)  this.vx *= -1;
    if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
    this.x = Math.max(0, Math.min(canvas.width,  this.x));
    this.y = Math.max(0, Math.min(canvas.height, this.y));
  }
  draw() {
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(0,212,255,${this.a})`;
    ctx.fill();
  }
}

function buildNodes() {
  nodes = [];
  for (let i = 0; i < NODE_COUNT; i++) nodes.push(new Node());
}

function drawLinks() {
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const dx = nodes[i].x - nodes[j].x;
      const dy = nodes[i].y - nodes[j].y;
      const d  = Math.sqrt(dx * dx + dy * dy);
      if (d >= LINK_DIST) continue;
      const alpha = (1 - d / LINK_DIST) * 0.28;
      const grad  = ctx.createLinearGradient(nodes[i].x, nodes[i].y, nodes[j].x, nodes[j].y);
      grad.addColorStop(0, `rgba(0,212,255,${alpha})`);
      grad.addColorStop(1, `rgba(124,58,237,${alpha})`);
      ctx.beginPath();
      ctx.moveTo(nodes[i].x, nodes[i].y);
      ctx.lineTo(nodes[j].x, nodes[j].y);
      ctx.strokeStyle = grad;
      ctx.lineWidth = 0.75;
      ctx.stroke();
    }
  }
}

function loop() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  nodes.forEach(n => { n.update(); n.draw(); });
  drawLinks();
  requestAnimationFrame(loop);
}

resize();
buildNodes();
loop();

window.addEventListener('resize', () => { resize(); buildNodes(); });

/* ============================================================
   NAVBAR — SCROLL & ACTIVE LINK
============================================================ */
const navbar = document.getElementById('navbar');

function onScroll() {
  navbar.classList.toggle('scrolled', window.scrollY > 50);
  highlightNav();
}

function highlightNav() {
  const sections  = document.querySelectorAll('section[id]');
  const navLinks  = document.querySelectorAll('.nav-link');
  let current = '';

  sections.forEach(sec => {
    if (window.scrollY >= sec.offsetTop - 110) current = sec.id;
  });

  navLinks.forEach(link => {
    link.classList.toggle('active', link.getAttribute('href') === `#${current}`);
  });
}

window.addEventListener('scroll', onScroll, { passive: true });

/* ============================================================
   MOBILE NAV TOGGLE
============================================================ */
const navToggle = document.getElementById('navToggle');
const navLinks  = document.getElementById('navLinks');

navToggle.addEventListener('click', () => {
  navLinks.classList.toggle('open');
});

document.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', () => navLinks.classList.remove('open'));
});

/* ============================================================
   TYPEWRITER
============================================================ */
const phrases  = ['AI Solutions', 'ML Pipelines', 'NLP Systems', 'Cloud Architecture', 'Foundation Models'];
let pIdx       = 0;
let cIdx       = 0;
let deleting   = false;
const twEl     = document.getElementById('typewriter');

function typewrite() {
  const phrase = phrases[pIdx];

  if (deleting) {
    twEl.textContent = phrase.slice(0, --cIdx);
  } else {
    twEl.textContent = phrase.slice(0, ++cIdx);
  }

  let delay = deleting ? 55 : 95;

  if (!deleting && cIdx === phrase.length) {
    delay = 2200;
    deleting = true;
  } else if (deleting && cIdx === 0) {
    deleting = false;
    pIdx = (pIdx + 1) % phrases.length;
    delay = 350;
  }

  setTimeout(typewrite, delay);
}

typewrite();

/* ============================================================
   SCROLL REVEAL (Intersection Observer)
============================================================ */
const revealIO = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry, i) => {
      if (entry.isIntersecting) {
        setTimeout(() => entry.target.classList.add('visible'), i * 75);
      }
    });
  },
  { threshold: 0.1, rootMargin: '0px 0px -40px 0px' }
);

document.querySelectorAll('.reveal').forEach(el => revealIO.observe(el));

/* ============================================================
   STATS COUNTER ANIMATION
============================================================ */
const statsIO = new IntersectionObserver(
  (entries) => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      entry.target.querySelectorAll('.stat-num').forEach(el => {
        const target = parseInt(el.dataset.target, 10);
        const suffix = el.dataset.suffix || '';
        let count    = 0;
        const inc    = target / 55;
        const timer  = setInterval(() => {
          count += inc;
          if (count >= target) {
            count = target;
            clearInterval(timer);
          }
          el.textContent = Math.floor(count) + suffix;
        }, 22);
      });
      statsIO.unobserve(entry.target);
    });
  },
  { threshold: 0.4 }
);

const statsGrid = document.querySelector('.stats-grid');
if (statsGrid) statsIO.observe(statsGrid);
