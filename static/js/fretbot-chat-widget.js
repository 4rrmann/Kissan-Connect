(function () {
  'use strict';

  /* ── DOM ── */
  const widget = document.createElement('div');
  widget.id = 'chatWidget';
  widget.setAttribute('aria-label', 'Kisaan Bot chat');
  widget.innerHTML = `
    <div id="chatWindow" role="dialog" aria-modal="true" aria-label="Kisaan Bot" hidden>
      <div class="chat-header">
        <div class="chat-avatar" aria-hidden="true">🤖</div>
        <div class="chat-header-info">
          <h4>Kisaan Bot</h4>
          <span class="chat-online">Online</span>
        </div>
        <button id="chatClose" aria-label="Close chat"
                style="background:none;border:none;color:var(--text-muted);cursor:pointer;
                       font-size:18px;padding:4px 8px;border-radius:6px;transition:var(--transition);"
                onmouseover="this.style.background='rgba(255,255,255,0.07)'"
                onmouseout="this.style.background='none'">✕</button>
      </div>
      <div id="chatMessages" role="log" aria-live="polite" aria-label="Chat messages"></div>
      <div class="chat-input-area">
        <input id="chatInput" type="text" placeholder="Apna sawaal likho…"
               autocomplete="off" aria-label="Type your message" maxlength="500">
        <button id="chatSend" aria-label="Send message" title="Send">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
            <line x1="22" y1="2" x2="11" y2="13"/>
            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>
    </div>
    <button id="chatToggle" aria-label="Open Kisaan Bot chat" aria-expanded="false">
      🌿
    </button>`;

  document.body.appendChild(widget);

  const chatWindow = document.getElementById('chatWindow');
  const chatToggle = document.getElementById('chatToggle');
  const chatClose  = document.getElementById('chatClose');
  const chatInput  = document.getElementById('chatInput');
  const chatSend   = document.getElementById('chatSend');
  const chatMsgs   = document.getElementById('chatMessages');

  let isOpen = false;

  /* ── Greeting ── */
  addMsg('Namaste! 🌱 Main Kisaan Bot hoon. Farming se related koi bhi sawaal poochho  -crop diseases, yield, govt schemes, kuch bhi!', 'bot');

  /* ── Toggle ── */
  function openChat()  {
    isOpen = true;
    chatWindow.hidden = false;
    chatWindow.classList.add('open');
    chatToggle.setAttribute('aria-expanded', 'true');
    chatToggle.innerHTML = '✕';
    setTimeout(() => chatInput.focus(), 80);
  }
  function closeChat() {
    isOpen = false;
    chatWindow.classList.remove('open');
    chatToggle.setAttribute('aria-expanded', 'false');
    chatToggle.innerHTML = '🌿';
    setTimeout(() => chatWindow.hidden = true, 240);
  }

  chatToggle.addEventListener('click', () => isOpen ? closeChat() : openChat());
  chatClose.addEventListener('click',  closeChat);

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && isOpen) closeChat();
  });

  /* ── Messages ── */
  function addMsg(text, type) {
    const div = document.createElement('div');
    div.className = `msg ${type}`;
    div.innerHTML = text.replace(/\n/g, '<br>');
    chatMsgs.appendChild(div);
    chatMsgs.scrollTop = chatMsgs.scrollHeight;
    return div;
  }

  /* ── Send ── */
  async function send() {
    const text = chatInput.value.trim();
    if (!text) return;
    chatInput.value = '';
    chatSend.disabled = true;

    addMsg(text, 'user');
    const typing = addMsg('Soch raha hoon…', 'bot typing');

    try {
      const res  = await fetch('/chat', {
        method : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body   : JSON.stringify({ message: text }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      typing.remove();
      addMsg(data.response || 'Kuch error aa gaya, dobara try karo.', 'bot');
    } catch (err) {
      typing.remove();
      addMsg('Server se connect nahi ho pa raha. Please refresh aur try karo.', 'bot');
    } finally {
      chatSend.disabled = false;
      chatInput.focus();
    }
  }

  chatSend.addEventListener('click', send);
  chatInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });
})();