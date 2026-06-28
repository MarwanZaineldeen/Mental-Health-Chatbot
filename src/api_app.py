from __future__ import annotations

import os
import sys
from functools import lru_cache
from threading import Thread
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "src" / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.append(str(MODELS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from chatbot_pipeline import ChatbotPipeline


app = FastAPI(
    title="Nura Mental Health Support",
    description="Nura integrates language detection, emotion classification, intent routing, RAG, safety guardrails, and supportive response generation.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    source: str = Field("both", pattern="^(both|cci|amod)$")
    top_k: int = Field(5, ge=1, le=10)
    collection: str | None = Field(None, pattern="^(mental_health_rag|mental_health_rag_v2)$")
    history: list[dict[str, str]] = Field(default_factory=list)
    include_state: bool = False


class ChatResponse(BaseModel):
    response: str
    suggested_questions: list[str] = Field(default_factory=list)
    state: dict[str, Any]


@lru_cache(maxsize=1)
def get_pipeline() -> ChatbotPipeline:
    return ChatbotPipeline()


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _warmup_pipeline() -> None:
    get_pipeline().warmup(include_retrieval=_env_flag("NURA_WARMUP_RETRIEVAL"))


@app.on_event("startup")
def startup_warmup() -> None:
    if _env_flag("NURA_WARMUP_ON_START"):
        Thread(target=_warmup_pipeline, daemon=True).start()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/warmup")
def warmup(retrieval: bool = False) -> dict[str, Any]:
    return get_pipeline().warmup(include_retrieval=retrieval)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    pipeline = get_pipeline()
    pipeline.retrieval_source = request.source
    pipeline.top_k = request.top_k
    pipeline.set_retrieval_collection(request.collection)
    output = pipeline.run(request.message, history=request.history)
    return ChatResponse(
        response=output["response"],
        suggested_questions=output.get("suggested_questions", []),
        state=output["state"] if request.include_state else {},
    )


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PRODUCTION_PAGE


@app.get("/developer", response_class=HTMLResponse)
def developer() -> str:
    return DEVELOPER_PAGE


PRODUCTION_PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Nura | Mental Health Support</title>
  <style>
    :root {
      --bg: #fbf5f1;
      --surface: #fffaf6;
      --surface-soft: #f5eee7;
      --ink: #221b2f;
      --muted: #746b7d;
      --line: #eadfd6;
      --plum: #2b193d;
      --plum-2: #4c276d;
      --teal: #087f73;
      --mint-soft: #ddf8ee;
      --coral: #e95778;
      --iris: #7567d6;
      --user: #4c276d;
      --shadow: 0 18px 48px rgba(34, 27, 47, 0.12);
      --radius: 8px;
    }
    body[data-theme="dark"] {
      --bg: #14101d;
      --surface: #211a2b;
      --surface-soft: #2b2335;
      --ink: #f7edf6;
      --muted: #c4b8c9;
      --line: #3e334a;
      --shadow: 0 18px 48px rgba(0, 0, 0, 0.28);
      background: linear-gradient(135deg, #130f1b 0%, #21172c 44%, #0d2f2d 100%);
    }
    body[data-theme="dark"] .topbar,
    body[data-theme="dark"] .composer {
      background: rgba(24, 19, 33, 0.88);
    }
    body[data-theme="dark"] .bubble.assistant,
    body[data-theme="dark"] textarea,
    body[data-theme="dark"] .clear,
    body[data-theme="dark"] .theme-toggle,
    body[data-theme="dark"] .chat-item {
      background: #211a2b;
      color: var(--ink);
      border-color: var(--line);
    }
    body[data-theme="dark"] .suggestion {
      background: #342642;
      color: #ffd7a8;
      border-color: #6f4d69;
    }
    body[data-theme="dark"] .suggestion:hover { background: #402f52; }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: linear-gradient(135deg, #fff8f0 0%, #f5eefb 42%, #ecfbf7 100%);
    }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 336px minmax(0, 1fr);
    }
    aside {
      background: linear-gradient(180deg, var(--plum) 0%, #24162f 55%, #123f3a 100%);
      color: white;
      padding: 26px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }
    .brand {
      display: grid;
      grid-template-columns: 52px 1fr;
      gap: 13px;
      align-items: center;
    }
    .logo-mark {
      width: 52px;
      height: 52px;
      display: grid;
      place-items: center;
      background: #fff8f0;
      color: var(--coral);
      border: 1px solid rgba(255,255,255,0.5);
      border-radius: 8px;
      box-shadow: 0 16px 36px rgba(0, 0, 0, 0.18);
      font-size: 24px;
      line-height: 1;
    }
    .brand h1 {
      margin: 0;
      font-size: 32px;
      line-height: 1;
      letter-spacing: 0;
    }
    .brand p {
      margin: 6px 0 0;
      color: #ffd7a8;
      font-size: 16px;
      line-height: 1.22;
      font-weight: 850;
    }
    .intro {
      color: #f5e9ff;
      line-height: 1.6;
      font-size: 15px;
      margin: 0;
    }
    .side-tabs {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .side-tab {
      border: 1px solid rgba(255,255,255,0.16);
      background: rgba(255,255,255,0.07);
      color: white;
      padding: 10px;
      font: inherit;
      font-size: 13px;
      font-weight: 850;
      cursor: pointer;
      border-radius: var(--radius);
    }
    .side-tab.active {
      background: #fff3de;
      color: #3a204f;
      border-color: #ffb067;
    }
    .side-panel { display: none; }
    .side-panel.active {
      display: grid;
      gap: 12px;
    }
    .chat-list {
      display: grid;
      gap: 8px;
      max-height: 330px;
      overflow-y: auto;
    }
    .chat-item {
      border: 1px solid rgba(255,255,255,0.16);
      background: rgba(255,255,255,0.07);
      color: white;
      padding: 10px;
      border-radius: var(--radius);
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 7px;
      align-items: center;
    }
    .chat-item button {
      border: 0;
      background: transparent;
      color: inherit;
      cursor: pointer;
      font: inherit;
      font-weight: 850;
      padding: 2px 4px;
    }
    .chat-name {
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
      cursor: pointer;
      font-weight: 750;
    }
    .empty-chats {
      color: #f5e9ff;
      font-size: 13px;
      line-height: 1.45;
      opacity: 0.84;
    }
    .mode-title {
      color: #ffcb8f;
      font-size: 12px;
      font-weight: 850;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 10px;
    }
    .mode-group {
      display: grid;
      gap: 10px;
    }
    .mode {
      border: 1px solid rgba(255,255,255,0.18);
      background: rgba(255,255,255,0.07);
      color: white;
      padding: 13px;
      text-align: left;
      cursor: pointer;
      border-radius: var(--radius);
      transition: transform 0.16s ease, background 0.16s ease, border-color 0.16s ease;
    }
    .mode:hover { transform: translateY(-1px); border-color: rgba(255,255,255,0.34); }
    .mode.active {
      background: #fff3de;
      color: #3a204f;
      border-color: #ffb067;
    }
    .mode b { display: block; margin-bottom: 4px; font-size: 14px; }
    .mode span { color: inherit; opacity: 0.76; font-size: 13px; line-height: 1.4; }
    .trust-panel {
      margin-top: auto;
      border: 1px solid rgba(255,255,255,0.16);
      background: rgba(255,255,255,0.07);
      padding: 14px;
      border-radius: var(--radius);
      color: #f5e9ff;
      font-size: 13px;
      line-height: 1.5;
    }
    main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto 1fr auto;
      height: 100vh;
    }
    .topbar {
      padding: 18px 24px;
      background: rgba(255,250,246,0.82);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(14px);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 14px;
    }
    .topbar b { display: block; font-size: 15px; }
    .topbar span { color: var(--muted); font-size: 13px; }
    .top-actions {
      display: flex;
      gap: 10px;
      align-items: center;
    }
    .theme-toggle {
      border: 1px solid var(--line);
      background: var(--surface);
      color: var(--ink);
      padding: 10px 12px;
      font: inherit;
      font-size: 13px;
      font-weight: 850;
      cursor: pointer;
      border-radius: var(--radius);
    }
    .clear {
      border: 1px solid var(--line);
      background: var(--surface);
      color: var(--ink);
      padding: 10px 13px;
      font: inherit;
      font-size: 13px;
      font-weight: 800;
      cursor: pointer;
      border-radius: var(--radius);
    }
    .chat {
      overflow-y: auto;
      padding: 26px min(5vw, 54px);
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .bubble {
      max-width: min(780px, 88%);
      padding: 15px 17px;
      border-radius: var(--radius);
      line-height: 1.58;
      white-space: pre-wrap;
      box-shadow: var(--shadow);
      font-size: 15px;
    }
    .bubble.user {
      align-self: flex-end;
      background: linear-gradient(135deg, var(--user), var(--teal));
      color: white;
      border-top-right-radius: 2px;
    }
    .bubble.assistant {
      align-self: flex-start;
      background: rgba(255,255,255,0.96);
      border: 1px solid var(--line);
      border-top-left-radius: 2px;
    }
    .suggestions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 13px;
      white-space: normal;
    }
    .suggestion {
      border: 1px solid #ffd19a;
      background: #fff1d9;
      color: #4a2768;
      padding: 8px 11px;
      font: inherit;
      font-size: 13px;
      font-weight: 800;
      cursor: pointer;
      border-radius: 999px;
      box-shadow: none;
      max-width: 100%;
      text-align: left;
    }
    .suggestion:hover { background: #ffe2b8; }
    .typing {
      display: inline-flex;
      gap: 5px;
      align-items: center;
      min-width: 54px;
    }
    .typing span {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--teal);
      animation: bounce 1.15s infinite ease-in-out;
    }
    .typing span:nth-child(2) { animation-delay: 0.15s; }
    .typing span:nth-child(3) { animation-delay: 0.3s; }
    @keyframes bounce {
      0%, 80%, 100% { transform: translateY(0); opacity: 0.42; }
      40% { transform: translateY(-5px); opacity: 1; }
    }
    .composer {
      padding: 18px min(5vw, 54px) 24px;
      border-top: 1px solid var(--line);
      background: rgba(255,248,240,0.92);
      backdrop-filter: blur(12px);
    }
    .composer-inner {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 112px;
      gap: 10px;
      max-width: 980px;
      margin: 0 auto;
    }
    textarea {
      width: 100%;
      min-height: 58px;
      max-height: 168px;
      resize: vertical;
      border: 1px solid var(--line);
      background: var(--surface);
      color: var(--ink);
      padding: 14px;
      font: inherit;
      line-height: 1.45;
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }
    textarea:focus {
      outline: 3px solid rgba(255,176,103,0.34);
      border-color: #ffb067;
    }
    .send {
      border: 1px solid #331a48;
      background: linear-gradient(135deg, var(--plum-2), var(--coral));
      color: white;
      padding: 0 18px;
      font: inherit;
      font-weight: 900;
      cursor: pointer;
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }
    .send:hover { filter: brightness(1.06); }
    .send:disabled { opacity: 0.68; cursor: wait; }
    @media (max-width: 900px) {
      .shell { grid-template-columns: 1fr; }
      aside { padding: 18px; gap: 16px; }
      .intro, .trust-panel { display: none; }
      .mode-group { grid-template-columns: 1fr; }
      main { height: auto; min-height: 72vh; }
      .chat { min-height: 54vh; padding: 18px; }
      .composer { padding: 14px 18px 18px; }
      .composer-inner { grid-template-columns: 1fr; }
      .send { min-height: 48px; }
      .bubble { max-width: 94%; }
    }
  </style>
</head>
<body data-theme="light">
  <div class="shell">
    <aside>
      <div class="brand">
        <div class="logo-mark" aria-hidden="true">&#10084;</div>
        <div>
          <h1>Nura</h1>
          <p>Your gentle mental wellness companion</p>
        </div>
      </div>
      <p class="intro">Feel heard. Find calm. Take the next step.</p>
      <div class="side-tabs">
        <button class="side-tab active" data-panel="support">Support</button>
        <button class="side-tab" data-panel="chats">Chats</button>
      </div>
      <div class="side-panel active" id="supportPanel">
        <div class="mode-title">Support style</div>
        <div class="mode-group">
          <button class="mode active" data-source="both"><b>Balanced Care</b><span>Supportive conversation with practical guidance.</span></button>
          <button class="mode" data-source="cci"><b>Learn and Cope</b><span>Clear skills, grounding ideas, and psychoeducation.</span></button>
          <button class="mode" data-source="amod"><b>Reflective Talk</b><span>Gentler counseling-style responses.</span></button>
        </div>
      </div>
      <div class="side-panel" id="chatsPanel">
        <div class="mode-title">Saved chats</div>
        <div class="chat-list" id="chatList"></div>
      </div>
      <div class="trust-panel">Nura offers educational support and reflection. It is not a replacement for a licensed professional or emergency care.</div>
    </aside>
    <main>
      <div class="topbar">
        <div><b>Your conversation with Nura</b><span>Share as little or as much as you want</span></div>
        <div class="top-actions">
          <button class="theme-toggle" id="themeToggle">&#127769; Dark</button>
          <button class="clear" id="clear">New chat</button>
        </div>
      </div>
      <div class="chat" id="chat">
        <div class="bubble assistant">Hi, I'm Nura &#10084;&#65039;. Tell me what feels heavy right now, and I'll help you sort it into one gentle next step.</div>
      </div>
      <div class="composer">
        <div class="composer-inner">
          <textarea id="message" placeholder="Write what is on your mind..."></textarea>
          <button class="send" id="send">Send</button>
        </div>
      </div>
    </main>
  </div>
  <script>
    const chat = document.getElementById("chat");
    const message = document.getElementById("message");
    const send = document.getElementById("send");
    const clear = document.getElementById("clear");
    const themeToggle = document.getElementById("themeToggle");
    const chatList = document.getElementById("chatList");
    const sideTabs = [...document.querySelectorAll(".side-tab")];
    const modeButtons = [...document.querySelectorAll(".mode")];
    let source = "both";
    let history = [];
    let shownSuggestions = new Set();
    let currentChatId = null;
    let savedChats = JSON.parse(localStorage.getItem("nuraChats") || "[]");

    function normalizeSuggestion(question) {
      return question.trim().toLowerCase().replace(/\s+/g, " ");
    }

    function freshSuggestions(suggestions) {
      const fresh = [];
      suggestions.forEach((question) => {
        const key = normalizeSuggestion(question);
        if (key && !shownSuggestions.has(key)) {
          shownSuggestions.add(key);
          fresh.push(question);
        }
      });
      return fresh.slice(0, 3);
    }

    function addBubble(role, text, suggestions = []) {
      const bubble = document.createElement("div");
      bubble.className = `bubble ${role}`;
      bubble.textContent = text;

      const visibleSuggestions = role === "assistant" ? freshSuggestions(suggestions) : [];
      if (visibleSuggestions.length) {
        const suggestionBox = document.createElement("div");
        suggestionBox.className = "suggestions";
        visibleSuggestions.forEach((question) => {
          const chip = document.createElement("button");
          chip.className = "suggestion";
          chip.type = "button";
          chip.textContent = question;
          chip.addEventListener("click", () => {
            message.value = question;
            submitMessage();
          });
          suggestionBox.appendChild(chip);
        });
        bubble.appendChild(suggestionBox);
      }

      chat.appendChild(bubble);
      chat.scrollTop = chat.scrollHeight;
      return bubble;
    }

    function addTypingBubble() {
      const bubble = document.createElement("div");
      bubble.className = "bubble assistant";
      bubble.innerHTML = "<div class='typing'><span></span><span></span><span></span></div>";
      chat.appendChild(bubble);
      chat.scrollTop = chat.scrollHeight;
      return bubble;
    }


    function initialMessage() {
      return "Hi, I'm Nura \u2764\ufe0f. Tell me what feels heavy right now, and I'll help you sort it into one gentle next step.";
    }

    function saveChats() {
      localStorage.setItem("nuraChats", JSON.stringify(savedChats));
      renderChatList();
    }

    function chatId() {
      if (crypto.randomUUID) return crypto.randomUUID();
      return `chat-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }

    function chatTitle(messages) {
      const firstUser = messages.find((item) => item.role === "user");
      if (!firstUser) return "New conversation";
      const text = firstUser.content.trim().replace(/\s+/g, " ");
      return text.length > 32 ? `${text.slice(0, 32)}...` : text;
    }

    function saveCurrentChat() {
      if (!history.length) return;
      const existing = savedChats.find((item) => item.id === currentChatId);
      if (existing) {
        existing.messages = history;
        existing.updatedAt = Date.now();
      } else {
        currentChatId = chatId();
        savedChats.unshift({
          id: currentChatId,
          title: chatTitle(history),
          messages: history,
          updatedAt: Date.now(),
        });
      }
      savedChats.sort((a, b) => b.updatedAt - a.updatedAt);
      saveChats();
    }

    function renderChatList() {
      chatList.innerHTML = "";
      if (!savedChats.length) {
        const empty = document.createElement("div");
        empty.className = "empty-chats";
        empty.textContent = "Saved conversations will appear here when you start a new chat.";
        chatList.appendChild(empty);
        return;
      }
      savedChats.forEach((item) => {
        const row = document.createElement("div");
        row.className = "chat-item";

        const name = document.createElement("div");
        name.className = "chat-name";
        name.textContent = item.title;
        name.title = item.title;
        name.addEventListener("click", () => loadChat(item.id));

        const rename = document.createElement("button");
        rename.type = "button";
        rename.textContent = "Edit";
        rename.addEventListener("click", () => renameChat(item.id));

        const del = document.createElement("button");
        del.type = "button";
        del.textContent = "Del";
        del.addEventListener("click", () => deleteChat(item.id));

        row.append(name, rename, del);
        chatList.appendChild(row);
      });
    }

    function renderHistory() {
      chat.innerHTML = "";
      if (!history.length) {
        addBubble("assistant", initialMessage());
        return;
      }
      history.forEach((item) => addBubble(item.role, item.content));
    }

    function loadChat(id) {
      saveCurrentChat();
      const item = savedChats.find((chatItem) => chatItem.id === id);
      if (!item) return;
      currentChatId = item.id;
      history = item.messages || [];
      shownSuggestions = new Set();
      renderHistory();
    }

    function renameChat(id) {
      const item = savedChats.find((chatItem) => chatItem.id === id);
      if (!item) return;
      const title = prompt("Rename chat", item.title);
      if (!title || !title.trim()) return;
      item.title = title.trim().slice(0, 60);
      item.updatedAt = Date.now();
      saveChats();
    }

    function deleteChat(id) {
      savedChats = savedChats.filter((item) => item.id !== id);
      if (currentChatId === id) {
        currentChatId = null;
        history = [];
        shownSuggestions = new Set();
        renderHistory();
      }
      saveChats();
    }
    sideTabs.forEach((button) => {
      button.addEventListener("click", () => {
        sideTabs.forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        document.getElementById("supportPanel").classList.toggle("active", button.dataset.panel === "support");
        document.getElementById("chatsPanel").classList.toggle("active", button.dataset.panel === "chats");
      });
    });

    modeButtons.forEach((button) => {
      button.addEventListener("click", () => {
        modeButtons.forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        source = button.dataset.source;
      });
    });

    function setTheme(theme) {
      document.body.dataset.theme = theme;
      themeToggle.textContent = theme === "dark" ? "\u2600\ufe0f Light" : "\ud83c\udf19 Dark";
      localStorage.setItem("nuraTheme", theme);
    }

    themeToggle.addEventListener("click", () => {
      setTheme(document.body.dataset.theme === "dark" ? "light" : "dark");
    });

    clear.addEventListener("click", () => {
      saveCurrentChat();
      currentChatId = null;
      history = [];
      shownSuggestions = new Set();
      chat.innerHTML = "";
      addBubble("assistant", "New chat started. I'm here with you \u2764\ufe0f. What would feel helpful to talk through today?");
      message.focus();
    });

    async function submitMessage() {
      const text = message.value.trim();
      if (!text) return;

      const previousHistory = history.slice(-10);
      addBubble("user", text);
      message.value = "";
      send.disabled = true;
      send.textContent = "...";
      const typingBubble = addTypingBubble();

      try {
        const response = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, source, top_k: 5, history: previousHistory }),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        typingBubble.remove();
        addBubble(
          "assistant",
          data.response || "I am here with you, but I could not generate a full response. Could you tell me a little more?",
          data.suggested_questions || []
        );
        history.push({ role: "user", content: text });
        history.push({ role: "assistant", content: data.response || "" });
        history = history.slice(-10);
        saveCurrentChat();
      } catch (error) {
        typingBubble.remove();
        addBubble("assistant", "I had trouble responding just now. Please try again in a moment.");
      } finally {
        send.disabled = false;
        send.textContent = "Send";
        message.focus();
      }
    }

    setTheme(localStorage.getItem("nuraTheme") || "light");
    renderChatList();

    send.addEventListener("click", submitMessage);
    message.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        submitMessage();
      }
    });
  </script>
</body>
</html>
"""

DEVELOPER_PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mental Health Support Chatbot</title>
  <style>
    :root {
      --ink: #111827;
      --muted: #4b5563;
      --line: #d1d5db;
      --paper: #f8fafc;
      --panel: #ffffff;
      --green: #059669;
      --green-soft: #d1fae5;
      --indigo: #4338ca;
      --rose: #be123c;
    }
   * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--paper);
    }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    header {
      background: #0f172a;
      color: white;
      border-bottom: 4px solid var(--green);
    }
    .header-inner {
      max-width: 1180px;
      margin: 0 auto;
      padding: 18px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }
    .brand {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .brand span {
      color: #86efac;
      font-size: 12px;
      text-transform: uppercase;
      font-weight: 800;
    }
    .brand h1 {
      margin: 0;
      font-size: clamp(22px, 3vw, 32px);
      letter-spacing: 0;
    }
    main {
      max-width: 1180px;
      width: 100%;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      grid-template-columns: minmax(300px, 420px) 1fr;
      gap: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 16px;
    }
    label {
      display: block;
      font-weight: 750;
      margin-bottom: 8px;
    }
    textarea, select, input {
      width: 100%;
      border: 1px solid var(--line);
      color: var(--ink);
      background: white;
      padding: 10px 11px;
      font: inherit;
    }
    textarea {
      min-height: 190px;
      resize: vertical;
      line-height: 1.45;
    }
    .controls {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin: 14px 0;
    }
    button {
      width: 100%;
      border: 1px solid #064e3b;
      background: var(--green);
      color: white;
      padding: 11px 14px;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.65;
      cursor: wait;
    }
    .developer-actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .secondary {
      background: white;
      color: var(--ink);
      border-color: var(--line);
    }
    .history-status {
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .answer {
      min-height: 190px;
      line-height: 1.58;
      white-space: pre-wrap;
    }
    .route {
      display: inline-block;
      background: var(--green-soft);
      border: 1px solid #065f46;
      color: #064e3b;
      padding: 4px 8px;
      font-size: 12px;
      font-weight: 850;
      margin-bottom: 12px;
      text-transform: uppercase;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-top: 14px;
    }
    .metric {
      border: 1px solid var(--line);
      background: #fafafa;
      padding: 12px;
      min-height: 72px;
    }
    .metric b {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .metric span {
      font-weight: 800;
    }
    details {
      margin-top: 14px;
      border: 1px solid var(--line);
      background: #fafafa;
      padding: 10px;
    }
    summary {
      cursor: pointer;
      font-weight: 800;
    }
    pre {
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      color: #1f2937;
    }
    .error {
      color: var(--rose);
      font-weight: 750;
    }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      .grid { grid-template-columns: 1fr; }
      .controls { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body data-theme="light">
  <div class="app">
    <header>
      <div class="header-inner">
        <div class="brand">
          <span>Integrated RAG Pipeline</span>
          <h1>Mental Health Support Chatbot</h1>
        </div>
      </div>
    </header>

    <main>
      <section class="panel">
        <label for="message">User message</label>
        <textarea id="message" placeholder="Example: I feel anxious every night and cannot sleep."></textarea>

        <div class="controls">
          <div>
            <label for="collection">Vector index</label>
            <select id="collection">
              <option value="mental_health_rag_v2">Current v2 index</option>
              <option value="mental_health_rag">Previous index</option>
            </select>
          </div>
          <div>
            <label for="source">Retrieval mode</label>
            <select id="source">
              <option value="both">Balanced Support</option>
              <option value="cci">Educational Guidance</option>
              <option value="amod">Counseling Style</option>
            </select>
          </div>
          <div>
            <label for="topK">Retrieved chunks</label>
            <input id="topK" type="number" min="1" max="10" value="5" />
          </div>
        </div>

        <div class="developer-actions">
          <button id="send">Generate Response</button>
          <button id="clearHistory" class="secondary">Clear Conversation</button>
        </div>
        <div id="historyStatus" class="history-status">Conversation history: 0 messages</div>
      </section>

      <section class="panel">
        <div id="route" class="route">Waiting</div>
        <div id="answer" class="answer">Enter a message to run the full pipeline.</div>

        <div class="grid">
          <div class="metric"><b>Language</b><span id="language">-</span></div>
          <div class="metric"><b>Emotion</b><span id="emotion">-</span></div>
          <div class="metric"><b>Intent</b><span id="intent">-</span></div>
          <div class="metric"><b>Total latency</b><span id="latency">-</span></div>
        </div>

        <details>
          <summary>Pipeline State</summary>
          <pre id="state">{}</pre>
        </details>
      </section>
    </main>
  </div>

  <script>
    const sendButton = document.getElementById("send");
    const answer = document.getElementById("answer");
    const stateBox = document.getElementById("state");
    const route = document.getElementById("route");
    const language = document.getElementById("language");
    const emotion = document.getElementById("emotion");
    const intent = document.getElementById("intent");
    const latency = document.getElementById("latency");
    const clearHistory = document.getElementById("clearHistory");
    const historyStatus = document.getElementById("historyStatus");
    let history = [];


    function pct(value) {
      if (typeof value !== "number") return "";
      return ` (${Math.round(value * 100)}%)`;
    }

    function updateHistoryStatus() {
      historyStatus.textContent = `Conversation history: ${history.length} messages`;
    }

    clearHistory.addEventListener("click", () => {
      history = [];
      updateHistoryStatus();
      stateBox.textContent = "{}";
      route.textContent = "Waiting";
      latency.textContent = "-";
      answer.textContent = "Conversation cleared. Enter a message to run the full pipeline.";
    });

    sendButton.addEventListener("click", async () => {
      const message = document.getElementById("message").value.trim();
      const source = document.getElementById("source").value;
      const collection = document.getElementById("collection").value;
      const topK = Number(document.getElementById("topK").value || 5);
      const previousHistory = history.slice(-10);

      if (!message) {
        answer.innerHTML = "<span class='error'>Please enter a message.</span>";
        return;
      }

      sendButton.disabled = true;
      sendButton.textContent = "Running pipeline...";
      answer.textContent = "Analyzing language, emotion, intent, safety, retrieval, and generation...";

      try {
        const response = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message, source, top_k: topK, collection, history: previousHistory, include_state: true }),
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        const state = data.state || {};

        answer.textContent = data.response || "";
        route.textContent = state.route || "unknown";
        language.textContent = `${state.language?.language_name || "-"}${pct(state.language?.confidence)}`;
        emotion.textContent = `${state.emotion?.emotion || "-"}${pct(state.emotion?.confidence)}`;
        intent.textContent = `${state.intent?.intent || "-"}${pct(state.intent?.confidence)}`;
        latency.textContent = state.timings_ms?.total ? `${state.timings_ms.total} ms` : "-";
        stateBox.textContent = JSON.stringify(state, null, 2);
        history.push({ role: "user", content: message });
        history.push({ role: "assistant", content: data.response || "" });
        history = history.slice(-10);

        updateHistoryStatus();
      } catch (error) {
        answer.innerHTML = `<span class='error'>Request failed: ${error.message}</span>`;
      } finally {
        sendButton.disabled = false;
        sendButton.textContent = "Generate Response";
      }
    });
  </script>
</body>
</html>
"""
