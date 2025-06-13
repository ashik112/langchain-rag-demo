import { queryRAG } from './rag-api.js';

// DOM Elements
let form = document.querySelector('#chatForm');
let promptInput = document.querySelector('#promptInput');
let chatHistory = document.querySelector('#chatHistory');
let loadingIndicator = document.querySelector('#loadingIndicator');
let submitButton = document.querySelector('#submitButton');
let welcomeMessage = document.querySelector('#welcomeMessage');
let clearChatButton = document.querySelector('#clearChat');
let statusIndicator = document.querySelector('#statusIndicator');
let charCount = document.querySelector('#charCount');

// Markdown configuration
let md = new markdownit({
  breaks: true,        // Convert \n to <br>
  linkify: true,       // Convert URLs to links
  typographer: true,   // Enable some language-neutral replacement + quotes beautification
  html: true,          // Enable HTML tags in source
  xhtmlOut: true,      // Use '/' to close single tags (<br />)
  langPrefix: 'language-',  // CSS language prefix for fenced blocks
  quotes: '""\'\'',    // Double + single quotes replacement pairs (fixed syntax)
});

// Configure markdown-it to better handle paragraphs and spacing
md.renderer.rules.paragraph_open = function() {
  return '<p>';
};

md.renderer.rules.paragraph_close = function() {
  return '</p>\n';
};

// State management
let isLoading = false;
let messageCount = 0;

// Auto-resize textarea
function autoResizeTextarea() {
  promptInput.style.height = 'auto';
  promptInput.style.height = Math.min(promptInput.scrollHeight, 128) + 'px';
}

// Update character counter
function updateCharCounter() {
  const count = promptInput.value.length;
  charCount.textContent = count;
  charCount.style.color = count > 1800 ? 'var(--error-color)' : 'var(--text-muted)';
}

// Set loading state
function setLoading(loading) {
  isLoading = loading;
  loadingIndicator.style.display = loading ? 'flex' : 'none';
  submitButton.disabled = loading;
  promptInput.disabled = loading;
  
  // Update status indicator
  const statusText = statusIndicator.querySelector('.status-text');
  const statusDot = statusIndicator.querySelector('.status-dot');
  
  if (loading) {
    statusText.textContent = 'Thinking...';
    statusDot.style.background = 'var(--warning-color)';
  } else {
    statusText.textContent = 'Ready';
    statusDot.style.background = 'var(--success-color)';
  }
}

// Hide welcome message
function hideWelcomeMessage() {
  if (welcomeMessage && messageCount === 0) {
    welcomeMessage.style.display = 'none';
  }
}

// Show welcome message
function showWelcomeMessage() {
  if (welcomeMessage && messageCount === 0) {
    welcomeMessage.style.display = 'flex';
  }
}

// Add message to chat
function addMessage(text, isUser = false) {
  hideWelcomeMessage();
  messageCount++;
  
  let messageDiv = document.createElement('div');
  messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
  
  let contentDiv = document.createElement('div');
  contentDiv.className = 'content';
  
  if (isUser) {
    contentDiv.textContent = text;
  } else {
    contentDiv.innerHTML = md.render(text);
  }
  
  messageDiv.appendChild(contentDiv);
  chatHistory.appendChild(messageDiv);
  
  // Smooth scroll to bottom
  setTimeout(() => {
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }, 100);
  
  return messageDiv;
}

// Clear chat
function clearChat() {
  chatHistory.innerHTML = '';
  messageCount = 0;
  showWelcomeMessage();
}

// Handle form submission
async function handleSubmit(question) {
  if (!question.trim() || isLoading) return;
  
  // Add user message
  addMessage(question, true);
  promptInput.value = '';
  updateCharCounter();
  autoResizeTextarea();
  
  // Show loading state
  setLoading(true);
  
  try {
    // Call the RAG system
    const response = await queryRAG(question);
    
    // Add assistant message
    addMessage(response.answer, false);
    
  } catch (e) {
    console.error('Error:', e);
    addMessage(`**Error:** ${e.message}`, false);
  } finally {
    setLoading(false);
  }
}

// Event Listeners
form.addEventListener('submit', (ev) => {
  ev.preventDefault();
  const question = promptInput.value.trim();
  handleSubmit(question);
});

// Auto-resize textarea on input
promptInput.addEventListener('input', () => {
  autoResizeTextarea();
  updateCharCounter();
});

// Handle Enter key (Shift+Enter for new line)
promptInput.addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter' && !ev.shiftKey && !isLoading) {
    ev.preventDefault();
    const question = promptInput.value.trim();
    handleSubmit(question);
  }
});

// Clear chat button
clearChatButton.addEventListener('click', () => {
  if (confirm('Are you sure you want to clear the chat history?')) {
    clearChat();
  }
});

// Quick action buttons
document.addEventListener('click', (ev) => {
  if (ev.target.classList.contains('quick-action-btn')) {
    const question = ev.target.getAttribute('data-question');
    if (question) {
      promptInput.value = question;
      updateCharCounter();
      autoResizeTextarea();
      handleSubmit(question);
    }
  }
});

// Focus input on page load
window.addEventListener('load', () => {
  promptInput.focus();
  updateCharCounter();
});

// Handle window resize for mobile
window.addEventListener('resize', () => {
  autoResizeTextarea();
});

// Keyboard shortcuts
document.addEventListener('keydown', (ev) => {
  // Ctrl/Cmd + K to focus input
  if ((ev.ctrlKey || ev.metaKey) && ev.key === 'k') {
    ev.preventDefault();
    promptInput.focus();
  }
  
  // Ctrl/Cmd + L to clear chat
  if ((ev.ctrlKey || ev.metaKey) && ev.key === 'l') {
    ev.preventDefault();
    if (confirm('Clear chat history?')) {
      clearChat();
    }
  }
  
  // Escape to blur input
  if (ev.key === 'Escape') {
    promptInput.blur();
  }
});

// Initialize
updateCharCounter();
autoResizeTextarea();
