import { queryRAG, initializeSession, destroySession } from './rag-api.js';

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
    
    // Update loading message text for better UX
    const loadingText = loadingIndicator.querySelector('.loading-text');
    if (loadingText) {
      loadingText.textContent = 'Thinking and preparing response...';
    }
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

// Add message to chat with enhanced hybrid response handling
function addMessage(text, isUser = false, responseData = null) {
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
    
    if (responseData) {
      // addHybridIndicators(contentDiv, responseData);
    }
  }
  
  messageDiv.appendChild(contentDiv);
  chatHistory.appendChild(messageDiv);
  
  setTimeout(() => {
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }, 100);
  
  return messageDiv;
}

// Add indicators for hybrid responses to show users what type of response they received
function addHybridIndicators(contentDiv, responseData) {
  /**
   * This function adds visual indicators to help users understand:
   * 1. Whether the response is purely from documents or includes general knowledge
   * 2. What topics are covered in their documents
   * 3. Whether code examples were provided
   */
  
  const indicatorContainer = document.createElement('div');
  indicatorContainer.className = 'response-indicators';
  
  const typeIndicator = document.createElement('div');
  typeIndicator.className = 'response-type-indicator';
  
  if (responseData.response_type === 'document_only') {
    typeIndicator.innerHTML = `
      <span class="indicator-icon">📚</span>
      <span class="indicator-text">Based entirely on your documents</span>
    `;
    typeIndicator.classList.add('document-only');
  } else if (responseData.response_type === 'hybrid') {
    typeIndicator.innerHTML = `
      <span class="indicator-icon">🔗</span>
      <span class="indicator-text">Document info + relevant examples</span>
    `;
    typeIndicator.classList.add('hybrid');
  }
  
  indicatorContainer.appendChild(typeIndicator);
  
  if (responseData.has_code_examples) {
    const codeIndicator = document.createElement('div');
    codeIndicator.className = 'code-indicator';
    codeIndicator.innerHTML = `
      <span class="indicator-icon">💻</span>
      <span class="indicator-text">Includes code examples</span>
    `;
    indicatorContainer.appendChild(codeIndicator);
  }
  
  if (responseData.document_topics && responseData.document_topics.length > 0) {
    const topicsIndicator = document.createElement('div');
    topicsIndicator.className = 'topics-indicator';
    const topicsList = responseData.document_topics.slice(0, 5).join(', ');
    const moreTopics = responseData.document_topics.length > 5 ? ` +${responseData.document_topics.length - 5} more` : '';
    
    topicsIndicator.innerHTML = `
      <span class="indicator-icon">🏷️</span>
      <span class="indicator-text">Document topics: ${topicsList}${moreTopics}</span>
    `;
    indicatorContainer.appendChild(topicsIndicator);
  }
  
  contentDiv.insertBefore(indicatorContainer, contentDiv.firstChild);
}

// Initialize session when page loads
document.addEventListener('DOMContentLoaded', async () => {
  console.log('🌟 Page loaded, initializing session...');
  
  try {
    await initializeSession();
    
    // Update status indicator
    const statusText = statusIndicator.querySelector('.status-text');
    const statusDot = statusIndicator.querySelector('.status-dot');
    statusText.textContent = 'Ready';
    statusDot.style.background = 'var(--success-color)';
    
    console.log('✅ Session ready for chat');
  } catch (error) {
    console.error('❌ Failed to initialize session:', error);
    
    // Update status to show error
    const statusText = statusIndicator.querySelector('.status-text');
    const statusDot = statusIndicator.querySelector('.status-dot');
    statusText.textContent = 'Connection Error';
    statusDot.style.background = 'var(--error-color)';
    
    // Show error message to user
    addMessage('❌ Failed to initialize chat session. Please reload the page.', false);
  }
});

// Clear chat and destroy session
async function clearChat() {
  try {
    // Destroy current session
    await destroySession();
    // Create new session
    await initializeSession();
    
    // Clear UI
    chatHistory.innerHTML = '';
    messageCount = 0;
    showWelcomeMessage();
    
    console.log('🧹 Chat cleared and new session created');
  } catch (error) {
    console.error('❌ Error clearing chat and session:', error);
    addMessage('❌ Error clearing chat. Please reload the page.', false);
  }
}

// Handle form submission with enhanced response processing
async function handleSubmit(question) {
  if (!question.trim() || isLoading) return;
  
  addMessage(question, true);
  promptInput.value = '';
  updateCharCounter();
  autoResizeTextarea();
  
  setLoading(true);
  
  try {
    console.log('🚀 Sending question to hybrid RAG system:', question);
    
    const response = await queryRAG(question);
    
    console.log('📊 Response metadata:', {
      type: response.response_type,
      sources: response.total_sources,
      topics: response.document_topics,
      hasCode: response.has_code_examples
    });
    
    addMessage(response.answer, false, response);
    
  } catch (e) {
    console.error('❌ Error in hybrid RAG system:', e);
    
    const errorMessage = `**Error:** ${e.message}\n\n> 💡 **Tip**: Try asking about topics mentioned in your documents, or ask for implementation examples related to your document content.`;
    addMessage(errorMessage, false);
    
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
  if (confirm('Clear chat history? This will remove all messages but keep your documents loaded.')) {
    console.log('🧹 Clearing chat history');
    clearChat();
  }
});

// Quick action buttons
document.addEventListener('click', (ev) => {
  if (ev.target.classList.contains('quick-action-btn')) {
    const question = ev.target.getAttribute('data-question');
    if (question) {
      console.log('🎯 Quick action selected:', question);
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
  if ((ev.ctrlKey || ev.metaKey) && ev.key === 'k') {
    ev.preventDefault();
    promptInput.focus();
    console.log('⌨️ Keyboard shortcut: Focus input');
  }
  
  if ((ev.ctrlKey || ev.metaKey) && ev.key === 'l') {
    ev.preventDefault();
    if (confirm('Clear chat history?')) {
      clearChat();
      console.log('⌨️ Keyboard shortcut: Clear chat');
    }
  }
  
  if (ev.key === 'Escape') {
    promptInput.blur();
    console.log('⌨️ Keyboard shortcut: Blur input');
  }
});

// Initialize with enhanced logging
console.log('🚀 Hybrid RAG Chat Interface Initialized');
console.log('📋 Features: Document search + relevant general knowledge');
console.log('⌨️ Shortcuts: Ctrl+K (focus), Ctrl+L (clear), Escape (blur)');

updateCharCounter();
autoResizeTextarea();
