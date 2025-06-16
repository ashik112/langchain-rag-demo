// Global session ID for this page
let currentSessionId = null;
let isDestroying = false;  // Flag to prevent multiple simultaneous destroy calls
let isNavigating = false;  // Flag to detect page navigation vs. site exit
let isFirstLoad = true;  // New flag to detect refresh
let isRefreshMode = false;

/**
 * Get session ID from localStorage or create new one
 */
function getStoredSessionId() {
  return localStorage.getItem('rag_session_id');
}

/**
 * Store session ID in localStorage
 */
function storeSessionId(sessionId) {
  localStorage.setItem('rag_session_id', sessionId);
}

/**
 * Remove session ID from localStorage
 */
function removeStoredSessionId() {
  localStorage.removeItem('rag_session_id');
}

/**
 * Store chat message in localStorage
 */
function storeChatMessage(message, isUser, responseData = null) {
  const chatHistory = getChatHistory();
  chatHistory.push({
    text: message,
    isUser: isUser,
    timestamp: Date.now(),
    responseData: responseData
  });
  localStorage.setItem('rag_chat_history', JSON.stringify(chatHistory));
}

/**
 * Get chat history from localStorage
 */
function getChatHistory() {
  const stored = localStorage.getItem('rag_chat_history');
  return stored ? JSON.parse(stored) : [];
}

/**
 * Clear chat history from localStorage
 */
function clearChatHistory() {
  localStorage.removeItem('rag_chat_history');
}

/**
 * Restore chat history to UI
 */
function restoreChatHistory() {
  const chatHistory = getChatHistory();
  if (chatHistory.length > 0) {
    console.log(`📜 Restoring ${chatHistory.length} messages from chat history`);
    
    // Import addMessage function from main.js
    if (typeof window.addMessage === 'function') {
      chatHistory.forEach(msg => {
        // Don't store these messages again (storeInHistory = false)
        window.addMessage(msg.text, msg.isUser, msg.responseData, false);
      });
    } else {
      console.warn('addMessage function not available for chat history restoration');
    }
  }
}

/**
 * Create a new session when page loads
 */
async function createSession() {
  console.log('🆕 Creating new session...');
  
  try {
    const response = await fetch("/api/create-session", {
      method: "POST",
      headers: { "content-type": "application/json" }
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    
    if (data.success) {
      currentSessionId = data.session_id;
      storeSessionId(currentSessionId);  // Persist to localStorage
      isDestroying = false;  // Reset destroy flag
      console.log(`✅ Session created: ${currentSessionId}`);
      return currentSessionId;
    } else {
      throw new Error(data.error || 'Failed to create session');
    }
  } catch (error) {
    console.error('❌ Error creating session:', error);
    throw error;
  }
}

/**
 * Destroy the current session
 */
export async function destroySession() {
  if (!currentSessionId || isDestroying) return;
  
  try {
    isDestroying = true;  // Set flag to prevent multiple destroy calls
    console.log(`🗑️ Destroying session: ${currentSessionId}`);
    
    const data = JSON.stringify({ session_id: currentSessionId });
    
    if ('sendBeacon' in navigator) {
      // sendBeacon is more reliable for cleanup during page unload
      navigator.sendBeacon('/api/destroy-session', data);
    } else {
      // Fallback for older browsers
      await fetch("/api/destroy-session", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: data,
        keepalive: true
      });
    }
    
    console.log(`✅ Session destroyed: ${currentSessionId}`);
    currentSessionId = null;
    removeStoredSessionId();  // Remove from localStorage
    clearChatHistory();  // Clear chat history when session is destroyed
  } catch (error) {
    console.error('❌ Error destroying session:', error);
  } finally {
    isDestroying = false;  // Reset flag
  }
}

/**
 * Check if we're navigating to another page on the same site
 */
function isInternalNavigation(event) {
  // If it's a programmatic navigation or click on internal link
  if (event && event.target) {
    const link = event.target.closest('a');
    if (link) {
      const href = link.href;
      const currentOrigin = window.location.origin;
      return href.startsWith(currentOrigin) || href.startsWith('/') || href.startsWith('./') || href.startsWith('../');
    }
  }
  return false;
}

/**
 * Ensure we have a valid session
 */
async function ensureSession() {
  const isRefresh = isPageRefresh();
  console.log(`🔄 Page load type: ${isRefresh ? 'Refresh' : 'First Load'}`);
  
  // First, try to get session from localStorage
  if (!currentSessionId) {
    const storedSessionId = getStoredSessionId();
    if (storedSessionId) {
      console.log('📦 Found stored session ID, verifying...');
      currentSessionId = storedSessionId;
    }
  }
  
  if (!currentSessionId) {
    console.log('No session found, creating new one...');
    await createSession();
  } else {
    // Verify session exists on server
    try {
      const response = await fetch("/api/verify-session", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ session_id: currentSessionId })
      });
      
      if (!response.ok) {
        console.log('Session invalid, creating new one...');
        removeStoredSessionId();  // Clean up invalid session
        await createSession();
      } else {
        console.log(`✅ Session ${currentSessionId} verified and restored`);
      }
    } catch (error) {
      console.error('Error verifying session:', error);
      removeStoredSessionId();  // Clean up on error
      await createSession();
    }
  }
}

/**
 * Calls the RAG system with a question and returns the response
 */
export async function queryRAG(question) {
  console.log('Starting RAG request...');
  
  // Ensure we have a valid session before proceeding
  await ensureSession();
  
  try {
    const response = await fetch("/api/rag", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ 
        question: question,
        session_id: currentSessionId 
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      if (errorData.error && errorData.error.includes('Session not found')) {
        // Session was lost, create new one and retry
        console.log('Session lost, retrying with new session...');
        await createSession();
        return queryRAG(question);  // Retry the query
      }
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('Got response:', data);
    
    if (data.error) {
      throw new Error(data.error);
    }
    
    return data;
  } catch (error) {
    console.error('Error in queryRAG:', error);
    throw error;
  }
}

/**
 * Initialize session management
 */
export async function initializeSession() {
  try {
    const isRefresh = isPageRefresh();
    console.log(`🔄 Page load type: ${isRefresh ? 'Refresh' : 'First Load'}`);
    
    await ensureSession();
    
    // Restore chat history from localStorage after session is ready
    setTimeout(() => {
      restoreChatHistory();
    }, 100);
    
    // Destroy session on tab/browser close, preserve for navigation and refresh
    window.addEventListener('beforeunload', (event) => {
      // Set refresh flag for detection on next load
      setRefreshFlag();
      
      // Don't destroy session if refreshing
      if (isCurrentlyRefreshing()) {
        console.log('🔄 Refresh in progress, preserving session...');
        return;
      }
      
      if (!isNavigating && currentSessionId && !isDestroying) {
        console.log('🚪 Tab/browser closing, destroying session...');
        destroySession();
      }
    });
    
    // Handle page hiding (more reliable than beforeunload)
    window.addEventListener('pagehide', (event) => {
      // Don't destroy session on refresh
      if (window.performance && window.performance.navigation && 
          window.performance.navigation.type === 1) {
        console.log('🔄 Page refresh detected, preserving session...');
        return; // Don't destroy session on refresh
      }
      
      // pagehide is more reliable for detecting actual page unload
      if (event.persisted) {
        // Page is going into back/forward cache (browser navigation), preserve session
        console.log('📱 Page cached for navigation, preserving session...');
      } else if (!isNavigating && currentSessionId && !isDestroying) {
        // Page is actually being unloaded (tab/browser close)
        console.log('🚪 Page unloading, destroying session...');
        destroySession();
      }
    });
    
    // Handle visibility changes - but don't destroy session on tab switches
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') {
        console.log('👁️ Tab became visible again');
        // Tab became visible - session should still be there
      } else {
        console.log('🙈 Tab hidden (tab switch or minimize) - preserving session');
        // Tab hidden - but preserve session, could be temporary
      }
    });
    
    // Track internal navigation
    document.addEventListener('click', (event) => {
      if (isInternalNavigation(event)) {
        isNavigating = true;
        setTimeout(() => { isNavigating = false; }, 1000);
      }
    });
    
    console.log('✅ Smart session management initialized');
  } catch (error) {
    console.error('❌ Failed to initialize session:', error);
    throw error;
  }
}

// Get current session ID
export function getCurrentSessionId() {
  return currentSessionId;
}

/**
 * Export chat history functions for use in main.js
 */
export { storeChatMessage, clearChatHistory }

// Add this new function
function isPageRefresh() {
  if (window.performance && window.performance.navigation) {
    return window.performance.navigation.type === 1; // 1 is TYPE_RELOAD
  }
  // Fallback: if not first load, assume it's a refresh
  const wasFirstLoad = isFirstLoad;
  isFirstLoad = false;
  return !wasFirstLoad;
}

function setRefreshFlag() {
  localStorage.setItem('rag_is_refreshing', 'true');
  setTimeout(() => {
    localStorage.removeItem('rag_is_refreshing');
  }, 1000); // Clear flag after 1 second
}

function isCurrentlyRefreshing() {
  return localStorage.getItem('rag_is_refreshing') === 'true';
} 