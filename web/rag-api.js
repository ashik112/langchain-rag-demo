// Global session ID for this page
let currentSessionId = null;
let isDestroying = false;  // Flag to prevent multiple simultaneous destroy calls

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
  } catch (error) {
    console.error('❌ Error destroying session:', error);
  } finally {
    isDestroying = false;  // Reset flag
  }
}

/**
 * Ensure we have a valid session
 */
async function ensureSession() {
  if (!currentSessionId) {
    console.log('No session found, creating new one...');
    await createSession();
  } else {
    // Verify session exists
    try {
      const response = await fetch("/api/verify-session", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ session_id: currentSessionId })
      });
      
      if (!response.ok) {
        console.log('Session invalid, creating new one...');
        await createSession();
      }
    } catch (error) {
      console.error('Error verifying session:', error);
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
    await createSession();
    
    // Set up cleanup when page unloads
    const cleanupHandler = async () => {
      if (currentSessionId && !isDestroying) {
        await destroySession();
      }
    };
    
    window.addEventListener('beforeunload', cleanupHandler);
    window.addEventListener('pagehide', cleanupHandler);
    
    // Also cleanup on visibility change (mobile browsers)
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') {
        cleanupHandler();
      }
    });
    
    console.log('✅ Session management initialized');
  } catch (error) {
    console.error('❌ Failed to initialize session:', error);
    throw error;
  }
}

// Get current session ID
export function getCurrentSessionId() {
  return currentSessionId;
} 