/**
 * Calls the RAG system with a question and returns the response
 */
export async function queryRAG(question) {
  console.log('Starting RAG request...');
  
  try {
    const response = await fetch("/api/rag", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
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