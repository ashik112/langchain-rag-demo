/**
 * Calls the RAG system with a question and streams the response
 */
export async function* streamRAGResponse(question) {
  console.log('Starting RAG request...');
  // Send the question to the Python backend
  let response = await fetch("/api/rag", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ question })
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  console.log('Got response, starting to read stream...');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      console.log('Read chunk:', { done, valueLength: value?.length });
      
      if (done) {
        console.log('Stream done, processing final buffer:', buffer);
        // Process any remaining data in the buffer
        if (buffer.trim()) {
          try {
            const chunk = buffer.replace(/^data:\s*/, '').trim();
            if (chunk) {
              const data = JSON.parse(chunk);
              if (data.error) throw new Error(data.error);
              console.log('Yielding final chunk:', data);
              yield data;
            }
          } catch (e) {
            console.error('Error processing final chunk:', e);
          }
        }
        break;
      }

      // Decode the chunk and add to buffer
      const decoded = decoder.decode(value, { stream: true });
      console.log('Decoded chunk:', decoded);
      buffer += decoded;
      
      // Process complete messages in the buffer
      const lines = buffer.split('\n\n');
      buffer = lines.pop() || ''; // Keep the last incomplete chunk in the buffer
      
      for (const line of lines) {
        if (!line.trim()) continue;
        
        try {
          const chunk = line.replace(/^data:\s*/, '').trim();
          if (chunk) {
            const data = JSON.parse(chunk);
            if (data.error) throw new Error(data.error);
            console.log('Yielding chunk:', data);
            yield data;
          }
        } catch (e) {
          console.error('Error processing chunk:', e, 'Chunk:', line);
        }
      }
    }
  } finally {
    console.log('Stream processing complete');
    reader.releaseLock();
  }
} 