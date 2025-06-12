import { streamRAGResponse } from './rag-api.js';

let form = document.querySelector('form');
let promptInput = document.querySelector('input[name="prompt"]');
let chatHistory = document.querySelector('#chatHistory');
let md = new markdownit();

function addMessage(text, isUser = false) {
  let messageDiv = document.createElement('div');
  messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
  
  let contentDiv = document.createElement('div');
  contentDiv.className = 'content';
  contentDiv.innerHTML = isUser ? text : md.render(text);
  messageDiv.appendChild(contentDiv);
  
  chatHistory.appendChild(messageDiv);
  chatHistory.scrollTop = chatHistory.scrollHeight;
  return messageDiv;
}

form.onsubmit = async (ev) => {
  ev.preventDefault();
  let question = promptInput.value.trim();
  if (!question) return;
  
  // Add user message
  addMessage(question, true);
  promptInput.value = '';
  
  // Create assistant message container
  let assistantMessage = addMessage('', false);
  let contentDiv = assistantMessage.querySelector('.content');
  let textBuffer = '';
  
  try {
    // Call the RAG system and stream the response
    let stream = streamRAGResponse(question);
    
    for await (let chunk of stream) {
      if (chunk.error) {
        contentDiv.innerHTML = md.render(`Error: ${chunk.error}`);
        break;
      }
      
      // Append new text to buffer
      textBuffer += chunk.text;
      
      // Update the content with markdown rendering
      contentDiv.innerHTML = md.render(textBuffer);
      
      // Scroll to bottom
      chatHistory.scrollTop = chatHistory.scrollHeight;
    }
  } catch (e) {
    contentDiv.innerHTML = md.render(`Error: ${e.message}`);
  }
};
