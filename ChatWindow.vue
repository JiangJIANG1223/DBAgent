<template>
  <div class="chat-window">
    <div class="chat-header">
      <span>DB-Agent</span>
      <button class="close-button" @click="$emit('close')">x</button>
    </div>
    <div class="messages">
      <div
        v-for="message in messages"
        :key="message.id"
        :class="['message', message.sender]"
      >
        <div
          class="message-content"
          v-html="message.sender === 'system' ? message.renderedContent : message.content"
        ></div>
      </div>
    </div>
    <div class="input-area">
      <input
        v-model="userInput"
        @keyup.enter="sendMessage"
        placeholder="Enter your question here..."
      />
      <button @click="sendMessage">Send</button>
    </div>
  </div>
</template>

<script>
import { marked } from 'marked';
import DOMPurify from 'dompurify';

export default {
  name: 'ChatWindow',
  data() {
    return {
      userInput: '',
      messages: [],
    };
  },
  methods: {
    generateSessionId() {
      return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    },
    async sendMessage() {
      if (this.userInput.trim() === '') return;

      // Ensure a unique session_id
      let sessionId = localStorage.getItem('session_id');
      if (!sessionId) {
        sessionId = this.generateSessionId();
        localStorage.setItem('session_id', sessionId);
      }

      // Show user's message
      this.messages.push({
        id: Date.now(),
        sender: 'user',
        content: this.userInput,
      });

      const question = this.userInput;
      this.userInput = '';

      // Create a temporary status message with initial content
      const progressMsg = {
        id: Date.now(),
        sender: 'system',
        content: "Processing your request, please wait",
        renderedContent: this.renderMarkdown("Processing your request, please wait"),
      };
      this.messages.push(progressMsg);

      // Start a loading indicator timer
      let baseMessage = progressMsg.content;
      let loadingCount = 0;
      const loadingInterval = setInterval(() => {
        loadingCount = (loadingCount + 1) % 4; // cycles 0,1,2,3
        const dots = '.'.repeat(loadingCount);
        // Update the message with dynamic loading dots appended to the base message
        progressMsg.content = baseMessage + dots;
        progressMsg.renderedContent = this.renderMarkdown(progressMsg.content);
        // Update the last message in the messages array
        this.messages.pop();
        this.messages.push(progressMsg);
      }, 500);

      // Use EventSource to connect to the SSE endpoint
      const evtSource = new EventSource(
        `http://127.0.0.1:8000/api/agent_stream?session_id=${sessionId}&question=${encodeURIComponent(question)}`
      );

      evtSource.onmessage = (event) => {
        console.log(event.data);
        // Update the base message with the latest event data
        baseMessage = event.data;
        // Check if the event data indicates completion
        if (baseMessage.includes('[DONE]')) {
          baseMessage = baseMessage.replace('[DONE]', '');
          clearInterval(loadingInterval); // Stop the loading animation
          progressMsg.content = baseMessage;
          progressMsg.renderedContent = this.renderMarkdown(progressMsg.content);
          this.messages.pop();
          this.messages.push(progressMsg);
          evtSource.close();
        } else {
          // For intermediate updates, the interval continues to append dots.
          // We update the base message immediately so that the next tick uses the new content.
          progressMsg.renderedContent = this.renderMarkdown(progressMsg.content);
          this.messages.pop();
          this.messages.push(progressMsg);
        }
      };

      evtSource.onerror = (error) => {
        console.error("EventSource failed:", error);
        clearInterval(loadingInterval);
        this.messages.push({
          id: Date.now(),
          sender: 'system',
          content: "An error occurred. Please try again later.",
          renderedContent: this.renderMarkdown("An error occurred. Please try again later."),
        });
        evtSource.close();
      };
    },
    renderMarkdown(content) {
      const html = marked(content);
      return DOMPurify.sanitize(html);
    },
  },
};
</script>

<style>
.chat-window {
  width: 40%;
  height: 70vh;             /* 高度自适应内容 */
  /* min-height: 60vh;
  max-height: 80vh;         最大高度为视口高度的80% */
  background-color: #f9f9f9;
  border: 1px solid #ccc;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);

  /* 新增定位样式 */
  position: fixed;
  bottom: 20px; /* 距离窗口底部20px */
  right: 20px;  /* 距离窗口右侧20px */
  /* top: 50%;
  left: 50%;
  transform: translate(-50%, -50%); */
  z-index: 1000; /* 确保在页面上层显示 */
}

.chat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: #2c3e50;
  color: #ecf0f1;
  padding: 10px;
  border-top-left-radius: 8px;
  border-top-right-radius: 8px;
}

.close-button {
  background: transparent;
  border: none;
  color: #ecf0f1;
  font-size: 16px;
  cursor: pointer;
}

.messages {
  flex: 1;
  padding: 10px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.message {
  display: flex;
}

.message.user {
  justify-content: flex-end;
}

.message.system {
  justify-content: flex-start;
}

.message-content {
  max-width: 70%;
  padding: 10px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.5;
  word-wrap: break-word;
  background-color: #ecf0f1;
  color: #2c3e50;
  text-align: left;
}

.message.user .message-content {
  background-color: #3498db;
  color: #ffffff;
}

.message.system .message-content {
  background-color: #ecf0f1;
  color: #2c3e50;
}

.input-area {
  display: flex;
  padding: 10px;
  border-top: 1px solid #ccc;
  background-color: #fff;
  border-bottom-left-radius: 8px;
  border-bottom-right-radius: 8px;
}

.input-area input {
  flex: 1;
  padding: 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  outline: none;
}

.input-area button {
  margin-left: 10px;
  padding: 8px 16px;
  border: none;
  background-color: #3498db;
  color: #fff;
  border-radius: 4px;
  cursor: pointer;
}

.input-area button:hover {
  background-color: #2980b9;
}
</style>
  
  
  