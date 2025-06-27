import { Component, ViewChild, ElementRef } from '@angular/core';
import { SidebarComponent, ChatSidebarItem } from '../sidebar/sidebar.component';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api.service';
import { EditChatModalComponent } from '../edit-chat-modal/edit-chat-modal.component';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';

interface Source {
  filename: string;
  url?: string;
  title?: string;
  source_type: string;
  display_name: string;
}

interface Message {
  content: SafeHtml;
  isUser: boolean;
  sources?: Source[];
}

interface Chat {
  id: number;
  title: string;
  messages: Message[];
  chatId?: string; // Backend chat ID for conversation history
  isWaitingForBot?: boolean; // Track loading state per chat
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [SidebarComponent, FormsModule, CommonModule, EditChatModalComponent],
  providers: [ApiService],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.scss'
})
export class ChatComponent {
  @ViewChild('chatInput') chatInput!: ElementRef<HTMLInputElement>;
  @ViewChild('messageContainer') messageContainer!: ElementRef<HTMLDivElement>;
  
  constructor(
    private apiService: ApiService,
    private sanitizer: DomSanitizer
  ) {
    // Log the API URL being used for debugging
    console.log('Chat component initialized with API URL:', this.apiService.getCurrentApiUrl());
  }

  inputValue: string = '';
  isSidebarOpen: boolean = false;
  hasStartedChat: boolean = false;

  chats: Chat[] = [
    { id: 1, title: 'New chat', messages: [], isWaitingForBot: false }
  ];
  selectedChatId: number = 1;
  chatIdCounter: number = 2;

  showEditModal = false;
  editingChat: ChatSidebarItem | null = null;

  get selectedChat(): Chat | undefined {
    const chat = this.chats.find(chat => chat.id === this.selectedChatId);
    // Ensure isWaitingForBot is initialized for existing chats
    if (chat && chat.isWaitingForBot === undefined) {
      chat.isWaitingForBot = false;
    }
    return chat;
  }

  get messages(): Message[] {
    return this.selectedChat?.messages ?? [];
  }

  get isCurrentChatWaiting(): boolean {
    const waiting = this.selectedChat?.isWaitingForBot === true;
    console.log('Template checking isCurrentChatWaiting:', waiting, 'selectedChat:', this.selectedChat?.id, 'isWaitingForBot:', this.selectedChat?.isWaitingForBot);
    return waiting;
  }

  onNewChat() {
    const newChat: Chat = {
      id: this.chatIdCounter++,
      title: 'New chat',
      messages: [],
      chatId: this.apiService.generateChatId(),
      isWaitingForBot: false
    };
    this.chats.unshift(newChat);
    this.selectedChatId = newChat.id;
    this.inputValue = '';
    this.hasStartedChat = false;
  }

  onSelectChat(id: number) {
    this.selectedChatId = id;
    this.inputValue = '';
    this.hasStartedChat = this.messages.length > 0;
  }

  private shouldScrollToBottom(): boolean {
    const container = this.messageContainer?.nativeElement;
    if (!container) return false;
    const threshold = 100;
    return container.scrollHeight - container.scrollTop - container.clientHeight < threshold;
  }

  private scrollToBottom(): void {
    if (this.shouldScrollToBottom()) {
      setTimeout(() => {
        const container = this.messageContainer?.nativeElement;
        if (container) {
          container.scrollTop = container.scrollHeight;
        }
      }, 0);
    }
  }

  private sanitizeHtml(text: string): SafeHtml {
    return this.sanitizer.bypassSecurityTrustHtml(text);
  }

  onArrowClick(): void {
    if (this.inputValue && !this.selectedChat?.isWaitingForBot && this.selectedChat) {
      const userQuestion = this.inputValue;
      const targetChat = this.selectedChat; // Capture reference to ensure response goes to correct chat
      
      this.hasStartedChat = true;
      targetChat.messages.push({
        content: this.sanitizeHtml(userQuestion),
        isUser: true
      });
      if (targetChat.messages.length === 1) {
        targetChat.title = userQuestion.slice(0, 30) + (userQuestion.length > 30 ? '...' : '');
      }
      this.scrollToBottom();
      targetChat.isWaitingForBot = true;
      this.inputValue = '';

      const chatId = targetChat.chatId || this.apiService.generateChatId();
      if (!targetChat.chatId) {
        targetChat.chatId = chatId;
      }
      
      this.apiService.getAnswer(userQuestion, chatId).subscribe({
        next: (response) => {
          const sources = response.sources || [];
          console.log('Sources:', sources);
          console.log('Response:', response.answer);
          targetChat.messages.push({
            content: this.sanitizeHtml(response.answer),
            isUser: false,
            sources: sources
          });
          targetChat.isWaitingForBot = false;
          // Only scroll to bottom if this is still the selected chat
          if (targetChat.id === this.selectedChatId) {
            this.scrollToBottom();
            setTimeout(() => this.chatInput?.nativeElement.focus(), 0);
          }
        },
        error: (error) => {
          console.error('Error getting answer from API:', error);
          console.error('API URL used:', this.apiService.getCurrentApiUrl());
          targetChat.messages.push({
            content: this.sanitizeHtml(`Error: Unable to get response from the API server. Please check if the server is running on ${this.apiService.getCurrentApiUrl()}`),
            isUser: false,
            sources: []
          });
          targetChat.isWaitingForBot = false;
          if (targetChat.id === this.selectedChatId) {
            this.scrollToBottom();
            setTimeout(() => this.chatInput?.nativeElement.focus(), 0);
          }
        }
      });
    }
  }

  toggleSidebar(): void {
    this.isSidebarOpen = !this.isSidebarOpen;
  }

  get sidebarChats(): ChatSidebarItem[] {
    return this.chats.map(chat => ({ id: chat.id, title: chat.title }));
  }

  deleteChat(id: number): void {
    const chatToDelete = this.chats.find(chat => chat.id === id);
    if (!chatToDelete) return;

    // Remove from frontend immediately for responsive UI
    this.removeChatFromFrontend(id);

    // If the chat has a backend chatId, delete it from the backend asynchronously
    if (chatToDelete.chatId) {
      this.apiService.clearChatHistory(chatToDelete.chatId).subscribe({
        next: (response) => {
          console.log('Chat deleted from backend:', response);
        },
        error: (error) => {
          console.error('Error deleting chat from backend (chat already removed from UI):', error);
          // Chat is already removed from frontend, so just log the error
        }
      });
    }
  }

  private removeChatFromFrontend(id: number): void {
    const index = this.chats.findIndex(chat => chat.id === id);
    if (index !== -1) {
      this.chats.splice(index, 1);
      if (this.selectedChatId === id) {
        if (this.chats.length === 0) {
          const newChat: Chat = {
            id: this.chatIdCounter++,
            title: 'New chat',
            messages: [],
            chatId: this.apiService.generateChatId(),
            isWaitingForBot: false
          };
          this.chats.push(newChat);
          this.selectedChatId = newChat.id;
        } else {
          this.selectedChatId = this.chats[0].id;
        }
        this.hasStartedChat = this.messages.length > 0;
      }
    }
  }

  renameChat(data: {id: number, newTitle: string}): void {
    const chat = this.chats.find(chat => chat.id === data.id);
    if (chat) {
      chat.title = data.newTitle;
    }
  }

  startEditing(event: Event, chat: ChatSidebarItem) {
    event.preventDefault();
    event.stopPropagation();
    this.editingChat = chat;
    this.showEditModal = true;
  }

  onSaveEdit(data: {id: number, newTitle: string}) {
    this.renameChat(data);
    this.showEditModal = false;
    this.editingChat = null;
  }

  onCancelEdit() {
    this.showEditModal = false;
    this.editingChat = null;
  }
}
