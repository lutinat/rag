import { Component, ViewChild, ElementRef } from '@angular/core';
import { SidebarComponent, ChatSidebarItem } from '../sidebar/sidebar.component';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api.service';
import { EditChatModalComponent } from '../edit-chat-modal/edit-chat-modal.component';

interface Source {
  name: string;
}

interface Message {
  content: string;
  isUser: boolean;
  sources?: Source[];
}

interface Chat {
  id: number;
  title: string;
  messages: Message[];
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
  
  constructor(private apiService: ApiService) {}

  inputValue: string = '';
  isSidebarOpen: boolean = false;
  hasStartedChat: boolean = false;
  isWaitingForBot: boolean = false;

  chats: Chat[] = [
    { id: 1, title: 'New chat', messages: [] }
  ];
  selectedChatId: number = 1;
  chatIdCounter: number = 2;

  showEditModal = false;
  editingChat: ChatSidebarItem | null = null;

  get selectedChat(): Chat | undefined {
    return this.chats.find(chat => chat.id === this.selectedChatId);
  }

  get messages(): Message[] {
    return this.selectedChat?.messages ?? [];
  }

  onNewChat() {
    const newChat: Chat = {
      id: this.chatIdCounter++,
      title: 'New chat',
      messages: []
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

  onArrowClick(): void {
    if (this.inputValue && !this.isWaitingForBot && this.selectedChat) {
      const userQuestion = this.inputValue;
      this.hasStartedChat = true;
      this.selectedChat.messages.push({
        content: userQuestion,
        isUser: true
      });
      if (this.selectedChat.messages.length === 1) {
        this.selectedChat.title = userQuestion.slice(0, 30) + (userQuestion.length > 30 ? '...' : '');
      }
      this.scrollToBottom();
      this.isWaitingForBot = true;
      this.inputValue = '';

      this.apiService.getAnswer(userQuestion).subscribe(response => {
        const sources = response.sources ? response.sources.map((source: string) => ({ name: source })) : [];
        console.log('Sources:', sources);
        console.log('Response:', response.answer);
        this.selectedChat?.messages.push({
          content: response.answer,
          isUser: false,
          sources: sources
        });
        this.scrollToBottom();
        this.isWaitingForBot = false;
        setTimeout(() => this.chatInput?.nativeElement.focus(), 0);
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
    const index = this.chats.findIndex(chat => chat.id === id);
    if (index !== -1) {
      this.chats.splice(index, 1);
      if (this.selectedChatId === id) {
        if (this.chats.length === 0) {
          const newChat: Chat = {
            id: this.chatIdCounter++,
            title: 'New chat',
            messages: []
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
