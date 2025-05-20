import { Component, ViewChild, ElementRef } from '@angular/core';
import { SidebarComponent, ChatSidebarItem } from '../sidebar/sidebar.component';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

interface Message {
  content: string;
  isUser: boolean;
}

interface Chat {
  id: number;
  title: string;
  messages: Message[];
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [SidebarComponent, FormsModule, CommonModule],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.scss'
})
export class ChatComponent {
  @ViewChild('chatInput') chatInput!: ElementRef<HTMLInputElement>;
  @ViewChild('messageContainer') messageContainer!: ElementRef<HTMLDivElement>;
  
  inputValue: string = '';
  isSidebarOpen: boolean = false;
  hasStartedChat: boolean = false;
  isWaitingForBot: boolean = false;

  chats: Chat[] = [
    { id: 1, title: 'New chat', messages: [] }
  ];
  selectedChatId: number = 1;
  chatIdCounter: number = 2;

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
      this.hasStartedChat = true;
      this.selectedChat.messages.push({
        content: this.inputValue,
        isUser: true
      });
      if (this.selectedChat.messages.length === 1) {
        this.selectedChat.title = this.inputValue.slice(0, 30) + (this.inputValue.length > 30 ? '...' : '');
      }
      this.scrollToBottom();
      this.isWaitingForBot = true;
      this.inputValue = '';

      setTimeout(() => {
        this.selectedChat?.messages.push({
          content: "I am a test response.",
          isUser: false
        });
        this.scrollToBottom();
        this.isWaitingForBot = false;
        setTimeout(() => this.chatInput?.nativeElement.focus(), 0);
      }, 1000);
    }
  }

  toggleSidebar(): void {
    this.isSidebarOpen = !this.isSidebarOpen;
  }

  get sidebarChats(): ChatSidebarItem[] {
    return this.chats.map(chat => ({ id: chat.id, title: chat.title }));
  }
}
