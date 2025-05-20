import { Component, ViewChild, ElementRef } from '@angular/core';
import { SidebarComponent } from '../sidebar/sidebar.component';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

interface Message {
  content: string;
  isUser: boolean;
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
  messages: Message[] = [];
  isWaitingForBot: boolean = false;

  private shouldScrollToBottom(): boolean {
    const container = this.messageContainer?.nativeElement;
    if (!container) return false;
    
    // On considère que l'utilisateur est en bas si il est à moins de 100px du bas
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
    if (this.inputValue && !this.isWaitingForBot) {
      this.hasStartedChat = true;
      this.messages.push({
        content: this.inputValue,
        isUser: true
      });
      this.scrollToBottom();
      this.isWaitingForBot = true;
      this.inputValue = '';

      setTimeout(() => {
        this.messages.push({
          content: "Je suis Chatlantis, votre assistant virtuel. Je suis là pour répondre à vos questions sur Satlantis.",
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
}
