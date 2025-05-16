import { Component } from '@angular/core';
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
  inputValue: string = '';
  isSidebarOpen: boolean = false;
  hasStartedChat: boolean = false;
  messages: Message[] = [];

  onArrowClick(): void {
    if (this.inputValue) {
      this.hasStartedChat = true;
      this.messages.push({
        content: this.inputValue,
        isUser: true
      });
      this.inputValue = ''; // Reset input after sending
      // TODO: Add chat logic here
    }
  }

  toggleSidebar(): void {
    this.isSidebarOpen = !this.isSidebarOpen;
  }
}
