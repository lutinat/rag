import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

export interface ChatSidebarItem {
  id: number;
  title: string;
}

@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './sidebar.component.html',
  styleUrl: './sidebar.component.scss'
})
export class SidebarComponent {
  @Input() chats: ChatSidebarItem[] = [];
  @Input() selectedChatId: number | null = null;
  @Output() newChat = new EventEmitter<void>();
  @Output() selectChat = new EventEmitter<number>();
  @Output() deleteChat = new EventEmitter<number>();

  onNewChat() {
    this.newChat.emit();
  }

  onSelectChat(id: number) {
    this.selectChat.emit(id);
  }

  onDeleteChat(event: Event, id: number) {
    event.stopPropagation();
    this.deleteChat.emit(id);
  }
}
