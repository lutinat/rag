import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

export interface ChatSidebarItem {
  id: number;
  title: string;
}

@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './sidebar.component.html',
  styleUrl: './sidebar.component.scss'
})
export class SidebarComponent {
  @Input() chats: ChatSidebarItem[] = [];
  @Input() selectedChatId: number | null = null;
  @Output() newChat = new EventEmitter<void>();
  @Output() selectChat = new EventEmitter<number>();
  @Output() deleteChat = new EventEmitter<number>();
  @Output() renameChat = new EventEmitter<{id: number, newTitle: string}>();
  @Output() editChat = new EventEmitter<{event: Event, chat: ChatSidebarItem}>();

  onNewChat() {
    this.newChat.emit();
  }

  onSelectChat(id: number) {
    this.selectChat.emit(id);
  }

  onDeleteChat(event: Event, id: number) {
    event.preventDefault();
    event.stopPropagation();
    this.deleteChat.emit(id);
  }

  startEditing(event: Event, chat: ChatSidebarItem) {
    event.preventDefault();
    event.stopPropagation();
    this.editChat.emit({ event, chat });
  }
}
