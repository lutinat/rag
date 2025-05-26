import { Component, EventEmitter, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-edit-chat-modal',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="fixed inset-0 backdrop-blur-sm bg-white/30 flex items-center justify-center z-[9999]" (click)="onCancel()">
      <div class="bg-white p-6 rounded-lg shadow-xl w-96" (click)="$event.stopPropagation()">
        <h2 class="text-xl font-bold mb-4 text-satlantis-blue">Edit Chat Title</h2>
        <input 
          type="text" 
          [(ngModel)]="title"
          (keydown.enter)="onSave()"
          (keydown.escape)="onCancel()"
          class="w-full p-2 border rounded mb-4"
          autofocus
        >
        <div class="flex justify-end gap-2">
          <button 
            (click)="onCancel()"
            class="px-4 py-2 text-gray-600 hover:text-gray-800 cursor-pointer"
          >
            Cancel
          </button>
          <button 
            (click)="onSave()"
            class="px-4 py-2 bg-satlantis-blue text-white rounded hover:bg-satlantis-blue/80 cursor-pointer"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  `
})
export class EditChatModalComponent {
  @Input() chatId: number = 0;
  @Input() initialTitle: string = '';
  @Output() save = new EventEmitter<{id: number, newTitle: string}>();
  @Output() cancel = new EventEmitter<void>();

  title: string = '';

  ngOnInit() {
    this.title = this.initialTitle;
  }

  onSave() {
    if (this.title.trim()) {
      this.save.emit({
        id: this.chatId,
        newTitle: this.title.trim()
      });
    }
  }

  onCancel() {
    this.cancel.emit();
  }
} 