import { Component } from '@angular/core';
import { SidebarComponent } from '../sidebar/sidebar.component';
import { FormsModule } from '@angular/forms';
@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [SidebarComponent, FormsModule],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.scss'
})
export class ChatComponent {
  inputValue: string = '';

  onArrowClick(): void {
    if (this.inputValue) {
      console.log('Arrow clicked with input:', this.inputValue);
    }
  }
}
