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
    // Check API health on startup
    this.checkApiHealth();
  }

  inputValue: string = '';
  isSidebarOpen: boolean = false;
  hasStartedChat: boolean = false;
  isApiOnline: boolean = true;
  showHelpModal: boolean = false;
  animationKey: number = 0; // Used to retrigger animations

  chats: Chat[] = [
    { id: 1, title: 'New chat', messages: [], isWaitingForBot: false }
  ];
  selectedChatId: number = 1;
  chatIdCounter: number = 2;

  showEditModal = false;
  editingChat: ChatSidebarItem | null = null;

  // Example questions for the welcome page
  exampleQuestions = [
    {
      category: "GARAI-A Mission",
      question: "What is the GARAI-A satellite's mission and when was it launched?",
      icon: "fa-solid fa-satellite"
    },
    {
      category: "GEISAT Specifications",
      question: "What are the main technical specifications of GEISAT?",
      icon: "fa-solid fa-globe"
    },
    {
      category: "Camera Systems",
      question: "What are the main differences between ISIM-90 and ISIM-170 camera systems?",
      icon: "fa-solid fa-microchip"
    },
    {
      category: "UHR Processing",
      question: "What are the main steps of the Ultra-High Resolution (UHR) processing pipeline?",
      icon: "fa-solid fa-route"
    },
    {
      category: "Satlantis Origin",
      question: "What is the origin story of the Satlantis company?",
      icon: "fa-solid fa-building"
    },
    {
      category: "Technology & Innovation",
      question: "How does Satlantis ensure high image quality and calibration in its payloads?",
      icon: "fa-solid fa-database"
    }
  ];

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
    // Just navigate to welcome page, don't create a new chat yet
    this.inputValue = '';
    const wasAlreadyOnWelcome = !this.hasStartedChat;
    this.hasStartedChat = false;
    // Clear the selected chat so no chat appears highlighted in sidebar
    this.selectedChatId = -1; // Use -1 to indicate no selection
    
    // If already on welcome page, retrigger animations
    if (wasAlreadyOnWelcome) {
      this.retriggerAnimations();
    }
  }

  private retriggerAnimations() {
    // Temporarily disable animations to reset them
    this.animationKey++;
    
    // Use setTimeout to ensure DOM updates before re-enabling animations
    setTimeout(() => {
      this.animationKey++;
    }, 50);
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
    if (this.inputValue) {
      const userQuestion = this.inputValue;
      
      // If we're on the welcome page (hasStartedChat = false), handle chat creation/selection
      if (!this.hasStartedChat) {
        // Check if we have an empty "New chat" to reuse
        const emptyChat = this.chats.find(chat => 
          chat.title === 'New chat' && chat.messages.length === 0
        );
        
        if (emptyChat) {
          // Reuse the existing empty chat
          this.selectedChatId = emptyChat.id;
          if (!emptyChat.chatId) {
            emptyChat.chatId = this.apiService.generateChatId();
          }
        } else {
          // Create a new chat only if no empty "New chat" exists
          const newChat: Chat = {
            id: this.chatIdCounter++,
            title: 'New chat',
            messages: [],
            chatId: this.apiService.generateChatId(),
            isWaitingForBot: false
          };
          this.chats.unshift(newChat);
          this.selectedChatId = newChat.id;
        }
      }
      
      const targetChat = this.selectedChat; // Capture reference to ensure response goes to correct chat
      
      // Check if we have a valid chat and it's not waiting for a response
      if (!targetChat || targetChat.isWaitingForBot) {
        return;
      }
      
      this.hasStartedChat = true;
      targetChat.messages.push({
        content: this.sanitizeHtml(userQuestion),
        isUser: true
      });
      if (targetChat.messages.length === 1) {
        targetChat.title = userQuestion.slice(0, 30) + (userQuestion.length > 30 ? '...' : '');
      }
      this.scrollToBottom();
      this.inputValue = '';

      // Check if API is offline
      if (!this.isApiOnline) {
        targetChat.messages.push({
          content: this.sanitizeHtml('Error: Unable to get response from the API server. The API appears to be offline. Please try again later.'),
          isUser: false,
          sources: []
        });
        if (targetChat.id === this.selectedChatId) {
          this.scrollToBottom();
          setTimeout(() => this.chatInput?.nativeElement.focus(), 0);
        }
        return;
      }

      targetChat.isWaitingForBot = true;
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
          // Update API status to offline if error occurs
          this.isApiOnline = false;
          targetChat.messages.push({
            content: this.sanitizeHtml('Sorry, the API is currently offline. Please try again later.'),
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

  selectExample(question: string): void {
    this.inputValue = question;
    // Optionally auto-send the question
    setTimeout(() => {
      this.onArrowClick();
    }, 100);
  }

  checkApiHealth(): void {
    this.apiService.checkApiHealth().subscribe({
      next: (response) => {
        this.isApiOnline = true;
        console.log('API health check successful:', response);
      },
      error: (error) => {
        this.isApiOnline = false;
        console.error('API health check failed:', error);
      }
    });
  }

  toggleHelpModal(): void {
    this.showHelpModal = !this.showHelpModal;
  }
}
