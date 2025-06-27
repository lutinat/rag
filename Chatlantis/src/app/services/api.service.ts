import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

export interface ConversationTurn {
    user: string;
    assistant: string;
}

export interface QuestionRequest {
    question: string;
    chat_id: string;
    conversation_history?: ConversationTurn[];
    quantization?: string;
}

export interface ChatInfo {
    chat_id: string;
    created_at: number;
    last_activity: number;
    total_turns: number;
}

@Injectable({
    providedIn: 'root'
})
export class ApiService {
    private apiUrl: string;

    constructor(private http: HttpClient) {
        // Dynamically set API URL based on current host and environment config
        const currentHost = window.location.hostname;
        this.apiUrl = `http://${currentHost}:${environment.apiPort}/api`;
        console.log(`API service initialized with URL: ${this.apiUrl}`);
    }

    getAnswer(question: string, chatId: string, conversationHistory?: ConversationTurn[]): Observable<any> {
        const requestBody: QuestionRequest = {
            question,
            chat_id: chatId,
            conversation_history: conversationHistory || []
        };
        
        return this.http.post(`${this.apiUrl}/question`, requestBody);
    }

    getChatHistory(chatId: string, maxTurns: number = 10): Observable<any> {
        return this.http.get(`${this.apiUrl}/chat/history`, {
            params: {
                chat_id: chatId,
                max_turns: maxTurns.toString()
            }
        });
    }

    clearChatHistory(chatId: string): Observable<any> {
        return this.http.post(`${this.apiUrl}/chat/clear`, { chat_id: chatId });
    }

    getAllChats(): Observable<any> {
        return this.http.get(`${this.apiUrl}/chats/all`);
    }

    cleanupInactiveChats(): Observable<any> {
        return this.http.post(`${this.apiUrl}/chats/cleanup`, {});
    }

    // Utility method to generate a unique chat ID
    generateChatId(): string {
        return 'chat_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    // Utility method to get current API URL
    getCurrentApiUrl(): string {
        return this.apiUrl;
    }

    // Utility method to check API health
    checkApiHealth(): Observable<any> {
        return this.http.get(`${this.apiUrl}/health`);
    }
}