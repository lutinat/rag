import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

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
    private apiUrl = 'http://10.3.18.2:5000/api';

    constructor(private http: HttpClient) {}

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
}