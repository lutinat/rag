import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
    providedIn: 'root'
})
export class ApiService {
    private apiUrl = 'http://10.3.18.2:5000/api';

    constructor(private http: HttpClient) {}

    getAnswer(question: string): Observable<any> {
        return this.http.post(`${this.apiUrl}/question`, { question });
    }
}