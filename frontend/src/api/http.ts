import axios from 'axios';
import { useAuthStore } from '@/stores/authStore';

/** Axios instance configured for the News Tracker API */
export const api = axios.create({
  baseURL: '/api',
  timeout: 30_000,
  headers: { 'Content-Type': 'application/json' },
});

// Request interceptor: auth + correlation ID
api.interceptors.request.use((config) => {
  const apiKey = useAuthStore.getState().apiKey;
  if (apiKey) {
    config.headers['X-API-Key'] = apiKey;
  }
  config.headers['x-request-id'] = crypto.randomUUID();
  return config;
});

// Response interceptor: error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (axios.isAxiosError(error)) {
      const status = error.response?.status;
      if (status === 401) {
        useAuthStore.getState().setShowAuthModal(true);
      }
    }
    return Promise.reject(error);
  },
);
