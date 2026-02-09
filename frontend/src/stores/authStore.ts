import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

interface AuthState {
  apiKey: string | null;
  showAuthModal: boolean;
  setApiKey: (key: string) => void;
  clearApiKey: () => void;
  setShowAuthModal: (show: boolean) => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      apiKey: null,
      showAuthModal: false,
      setApiKey: (key: string) => set({ apiKey: key, showAuthModal: false }),
      clearApiKey: () => set({ apiKey: null }),
      setShowAuthModal: (show: boolean) => set({ showAuthModal: show }),
    }),
    {
      name: 'news-tracker-auth',
      storage: createJSONStorage(() => sessionStorage),
      partialize: (state) => ({ apiKey: state.apiKey }),
    },
  ),
);
