import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

interface UIState {
  sidebarOpen: boolean;
  darkMode: boolean;
  toggleSidebar: () => void;
  toggleDarkMode: () => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      sidebarOpen: true,
      darkMode: true,
      toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
      toggleDarkMode: () =>
        set((s) => {
          const next = !s.darkMode;
          document.documentElement.classList.toggle('dark', next);
          return { darkMode: next };
        }),
    }),
    {
      name: 'news-tracker-ui',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({ darkMode: state.darkMode, sidebarOpen: state.sidebarOpen }),
    },
  ),
);
