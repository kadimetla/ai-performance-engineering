'use client';

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

export interface ThemeColors {
  primary: string;
  secondary: string;
  bg: string;
  card: string;
}

export interface Theme {
  id: string;
  name: string;
  description?: string;
  colors: ThemeColors;
}

// Built-in themes
export const builtInThemes: Theme[] = [
  {
    id: 'cyberpunk',
    name: 'Cyberpunk',
    description: 'Neon cyan and purple on dark background',
    colors: {
      primary: '#00f5d4',
      secondary: '#9d4edd',
      bg: '#06060a',
      card: 'rgba(16, 16, 24, 0.9)',
    },
  },
  {
    id: 'midnight-blue',
    name: 'Midnight Blue',
    description: 'Deep blue with electric accents',
    colors: {
      primary: '#4cc9f0',
      secondary: '#7209b7',
      bg: '#0a1628',
      card: 'rgba(15, 30, 50, 0.9)',
    },
  },
  {
    id: 'forest',
    name: 'Forest',
    description: 'Natural greens with warm highlights',
    colors: {
      primary: '#00f5a0',
      secondary: '#ffc43d',
      bg: '#0a140a',
      card: 'rgba(10, 25, 15, 0.9)',
    },
  },
  {
    id: 'sunset',
    name: 'Sunset',
    description: 'Warm oranges and pinks',
    colors: {
      primary: '#ff6b6b',
      secondary: '#ffc43d',
      bg: '#1a0a0a',
      card: 'rgba(30, 15, 15, 0.9)',
    },
  },
  {
    id: 'monochrome',
    name: 'Monochrome',
    description: 'Clean grayscale aesthetic',
    colors: {
      primary: '#ffffff',
      secondary: '#888888',
      bg: '#0a0a0a',
      card: 'rgba(20, 20, 20, 0.9)',
    },
  },
  {
    id: 'aurora',
    name: 'Aurora',
    description: 'Northern lights inspired',
    colors: {
      primary: '#7ee8fa',
      secondary: '#80ff72',
      bg: '#050510',
      card: 'rgba(10, 15, 25, 0.9)',
    },
  },
];

interface ThemeContextValue {
  currentTheme: Theme;
  colorMode: 'dark' | 'light' | 'system';
  themes: Theme[];
  setTheme: (themeId: string) => void;
  setColorMode: (mode: 'dark' | 'light' | 'system') => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [currentThemeId, setCurrentThemeId] = useState('cyberpunk');
  const [colorMode, setColorMode] = useState<'dark' | 'light' | 'system'>('dark');
  const [themes, setThemes] = useState<Theme[]>(builtInThemes);

  // Load saved theme on mount
  useEffect(() => {
    try {
      const savedThemeId = localStorage.getItem('dashboard_theme');
      const savedColorMode = localStorage.getItem('dashboard_color_mode');
      if (savedThemeId) setCurrentThemeId(savedThemeId);
      if (savedColorMode) setColorMode(savedColorMode as 'dark' | 'light' | 'system');
    } catch {
      // Ignore localStorage errors
    }
  }, []);

  // Apply theme CSS variables whenever theme changes
  useEffect(() => {
    const theme = themes.find((t) => t.id === currentThemeId) || themes[0];
    const root = document.documentElement;

    root.style.setProperty('--accent-primary', theme.colors.primary);
    root.style.setProperty('--accent-secondary', theme.colors.secondary);
    root.style.setProperty('--bg-primary', theme.colors.bg);
    root.style.setProperty('--bg-card', theme.colors.card);

    // Also update body background
    document.body.style.backgroundColor = theme.colors.bg;

    // Save to localStorage
    try {
      localStorage.setItem('dashboard_theme', currentThemeId);
      localStorage.setItem('dashboard_color_mode', colorMode);
    } catch {
      // Ignore localStorage errors
    }
  }, [currentThemeId, colorMode, themes]);

  const setTheme = useCallback((themeId: string) => {
    setCurrentThemeId(themeId);
  }, []);

  const currentTheme = themes.find((t) => t.id === currentThemeId) || themes[0];

  return (
    <ThemeContext.Provider
      value={{
        currentTheme,
        colorMode,
        themes,
        setTheme,
        setColorMode,
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}

