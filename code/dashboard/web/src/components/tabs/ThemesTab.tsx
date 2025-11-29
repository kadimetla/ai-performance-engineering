'use client';

import { Palette, Check, Moon, Sun, Monitor, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useTheme } from '@/lib/ThemeContext';
import { useToast } from '@/components/Toast';

export function ThemesTab() {
  const { currentTheme, colorMode, themes, setTheme, setColorMode } = useTheme();
  const { showToast } = useToast();

  const handleThemeChange = (themeId: string) => {
    setTheme(themeId);
    const theme = themes.find((t) => t.id === themeId);
    showToast(`Theme changed to ${theme?.name || themeId}`, 'success');
  };

  const handleColorModeChange = (mode: 'dark' | 'light' | 'system') => {
    setColorMode(mode);
    showToast(`Color mode set to ${mode}`, 'info');
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Palette className="w-5 h-5 text-accent-secondary" />
            <h2 className="text-lg font-semibold text-white">Theme Settings</h2>
          </div>
          <div className="flex items-center gap-2 text-sm text-white/60">
            <Sparkles className="w-4 h-4" />
            <span>Changes apply instantly</span>
          </div>
        </div>
        <div className="card-body">
          {/* Color mode toggle */}
          <div className="mb-6">
            <h3 className="text-sm text-white/50 mb-3">Color Mode</h3>
            <div className="flex gap-2">
              {[
                { id: 'dark', label: 'Dark', icon: Moon },
                { id: 'light', label: 'Light', icon: Sun },
                { id: 'system', label: 'System', icon: Monitor },
              ].map((mode) => {
                const Icon = mode.icon;
                return (
                  <button
                    key={mode.id}
                    onClick={() => handleColorModeChange(mode.id as typeof colorMode)}
                    className={cn(
                      'flex items-center gap-2 px-4 py-2 rounded-lg transition-all',
                      colorMode === mode.id
                        ? 'bg-accent-primary/20 text-accent-primary border border-accent-primary/30'
                        : 'bg-white/5 text-white/60 hover:text-white hover:bg-white/10'
                    )}
                  >
                    <Icon className="w-4 h-4" />
                    {mode.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Current theme indicator */}
          <div className="p-4 rounded-lg bg-accent-primary/10 border border-accent-primary/30 mb-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-white/60">Current Theme</div>
                <div className="text-xl font-bold text-white">{currentTheme.name}</div>
              </div>
              <div className="flex gap-2">
                <div
                  className="w-8 h-8 rounded-lg"
                  style={{ backgroundColor: currentTheme.colors.primary }}
                  title="Primary"
                />
                <div
                  className="w-8 h-8 rounded-lg"
                  style={{ backgroundColor: currentTheme.colors.secondary }}
                  title="Secondary"
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Theme grid */}
      <div className="card">
        <div className="card-header">
          <h3 className="font-medium text-white">Available Themes</h3>
          <span className="text-sm text-white/50">{themes.length} themes</span>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {themes.map((theme) => (
              <button
                key={theme.id}
                onClick={() => handleThemeChange(theme.id)}
                className={cn(
                  'relative p-4 rounded-xl border transition-all text-left group',
                  currentTheme.id === theme.id
                    ? 'border-accent-primary bg-accent-primary/10 ring-2 ring-accent-primary/20'
                    : 'border-white/10 bg-white/5 hover:bg-white/10 hover:border-white/20'
                )}
              >
                {currentTheme.id === theme.id && (
                  <div className="absolute top-3 right-3 w-6 h-6 bg-accent-primary rounded-full flex items-center justify-center">
                    <Check className="w-4 h-4 text-black" />
                  </div>
                )}

                {/* Color preview */}
                <div className="flex gap-2 mb-3">
                  <div
                    className="w-8 h-8 rounded-lg shadow-lg transition-transform group-hover:scale-110"
                    style={{ backgroundColor: theme.colors.primary }}
                    title="Primary"
                  />
                  <div
                    className="w-8 h-8 rounded-lg shadow-lg transition-transform group-hover:scale-110"
                    style={{ backgroundColor: theme.colors.secondary }}
                    title="Secondary"
                  />
                  <div
                    className="w-8 h-8 rounded-lg border border-white/20 transition-transform group-hover:scale-110"
                    style={{ backgroundColor: theme.colors.bg }}
                    title="Background"
                  />
                  <div
                    className="w-8 h-8 rounded-lg border border-white/20 transition-transform group-hover:scale-110"
                    style={{ backgroundColor: theme.colors.card }}
                    title="Card"
                  />
                </div>

                <h4 className="font-medium text-white mb-1">{theme.name}</h4>
                {theme.description && (
                  <p className="text-sm text-white/50">{theme.description}</p>
                )}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Live Preview */}
      <div className="card">
        <div className="card-header">
          <h3 className="font-medium text-white">Live Preview</h3>
        </div>
        <div
          className="p-6 rounded-b-xl transition-colors"
          style={{ backgroundColor: currentTheme.colors.bg }}
        >
          <div
            className="p-6 rounded-xl border border-white/10 transition-colors"
            style={{ backgroundColor: currentTheme.colors.card }}
          >
            <div className="flex items-center gap-4 mb-4">
              <div
                className="w-12 h-12 rounded-lg transition-all"
                style={{
                  background: `linear-gradient(135deg, ${currentTheme.colors.primary}, ${currentTheme.colors.secondary})`,
                }}
              />
              <div>
                <h4
                  className="text-lg font-bold transition-colors"
                  style={{ color: currentTheme.colors.primary }}
                >
                  Sample Card Title
                </h4>
                <p className="text-sm" style={{ color: 'rgba(255,255,255,0.5)' }}>
                  This is how cards will look with the current theme
                </p>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-3 mb-4">
              <div className="p-3 rounded-lg bg-white/5">
                <div className="text-sm text-white/50">Metric 1</div>
                <div className="text-xl font-bold" style={{ color: currentTheme.colors.primary }}>
                  2.5x
                </div>
              </div>
              <div className="p-3 rounded-lg bg-white/5">
                <div className="text-sm text-white/50">Metric 2</div>
                <div className="text-xl font-bold" style={{ color: currentTheme.colors.secondary }}>
                  156ms
                </div>
              </div>
              <div className="p-3 rounded-lg bg-white/5">
                <div className="text-sm text-white/50">Metric 3</div>
                <div className="text-xl font-bold text-white">95%</div>
              </div>
            </div>

            <div className="flex gap-2">
              <button
                className="px-4 py-2 rounded-lg font-medium transition-all hover:opacity-90"
                style={{
                  backgroundColor: currentTheme.colors.primary,
                  color: '#000',
                }}
              >
                Primary Button
              </button>
              <button
                className="px-4 py-2 rounded-lg font-medium transition-all hover:opacity-90"
                style={{
                  backgroundColor: `${currentTheme.colors.secondary}30`,
                  color: currentTheme.colors.secondary,
                }}
              >
                Secondary Button
              </button>
              <button className="px-4 py-2 rounded-lg font-medium bg-white/10 text-white/80 hover:bg-white/20 transition-all">
                Ghost Button
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Theme info */}
      <div className="card">
        <div className="card-header">
          <h3 className="font-medium text-white">Theme Details</h3>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 bg-white/5 rounded-lg">
              <div className="text-sm text-white/50 mb-2">Primary Color</div>
              <div className="flex items-center gap-2">
                <div
                  className="w-6 h-6 rounded"
                  style={{ backgroundColor: currentTheme.colors.primary }}
                />
                <code className="text-sm text-white font-mono">{currentTheme.colors.primary}</code>
              </div>
            </div>
            <div className="p-4 bg-white/5 rounded-lg">
              <div className="text-sm text-white/50 mb-2">Secondary Color</div>
              <div className="flex items-center gap-2">
                <div
                  className="w-6 h-6 rounded"
                  style={{ backgroundColor: currentTheme.colors.secondary }}
                />
                <code className="text-sm text-white font-mono">{currentTheme.colors.secondary}</code>
              </div>
            </div>
            <div className="p-4 bg-white/5 rounded-lg">
              <div className="text-sm text-white/50 mb-2">Background</div>
              <div className="flex items-center gap-2">
                <div
                  className="w-6 h-6 rounded border border-white/20"
                  style={{ backgroundColor: currentTheme.colors.bg }}
                />
                <code className="text-sm text-white font-mono">{currentTheme.colors.bg}</code>
              </div>
            </div>
            <div className="p-4 bg-white/5 rounded-lg">
              <div className="text-sm text-white/50 mb-2">Card</div>
              <div className="flex items-center gap-2">
                <div
                  className="w-6 h-6 rounded border border-white/20"
                  style={{ backgroundColor: currentTheme.colors.card }}
                />
                <code className="text-sm text-white font-mono text-xs">{currentTheme.colors.card.slice(0, 20)}...</code>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
