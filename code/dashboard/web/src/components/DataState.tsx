'use client';

import { AlertTriangle, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface LoadingStateProps {
  message?: string;
  className?: string;
  inline?: boolean;
}

export function LoadingState({ message = 'Loading...', className, inline }: LoadingStateProps) {
  return (
    <div
      className={cn(
        inline ? 'inline-flex items-center gap-2 text-sm text-white/70' : 'flex items-center justify-center gap-3 py-8 text-white/70',
        className
      )}
      role="status"
      aria-live="polite"
    >
      <Loader2 className={cn('animate-spin text-accent-primary', inline ? 'w-4 h-4' : 'w-6 h-6')} />
      <span>{message}</span>
    </div>
  );
}

interface ErrorStateProps {
  message: string;
  onRetry?: () => void;
  className?: string;
}

export function ErrorState({ message, onRetry, className }: ErrorStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center gap-3 rounded-lg border border-accent-warning/30 bg-accent-warning/10 px-4 py-6 text-center text-white/80',
        className
      )}
      role="alert"
    >
      <div className="flex items-center gap-2 text-accent-warning">
        <AlertTriangle className="w-5 h-5" />
        <span className="font-medium">Something went wrong</span>
      </div>
      <div className="text-sm text-white/70">{message}</div>
      {onRetry && (
        <button
          className="px-3 py-1.5 rounded-lg bg-white/10 text-sm text-white hover:bg-white/20 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-warning"
          onClick={onRetry}
        >
          Try again
        </button>
      )}
    </div>
  );
}

interface EmptyStateProps {
  title: string;
  description?: string;
  actionLabel?: string;
  onAction?: () => void;
  className?: string;
}

export function EmptyState({ title, description, actionLabel, onAction, className }: EmptyStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center gap-2 rounded-lg border border-white/10 bg-white/5 px-4 py-6 text-center text-white/70',
        className
      )}
      role="status"
      aria-live="polite"
    >
      <div className="text-white font-semibold">{title}</div>
      {description && <div className="text-sm text-white/60 max-w-md">{description}</div>}
      {actionLabel && onAction && (
        <button
          className="mt-2 rounded-lg bg-accent-primary/20 px-3 py-1.5 text-sm text-accent-primary hover:bg-accent-primary/30 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-primary"
          onClick={onAction}
        >
          {actionLabel}
        </button>
      )}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn('animate-pulse rounded-md bg-white/5', className)} aria-hidden="true" />;
}

// Skeleton loader for cards
export function SkeletonCard() {
  return (
    <div className="card p-5 animate-pulse">
      <div className="h-4 w-1/3 bg-white/10 rounded mb-4" />
      <div className="space-y-3">
        <div className="h-3 w-full bg-white/5 rounded" />
        <div className="h-3 w-2/3 bg-white/5 rounded" />
        <div className="h-3 w-4/5 bg-white/5 rounded" />
      </div>
    </div>
  );
}

// Skeleton loader for tables
export function SkeletonTable({ rows = 5 }: { rows?: number }) {
  return (
    <div className="card animate-pulse">
      <div className="card-header">
        <div className="h-5 w-32 bg-white/10 rounded" />
        <div className="h-5 w-24 bg-white/5 rounded" />
      </div>
      <div className="p-5 space-y-3">
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="flex items-center gap-4">
            <div className="h-4 w-1/4 bg-white/5 rounded" />
            <div className="h-4 w-1/3 bg-white/5 rounded" />
            <div className="h-4 w-1/5 bg-white/5 rounded" />
            <div className="h-4 w-16 bg-white/5 rounded ml-auto" />
          </div>
        ))}
      </div>
    </div>
  );
}

// Skeleton loader for charts
export function SkeletonChart() {
  return (
    <div className="card p-5 animate-pulse">
      <div className="h-5 w-40 bg-white/10 rounded mb-6" />
      <div className="h-64 bg-white/5 rounded-lg flex items-end justify-around p-4">
        {[0.6, 0.8, 0.4, 0.9, 0.5, 0.7, 0.3].map((h, i) => (
          <div
            key={i}
            className="w-8 bg-white/10 rounded-t"
            style={{ height: `${h * 100}%` }}
          />
        ))}
      </div>
    </div>
  );
}

// Skeleton loader for stats cards
export function SkeletonStats() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {Array.from({ length: 4 }).map((_, i) => (
        <div key={i} className="card p-5 animate-pulse">
          <div className="h-4 w-20 bg-white/10 rounded mb-2" />
          <div className="h-8 w-16 bg-white/5 rounded mb-1" />
          <div className="h-3 w-24 bg-white/5 rounded" />
        </div>
      ))}
    </div>
  );
}
