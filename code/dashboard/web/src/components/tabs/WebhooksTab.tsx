'use client';

import { useState, useEffect, useCallback } from 'react';
import { Bell, Plus, Trash2, TestTube, CheckCircle, XCircle, Loader2, Send, RefreshCw, Save } from 'lucide-react';
import { getWebhooks, saveWebhooks, testWebhook, sendWebhookNotification } from '@/lib/api';
import { useToast } from '@/components/Toast';

interface Webhook {
  id: string;
  name: string;
  url: string;
  events: string[];
  enabled: boolean;
  platform?: string;
}

export function WebhooksTab() {
  const [webhooks, setWebhooks] = useState<Webhook[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newName, setNewName] = useState('');
  const [newUrl, setNewUrl] = useState('');
  const [newPlatform, setNewPlatform] = useState('slack');
  const [newEvents, setNewEvents] = useState<string[]>(['optimization_complete']);
  const [testing, setTesting] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<{ id: string; success: boolean } | null>(null);
  const [hasChanges, setHasChanges] = useState(false);
  const { showToast } = useToast();

  const eventTypes = [
    { id: 'optimization_complete', label: 'Optimization Complete' },
    { id: 'regression_detected', label: 'Regression Detected' },
    { id: 'benchmark_failed', label: 'Benchmark Failed' },
    { id: 'daily_summary', label: 'Daily Summary' },
  ];

  // Load webhooks from backend
  const loadWebhooks = useCallback(async () => {
    try {
      setLoading(true);
      const data = await getWebhooks();
      const webhookList = (data as any)?.webhooks || [];
      setWebhooks(webhookList);
      setHasChanges(false);
    } catch (e) {
      showToast('Failed to load webhooks', 'error');
    } finally {
      setLoading(false);
    }
  }, [showToast]);

  useEffect(() => {
    loadWebhooks();
  }, [loadWebhooks]);

  // Save webhooks to backend
  const handleSave = async () => {
    try {
      setSaving(true);
      await saveWebhooks(webhooks);
      setHasChanges(false);
      showToast('Webhooks saved successfully', 'success');
    } catch (e) {
      showToast('Failed to save webhooks', 'error');
    } finally {
      setSaving(false);
    }
  };

  function addWebhook() {
    if (!newName || !newUrl) return;

    const webhook: Webhook = {
      id: Date.now().toString(),
      name: newName,
      url: newUrl,
      events: newEvents.length ? newEvents : ['optimization_complete'],
      enabled: true,
      platform: newPlatform,
    };

    setWebhooks([...webhooks, webhook]);
    setNewName('');
    setNewUrl('');
    setNewEvents(['optimization_complete']);
    setNewPlatform('slack');
    setShowAddForm(false);
    setHasChanges(true);
    showToast('Webhook added - remember to save!', 'info');
  }

  function deleteWebhook(id: string) {
    setWebhooks(webhooks.filter((w) => w.id !== id));
    setHasChanges(true);
  }

  function toggleWebhook(id: string) {
    setWebhooks(
      webhooks.map((w) => (w.id === id ? { ...w, enabled: !w.enabled } : w))
    );
    setHasChanges(true);
  }

  async function handleTestWebhook(id: string) {
    setTesting(id);
    setTestResult(null);

    const webhook = webhooks.find((w) => w.id === id);
    try {
      if (!webhook) throw new Error('Webhook not found');
      const res = await testWebhook({
        name: webhook.name,
        url: webhook.url,
        events: webhook.events,
        platform: webhook.platform,
      });
      const success = (res as any).success !== false;
      setTestResult({ id, success });
      showToast(success ? 'Test successful!' : 'Test failed', success ? 'success' : 'error');
    } catch (e) {
      setTestResult({ id, success: false });
      showToast('Test failed', 'error');
    } finally {
      setTesting(null);
    }
  }

  async function sendReport(id: string) {
    setTesting(id);
    setTestResult(null);
    const webhook = webhooks.find((w) => w.id === id);
    try {
      if (!webhook) throw new Error('Webhook not found');
      const res = await sendWebhookNotification({
        url: webhook.url,
        type: webhook.platform || 'slack',
        message_type: 'summary',
      });
      const success = (res as any).success !== false;
      setTestResult({ id, success });
      showToast(success ? 'Summary sent!' : 'Failed to send', success ? 'success' : 'error');
    } catch (e) {
      setTestResult({ id, success: false });
      showToast('Failed to send', 'error');
    } finally {
      setTesting(null);
    }
  }

  if (loading) {
    return (
      <div className="card">
        <div className="card-body flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 animate-spin text-accent-info" />
          <span className="ml-3 text-white/50">Loading webhooks...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Bell className="w-5 h-5 text-accent-info" />
            <h2 className="text-lg font-semibold text-white">Webhooks & Notifications</h2>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={loadWebhooks}
              className="p-2 hover:bg-white/5 rounded-lg text-white/50 hover:text-white"
              title="Refresh"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            {hasChanges && (
              <button
                onClick={handleSave}
                disabled={saving}
                className="flex items-center gap-2 px-4 py-2 bg-accent-success/20 text-accent-success rounded-lg hover:bg-accent-success/30 disabled:opacity-50"
              >
                {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                Save Changes
              </button>
            )}
            <button
              onClick={() => setShowAddForm(true)}
              className="flex items-center gap-2 px-4 py-2 bg-accent-primary/20 text-accent-primary rounded-lg hover:bg-accent-primary/30"
            >
              <Plus className="w-4 h-4" />
              Add Webhook
            </button>
          </div>
        </div>
        {hasChanges && (
          <div className="px-5 py-2 bg-accent-warning/10 border-t border-accent-warning/20 text-sm text-accent-warning">
            ⚠️ You have unsaved changes. Click "Save Changes" to persist them.
          </div>
        )}
      </div>

      {/* Add form */}
      {showAddForm && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">Add New Webhook</h3>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-sm text-white/50 mb-2">Name</label>
                <input
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="e.g., Slack Notifications"
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-white/30 focus:outline-none focus:border-accent-primary/50"
                />
              </div>
              <div>
                <label className="block text-sm text-white/50 mb-2">Webhook URL</label>
                <input
                  type="url"
                  value={newUrl}
                  onChange={(e) => setNewUrl(e.target.value)}
                  placeholder="https://..."
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-white/30 focus:outline-none focus:border-accent-primary/50"
                />
              </div>
              <div>
                <label className="block text-sm text-white/50 mb-2">Platform</label>
                <select
                  value={newPlatform}
                  onChange={(e) => setNewPlatform(e.target.value)}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-accent-primary/50"
                >
                  <option value="slack" className="bg-brand-bg">Slack</option>
                  <option value="teams" className="bg-brand-bg">Microsoft Teams</option>
                  <option value="discord" className="bg-brand-bg">Discord</option>
                  <option value="custom" className="bg-brand-bg">Custom (JSON)</option>
                </select>
              </div>
            </div>
            <div className="mb-4">
              <div className="block text-sm text-white/50 mb-2">Events</div>
              <div className="flex flex-wrap gap-2">
                {eventTypes.map((event) => {
                  const checked = newEvents.includes(event.id);
                  return (
                    <label
                      key={event.id}
                      className={`flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-colors ${
                        checked ? 'bg-accent-primary/20 border border-accent-primary/30' : 'bg-white/5 border border-white/10 hover:bg-white/10'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => {
                          if (checked) {
                            setNewEvents(newEvents.filter((e) => e !== event.id));
                          } else {
                            setNewEvents([...newEvents, event.id]);
                          }
                        }}
                        className="w-4 h-4 accent-accent-primary"
                      />
                      <span className="text-sm text-white/80">{event.label}</span>
                    </label>
                  );
                })}
              </div>
            </div>
            <div className="flex gap-2">
              <button
                onClick={addWebhook}
                disabled={!newName || !newUrl}
                className="px-4 py-2 bg-accent-primary text-black rounded-lg font-medium disabled:opacity-50 hover:opacity-90 transition-opacity"
              >
                Add Webhook
              </button>
              <button
                onClick={() => setShowAddForm(false)}
                className="px-4 py-2 bg-white/5 text-white rounded-lg hover:bg-white/10 transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Webhooks list */}
      <div className="card">
        <div className="card-header">
          <h3 className="font-medium text-white">Configured Webhooks</h3>
          <span className="text-sm text-white/50">{webhooks.length} webhook{webhooks.length !== 1 ? 's' : ''}</span>
        </div>
        <div className="card-body">
          {webhooks.length === 0 ? (
            <div className="text-center py-8">
              <Bell className="w-12 h-12 text-white/20 mx-auto mb-3" />
              <p className="text-white/50 mb-2">No webhooks configured</p>
              <p className="text-sm text-white/30">Add a webhook to receive notifications about benchmark results and optimizations.</p>
            </div>
          ) : (
            <div className="space-y-4">
              {webhooks.map((webhook) => (
                <div
                  key={webhook.id}
                  className={`p-4 rounded-xl border transition-all ${
                    webhook.enabled
                      ? 'bg-white/5 border-white/10'
                      : 'bg-white/[0.02] border-white/5 opacity-60'
                  }`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1 min-w-0">
                      <h4 className="font-medium text-white mb-1">{webhook.name}</h4>
                      <p className="text-sm text-white/50 font-mono truncate">
                        {webhook.url}
                      </p>
                      {webhook.platform && (
                        <span className="inline-flex items-center px-2 py-0.5 bg-accent-secondary/20 text-accent-secondary text-xs rounded mt-2">
                          {webhook.platform}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2 ml-4">
                      <button
                        onClick={() => handleTestWebhook(webhook.id)}
                        disabled={testing === webhook.id}
                        className="p-2 hover:bg-white/5 rounded-lg transition-colors"
                        title="Test webhook"
                      >
                        {testing === webhook.id ? (
                          <Loader2 className="w-4 h-4 animate-spin text-accent-info" />
                        ) : testResult?.id === webhook.id ? (
                          testResult.success ? (
                            <CheckCircle className="w-4 h-4 text-accent-success" />
                          ) : (
                            <XCircle className="w-4 h-4 text-accent-danger" />
                          )
                        ) : (
                          <TestTube className="w-4 h-4 text-white/50 hover:text-white" />
                        )}
                      </button>
                      <button
                        onClick={() => sendReport(webhook.id)}
                        disabled={testing === webhook.id}
                        className="p-2 hover:bg-white/5 rounded-lg transition-colors"
                        title="Send performance summary"
                      >
                        <Send className="w-4 h-4 text-accent-primary" />
                      </button>
                      <button
                        onClick={() => deleteWebhook(webhook.id)}
                        className="p-2 hover:bg-accent-danger/10 rounded-lg text-accent-danger transition-colors"
                        title="Delete webhook"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => toggleWebhook(webhook.id)}
                        className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                          webhook.enabled
                            ? 'bg-accent-success/20 text-accent-success'
                            : 'bg-white/10 text-white/50'
                        }`}
                      >
                        {webhook.enabled ? 'Enabled' : 'Disabled'}
                      </button>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {webhook.events.map((event) => (
                      <span
                        key={event}
                        className="px-2 py-1 bg-accent-info/10 text-accent-info text-xs rounded-lg"
                      >
                        {eventTypes.find((e) => e.id === event)?.label || event}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Event types reference */}
      <div className="card">
        <div className="card-header">
          <h3 className="font-medium text-white">Available Event Types</h3>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {eventTypes.map((event) => (
              <div key={event.id} className="p-3 bg-white/5 rounded-lg border border-white/5">
                <span className="font-medium text-white">{event.label}</span>
                <p className="text-sm text-white/40 font-mono mt-1">{event.id}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
