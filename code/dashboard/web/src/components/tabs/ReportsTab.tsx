import { useState } from 'react';
import {
  exportCSV,
  exportCSVDetailed,
  exportPDF,
  exportHTML,
  exportGeneric,
  compareRuns as compareRunsApi,
  generateLaunchPlan as generateLaunchPlanApi,
  getRooflineSweep,
} from '@/lib/api';
import { useToast } from '@/components/Toast';

export function ReportsTab() {
  const { showToast } = useToast();
  const [compareInputs, setCompareInputs] = useState({ baseline: 'benchmark_test_results.json', candidate: 'benchmark_test_results.json', top: 5 });
  const [compareResult, setCompareResult] = useState<{ regressions: any[]; improvements: any[] } | null>(null);
  const [launchPlanInputs, setLaunchPlanInputs] = useState({
    model_params: 70,
    nodes: 1,
    gpus: 8,
    tp: 1,
    pp: 1,
    dp: 1,
    batch_size: 1,
    script: 'train.py',
    extra_args: '',
  });
  const [launchPlan, setLaunchPlan] = useState<{ command: string; plan: any } | null>(null);
  const [rooflineSweep, setRooflineSweep] = useState<{ size_mb: number; rows: { stride: number; bandwidth_gbps: number }[] }>({ size_mb: 32, rows: [] });

  const downloadBlob = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleExport = async (type: 'csv' | 'csv-detailed' | 'pdf' | 'html' | 'md' | 'json') => {
    try {
      let blob: Blob;
      let name = `benchmarks_${new Date().toISOString().split('T')[0]}`;
      if (type === 'csv') {
        blob = await exportCSV();
        name += '.csv';
      } else if (type === 'csv-detailed') {
        blob = await exportCSVDetailed();
        name = `benchmarks_detailed_${new Date().toISOString().split('T')[0]}.csv`;
      } else if (type === 'pdf') {
        blob = await exportPDF();
        name = `report_${new Date().toISOString().split('T')[0]}.pdf`;
      } else if (type === 'html') {
        blob = await exportHTML();
        name = `report_${new Date().toISOString().split('T')[0]}.html`;
      } else {
        const fmt = type === 'md' ? 'markdown' : 'json';
        const generic = await exportGeneric(fmt as any);
        const payload = typeof generic.payload === 'string' ? generic.payload : JSON.stringify(generic.payload, null, 2);
        blob = new Blob([payload], { type: type === 'md' ? 'text/markdown' : 'application/json' });
        name = `benchmarks_${new Date().toISOString().split('T')[0]}.${type === 'md' ? 'md' : 'json'}`;
      }
      downloadBlob(blob, name);
      showToast(`Exported ${name}`, 'success');
    } catch (e) {
      showToast('Export failed', 'error');
    }
  };

  const handleCompare = async () => {
    try {
      const res = await compareRunsApi(compareInputs);
      setCompareResult(res as any);
      showToast('Compared runs', 'success');
    } catch (e) {
      showToast('Compare failed', 'error');
    }
  };

  const handleLaunchPlan = async () => {
    try {
      const res = await generateLaunchPlanApi(launchPlanInputs as any);
      setLaunchPlan(res as any);
      showToast('Launch plan ready', 'success');
    } catch (e) {
      showToast('Launch plan failed', 'error');
    }
  };

  const handleRoofline = async () => {
    try {
      const res = await getRooflineSweep(rooflineSweep.size_mb);
      const data = res as any;
      setRooflineSweep(data);
      const hasData = Array.isArray(data?.rows) && data.rows.some((r: any) => (r.bandwidth_gbps ?? 0) > 0);
      showToast(hasData ? 'Roofline sweep updated' : 'Roofline sweep returned no data', hasData ? 'success' : 'warning');
    } catch (e) {
      showToast('Roofline sweep failed', 'error');
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <Card title="Exports & Reports" subtitle="One-click downloads">
          <div className="flex flex-wrap gap-2">
            <Button onClick={() => handleExport('csv')}>CSV</Button>
            <Button onClick={() => handleExport('csv-detailed')}>CSV (Detailed)</Button>
            <Button onClick={() => handleExport('pdf')}>PDF</Button>
            <Button onClick={() => handleExport('html')}>HTML</Button>
            <Button onClick={() => handleExport('md')}>Markdown</Button>
            <Button onClick={() => handleExport('json')}>JSON</Button>
          </div>
        </Card>

        <Card title="Compare Runs" subtitle="Baseline vs candidate JSON">
          <div className="space-y-2 text-sm">
            <input
              className="w-full rounded bg-white/10 px-3 py-2 text-white"
              value={compareInputs.baseline}
              onChange={(e) => setCompareInputs((p) => ({ ...p, baseline: e.target.value }))}
              placeholder="baseline benchmark_test_results.json"
            />
            <input
              className="w-full rounded bg-white/10 px-3 py-2 text-white"
              value={compareInputs.candidate}
              onChange={(e) => setCompareInputs((p) => ({ ...p, candidate: e.target.value }))}
              placeholder="candidate benchmark_test_results.json"
            />
            <div className="flex items-center gap-2">
              <label className="text-xs text-white/60">Top</label>
              <input
                type="number"
                className="w-16 rounded bg-white/10 px-2 py-1 text-white"
                value={compareInputs.top}
                onChange={(e) => setCompareInputs((p) => ({ ...p, top: Number(e.target.value) || 0 }))}
              />
              <Button className="ml-auto" onClick={handleCompare}>
                Diff
              </Button>
            </div>
            {compareResult && (
              <div className="space-y-2">
                <ChartList title="Regressions" colorClass="bg-accent-warning" data={compareResult.regressions || []} />
                <ChartList title="Improvements" colorClass="bg-accent-success" data={compareResult.improvements || []} />
              </div>
            )}
          </div>
        </Card>

        <Card title="Launch Plan" subtitle="Torchrun layout (TP/PP/DP)">
          <div className="grid grid-cols-2 gap-2 text-xs">
            {(["model_params","nodes","gpus","tp","pp","dp","batch_size"] as const).map((key) => (
              <div key={key} className="flex flex-col gap-1">
                <label className="text-white/60">{key.replace('_',' ').toUpperCase()}</label>
                <input
                  type="number"
                  className="rounded bg-white/10 px-2 py-1 text-white text-sm"
                  value={launchPlanInputs[key]}
                  onChange={(e) => setLaunchPlanInputs((p) => ({ ...p, [key]: Number(e.target.value) || 0 }))}
                />
              </div>
            ))}
            <div className="col-span-2 flex flex-col gap-1">
              <label className="text-white/60">Script</label>
              <input
                className="rounded bg-white/10 px-2 py-1 text-white text-sm"
                value={launchPlanInputs.script}
                onChange={(e) => setLaunchPlanInputs((p) => ({ ...p, script: e.target.value }))}
              />
            </div>
            <div className="col-span-2 flex flex-col gap-1">
              <label className="text-white/60">Extra Args</label>
              <input
                className="rounded bg-white/10 px-2 py-1 text-white text-sm"
                value={launchPlanInputs.extra_args}
                onChange={(e) => setLaunchPlanInputs((p) => ({ ...p, extra_args: e.target.value }))}
              />
            </div>
            <Button className="col-span-2" onClick={handleLaunchPlan}>
              Build Command
            </Button>
            {launchPlan && (
              <div className="col-span-2 text-white/80 text-xs bg-black/40 rounded p-2">
                <div className="font-semibold mb-1">Command</div>
                <code className="whitespace-pre-wrap break-words">{launchPlan.command}</code>
              </div>
            )}
          </div>
        </Card>
      </div>

      <Card title="Memory Roofline Sweep" subtitle="Stride sweep bandwidth">
        <div className="flex items-center gap-2 mb-3">
          <label className="text-xs text-white/60">Size MB</label>
          <input
            type="number"
            className="w-20 rounded bg-white/10 px-2 py-1 text-white text-sm"
            value={rooflineSweep.size_mb}
            onChange={(e) => setRooflineSweep((p) => ({ ...p, size_mb: Number(e.target.value) || 0 }))}
          />
          <Button onClick={handleRoofline}>Sweep</Button>
        </div>
        <div className="text-xs text-white/80 space-y-2">
          {rooflineSweep.rows.map((row, idx) => (
            <div key={idx} className="flex items-center gap-2">
              <span className="w-16 text-white/60">{row.stride}B</span>
              <div className="flex-1 h-2 bg-white/5 rounded">
                <div
                  className="h-2 bg-accent-primary rounded"
                  style={{
                    width: `${Math.min(100, (row.bandwidth_gbps || 0) * 3)}%`,
                  }}
                  title={row.bandwidth_gbps ? `${row.bandwidth_gbps.toFixed(3)} GB/s` : 'No data - try larger size'}
                />
              </div>
              <span className="w-20 text-right">{row.bandwidth_gbps?.toFixed(3) ?? '0.000'} GB/s</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function ChartList({
  title,
  colorClass,
  data,
}: {
  title: string;
  colorClass: string;
  data: { name: string; delta: number; baseline?: number; candidate?: number }[];
}) {
  const maxAbs = Math.max(...data.map((d) => Math.abs(d.delta || 0)), 1);
  return (
    <div className="text-xs text-white/80 space-y-1 max-h-40 overflow-y-auto">
      <div className="font-semibold">{title}</div>
      {data.length === 0 && <div className="text-white/50">None</div>}
      {data.map((r, idx) => {
        const width = Math.min(100, (Math.abs(r.delta || 0) / maxAbs) * 100);
        return (
          <div key={idx} className="flex items-center gap-2">
            <span className="flex-1 truncate" title={r.name}>
              {r.name}
            </span>
            <div className="w-32 h-2 bg-white/5 rounded">
              <div className={`h-2 ${colorClass} rounded`} style={{ width: `${width}%` }} />
            </div>
            <span className="w-28 text-right">
              {r.baseline?.toFixed?.(2)}x → {r.candidate?.toFixed?.(2)}x
            </span>
          </div>
        );
      })}
    </div>
  );
}

function Card({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <div className="p-4 rounded-lg border border-white/10 bg-white/5 space-y-2">
      <div>
        <div className="text-white font-semibold">{title}</div>
        {subtitle && <div className="text-xs text-white/60">{subtitle}</div>}
      </div>
      {children}
    </div>
  );
}

function Button({ children, className, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={`px-3 py-1.5 bg-accent-primary/20 text-accent-primary rounded text-sm hover:bg-accent-primary/30 transition ${className || ''}`}
      {...props}
    >
      {children}
    </button>
  );
}
