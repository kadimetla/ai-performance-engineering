import React, { type ReactNode } from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { RooflineTab } from '../components/tabs/RooflineTab';

const mockUseApiQuery = jest.fn();

jest.mock('@/lib/useApi', () => ({
  useApiQuery: (...args: unknown[]) => mockUseApiQuery(...(args as any)),
  getErrorMessage: (err: any) => String(err),
}));

jest.mock('recharts', () => {
  const Wrapper = ({ children }: { children: ReactNode }) => <div>{children}</div>;
  const Component = ({ children }: { children: ReactNode }) => <div>{children}</div>;
  return {
    ResponsiveContainer: Wrapper,
    ScatterChart: Component,
    Scatter: Component,
    XAxis: Component,
    YAxis: Component,
    CartesianGrid: Component,
    Tooltip: Component,
    ReferenceLine: Component,
  };
});

describe('RooflineTab', () => {
  beforeEach(() => {
    mockUseApiQuery.mockReturnValue({
      data: {
        data: {
          peak_flops: 500,
          memory_bandwidth: 1000,
          kernels: [
            { name: 'kernelA', arithmetic_intensity: 2, performance: 150, efficiency: 0.6 },
            { name: 'kernelB', arithmetic_intensity: 4, performance: 300, efficiency: 0.8 },
          ],
        },
        hardware: { peak_flops: 500, memory_bandwidth: 1000 },
      },
      isLoading: false,
      error: null,
      mutate: jest.fn(),
      isValidating: false,
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('renders roofline stats and kernel list', () => {
    render(<RooflineTab />);

    expect(screen.getByText(/Roofline Model Analysis/)).toBeInTheDocument();
    expect(screen.getByText(/Peak Performance/)).toBeInTheDocument();
    expect(screen.getByText(/Kernels Analyzed/)).toBeInTheDocument();
    expect(screen.getByText(/kernelA/)).toBeInTheDocument();
    expect(screen.getByText(/kernelB/)).toBeInTheDocument();
  });

  it('shows error state when query fails', () => {
    mockUseApiQuery.mockReturnValueOnce({
      data: null,
      isLoading: false,
      error: new Error('roofline failed'),
      mutate: vi.fn(),
      isValidating: false,
    });

    render(<RooflineTab />);
    expect(screen.getByText(/roofline failed/)).toBeInTheDocument();
  });
});
