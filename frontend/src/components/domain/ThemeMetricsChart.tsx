import {
  ComposedChart,
  Area,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import type { ThemeMetricsItem } from '@/api/hooks/useThemes';

interface ThemeMetricsChartProps {
  metrics: ThemeMetricsItem[];
}

export function ThemeMetricsChart({ metrics }: ThemeMetricsChartProps) {
  if (metrics.length === 0) {
    return (
      <div className="flex items-center justify-center py-12 text-sm text-muted-foreground">
        No metrics data available
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={350}>
      <ComposedChart data={metrics} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
          tickLine={false}
        />
        <YAxis
          yAxisId="left"
          tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
          tickLine={false}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
          tickLine={false}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: 'hsl(var(--card))',
            border: '1px solid hsl(var(--border))',
            borderRadius: '0.5rem',
            fontSize: 12,
            color: 'hsl(var(--foreground))',
          }}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Area
          yAxisId="left"
          type="monotone"
          dataKey="document_count"
          name="Documents"
          fill="hsl(210 100% 50% / 0.15)"
          stroke="hsl(210 100% 50%)"
          strokeWidth={1.5}
        />
        <Line
          yAxisId="right"
          type="monotone"
          dataKey="sentiment_score"
          name="Sentiment"
          stroke="hsl(142 71% 45%)"
          strokeWidth={2}
          dot={false}
        />
        <Bar
          yAxisId="left"
          dataKey="volume_zscore"
          name="Volume Z-Score"
          fill="hsl(280 67% 55% / 0.5)"
          barSize={12}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
