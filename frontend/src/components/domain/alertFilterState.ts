import { TRIGGER_TYPE_LABELS } from '@/lib/constants';

export interface AlertFilterValues {
  severity: string;
  triggerType: string;
  subjectType: string;
  acknowledged: string;
  limit: number;
}

export const DEFAULT_ALERT_FILTERS: AlertFilterValues = {
  severity: '',
  triggerType: '',
  subjectType: '',
  acknowledged: '',
  limit: 25,
};

export const SEVERITIES = ['critical', 'warning', 'info'];
export const TRIGGER_KEYS = Object.keys(TRIGGER_TYPE_LABELS);
export const SUBJECT_TYPES = [
  { value: 'theme', label: 'Theme' },
  { value: 'narrative_run', label: 'Narrative Run' },
  { value: 'graph_node', label: 'Graph Node' },
];
export const LIMIT_OPTIONS = [25, 50, 100];
