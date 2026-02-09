import { useParams } from 'react-router-dom';

import { Header } from '@/components/layout/Header';

export default function ThemeDetail() {
  const { themeId } = useParams();

  return (
    <>
      <Header title="Theme Detail" />
      <div className="p-6">
        <p className="text-sm text-muted-foreground">Theme Detail — coming soon. ID: {themeId}</p>
      </div>
    </>
  );
}
