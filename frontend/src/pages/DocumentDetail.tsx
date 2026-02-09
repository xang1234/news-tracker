import { useParams } from 'react-router-dom';

import { Header } from '@/components/layout/Header';

export default function DocumentDetail() {
  const { documentId } = useParams();

  return (
    <>
      <Header title="Document Detail" />
      <div className="p-6">
        <p className="text-sm text-muted-foreground">Document Detail — coming soon. ID: {documentId}</p>
      </div>
    </>
  );
}
