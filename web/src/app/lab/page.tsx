import LabNotebook from '@/components/LabNotebook'

export const metadata = {
  title: 'Lab Notebook — CFR Strategy Explorer',
  description: 'Metric evolution, seed variance, and experimental verdicts',
}

export default function LabPage() {
  return (
    <main className="main">
      <LabNotebook />
    </main>
  )
}
