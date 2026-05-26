import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'CFR Strategy Explorer',
  description: 'NLHE strategy visualiser powered by Deep CFR',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fi">
      <body>{children}</body>
    </html>
  )
}
