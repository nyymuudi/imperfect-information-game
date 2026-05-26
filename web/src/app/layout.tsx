import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'CFR Strategy Explorer',
  description: 'NLHE strategy visualiser powered by Deep CFR',
  icons: { icon: '/favicon.svg' },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
