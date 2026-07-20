'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

const TABS = [
  { href: '/',       label: 'Strategy Explorer' },
  { href: '/matrix', label: 'Range Matrix'      },
  { href: '/lab',    label: 'Lab Notebook'      },
]

export default function Nav() {
  const path = usePathname()

  return (
    <nav className="nav">
      <div className="nav-inner">
        <span className="nav-brand">IIG // CFR</span>
        <div className="nav-tabs">
          {TABS.map(t => (
            <Link
              key={t.href}
              href={t.href}
              className={`nav-tab ${path === t.href ? 'active' : ''}`}
            >
              {t.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  )
}
