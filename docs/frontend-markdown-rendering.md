# Frontend Markdown Rendering

The 2Quip Agent (Alex) returns markdown-formatted tables when responding with structured data such as metric mappings, part cross-references, telematics field comparisons, and multi-column reports. Your Next.js frontend needs `react-markdown` with the `remark-gfm` plugin to render these properly.

## When Alex Uses Tables

Alex uses markdown tables for:

- AEMP-to-OEM data set mappings
- Telematics metric comparisons
- Part number cross-references
- Equipment metric reports and rankings
- Cost analysis and financial breakdowns
- Any multi-column structured data

For general conversation, troubleshooting steps, and simple answers, Alex responds in plain text paragraphs (no emojis or decorative symbols).

## Installation

```bash
npm install react-markdown remark-gfm
```

## Basic Usage

```tsx
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

function ChatMessage({ content }: { content: string }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]}>
      {content}
    </ReactMarkdown>
  )
}
```

The `remark-gfm` plugin is required. Without it, `react-markdown` will not parse pipe tables (`| col | col |`) into `<table>` elements.

## Streaming (SSE) Usage

When consuming the `/chat/stream` endpoint, accumulate content chunks into a single string before passing to `react-markdown`. Do not pass individual chunks separately, as partial markdown will not render correctly.

```tsx
'use client'

import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

function ChatStream({ message }: { message: string }) {
  const [content, setContent] = useState('')

  async function sendMessage() {
    setContent('')

    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    })

    const reader = response.body?.getReader()
    const decoder = new TextDecoder()

    while (reader) {
      const { done, value } = await reader.read()
      if (done) break

      const text = decoder.decode(value)
      const lines = text.split('\n').filter(line => line.startsWith('data: '))

      for (const line of lines) {
        const data = JSON.parse(line.slice(6))
        if (data.type === 'content') {
          setContent(prev => prev + data.content)
        }
      }
    }
  }

  return (
    <div>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {content}
      </ReactMarkdown>
    </div>
  )
}
```

## Table Styling

`react-markdown` renders tables as plain `<table>`, `<thead>`, `<tbody>`, `<tr>`, `<th>`, and `<td>` elements. Style them with CSS or Tailwind.

### CSS example

```css
.chat-message table {
  border-collapse: collapse;
  width: 100%;
  margin: 1rem 0;
}

.chat-message th,
.chat-message td {
  border: 1px solid #d1d5db;
  padding: 0.5rem 0.75rem;
  text-align: left;
}

.chat-message th {
  background-color: #f3f4f6;
  font-weight: 600;
}

.chat-message tr:nth-child(even) {
  background-color: #f9fafb;
}
```

### Tailwind example (custom components)

```tsx
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

function ChatMessage({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        table: ({ children }) => (
          <table className="w-full border-collapse my-4">{children}</table>
        ),
        th: ({ children }) => (
          <th className="border border-gray-300 bg-gray-100 px-3 py-2 text-left font-semibold">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="border border-gray-300 px-3 py-2">{children}</td>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  )
}
```

## Copy and Download

Tables rendered by Alex can be made copyable to clipboard and downloadable as CSV. This uses a wrapper component that detects `<table>` elements and adds action buttons.

### Helper: extract table data from DOM

```ts
// lib/table-utils.ts

export function extractTableData(tableEl: HTMLTableElement): string[][] {
  const rows: string[][] = []
  tableEl.querySelectorAll('tr').forEach(tr => {
    const cells: string[] = []
    tr.querySelectorAll('th, td').forEach(cell => {
      cells.push(cell.textContent?.trim() ?? '')
    })
    rows.push(cells)
  })
  return rows
}

export function toCsv(rows: string[][]): string {
  return rows
    .map(row => row.map(cell => `"${cell.replace(/"/g, '""')}"`).join(','))
    .join('\n')
}

export function toTsv(rows: string[][]): string {
  return rows.map(row => row.join('\t')).join('\n')
}

export function downloadCsv(rows: string[][], filename = 'table-export.csv') {
  const csv = toCsv(rows)
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}
```

### Table wrapper component

```tsx
// components/copyable-table.tsx
'use client'

import { useRef, useState } from 'react'
import { extractTableData, toTsv, downloadCsv } from '@/lib/table-utils'

export function CopyableTable({ children }: { children: React.ReactNode }) {
  const tableRef = useRef<HTMLDivElement>(null)
  const [copied, setCopied] = useState(false)

  function getRows() {
    const table = tableRef.current?.querySelector('table')
    if (!table) return null
    return extractTableData(table)
  }

  function handleCopy() {
    const rows = getRows()
    if (!rows) return
    navigator.clipboard.writeText(toTsv(rows))
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  function handleDownload() {
    const rows = getRows()
    if (!rows) return
    downloadCsv(rows)
  }

  return (
    <div className="relative group">
      <div className="absolute right-2 top-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={handleCopy}
          className="px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded border border-gray-300"
        >
          {copied ? 'Copied' : 'Copy'}
        </button>
        <button
          onClick={handleDownload}
          className="px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded border border-gray-300"
        >
          Download CSV
        </button>
      </div>
      <div ref={tableRef} className="overflow-x-auto">
        {children}
      </div>
    </div>
  )
}
```

### Wire into react-markdown

Override the `table` component to wrap every table with the copy/download buttons:

```tsx
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { CopyableTable } from '@/components/copyable-table'

function ChatMessage({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        table: ({ children }) => (
          <CopyableTable>
            <table className="w-full border-collapse my-4">{children}</table>
          </CopyableTable>
        ),
        th: ({ children }) => (
          <th className="border border-gray-300 bg-gray-100 px-3 py-2 text-left font-semibold">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="border border-gray-300 px-3 py-2">{children}</td>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  )
}
```

When a user hovers over a table, "Copy" and "Download CSV" buttons appear in the top-right corner. Copy puts tab-separated data on the clipboard (pastes cleanly into Excel and Google Sheets). Download saves a `.csv` file.

## Example Response

When a user asks "Give me an AEMP mapping to a John Deere OEM data set", Alex returns:

```markdown
Here is a mapping of AEMP telematics standard fields to their John Deere JDLink equivalents.

| AEMP Standard Field | John Deere JDLink Field | Description |
|---|---|---|
| Equipment ID | Serial Number | Unique machine identifier |
| Cumulative Operating Hours | Engine Hours | Total runtime from engine sensors |
| Fuel Used | Fuel Consumption | Gallons consumed over period |
| Location | GPS Coordinates | Lat/lon from onboard GPS |
| Odometer | Distance Traveled | Cumulative distance from wheel sensors |
| Idle Time | Engine Idle Hours | Duration below threshold RPM |
| Fuel Level | Tank Level | Current fuel percentage remaining |

You can use this mapping to normalize John Deere JDLink telematics data into the AEMP v2 standard format for cross-fleet reporting.
```

This renders as a clean HTML table with the `react-markdown` + `remark-gfm` setup above.

## Non-Chat Endpoint

The `/chat` (non-streaming) endpoint returns the full response in one JSON payload:

```json
{
  "response": "Here is a mapping...\n\n| AEMP Standard Field | John Deere JDLink Field | ...",
  "session_id": "abc-123"
}
```

Pass `response` directly to `<ReactMarkdown>`.

## Voice Agent

The LiveKit voice agent does not use markdown tables. Its output is spoken aloud by a TTS engine, so it uses plain conversational text only. No frontend rendering changes are needed for voice.
