import type { AnyMessage } from '@agentclientprotocol/sdk'
import { once } from 'node:events'
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http'
import { setTimeout as delay } from 'node:timers/promises'
import { expect, it, vi } from 'vitest'
import type { OpenCodeClassificationDiagnostic } from '../diagnostics/opencode-classification'
import { createOpenCodeCompactionStream } from './opencode-compaction-stream'

const metadata = JSON.stringify({ info: { sessionID: 's', id: 'm', role: 'assistant' } })
const contentChunk = (messageId: string, sessionUpdate = 'agent_message_chunk'): AnyMessage => ({
  jsonrpc: '2.0',
  method: 'session/update',
  params: {
    sessionId: 's',
    update: {
      sessionUpdate,
      messageId,
      content: { type: 'text', text: 'classified output' }
    }
  }
})
const chunk = contentChunk('m')
const response = (id: number): AnyMessage => ({
  jsonrpc: '2.0',
  id,
  result: { stopReason: 'end_turn' }
})

const withServer = async (
  handle: (response: ServerResponse, attempt: number, request: IncomingMessage) => void,
  run: (baseUrl: string, attempts: () => number) => Promise<void>
): Promise<void> => {
  let attempts = 0
  const server = createServer((request, response) => handle(response, ++attempts, request))
  try {
    server.listen(0, '127.0.0.1')
    await once(server, 'listening')
    const address = server.address()
    if (!address || typeof address === 'string') throw new Error('Expected TCP address')
    await run(`http://127.0.0.1:${address.port}`, () => attempts)
  } finally {
    server.closeAllConnections()
    await new Promise<void>((resolve) => server.close(() => resolve()))
  }
}

const fixture = (
  baseUrl: string,
  onHeaders?: () => void,
  incoming = [chunk]
): {
  reader: ReadableStreamDefaultReader<AnyMessage>
  prompt: (id: number, method?: 'session/prompt' | 'session/load') => Promise<void>
  abort: AbortController
  headers: number[]
  diagnostics: OpenCodeClassificationDiagnostic[]
} => {
  const abort = new AbortController()
  const headers: number[] = []
  const diagnostics: OpenCodeClassificationDiagnostic[] = []
  let source!: ReadableStreamDefaultController<AnyMessage>
  const stream = createOpenCodeCompactionStream(
    {
      readable: new ReadableStream({
        start(controller) {
          source = controller
        }
      }),
      writable: new WritableStream()
    },
    {
      api: { baseUrl, authorization: 'Basic test' },
      resolveSessionCwd: () => '/workspace',
      reportFailure: (diagnostic) => diagnostics.push(diagnostic),
      signal: abort.signal,
      fetchImpl: async (input, init) => {
        const response = await fetch(input, init)
        headers.push(response.status)
        onHeaders?.()
        return response
      }
    }
  )
  const writer = stream.writable.getWriter()
  return {
    reader: stream.readable.getReader(),
    abort,
    headers,
    diagnostics,
    prompt: async (id, method = 'session/prompt') => {
      await writer.write({
        jsonrpc: '2.0',
        id,
        method,
        params: { sessionId: 's', prompt: [] }
      })
      incoming.forEach((message) => source.enqueue(message))
      source.enqueue(response(id))
    }
  }
}

it('rechecks metadata after a temporary not-found response', async () => {
  await withServer(
    (res, attempt) => {
      if (attempt === 1) res.writeHead(404).end()
      else res.writeHead(200, { 'content-type': 'application/json' }).end(metadata)
    },
    async (url, attempts) => {
      const test = fixture(url)
      try {
        await test.prompt(1)
        expect((await test.reader.read()).value).toEqual(chunk)
        expect((await test.reader.read()).value).toEqual(response(1))
        expect(test.headers).toEqual([404, 200])
        expect(attempts()).toBe(2)
      } finally {
        test.abort.abort()
      }
    }
  )
})

it('retries the recent window and classifies multiple reasoning parts from one history hit', async () => {
  const info = { sessionID: 's', id: 'msg_m', role: 'assistant' }
  const parts = [{ id: 'prt_first' }, { id: 'prt_second' }]
  const thought = 'agent_thought_chunk'
  const urls: URL[] = []
  await withServer(
    (res, attempt, request) => {
      urls.push(new URL(request.url!, 'http://localhost'))
      const body = attempt <= 2 ? [] : [{ info, parts }]
      res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify(body))
    },
    async (url) => {
      const test = fixture(
        url,
        undefined,
        parts.map(({ id }) => contentChunk(id, thought))
      )
      try {
        await test.prompt(1)
        expect((await test.reader.read()).value).toEqual(contentChunk('msg_m', thought))
        expect((await test.reader.read()).value).toEqual(contentChunk('msg_m', thought))
        expect((await test.reader.read()).value).toEqual(response(1))
        expect(urls.map((url) => [url.pathname, url.searchParams.get('limit')])).toEqual(
          Array.from({ length: 3 }, () => ['/session/s/message', '20'])
        )
      } finally {
        test.abort.abort()
      }
    }
  )
})

it.each(['session/prompt', 'session/load'] as const)(
  'finds older parts in full history for %s',
  async (method) => {
    const limits: Array<string | null> = []
    const thought = 'agent_thought_chunk'
    const target = {
      info: { sessionID: 's', id: 'msg_old', role: 'assistant' },
      parts: [{ id: 'prt_old' }]
    }
    const recent = Array.from({ length: 20 }, (_, index) => ({
      info: { sessionID: 's', id: `msg_recent${index}`, role: 'assistant' },
      parts: [{ id: `prt_recent${index}` }]
    }))
    await withServer(
      (res, _attempt, request) => {
        const url = new URL(request.url!, 'http://localhost')
        const limit = url.searchParams.get('limit')
        limits.push(limit)
        const body = limit ? recent : [target, ...recent]
        res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify(body))
      },
      async (url) => {
        const test = fixture(url, undefined, [contentChunk('prt_old', thought)])
        try {
          await test.prompt(1, method)
          expect((await test.reader.read()).value).toEqual(contentChunk('msg_old', thought))
          expect((await test.reader.read()).value).toEqual(response(1))
          expect(limits).toEqual(method === 'session/prompt' ? ['20', '20', '20', null] : [null])
        } finally {
          test.abort.abort()
        }
      }
    )
  }
)

it('accepts a history body that takes longer than the old two-second deadline', async () => {
  const info = { sessionID: 's', id: 'msg_slow', role: 'assistant' }
  await withServer(
    (res) => {
      res.writeHead(200, { 'content-type': 'application/json' }).flushHeaders()
      void delay(2_200).then(() => res.end(JSON.stringify([{ info, parts: [{ id: 'prt_slow' }] }])))
    },
    async (url, attempts) => {
      const test = fixture(url, undefined, [contentChunk('prt_slow')])
      try {
        await test.prompt(1)
        expect((await test.reader.read()).value).toEqual(contentChunk('msg_slow'))
        expect((await test.reader.read()).value).toEqual(response(1))
        expect(attempts()).toBe(1)
      } finally {
        test.abort.abort()
      }
    }
  )
})

it('reports elapsed time and the full-history count when a part stays absent', async () => {
  await withServer(
    (res, _attempt, request) => {
      const limit = new URL(request.url!, 'http://localhost').searchParams.get('limit')
      const body = Array.from({ length: limit ? 20 : 580 }, () => ({}))
      res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify(body))
    },
    async (url, attempts) => {
      const test = fixture(url, undefined, [contentChunk('prt_missing')])
      try {
        await test.prompt(1)
        expect((await test.reader.read()).value).toMatchObject({
          id: 1,
          error: { code: -32603, message: expect.stringContaining('(part-not-found)') }
        })
        expect(test.diagnostics).toEqual([
          {
            stage: 'chunk',
            reason: 'part-not-found',
            endpoint: 'history',
            attempt: 4,
            elapsedMs: expect.any(Number),
            messageCount: 580
          }
        ])
        expect(test.diagnostics[0].elapsedMs).toBeGreaterThanOrEqual(800)
        expect(attempts()).toBe(4)
      } finally {
        test.abort.abort()
      }
    }
  )
})

it('retries a reset after successful headers and delivers the classified response', async () => {
  let interrupted: ServerResponse | undefined
  await withServer(
    (res, attempt) => {
      res.writeHead(200, { 'content-type': 'application/json' })
      if (attempt === 1) {
        res.write('{"info":')
        interrupted = res
      } else res.end(metadata)
    },
    async (url, attempts) => {
      const test = fixture(url, () => {
        interrupted?.destroy()
        interrupted = undefined
      })
      try {
        await test.prompt(1)
        expect((await test.reader.read()).value).toEqual(chunk)
        expect((await test.reader.read()).value).toEqual(response(1))
        expect(attempts()).toBe(2)
        // Both fetches resolved; the first failure was in the body, not before headers.
        expect(test.headers).toEqual([200, 200])
      } finally {
        test.abort.abort()
      }
    }
  )
})

it('bounds repeated body resets and permits a later prompt on the same stream', async () => {
  let healthy = false
  let interrupted: ServerResponse | undefined
  await withServer(
    (res) => {
      res.writeHead(200, { 'content-type': 'application/json' })
      if (healthy) res.end(metadata)
      else {
        res.write('{"private_context":"INTERNAL_PATH_AND_INSTRUCTIONS",')
        interrupted = res
      }
    },
    async (url, attempts) => {
      const test = fixture(url, () => {
        interrupted?.destroy()
        interrupted = undefined
      })
      try {
        await test.prompt(1)
        const failed = (await test.reader.read()).value
        expect(failed).toMatchObject({ id: 1, error: { code: -32603 } })
        expect(JSON.stringify(failed)).not.toContain('INTERNAL_PATH_AND_INSTRUCTIONS')
        expect(attempts()).toBe(4)
        healthy = true
        await test.prompt(2)
        expect((await test.reader.read()).value).toEqual(chunk)
        expect((await test.reader.read()).value).toEqual(response(2))
        expect(attempts()).toBe(5)
      } finally {
        test.abort.abort()
      }
    }
  )
})

it('does not retry a complete malformed JSON response or expose its body', async () => {
  await withServer(
    (res) => {
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end('{"private_context":"INTERNAL_PATH_AND_INSTRUCTIONS",')
    },
    async (url, attempts) => {
      const test = fixture(url)
      try {
        await test.prompt(1)
        const failed = (await test.reader.read()).value
        expect(failed).toMatchObject({ id: 1, error: { code: -32603 } })
        expect(JSON.stringify(failed)).not.toContain('INTERNAL_PATH_AND_INSTRUCTIONS')
        expect(attempts()).toBe(1)
      } finally {
        test.abort.abort()
      }
    }
  )
})

it('retries a stalled body after its deadline without closing ACP', async () => {
  await withServer(
    (res, attempt) => {
      res.writeHead(200, { 'content-type': 'application/json' })
      if (attempt === 1) res.write('{"info":')
      else res.end(metadata)
    },
    async (url, attempts) => {
      const test = fixture(url)
      try {
        await test.prompt(1)
        expect((await test.reader.read()).value).toEqual(chunk)
        expect((await test.reader.read()).value).toEqual(response(1))
        expect(attempts()).toBe(2)
        expect(test.headers).toEqual([200, 200])
      } finally {
        test.abort.abort()
      }
    }
  )
})

it('cancels a pending body read on cleanup without retrying', async () => {
  await withServer(
    (res) => {
      res.writeHead(200, { 'content-type': 'application/json' })
      res.write('{"info":')
    },
    async (url, attempts) => {
      const test = fixture(url)
      try {
        await test.prompt(1)
        const reading = expect(test.reader.read()).rejects.toThrow()
        await vi.waitFor(() => expect(test.headers).toEqual([200]))
        test.abort.abort()
        await reading
        expect(attempts()).toBe(1)
      } finally {
        test.abort.abort()
      }
    }
  )
})
