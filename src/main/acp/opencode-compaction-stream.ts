import type { AnyMessage, JsonRpcId, Stream } from '@agentclientprotocol/sdk'
import { setTimeout as delay } from 'node:timers/promises'
import type { ResolvedAgentBackend } from '../agent-framework'
import type { OpenCodeClassificationDiagnostic } from '../diagnostics/opencode-classification'
import { isRecord } from '../value-guards'

type MessageIdentity = { sessionId: string; messageId: string }
type Classification = { summary: boolean; finished: boolean; succeeded: boolean }
type TrackedRequest = {
  id: JsonRpcId
  method: string
  sessionId?: string
  cwd?: string
  classificationFailed?: boolean
  withheldAfterStop?: boolean
  isUserStopped?: () => boolean
  cancellation?: AbortController
}
type PendingSummary = MessageIdentity & { owners: readonly TrackedRequest[] }
type MetadataEndpoint = NonNullable<OpenCodeClassificationDiagnostic['endpoint']>
type ClassificationCause = Omit<OpenCodeClassificationDiagnostic, 'stage'> | { reason: 'cancelled' }

const MAX_CACHE_ENTRIES = 256
const METADATA_LOOKUP_ATTEMPTS = 2
const METADATA_RETRY_DELAY_MS = 100
const RETRYABLE_METADATA_STATUS = new Set([404, 408, 425, 429, 500, 502, 503, 504])

const CLASSIFICATION_ERROR_MESSAGE =
  'OpenCode message classification failed. Output was withheld. Check the local OpenCode service and use an OpenCode 1.x version that supplies ACP message IDs.'

class OpenCodeClassificationError extends Error {
  constructor(readonly detail: ClassificationCause) {
    super(CLASSIFICATION_ERROR_MESSAGE)
  }
}

const lifecycle = (
  identity: MessageIdentity,
  status: 'started' | 'completed' | 'failed'
): AnyMessage => ({
  jsonrpc: '2.0',
  method: 'session/update',
  params: {
    sessionId: identity.sessionId,
    update: {
      sessionUpdate: 'session_info_update',
      _meta: { 'opencode/compaction': { messageId: identity.messageId, reason: 'auto', status } }
    }
  }
})

// OpenCode 1.x can put a part ID on reasoning chunks. Normalize it before the SDK queues or
// observes the update, including session/load replay. Never inspect prose or modify native history.
export function createOpenCodeCompactionStream(
  stream: Stream,
  options: {
    api: ResolvedAgentBackend['opencodeUsageApi']
    captureUserStop?: (providerSessionId: string) => (() => boolean) | undefined
    reportFailure?: (diagnostic: OpenCodeClassificationDiagnostic) => void
    resolveSessionCwd: (sessionId: string) => string
    signal: AbortSignal
    fetchImpl?: typeof fetch
  }
): Stream {
  const cache = new Map<string, Classification>()
  const pending = new Map<string, PendingSummary>()
  const requests = new Map<JsonRpcId, TrackedRequest>()
  const sessionCwds = new Map<string, string>()
  const partMessages = new Map<string, string>()
  const abort = new AbortController()
  const signal = AbortSignal.any([options.signal, abort.signal])
  const clear = (): void => {
    cache.clear()
    pending.clear()
    requests.clear()
    sessionCwds.clear()
    partMessages.clear()
  }
  signal.addEventListener('abort', clear, { once: true })
  const contentRequest = (method: string): boolean =>
    ['session/prompt', 'session/load', 'session/resume'].includes(method)
  const sharesOwnership = (
    left: readonly TrackedRequest[],
    right: readonly TrackedRequest[]
  ): boolean => left.some((owner) => right.includes(owner))
  const failRequests = (owners: readonly TrackedRequest[]): boolean => {
    const activeOwners = owners.filter((request) => requests.get(request.id) === request)
    if (activeOwners.length === 0) return false
    for (const request of activeOwners) {
      if (request.cancellation?.signal.aborted) request.withheldAfterStop = true
      else request.classificationFailed = true
    }
    return true
  }
  const reportFailure = (
    stage: OpenCodeClassificationDiagnostic['stage'],
    error: OpenCodeClassificationError
  ): void => {
    if (signal.aborted || error.detail.reason === 'cancelled') return
    try {
      // 仅记录预定义原因和状态，不记录服务地址、消息 ID 或响应内容。
      options.reportFailure?.({ ...error.detail, stage })
    } catch {
      // 诊断写入失败不影响 ACP 请求的收尾。
    }
  }
  const readMetadata = async <T>(
    url: URL,
    sessionId: string,
    endpoint: MetadataEndpoint,
    decode: (body: unknown) => T | undefined
  ): Promise<T> => {
    // Stop is written while metadata may be blocking the inbound ACP stream. Interrupt
    // only this prompt's lookup so the native cancellation acknowledgement can drain.
    const requestSignal = AbortSignal.any([
      signal,
      ...[...requests.values()].flatMap((request) =>
        request.sessionId === sessionId && request.cancellation ? [request.cancellation.signal] : []
      )
    ])
    if (requestSignal.aborted) throw new OpenCodeClassificationError({ reason: 'cancelled' })
    if (!options.api) throw new OpenCodeClassificationError({ reason: 'api-unavailable', endpoint })
    for (let attempt = 1; attempt <= METADATA_LOOKUP_ATTEMPTS; attempt++) {
      if (attempt > 1) {
        // ACP 内容块到达时，HTTP 查询可能尚不可见。只重读元数据，并允许停止立即打断等待。
        try {
          await delay(METADATA_RETRY_DELAY_MS, undefined, { signal: requestSignal })
        } catch {
          throw new OpenCodeClassificationError({ reason: 'cancelled' })
        }
      }
      const deadline = AbortSignal.timeout(2_000)
      let response: Response
      try {
        response = await (options.fetchImpl ?? fetch)(url, {
          headers: { authorization: options.api.authorization },
          signal: AbortSignal.any([requestSignal, deadline])
        })
      } catch {
        if (requestSignal.aborted) {
          throw new OpenCodeClassificationError({ reason: 'cancelled' })
        }
        if (attempt < METADATA_LOOKUP_ATTEMPTS) continue
        throw new OpenCodeClassificationError({
          reason: deadline.aborted ? 'timeout' : 'transport',
          endpoint,
          attempt
        })
      }
      if (!response.ok) {
        void response.body?.cancel().catch(() => undefined)
        if (requestSignal.aborted) throw new OpenCodeClassificationError({ reason: 'cancelled' })
        if (attempt < METADATA_LOOKUP_ATTEMPTS && RETRYABLE_METADATA_STATUS.has(response.status))
          continue
        throw new OpenCodeClassificationError({
          reason: 'http-status',
          endpoint,
          attempt,
          status: response.status
        })
      }
      let body: unknown
      try {
        body = await response.json()
      } catch (error) {
        if (requestSignal.aborted) {
          throw new OpenCodeClassificationError({ reason: 'cancelled' })
        }
        const invalidJson = error instanceof SyntaxError
        if (attempt < METADATA_LOOKUP_ATTEMPTS && !invalidJson) continue
        throw new OpenCodeClassificationError({
          reason: invalidJson ? 'invalid-json' : deadline.aborted ? 'timeout' : 'transport',
          endpoint,
          attempt
        })
      }
      if (requestSignal.aborted) throw new OpenCodeClassificationError({ reason: 'cancelled' })
      const metadata = decode(body)
      if (metadata !== undefined) return metadata
      if (attempt === METADATA_LOOKUP_ATTEMPTS)
        throw new OpenCodeClassificationError({ reason: 'part-not-found', endpoint, attempt })
    }
    throw new OpenCodeClassificationError({ reason: 'unexpected', endpoint })
  }
  const resolveMessageId = async (identity: MessageIdentity): Promise<MessageIdentity> => {
    try {
      if (!identity.messageId.startsWith('prt_')) return identity
      const cached = partMessages.get(`${identity.sessionId}:${identity.messageId}`)
      if (cached) return { ...identity, messageId: cached }
      if (signal.aborted) throw new OpenCodeClassificationError({ reason: 'cancelled' })
      if (!options.api)
        throw new OpenCodeClassificationError({ reason: 'api-unavailable', endpoint: 'history' })
      const url = new URL(
        `/session/${encodeURIComponent(identity.sessionId)}/message`,
        options.api.baseUrl
      )
      const loading = [...requests.values()].find(
        (request) => request.sessionId === identity.sessionId && request.cwd
      )
      url.searchParams.set(
        'directory',
        loading?.cwd ??
          sessionCwds.get(identity.sessionId) ??
          options.resolveSessionCwd(identity.sessionId)
      )
      const messageId = await readMetadata(url, identity.sessionId, 'history', (body) => {
        const messages = Array.isArray(body) ? body : isRecord(body) ? body.data : undefined
        if (!Array.isArray(messages))
          throw new OpenCodeClassificationError({
            reason: 'invalid-history-metadata',
            endpoint: 'history'
          })
        for (const message of messages) {
          if (
            isRecord(message) &&
            isRecord(message.info) &&
            message.info.sessionID === identity.sessionId &&
            message.info.role === 'assistant' &&
            typeof message.info.id === 'string' &&
            Array.isArray(message.parts) &&
            message.parts.some((part) => isRecord(part) && part.id === identity.messageId)
          ) {
            if (!message.info.id.startsWith('msg_'))
              throw new OpenCodeClassificationError({
                reason: 'invalid-history-metadata',
                endpoint: 'history'
              })
            return message.info.id
          }
        }
        return undefined
      })
      if (partMessages.size >= MAX_CACHE_ENTRIES)
        partMessages.delete(partMessages.keys().next().value!)
      partMessages.set(`${identity.sessionId}:${identity.messageId}`, messageId)
      return { ...identity, messageId }
    } catch (error) {
      if (error instanceof OpenCodeClassificationError) throw error
      throw new OpenCodeClassificationError({ reason: 'unexpected', endpoint: 'history' })
    }
  }
  const classify = async (identity: MessageIdentity): Promise<Classification> => {
    try {
      if (signal.aborted) throw new OpenCodeClassificationError({ reason: 'cancelled' })
      if (!options.api)
        throw new OpenCodeClassificationError({ reason: 'api-unavailable', endpoint: 'message' })
      const url = new URL(
        `/session/${encodeURIComponent(identity.sessionId)}/message/${encodeURIComponent(identity.messageId)}`,
        options.api.baseUrl
      )
      // Replay arrives before session/load attaches the ActiveSession to the registry.
      const loading = [...requests.values()].find(
        (request) => request.sessionId === identity.sessionId && request.cwd
      )
      url.searchParams.set(
        'directory',
        loading?.cwd ??
          sessionCwds.get(identity.sessionId) ??
          options.resolveSessionCwd(identity.sessionId)
      )
      return await readMetadata(url, identity.sessionId, 'message', (body) => {
        const info = isRecord(body) ? body.info : undefined
        if (
          !isRecord(info) ||
          info.id !== identity.messageId ||
          info.sessionID !== identity.sessionId ||
          info.role !== 'assistant' ||
          (info.summary !== undefined && typeof info.summary !== 'boolean')
        )
          throw new OpenCodeClassificationError({
            reason: 'invalid-message-metadata',
            endpoint: 'message'
          })
        return {
          summary: info.summary === true,
          finished: typeof info.finish === 'string' || info.error != null,
          succeeded: info.finish === 'stop' && info.error == null
        }
      })
    } catch (error) {
      // Native response bodies, paths and transport errors may contain internal context.
      if (error instanceof OpenCodeClassificationError) throw error
      throw new OpenCodeClassificationError({ reason: 'unexpected', endpoint: 'message' })
    }
  }
  const finish = async (
    sessionId: string,
    owners: readonly TrackedRequest[],
    controller: TransformStreamDefaultController<AnyMessage>
  ): Promise<boolean> => {
    let verified = true
    for (const [key, identity] of pending) {
      if (
        identity.sessionId !== sessionId ||
        (identity.owners.length > 0 && !sharesOwnership(identity.owners, owners))
      )
        continue
      let info: Classification
      try {
        info = await classify(identity)
      } catch (error) {
        if (error instanceof OpenCodeClassificationError)
          reportFailure('summary-finalization', error)
        // A summary keeps its first owners. Unowned lookup failures remain transport-fatal.
        if (signal.aborted || !failRequests(identity.owners)) throw error
        // One failed probe does not establish the outcome of other pending summaries.
        controller.enqueue(lifecycle(identity, 'failed'))
        pending.delete(key)
        cache.delete(key)
        verified = false
        continue
      }
      controller.enqueue(lifecycle(identity, info.succeeded ? 'completed' : 'failed'))
      pending.delete(key)
    }
    return verified
  }
  const writer = stream.writable.getWriter()
  return {
    writable: new WritableStream<AnyMessage>({
      async write(message) {
        if (
          'method' in message &&
          message.method === 'session/cancel' &&
          isRecord(message.params) &&
          typeof message.params.sessionId === 'string'
        ) {
          for (const request of requests.values()) {
            if (request.sessionId === message.params.sessionId) request.cancellation?.abort()
          }
        }
        if (
          'method' in message &&
          'id' in message &&
          [
            'session/new',
            'session/load',
            'session/resume',
            'session/fork',
            'session/prompt',
            'session/close',
            'session/delete'
          ].includes(message.method) &&
          isRecord(message.params)
        ) {
          requests.set(message.id, {
            id: message.id,
            method: message.method,
            ...(message.method === 'session/prompt' && typeof message.params.sessionId === 'string'
              ? {
                  cancellation: new AbortController(),
                  isUserStopped: options.captureUserStop?.(message.params.sessionId)
                }
              : {}),
            ...(typeof message.params.sessionId === 'string'
              ? { sessionId: message.params.sessionId }
              : {}),
            ...(typeof message.params.cwd === 'string' ? { cwd: message.params.cwd } : {})
          })
        }
        await writer.write(message)
      },
      close: () => writer.close(),
      abort: (reason) => {
        abort.abort()
        return writer.abort(reason)
      }
    }),
    readable: stream.readable.pipeThrough(
      new TransformStream<AnyMessage, AnyMessage>({
        async transform(message, controller) {
          if (signal.aborted) throw new OpenCodeClassificationError({ reason: 'cancelled' })
          if (!('method' in message)) {
            const request = requests.get(message.id)
            if (request?.sessionId && contentRequest(request.method)) {
              await finish(request.sessionId, [request], controller)
            }
            if (request && 'result' in message) {
              const providerSessionId =
                isRecord(message.result) && typeof message.result.sessionId === 'string'
                  ? message.result.sessionId
                  : request.sessionId
              if (request.cwd && providerSessionId) sessionCwds.set(providerSessionId, request.cwd)
              if (
                request.sessionId &&
                ['session/close', 'session/delete'].includes(request.method)
              ) {
                sessionCwds.delete(request.sessionId)
                // Cached identity contains no content and is bounded. Reclassify after a close,
                // including a later load of the same native ID in another working directory.
                cache.clear()
                partMessages.clear()
                for (const [key, identity] of pending) {
                  if (identity.sessionId === request.sessionId) pending.delete(key)
                }
              }
            }
            requests.delete(message.id)
            const reason =
              'result' in message && isRecord(message.result)
                ? message.result.stopReason
                : undefined
            // Protocol-only cancellation cannot excuse missing output. For a human Stop the
            // prompt owner settles the native terminal after Notebook and Artifact cleanup.
            const withheldFailure =
              request?.withheldAfterStop && reason !== 'cancelled' && !request.isUserStopped?.()
            controller.enqueue(
              request?.classificationFailed || withheldFailure
                ? {
                    jsonrpc: '2.0',
                    id: message.id,
                    error: { code: -32603, message: CLASSIFICATION_ERROR_MESSAGE }
                  }
                : message
            )
            return
          }
          const params = message.params
          const update = isRecord(params) ? params.update : undefined
          if (
            message.method !== 'session/update' ||
            !isRecord(update) ||
            !['agent_message_chunk', 'agent_thought_chunk'].includes(String(update.sessionUpdate))
          ) {
            controller.enqueue(message)
            return
          }
          if (!isRecord(params) || typeof params.sessionId !== 'string') {
            const error = new OpenCodeClassificationError({ reason: 'missing-session-id' })
            reportFailure('chunk', error)
            throw error
          }
          // Preserve arrival-time ownership across metadata awaits; later writes cannot claim it.
          const owners = [...requests.values()].filter(
            (request) => request.sessionId === params.sessionId && contentRequest(request.method)
          )
          if (
            owners.some(
              (request) => request.classificationFailed || request.cancellation?.signal.aborted
            )
          ) {
            // Withheld output must not leave another possible owner reporting success.
            failRequests(owners)
            return
          }
          let normalized: AnyMessage = message
          try {
            if (typeof update.messageId !== 'string' || !update.messageId)
              throw new OpenCodeClassificationError({ reason: 'missing-message-id' })
            const incoming = { sessionId: params.sessionId, messageId: update.messageId }
            const identity = await resolveMessageId(incoming)
            normalized =
              identity.messageId === incoming.messageId
                ? message
                : ({
                    ...message,
                    params: {
                      ...params,
                      update: { ...update, messageId: identity.messageId }
                    }
                  } as AnyMessage)
            const key = JSON.stringify([identity.sessionId, identity.messageId])
            let info = cache.get(key)
            if (!info) {
              info = await classify(identity)
              // Bound per-connection identity memory; eviction only causes another safe lookup.
              if (cache.size >= MAX_CACHE_ENTRIES) cache.delete(cache.keys().next().value!)
              cache.set(key, info)
              if (info.summary) {
                controller.enqueue(
                  lifecycle(
                    identity,
                    info.finished ? (info.succeeded ? 'completed' : 'failed') : 'started'
                  )
                )
                if (!info.finished && !pending.has(key)) pending.set(key, { ...identity, owners })
              }
            }
            if (info.summary) return
          } catch (error) {
            if (error instanceof OpenCodeClassificationError) reportFailure('chunk', error)
            // Unknown output stays withheld. A metadata failure belongs to its content request;
            // keep consuming control traffic until its native terminal can reject that request.
            if (signal.aborted || !failRequests(owners)) throw error
            return
          }
          if (!(await finish(params.sessionId, owners, controller))) {
            failRequests(owners)
            return
          }
          controller.enqueue(normalized)
        },
        flush: clear
      }),
      { signal }
    )
  }
}
