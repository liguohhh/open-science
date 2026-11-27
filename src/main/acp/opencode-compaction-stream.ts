import type { AnyMessage, JsonRpcId, Stream } from '@agentclientprotocol/sdk'
import { setTimeout as delay } from 'node:timers/promises'
import type { ResolvedAgentBackend } from '../agent-framework'
import type { OpenCodeClassificationDiagnostic } from '../diagnostics/opencode-classification'
import { isRecord } from '../value-guards'

type MessageIdentity = { sessionId: string; messageId: string }
type Classification = { summary: boolean; finished: boolean; succeeded: boolean }
type ResolvedMessage = { identity: MessageIdentity; classification?: Classification }
type HistoryAssistant = { info: Record<string, unknown>; messageId: string; parts: unknown[] }
type MetadataResult<T> =
  | { value: T }
  | { reason: 'part-not-found'; messageCount: number }
  | { reason: 'invalid-history-metadata' | 'invalid-message-metadata' }
type TrackedRequest = {
  id: JsonRpcId
  method: string
  sessionId?: string
  cwd?: string
  classificationFailure?: string
  withheldAfterStop?: boolean
  isUserStopped?: () => boolean
  cancellation?: AbortController
}
type PendingSummary = MessageIdentity & { owners: readonly TrackedRequest[] }
type MetadataEndpoint = NonNullable<OpenCodeClassificationDiagnostic['endpoint']>
type ClassificationCause = Omit<OpenCodeClassificationDiagnostic, 'stage'> | { reason: 'cancelled' }

const MAX_CACHE_ENTRIES = 256
const RECENT_HISTORY_LIMIT = 20
const METADATA_RETRY_DELAYS_MS = [100, 250, 500]
const METADATA_LOOKUP_ATTEMPTS = METADATA_RETRY_DELAYS_MS.length + 1
const RETRYABLE_METADATA_STATUS = new Set([404, 408, 425, 429, 500, 502, 503, 504])

const CLASSIFICATION_ERROR_MESSAGE = 'OpenCode message classification failed. Output was withheld.'

class OpenCodeClassificationError extends Error {
  constructor(readonly detail: ClassificationCause) {
    let message = CLASSIFICATION_ERROR_MESSAGE
    if (detail.reason === 'missing-message-id') {
      message +=
        ' The ACP update is missing a message ID. Check that the OpenCode 1.x version supplies ACP message IDs.'
    } else if (detail.reason === 'missing-session-id') {
      message += ' The ACP update is missing a session ID.'
    } else if (detail.reason !== 'cancelled') {
      message += ` Local OpenCode metadata is unavailable (${detail.reason}).`
    }
    super(message)
  }
}

const readClassification = (
  info: unknown,
  identity: MessageIdentity
): Classification | undefined => {
  if (
    !isRecord(info) ||
    info.id !== identity.messageId ||
    info.sessionID !== identity.sessionId ||
    info.role !== 'assistant' ||
    (info.summary !== undefined && typeof info.summary !== 'boolean')
  )
    return undefined

  return {
    summary: info.summary === true,
    finished: typeof info.finish === 'string' || info.error != null,
    succeeded: info.finish === 'stop' && info.error == null
  }
}

const historyAssistant = (message: unknown, sessionId: string): HistoryAssistant | undefined =>
  isRecord(message) &&
  isRecord(message.info) &&
  message.info.sessionID === sessionId &&
  message.info.role === 'assistant' &&
  typeof message.info.id === 'string' &&
  Array.isArray(message.parts)
    ? { info: message.info, messageId: message.info.id, parts: message.parts }
    : undefined

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
  const failRequests = (
    owners: readonly TrackedRequest[],
    message = CLASSIFICATION_ERROR_MESSAGE
  ): boolean => {
    const activeOwners = owners.filter((request) => requests.get(request.id) === request)
    if (activeOwners.length === 0) return false
    for (const request of activeOwners) {
      if (request.cancellation?.signal.aborted) request.withheldAfterStop = true
      else request.classificationFailure ??= message
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
    decode: (body: unknown) => MetadataResult<T>,
    fullHistoryUrl?: URL
  ): Promise<T> => {
    // Stop is written while metadata may be blocking the inbound ACP stream. Interrupt
    // only this prompt's lookup so the native cancellation acknowledgement can drain.
    const requestSignal = AbortSignal.any([
      signal,
      ...[...requests.values()].flatMap((request) =>
        request.sessionId === sessionId && request.cancellation ? [request.cancellation.signal] : []
      )
    ])
    const startedAt = performance.now()
    const failure = (
      reason: OpenCodeClassificationDiagnostic['reason'],
      attempt?: number,
      details?: Pick<OpenCodeClassificationDiagnostic, 'status' | 'messageCount'>
    ): OpenCodeClassificationError =>
      new OpenCodeClassificationError({
        reason,
        endpoint,
        elapsedMs: Math.round(performance.now() - startedAt),
        ...(attempt === undefined ? {} : { attempt }),
        ...details
      })

    if (requestSignal.aborted) throw new OpenCodeClassificationError({ reason: 'cancelled' })
    if (!options.api) throw failure('api-unavailable')
    let historyMiss = false
    for (let attempt = 1; attempt <= METADATA_LOOKUP_ATTEMPTS; attempt++) {
      const lastAttempt = attempt === METADATA_LOOKUP_ATTEMPTS
      if (attempt > 1) {
        // ACP 内容块可能先于元数据可见。停止可以打断查询和重试等待。
        try {
          await delay(METADATA_RETRY_DELAYS_MS[attempt - 2], undefined, { signal: requestSignal })
        } catch {
          throw new OpenCodeClassificationError({ reason: 'cancelled' })
        }
      }
      const attemptUrl = lastAttempt && historyMiss && fullHistoryUrl ? fullHistoryUrl : url
      const timeoutMs =
        endpoint === 'history'
          ? Math.min(10_000, 5_000 + (attempt - 1) * 2_000)
          : Math.min(3_000, 2_000 + (attempt - 1) * 500)
      const deadline = AbortSignal.timeout(timeoutMs)
      let response: Response
      try {
        response = await (options.fetchImpl ?? fetch)(attemptUrl, {
          headers: { authorization: options.api.authorization },
          signal: AbortSignal.any([requestSignal, deadline])
        })
      } catch {
        if (requestSignal.aborted) throw new OpenCodeClassificationError({ reason: 'cancelled' })
        if (!lastAttempt) continue
        throw failure(deadline.aborted ? 'timeout' : 'transport', attempt)
      }
      if (!response.ok) {
        void response.body?.cancel().catch(() => undefined)
        if (requestSignal.aborted) throw new OpenCodeClassificationError({ reason: 'cancelled' })
        if (!lastAttempt && RETRYABLE_METADATA_STATUS.has(response.status)) continue
        throw failure('http-status', attempt, { status: response.status })
      }
      let body: unknown
      try {
        body = await response.json()
      } catch (error) {
        if (requestSignal.aborted) throw new OpenCodeClassificationError({ reason: 'cancelled' })
        const invalidJson = error instanceof SyntaxError
        if (!lastAttempt && !invalidJson) continue
        throw failure(
          invalidJson ? 'invalid-json' : deadline.aborted ? 'timeout' : 'transport',
          attempt
        )
      }
      if (requestSignal.aborted) throw new OpenCodeClassificationError({ reason: 'cancelled' })
      const metadata = decode(body)
      if ('value' in metadata) return metadata.value
      if (metadata.reason !== 'part-not-found') throw failure(metadata.reason, attempt)
      if (lastAttempt)
        throw failure(metadata.reason, attempt, { messageCount: metadata.messageCount })
      historyMiss = true
    }
    throw failure('unexpected')
  }
  const metadataUrl = (sessionId: string, endpoint: MetadataEndpoint, messageId = ''): URL => {
    if (signal.aborted) throw new OpenCodeClassificationError({ reason: 'cancelled' })
    if (!options.api) throw new OpenCodeClassificationError({ reason: 'api-unavailable', endpoint })
    const url = new URL(
      `/session/${encodeURIComponent(sessionId)}/message${messageId ? `/${encodeURIComponent(messageId)}` : ''}`,
      options.api.baseUrl
    )
    // Replay arrives before session/load attaches the ActiveSession to the registry.
    const loading = [...requests.values()].find(
      (request) => request.sessionId === sessionId && request.cwd
    )
    url.searchParams.set(
      'directory',
      loading?.cwd ?? sessionCwds.get(sessionId) ?? options.resolveSessionCwd(sessionId)
    )
    return url
  }
  const rememberPart = (sessionId: string, partId: string, messageId: string): void => {
    const key = JSON.stringify([sessionId, partId])
    partMessages.delete(key)
    if (partMessages.size >= MAX_CACHE_ENTRIES)
      partMessages.delete(partMessages.keys().next().value!)
    partMessages.set(key, messageId)
  }
  const resolveMessageId = async (
    incoming: MessageIdentity,
    owners: readonly TrackedRequest[]
  ): Promise<ResolvedMessage> => {
    const { sessionId, messageId: partId } = incoming
    if (!partId.startsWith('prt_')) return { identity: incoming }
    const cached = partMessages.get(JSON.stringify([sessionId, partId]))
    if (cached) return { identity: { sessionId, messageId: cached } }
    try {
      const historyUrl = metadataUrl(sessionId, 'history')
      const live =
        owners.length > 0 && owners.every((request) => request.method === 'session/prompt')
      const url = new URL(historyUrl)
      if (live) url.searchParams.set('limit', String(RECENT_HISTORY_LIMIT))
      const decode = (body: unknown): MetadataResult<ResolvedMessage> => {
        const messages = Array.isArray(body) ? body : isRecord(body) ? body.data : undefined
        if (!Array.isArray(messages)) return { reason: 'invalid-history-metadata' }

        const match = messages.findLast((message) =>
          historyAssistant(message, sessionId)?.parts.some(
            (part) => isRecord(part) && part.id === partId
          )
        )
        const matched = historyAssistant(match, sessionId)
        if (matched && !matched.messageId.startsWith('msg_'))
          return { reason: 'invalid-history-metadata' }

        const rememberParts = (candidate: HistoryAssistant | undefined): void => {
          if (!candidate?.messageId.startsWith('msg_')) return
          for (const part of candidate.parts) {
            if (isRecord(part) && typeof part.id === 'string' && part.id.startsWith('prt_'))
              rememberPart(sessionId, part.id, candidate.messageId)
          }
        }
        // 历史按时间顺序返回。先索引近期消息，最后写入命中消息，避免旧消息的当前 part 被淘汰。
        for (const message of messages.slice(-RECENT_HISTORY_LIMIT)) {
          if (message !== match) rememberParts(historyAssistant(message, sessionId))
        }
        if (!matched) return { reason: 'part-not-found', messageCount: messages.length }

        rememberParts(matched)
        rememberPart(sessionId, partId, matched.messageId)
        const identity = { sessionId, messageId: matched.messageId }
        return { value: { identity, classification: readClassification(matched.info, identity) } }
      }
      // 给新写入的 part 留出可见时间，只在最后一次查询扩大到全量历史。
      return await readMetadata(url, sessionId, 'history', decode, live ? historyUrl : undefined)
    } catch (error) {
      if (error instanceof OpenCodeClassificationError) throw error
      throw new OpenCodeClassificationError({ reason: 'unexpected', endpoint: 'history' })
    }
  }
  const classify = async (identity: MessageIdentity): Promise<Classification> => {
    try {
      const url = metadataUrl(identity.sessionId, 'message', identity.messageId)
      return await readMetadata<Classification>(url, identity.sessionId, 'message', (body) => {
        const classification = readClassification(isRecord(body) ? body.info : undefined, identity)
        return classification ? { value: classification } : { reason: 'invalid-message-metadata' }
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
        if (
          signal.aborted ||
          !failRequests(
            identity.owners,
            error instanceof OpenCodeClassificationError ? error.message : undefined
          )
        )
          throw error
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
              request?.classificationFailure || withheldFailure
                ? {
                    jsonrpc: '2.0',
                    id: message.id,
                    error: {
                      code: -32603,
                      message: request?.classificationFailure ?? CLASSIFICATION_ERROR_MESSAGE
                    }
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
          const priorFailure = owners.find(
            (request) => request.classificationFailure
          )?.classificationFailure
          if (priorFailure || owners.some((request) => request.cancellation?.signal.aborted)) {
            // Withheld output must not leave another possible owner reporting success.
            failRequests(owners, priorFailure)
            return
          }
          let normalized: AnyMessage = message
          try {
            if (typeof update.messageId !== 'string' || !update.messageId)
              throw new OpenCodeClassificationError({ reason: 'missing-message-id' })
            const incoming = { sessionId: params.sessionId, messageId: update.messageId }
            const { identity, classification } = await resolveMessageId(incoming, owners)
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
              info = classification ?? (await classify(identity))
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
            if (
              signal.aborted ||
              !failRequests(
                owners,
                error instanceof OpenCodeClassificationError ? error.message : undefined
              )
            )
              throw error
            return
          }
          if (!(await finish(params.sessionId, owners, controller))) {
            failRequests(
              owners,
              owners.find((request) => request.classificationFailure)?.classificationFailure
            )
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
