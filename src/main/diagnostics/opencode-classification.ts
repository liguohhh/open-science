import { z } from 'zod'

export const OPENCODE_CLASSIFICATION_FAILURE_EVENT = 'OpenCode output classification failed'

export const openCodeClassificationDiagnosticSchema = z.object({
  stage: z.enum(['chunk', 'summary-finalization']),
  reason: z.enum([
    'missing-session-id',
    'missing-message-id',
    'part-not-found',
    'invalid-history-metadata',
    'invalid-message-metadata',
    'api-unavailable',
    'unexpected',
    'timeout',
    'transport',
    'invalid-json',
    'http-status'
  ]),
  endpoint: z.enum(['history', 'message']).optional(),
  attempt: z.number().int().positive().optional(),
  status: z.number().int().min(100).max(599).optional()
})

export type OpenCodeClassificationDiagnostic = z.infer<
  typeof openCodeClassificationDiagnosticSchema
>
