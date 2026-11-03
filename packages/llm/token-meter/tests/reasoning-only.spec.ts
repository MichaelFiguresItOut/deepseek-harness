/** Reasoning-only assistant messages dropped by the route must not inflate later context estimates. */
import { describe, expect, it, onTestFinished } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import { AssistantStreamAccumulator, LlmAdapter, LlmRuntime, createAssistantMessage, createUserMessage, createSystemMessage } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, StreamChunk, TokenUsage } from '@deepseek-ai/dsh-llm'
import { Session, SessionId, canonicalHeader } from '@deepseek-ai/dsh-session'
import type { EpochHeader } from '@deepseek-ai/dsh-session'
import SessionProjectionRegistry from '@deepseek-ai/dsh-session-projection'
import TokenMeter from '@deepseek-ai/dsh-token-meter'
import { estimateMessage } from '../src/estimate.ts'

/** Adapter double that declares whether its route omits reasoning-only assistant messages. */
class HookAdapter extends LlmAdapter {
  constructor(private readonly omits: (provider: string, model: string) => boolean) { super() }
  override omitsReasoningOnlyAssistantMessages(provider: string, model: string): boolean {
    return this.omits(provider, model)
  }
  async * stream(_options: GenerateOptions): AsyncIterable<StreamChunk> {
    throw new Error('the hook adapter double does not stream')
  }
}

function header(model = 'mock'): EpochHeader {
  return canonicalHeader({ config: { provider: 'mock', model } })
}

async function harness(omits: (provider: string, model: string) => boolean): Promise<TokenMeter> {
  const ctx = new Context()
  onTestFinished(async () => { await ctx.fiber.dispose() })
  new SessionProjectionRegistry(ctx)
  const llm = new LlmRuntime(ctx)
  llm.registerAdapter(['mock'], new HookAdapter(omits))
  await ctx.plugin(TokenMeter)
  return ctx.tokenMeter
}

/** Bare meter without an llm service: the hook degrades to false. */
function bareMeter(): TokenMeter {
  const ctx = new Context()
  onTestFinished(async () => { await ctx.fiber.dispose() })
  new SessionProjectionRegistry(ctx)
  return new TokenMeter(ctx)
}

const SYSTEM_TEXT = 'You are a concise assistant.'
const USER_TEXT = 'hi'
const REASONING_TEXT = 'The user is asking for the capital of France, which is Paris.'
const USAGE: TokenUsage = { inputTokens: 1200, outputTokens: 50 }

function sysPrice(): number {
  return estimateMessage(createSystemMessage(SYSTEM_TEXT))
}

function userPrice(text: string): number {
  return estimateMessage(createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } }))
}

function reasoningPrice(): number {
  return estimateMessage(createAssistantMessage({ content: [{ type: 'reasoning', text: REASONING_TEXT }], source: { provider: 'mock', model: 'mock' } }))
}

/** The prompt side of the anchored call: system plus user nodes, no tools. */
const PROMPT_SIDE = sysPrice() + userPrice(USER_TEXT)

interface AssistantCallOptions {
  turn?: number
  step?: number
  kind: 'reasoning' | 'text'
  text: string
  usage?: TokenUsage
}

function appendAssistantCall(session: Session, value: EpochHeader, options: AssistantCallOptions): void {
  const turn = options.turn ?? 1
  const step = options.step ?? 1
  session.append('step/start', { turn, step })
  session.append('request/header', { header: value, reason: 'initial' })
  const block = options.kind === 'reasoning'
    ? { type: 'reasoning' as const, text: options.text }
    : { type: 'text' as const, text: options.text }
  const delta = options.kind === 'reasoning'
    ? { type: 'reasoning-delta' as const, index: 0, text: options.text }
    : { type: 'text-delta' as const, index: 0, text: options.text }
  const chunks: StreamChunk[] = [
    { type: 'block-start', index: 0, blockType: options.kind },
    delta,
    { type: 'block-end', index: 0, block },
    ...options.usage === undefined ? [] : [{ type: 'usage' as const, usage: options.usage }],
    { type: 'finish', reason: { kind: 'stop' } },
  ]
  const accumulator = new AssistantStreamAccumulator()
  for (const [index, chunk] of chunks.entries()) accumulator.push({ time: index, chunk })
  session.append('assistant/message', {
    stream: [...accumulator.snapshot()],
    turn,
    step,
    message: createAssistantMessage({ content: [block], source: { provider: value.config.provider, model: value.config.model } }),
    ...options.usage === undefined ? {} : { usage: options.usage },
  }, { surfaceOp: 'append' })
  session.append('step/end', { turn, step })
}

function seedSession(session: Session): void {
  session.append('system/message', { turn: 1, step: 1, message: createSystemMessage(SYSTEM_TEXT) }, { surfaceOp: 'append' })
  session.append('user/message', createUserMessage({ content: [{ type: 'text', text: USER_TEXT }], source: { kind: 'user' } }), { surfaceOp: 'append' })
}

function reasoningOnlyCall(session: Session, usage?: TokenUsage, model = 'mock'): void {
  appendAssistantCall(session, header(model), { kind: 'reasoning', text: REASONING_TEXT, ...usage === undefined ? {} : { usage } })
}

describe('reasoning-only anchor accounting', () => {
  it('prices the prompt side only when a dropping route anchors on a reasoning-only message', async () => {
    const meter = await harness(() => true)
    const session = Session.create(SessionId('reasoning-only-drop'))
    seedSession(session)
    reasoningOnlyCall(session, USAGE)

    const measurement = meter.measure(session)
    expect(measurement.baseline).toEqual({ kind: 'usage', tokens: USAGE.inputTokens, usage: USAGE })
    expect(measurement.surfaceDeltaTokens).toBe(0)
    // The completion tokens of the dropped message are never resent to the provider.
    expect(measurement.totalTokens).toBe(USAGE.inputTokens)

    // The reasoning node stays on the surface for replay and display.
    expect(measurement.nodes.map(node => node.tokens)).toEqual([sysPrice(), userPrice(USER_TEXT), reasoningPrice()])
  })

  it('keeps the full provider total when the route retains reasoning messages', async () => {
    const meter = await harness(() => false)
    const session = Session.create(SessionId('reasoning-only-retain'))
    seedSession(session)
    reasoningOnlyCall(session, USAGE)

    const measurement = meter.measure(session)
    expect(measurement.baseline).toEqual({ kind: 'usage', tokens: USAGE.inputTokens + USAGE.outputTokens, usage: USAGE })
    expect(measurement.totalTokens).toBe(USAGE.inputTokens + USAGE.outputTokens)
  })

  it('keeps the full provider total for a non-reasoning-only anchor on a dropping route', async () => {
    const meter = await harness(() => true)
    const session = Session.create(SessionId('reasoning-only-text'))
    seedSession(session)
    appendAssistantCall(session, header(), { kind: 'text', text: 'answer', usage: USAGE })

    const measurement = meter.measure(session)
    expect(measurement.baseline).toEqual({ kind: 'usage', tokens: USAGE.inputTokens + USAGE.outputTokens, usage: USAGE })
    expect(measurement.totalTokens).toBe(USAGE.inputTokens + USAGE.outputTokens)
  })

  it('falls back to the prompt-side estimate when provider usage is below it on a dropping route', async () => {
    const meter = await harness(() => true)
    const session = Session.create(SessionId('reasoning-only-low-usage'))
    seedSession(session)
    reasoningOnlyCall(session, { inputTokens: 8, outputTokens: 50 })

    const measurement = meter.measure(session)
    expect(measurement.baseline).toEqual({ kind: 'estimated', tokens: PROMPT_SIDE })
    expect(measurement.totalTokens).toBe(PROMPT_SIDE)
  })

  it('adds only the post-anchor surface delta on a dropping route', async () => {
    const meter = await harness(() => true)
    const session = Session.create(SessionId('reasoning-only-delta'))
    seedSession(session)
    reasoningOnlyCall(session, USAGE)
    session.append('user/message', createUserMessage({ content: [{ type: 'text', text: 'follow-up' }], source: { kind: 'user' } }), { surfaceOp: 'append' })

    const followUpPrice = userPrice('follow-up')
    const measurement = meter.measure(session)
    // The dropped node's own heuristic price cancels against the anchored snapshot.
    expect(measurement.surfaceDeltaTokens).toBe(followUpPrice)
    expect(measurement.totalTokens).toBe(USAGE.inputTokens + followUpPrice)
  })

  it('consults the hook with the routed provider and model', async () => {
    const seen: Array<[string, string]> = []
    const meter = await harness((provider, model) => {
      seen.push([provider, model])
      return model === 'dropper'
    })
    const session = Session.create(SessionId('reasoning-only-route'))
    seedSession(session)
    appendAssistantCall(session, header('keeper'), { kind: 'text', text: 'answer', usage: { inputTokens: 300, outputTokens: 40 } })

    let measurement = meter.measure(session)
    expect(measurement.baseline).toEqual({ kind: 'usage', tokens: 340, usage: { inputTokens: 300, outputTokens: 40 } })

    reasoningOnlyCall(session, USAGE, 'dropper')
    measurement = meter.measure(session)
    expect(measurement.baseline).toEqual({ kind: 'usage', tokens: USAGE.inputTokens, usage: USAGE })
    expect(seen.some(([provider, model]) => provider === 'mock' && model === 'keeper')).toBe(true)
    expect(seen.some(([provider, model]) => provider === 'mock' && model === 'dropper')).toBe(true)
  })

  it('degrades to the full provider total when no llm service is mounted', () => {
    const meter = bareMeter()
    const session = Session.create(SessionId('reasoning-only-bare'))
    seedSession(session)
    reasoningOnlyCall(session, USAGE)

    const measurement = meter.measure(session)
    expect(measurement.baseline).toEqual({ kind: 'usage', tokens: USAGE.inputTokens + USAGE.outputTokens, usage: USAGE })
  })
})
