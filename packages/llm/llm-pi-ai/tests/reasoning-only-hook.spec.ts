/** The route's declaration of whether it drops reasoning-only assistant messages. */
import { describe, expect, it } from 'vitest'
import * as LlmPiAi from '@deepseek-ai/dsh-llm-pi-ai'
import { PiAiAdapter } from '@deepseek-ai/dsh-llm-pi-ai'
import { resolveProfiles } from '../src/config.ts'
import { memoryAuth } from './auth-double.ts'

/** Direct adapter over the real profile resolver, with a fixed key per call. */
function adapterOf(providers: Record<string, LlmPiAi.PiAiProviderProfile>): PiAiAdapter {
  return new PiAiAdapter({
    profiles: () => resolveProfiles(providers),
    resolveApiKey: () => Promise.resolve('test-key'),
    auth: memoryAuth(),
  })
}

const completions = {
  'acme-gateway': {
    api: 'openai-completions',
    baseURL: 'https://acme.test',
    models: [
      { id: 'm' },
      { id: 't', compat: { requiresThinkingAsText: true } },
    ],
  },
}

const anthropic = {
  'claude-gateway': { api: 'anthropic-messages', baseURL: 'https://acme.test', models: [{ id: 'c' }] },
}

describe('omitsReasoningOnlyAssistantMessages', () => {
  it('reports true for an openai-completions route whose serializer drops empty-content messages', () => {
    const adapter = adapterOf(completions)
    expect(adapter.omitsReasoningOnlyAssistantMessages('acme-gateway', 'm')).toBe(true)
  })

  it('reports false when the model sends thinking as ordinary text, keeping the message on the wire', () => {
    const adapter = adapterOf(completions)
    expect(adapter.omitsReasoningOnlyAssistantMessages('acme-gateway', 't')).toBe(false)
  })

  it('reports false for an anthropic-messages route, which retains thinking blocks', () => {
    const adapter = adapterOf(anthropic)
    expect(adapter.omitsReasoningOnlyAssistantMessages('claude-gateway', 'c')).toBe(false)
  })

  it('degrades to false for a provider this adapter does not own', () => {
    const adapter = adapterOf(completions)
    expect(adapter.omitsReasoningOnlyAssistantMessages('other-route', 'm')).toBe(false)
  })

  it('degrades to false for a model the route no longer configures', () => {
    const adapter = adapterOf(completions)
    expect(adapter.omitsReasoningOnlyAssistantMessages('acme-gateway', 'gone')).toBe(false)
  })
})
