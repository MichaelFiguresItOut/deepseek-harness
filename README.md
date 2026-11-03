# DSH Compaction Fix Fork

**Important Notice:** This is an independent fork of the [original DSH repository](https://github.com/deepseek-ai/deepseek-harness) by [MichaelFiguresItOut](https://www.youtube.com/@MichaelFiguresItOut). Original Read.me info further below.

Because the upstream DSH project does not accept GitHub Pull Requests, this repository was created to share this fix with the community.

* **Base Version:** Forked on October 30, 2026 from upstream master at commit [`ce7073d512`](https://github.com/deepseek-ai/deepseek-harness/commit/ce7073d512). All fixes were committed to this fork by November 3, 2026.

##  Summary
Compaction issues arose while running DSH against a local reasoning model ([unsloth/Qwen3.8 27B GGUF UD-IQ4-XS](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)) served through an OpenAI-compatible endpoint (LM Studio) on the `openai-completions` route. Forked DSH and added these reliability and performance fixes to the compaction process: provider-reported token usage now survives session replay, reasoning-only messages are priced correctly on routes that drop them from later requests, and truncated compaction summaries fail closed instead of being checkpointed — with summarizer reasoning disabled where the model allows it.

## Installing this fork

### From scratch (no existing DSH)

Prerequisites: [Node.js](https://nodejs.org/) **24.9 or newer** and pnpm (this repo pins pnpm 11.28.5 via its `packageManager` field):

```sh
# one-time setup
npm install -g pnpm@11.28.5
```

Then clone this fork and build it. The bundled dependency patches are applied automatically during install — no manual steps:

```sh
git clone https://github.com/MichaelFiguresItOut/deepseek-harness.git dsh-fork
cd dsh-fork
pnpm install
pnpm run build
pnpm dsh web
```

Note: the upstream quick start (`npx @deepseek-ai/dsh web`) installs the **upstream** package from npm, not this fork. Use the steps above to get the compaction fixes.

### Keeping your current settings

DSH keeps all user state — settings, profiles, API keys (`.env`), and session history — in `~/.dsh` on macOS/Linux or `%USERPROFILE%\.dsh` on Windows (overridable with the `DSH_HOME` environment variable). That folder lives **outside** any repository checkout, so installing this fork never touches it.

To move an existing install over to this fork without losing anything:

1. Leave your `.dsh` folder exactly where it is — do not delete or overwrite it.
2. Clone this fork into a new directory (next to your current setup is fine):
   ```sh
   git clone https://github.com/MichaelFiguresItOut/deepseek-harness.git dsh-fork
   cd dsh-fork
   pnpm install
   pnpm run build
   ```
3. Start it with `pnpm dsh web`. Your settings, profiles, and sessions load exactly as before — only the harness code is different.

Both checkouts share the same `.dsh` state, so you can start whichever one you want at any time, including going back to upstream.

### Updating this fork later

In your `dsh-fork` directory:

```sh
git pull
pnpm install
pnpm run build
```

Your settings and sessions are unaffected — they live in `.dsh`, not in the checkout.

## Issues encountered / bugs discovered

### 1. Compaction summaries silently truncated
Compaction repeatedly failed with `summarization truncated at the token cap` (ten consecutive failures were recorded in a single session). Two root causes:

- **No post-stream guard.** When a provider ended the summarization stream exactly at the output cap *without* reporting a max-tokens finish reason, the harness treated the response as successful and checkpointed the incomplete summary as if it were complete — corrupting the compacted history for every subsequent turn.
- **Reasoning ate the summary budget.** Models that think by default spent part of the summarizer's output-token budget on reasoning tokens before producing any summary text, making truncation far more likely on exactly the models where compaction matters most.

### 2. Provider-reported token usage lost across session replay
The pi-ai adapter's durable replay state stored each assistant turn's content and stop reason but dropped the provider-reported `usage` record. After a session was resumed, the harness had to re-estimate those tokens heuristically, so its view of conversation size drifted from what the provider actually reported — making compaction trigger at the wrong time (too late, handing the summarizer an oversized input; or too early).

### 3. Spurious double-compaction on reasoning models
The `openai-completions` serializer omits assistant messages that carry no text and no tool calls, so a reasoning-only message never reaches the provider again on later turns. The token meter nevertheless carried its *full* usage total (prompt + output) forward as the baseline for context size — inflating estimates by exactly the tokens the provider would never re-bill, and triggering compaction more often than necessary (sometimes twice in a row with nothing new to say).

## Detailed changes (commit by commit)

### 1. [`c26591abbf`](https://github.com/MichaelFiguresItOut/deepseek-harness/commit/c26591abbf) — Preserve provider-reported token usage in pi-ai replay state
*Files: `packages/llm/llm-pi-ai/src/{adapter,context,replay,stream}.ts` + tests (+323/−17)*

The normalized provider usage is now stored **verbatim** on the replay response (`PiAiReplayResponse.usage`) alongside a SHA-256 digest of the exact normalized request prefix (system prompt, messages, tools) that produced it. On replay, the digest for the assistant's preceding context is recomputed and usage is restored **only when it matches**; missing, malformed, or mismatched pairs degrade to zero usage so estimation never anchors on stale numbers. Stored records are validated field-by-field without normalization (forward-compatible fields survive), and legacy envelopes without the pair keep working unchanged.

### 2. [`ea2855fc2f`](https://github.com/MichaelFiguresItOut/deepseek-harness/commit/ea2855fc2f) — Price reasoning-only messages by prompt side when routes drop them
*Files: `packages/llm/{llm,token-meter,llm-pi-ai}` + docs (+378/−11)*

Adds an `LlmAdapter.omitsReasoningOnlyAssistantMessages` hook (default `false`, with `LlmRuntime` delegation). `PiAiAdapter` answers `true` only for the `openai-completions` API unless the model requires thinking as text, so routes that send reasoning on the wire keep full-total accounting. The token meter records a `reasoningOnly` flag on usage anchors and, when the routed model's route drops such messages, baselines from **provider usage minus output tokens** (prompt side), falling back to a prompt-side estimate when reported usage is too low to trust. Surface deltas are unchanged, so the dropped node's own price cancels out of later measurements. Measurement `stateVersion` bumps to 3; mismatched checkpoints replay instead of mispricing.

### 3. [`fcc63e543a`](https://github.com/MichaelFiguresItOut/deepseek-harness/commit/fcc63e543a) — Detect truncated compaction summaries and skip summarizer reasoning
*Files: `packages/compaction/compaction-basic/src/summarizer.ts` + tests (+126/−4)*

Two fixes in the summarizer:

- **Reasoning off where possible.** Summarization requests now ask for the model's *off* reasoning effort when its resolved metadata offers one; models without selectable efforts keep their existing behavior (no thinking control is sent).
- **Fail closed on truncation.** One shared truncation error covers both the max-tokens finish case and a new post-stream guard: a success-kind finish whose reported output tokens reach the configured cap now fails compaction with `compaction/summary-error` instead of checkpointing an incomplete summary.

### 4. [`e1986442af`](https://github.com/MichaelFiguresItOut/deepseek-harness/commit/e1986442af) — Disable reasoning for compaction without changing normal turns
*Files: `packages/llm/llm-pi-ai/src/adapter.ts` + tests (+113/−1)*

Makes fix 3's "off effort" actually reach the wire on `openai-completions` routes without touching anything else. The adapter's model handoff now gates on route + explicit effort: with **no** reasoning effort requested, pi-ai receives `{...model, reasoning:false}` so no thinking control is sent at all — normal turns are byte-identical to upstream. With an **explicit** effort (e.g., the summarizer's `off`), the declared model is preserved and pi-ai sends the route's wire spelling (`"none"` / `"minimal"`).

## What this fork does not change
- Normal (non-compaction) turns: identical wire payloads to upstream on every route.
- Routes that keep reasoning content on the wire: full-total usage accounting unchanged.
- Models without selectable reasoning efforts: summarizer behavior unchanged.
- Legacy replay envelopes: read and written exactly as before when no usage pair is present.

## Verification
- `packages/llm` full suite: **1739/1739** passing (includes new suites for replay usage preservation, reasoning-only pricing ×7, and the pi-ai hook ×5)
- `compaction-basic`: **174/174** passing (new regression tests for truncation detection at/over budget, missing terminal finish chunks, and effort routing)
- Focused regression run across all touched packages: **89/89**
- Typecheck: all 5 repository gates pass
- Live end-to-end against a local reasoning model (LM Studio): real compaction **2/2**, confirming the endpoint accepts `role:"developer"` and the `"none"` effort wire spelling

---
---

# DeepSeek Harness

English | [中文](README.zh.md)

DeepSeek Harness (`dsh`) is an open-source agent harness developed by [DeepSeek AI](https://deepseek.com).

It is built on an **everything-is-a-plugin** architecture and powered by [Cordis](https://github.com/cordiverse/cordis), whose design is described in [_A Programming Paradigm for Spatiotemporal Composability_](https://arxiv.org/abs/2608.25512).

Documentation: [https://deepseek-harness.github.io/deepseek-harness/](https://deepseek-harness.github.io/deepseek-harness/)

## Developer preview

DeepSeek Harness is in _developer preview_ and iterating rapidly. **THERE WILL BE COMPATIBILITY-BREAKING CHANGES.**

Review the [safety notice](SAFETY.md) before running the project.

## Run

### Run from `npm`

Install `Node.js`, then run:

```sh
npx @deepseek-ai/dsh web
```

The command starts the Web UI at `http://127.0.0.1:3080` by default and opens it in the default browser for a local launch. An SSH launch only prints the host URL because the SSH client or editor owns the local forwarded address. Pass `--no-open` to run the server without opening a browser. See [Web UI guide](docs/user/guide/index.md).

### Run from source

To run from a repository checkout:

```sh
git clone https://github.com/deepseek-ai/deepseek-harness.git
cd deepseek-harness
pnpm install
pnpm run build
pnpm dsh web
```

`pnpm run build` prepares the repository artifacts. `pnpm dsh web` uses those built artifacts without rebuilding.

## Community and support

- Submit feedback or bug reports through [GitHub Discussions](https://github.com/deepseek-ai/deepseek-harness/discussions).
- Add the [`dsh-plugin`](https://github.com/topics/dsh-plugin) topic to your plugin repository for discoverability.
- Join <a href="https://discord.gg/4MrtZUhpxg">DeepSeek Harness Discord community</a>.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Development

Start with the [development guide](docs/development.md) and [architecture documentation](docs/architecture.md).

`pnpm run dev:web` builds, serves, and rebuilds client bundles on source edits in one terminal, and `make help` lists the matching Make targets for Web and Desktop; the guide's application commands section owns the full table.

For agents, follow [AGENTS.md](AGENTS.md).

## Citation

```bibtex
@misc{deepseek-harness2026,
  title={DeepSeek Harness: Everything is a Plugin},
  author={DeepSeek-AI},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/deepseek-ai/deepseek-harness}},
}
```

## License

[MIT](LICENSE)

Third-party dependencies and their licenses are disclosed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
