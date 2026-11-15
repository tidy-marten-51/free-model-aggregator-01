// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// Routing: for each client request, every free model that meets the manifest's requirements for it is
// a candidate, ranked by the quality table's rating_milli (best first). Candidates are tried in that
// order until one answers 2xx: a refusal (402, 429, 5xx, a network error, an unparseable body) moves
// to the next candidate, so a provider's spent budget, failure schedule or outage costs the request
// nothing but the extra call. Requests the client marks as JSON mode never route to an anthropic-wire
// provider, whose wire cannot carry the JSON-mode flag the provider echoes back.
//
// Wires: the client request is normalized to the Chat Completions shape, then rebuilt in the serving
// provider's own wire (OpenAI, Anthropic or Gemini), always as one non-streaming call; the answer is
// converted back to the client's wire, as JSON or as a rendered SSE stream when the client streamed.
// Every field the providers echo (system prompt, last user text, tool names, max tokens, stops, image
// bytes, JSON mode, usage, stop reason) is carried through each translation unchanged.
//
// The brief, the catalog probe, the register and the /work poll are the kit's: the catalog built once
// from the probes is both the `--list` document and the register body, byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicResponseToOpenAI,
  anthropicSseFromMessage,
  anthropicToOpenAI,
  geminiResponseToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  openAISseFromCompletion,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 15000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Inference attempts per provider id, for spreading equal-rated picks across providers. */
const providerCalls = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every legal pick for one request, best first: free models meeting the requirements, ranked by the
 * quality table's rating; equal ratings spread over the providers used least so far, then by provider
 * and model id so the order is a pure function of the brief and the calls already made. A JSON-mode
 * request skips anthropic-wire providers, whose wire has no JSON-mode flag to echo.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      (providerCalls.get(a.probe.provider.id) ?? 0) - (providerCalls.get(b.probe.provider.id) ?? 0) ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamResult {
  status: number;
  body: Json | null;
  raw: string;
}

/** One non-streaming inference call to a candidate, the request rebuilt in the provider's own wire. */
async function callCandidate(candidate: Candidate, canonical: Json): Promise<UpstreamResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = { ...canonical, model: candidate.model.id };
    delete payload.stream;
    delete payload.stream_options;
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    payload = openAIToAnthropicRequest(canonical, candidate.model.id);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:generateContent`;
    payload = openAIToGeminiRequest(canonical);
  }
  providerCalls.set(provider.id, (providerCalls.get(provider.id) ?? 0) + 1);
  const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  const raw = await res.text();
  let body: Json | null = null;
  try {
    const value: unknown = JSON.parse(raw);
    if (isRecord(value)) body = value;
  } catch {
    // Not JSON: the caller treats the call as failed and moves to the next candidate.
  }
  return { status: res.status, body, raw };
}

/** A provider's answer as one Chat Completions object, or null when the body is not a usable answer. */
function toCanonical(candidate: Candidate, body: Json): Json | null {
  const wire = candidate.probe.provider.wire;
  if (wire === 'openai') {
    if (!Array.isArray(body.choices) || body.choices.length === 0) return null;
    return body;
  }
  if (wire === 'anthropic') {
    if (!Array.isArray(body.content)) return null;
    return anthropicResponseToOpenAI(body, candidate.model.id);
  }
  if (!Array.isArray(body.candidates) || body.candidates.length === 0) return null;
  return geminiResponseToOpenAI(body, candidate.model.id);
}

/**
 * Serve one client request on either client wire: normalize to the Chat Completions shape, try the
 * ranked candidates until one answers, then render the answer in the client's wire, as JSON or as an
 * SSE stream when the client asked for one.
 */
async function serve(clientWire: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const canonical = clientWire === 'anthropic' ? anthropicToOpenAI(body, 'aggregate') : body;
  const format = isRecord(canonical.response_format) ? canonical.response_format.type : undefined;
  const jsonMode = format === 'json_object' || format === 'json_schema';
  const candidates = candidatesFor(requirementsFor(body), jsonMode);
  if (candidates.length === 0) {
    return sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  let lastStatus = 502;
  let lastText = '';
  for (const candidate of candidates) {
    let result: UpstreamResult;
    try {
      result = await callCandidate(candidate, canonical);
    } catch (err) {
      lastStatus = 502;
      lastText = (err as Error).message;
      continue;
    }
    if (result.status < 200 || result.status >= 300 || result.body === null) {
      lastStatus = result.status;
      lastText = result.raw.slice(0, 500);
      continue;
    }
    const completion = toCanonical(candidate, result.body);
    if (completion === null) {
      lastStatus = 502;
      lastText = 'the provider answered a body that is not a usable answer';
      continue;
    }
      if (clientWire === 'openai') {
        if (body.stream === true) sendSse(res, openAISseFromCompletion(completion));
        else sendJson(res, 200, completion);
      } else {
        const requested = typeof body.model === 'string' && body.model !== '' ? body.model : candidate.model.id;
        const message = openAIToAnthropic(completion, requested);
        if (body.stream === true) sendSse(res, anthropicSseFromMessage(message));
        else sendJson(res, 200, message);
      }
      return;
  }
  const status = lastStatus >= 400 && lastStatus <= 599 ? lastStatus : 502;
  sendError(res, clientWire, status, clientWire === 'anthropic' ? 'api_error' : 'upstream_error', `every free model that meets this request refused it; the last answered HTTP ${lastStatus}: ${lastText}`);
}

/** POST /v1/chat/completions: an OpenAI-shaped client request. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client request. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
