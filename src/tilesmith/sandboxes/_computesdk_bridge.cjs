"use strict";

const readline = require("node:readline");
const path = require("node:path");
const util = require("node:util");
const { createRequire } = require("node:module");

let compute;
let computesdk;
const providers = new Map();
const sandboxes = new Map();
const ownedSandboxIds = new Set();
const createOperations = new Map();
const destroyOperations = new Map();
const maxResponseBytes = 128 * 1024 * 1024;
const maxQueuedStreamBytes = 8 * 1024 * 1024;
let outputQueue = Promise.resolve();
let queuedStreamBytes = 0;
let shuttingDown = false;

/** Keep provider diagnostics away from the JSON-lines protocol on stdout. */
function redirectConsole() {
  const write = (...values) => process.stderr.write(`${util.format(...values)}\n`);
  for (const name of ["log", "info", "debug", "warn", "error"]) {
    console[name] = write;
  }
}

/** Resolve packages installed into npm exec's temporary node_modules tree. */
function packageLoader() {
  const entries = (process.env.PATH || "").split(path.delimiter);
  for (const entry of entries) {
    if (path.basename(entry) !== ".bin" || path.basename(path.dirname(entry)) !== "node_modules") {
      continue;
    }
    const loader = createRequire(path.join(path.dirname(entry), "tilesmith-computesdk-bridge.cjs"));
    try {
      loader.resolve("computesdk");
      return loader;
    } catch {
      // npm may place another executable directory earlier on PATH.
    }
  }
  throw new Error("npm exec did not expose the requested ComputeSDK packages");
}

/** Convert values such as bigint into JSON-safe protocol values. */
function serialize(value) {
  return JSON.stringify(value, (_key, item) => typeof item === "bigint" ? item.toString() : item);
}

/** Queue one bounded protocol message and honor stdout backpressure. */
function send(value) {
  let encoded = serialize(value);
  if (Buffer.byteLength(encoded) > maxResponseBytes) {
    const id = value.id ?? value.requestId;
    encoded = serialize({ id, error: { name: "RangeError", message: "ComputeSDK response exceeds 128 MiB" } });
  }
  const line = `${encoded}\n`;
  const size = Buffer.byteLength(line);
  if (value.event && queuedStreamBytes + size > maxQueuedStreamBytes) {
    return Promise.resolve();
  }
  if (value.event) {
    queuedStreamBytes += size;
  }
  const completion = outputQueue.then(() => new Promise((resolve, reject) => {
    if (process.stdout.write(line)) {
      resolve();
      return;
    }
    const onDrain = () => {
      process.stdout.off("error", onError);
      resolve();
    };
    const onError = (error) => {
      process.stdout.off("drain", onDrain);
      reject(error);
    };
    process.stdout.once("drain", onDrain);
    process.stdout.once("error", onError);
  }));
  outputQueue = completion.catch(() => {});
  return completion.finally(() => {
    if (value.event) {
      queuedStreamBytes -= size;
    }
  });
}

/** Return the stable identity needed to reconstruct a Python Sandbox handle. */
function describeSandbox(sandbox) {
  return { sandboxId: sandbox.sandboxId, provider: sandbox.provider };
}

/** Strip provider-native objects from ComputeSDK's universal snapshot result. */
function describeSnapshot(snapshot) {
  return {
    id: snapshot.id,
    provider: snapshot.provider,
    createdAt: snapshot.createdAt,
    metadata: snapshot.metadata,
  };
}

/** Resolve a sandbox handle, reconnecting through ComputeSDK when necessary. */
async function getSandbox(sandboxId) {
  let sandbox = sandboxes.get(sandboxId);
  if (!sandbox) {
    sandbox = await compute.sandbox.getById(sandboxId);
    if (sandbox) {
      sandboxes.set(sandboxId, sandbox);
    }
  }
  if (!sandbox) {
    throw new Error(`Sandbox "${sandboxId}" was not found`);
  }
  return sandbox;
}

/** Resolve a configured provider for provider-scoped template operations. */
function getProvider(providerName) {
  const provider = providers.get(providerName);
  if (!provider) {
    throw new Error(`Provider "${providerName}" is not configured`);
  }
  return provider;
}

/** Load provider factories and create one isolated ComputeSDK manager. */
async function configure(request) {
  const load = packageLoader();
  computesdk = load("computesdk");
  const configured = [];
  for (const specification of request.providers) {
    const module = load(specification.package);
    const factory = module[specification.factory];
    if (typeof factory !== "function") {
      throw new Error(`${specification.package} does not export provider factory "${specification.factory}"`);
    }
    const provider = await factory(specification.config || {});
    if (!provider || typeof provider !== "object" || typeof provider.name !== "string") {
      throw new Error(`${specification.package}.${specification.factory} did not return a ComputeSDK provider`);
    }
    // ComputeSDK keeps the first provider with a given name; template routing must agree.
    if (providers.has(provider.name)) {
      continue;
    }
    configured.push(provider);
    providers.set(provider.name, provider);
  }
  compute = computesdk.compute({
    providers: configured,
    providerStrategy: request.providerStrategy,
    fallbackOnError: request.fallbackOnError,
  });
  return { providers: configured.map((provider) => provider.name) };
}

/** Abort and settle sandbox creations before process-level cleanup starts. */
async function abortCreateOperations() {
  const operations = [...createOperations.values()];
  for (const operation of operations) {
    operation.controller.abort();
  }
  await Promise.allSettled(operations.map((operation) => operation.completion));
}

/** Terminate a provider sandbox and wait for provider-confirmed teardown when available. */
async function terminateSandbox(sandbox) {
  const instance = typeof sandbox.getInstance === "function" ? sandbox.getInstance() : undefined;
  const nativeSandbox = instance?.sandbox ?? instance;
  if (nativeSandbox && typeof nativeSandbox.terminate === "function") {
    await nativeSandbox.terminate({ wait: true });
    return;
  }
  await sandbox.destroy();
}

/** Reuse one in-flight provider teardown for every request targeting an ID. */
async function destroySandboxById(sandboxId) {
  let operation = destroyOperations.get(sandboxId);
  if (!operation) {
    operation = (async () => {
      const sandbox = sandboxes.get(sandboxId);
      if (sandbox) {
        await terminateSandbox(sandbox);
      } else {
        await compute.sandbox.destroy(sandboxId);
      }
      sandboxes.delete(sandboxId);
      ownedSandboxIds.delete(sandboxId);
    })();
    destroyOperations.set(sandboxId, operation);
  }
  try {
    await operation;
  } finally {
    if (destroyOperations.get(sandboxId) === operation) {
      destroyOperations.delete(sandboxId);
    }
  }
}

/** Destroy every sandbox created by this bridge before the Node process exits. */
async function destroyOwnedSandboxes() {
  const results = await Promise.allSettled([...ownedSandboxIds].map(async (sandboxId) => {
    await destroySandboxById(sandboxId);
  }));
  const failures = results.filter((result) => result.status === "rejected");
  if (failures.length > 0) {
    throw new AggregateError(failures.map((result) => result.reason), "Failed to destroy every owned ComputeSDK sandbox");
  }
}

/** Make abnormal bridge termination perform the same best-effort resource cleanup. */
async function shutdown(exitCode) {
  if (shuttingDown) {
    return;
  }
  shuttingDown = true;
  try {
    if (compute) {
      await abortCreateOperations();
      await destroyOwnedSandboxes();
    }
  } catch (error) {
    console.error(error);
  } finally {
    process.exit(exitCode);
  }
}

/** Dispatch one Python request onto the exact modern ComputeSDK operation. */
async function dispatch(request) {
  switch (request.action) {
    case "configure":
      return configure(request);
    case "create": {
      const controller = new AbortController();
      let finish;
      const completion = new Promise((resolve) => { finish = resolve; });
      createOperations.set(request.id, { controller, completion });
      try {
        const options = { ...(request.options || {}), signal: controller.signal };
        const sandbox = await compute.sandbox.create(options);
        sandboxes.set(sandbox.sandboxId, sandbox);
        ownedSandboxIds.add(sandbox.sandboxId);
        return describeSandbox(sandbox);
      } finally {
        finish();
        createOperations.delete(request.id);
      }
    }
    case "cancel": {
      const operation = createOperations.get(request.targetId);
      if (operation) {
        operation.controller.abort();
        await operation.completion;
      }
      return null;
    }
    case "getById": {
      const sandbox = await compute.sandbox.getById(request.sandboxId);
      if (!sandbox) {
        return null;
      }
      sandboxes.set(sandbox.sandboxId, sandbox);
      return describeSandbox(sandbox);
    }
    case "listSandboxes": {
      const listed = await compute.sandbox.list();
      for (const sandbox of listed) {
        sandboxes.set(sandbox.sandboxId, sandbox);
      }
      return listed.map(describeSandbox);
    }
    case "destroySandbox":
      await destroySandboxById(request.sandboxId);
      return null;
    case "sandboxDestroy": {
      await getSandbox(request.sandboxId);
      await destroySandboxById(request.sandboxId);
      return null;
    }
    case "runCommand": {
      const sandbox = await getSandbox(request.sandboxId);
      const options = { ...(request.options || {}) };
      if (request.streamStdout) {
        options.onStdout = (data) => void send({ event: "stdout", requestId: request.id, data });
      }
      if (request.streamStderr) {
        options.onStderr = (data) => void send({ event: "stderr", requestId: request.id, data });
      }
      return sandbox.runCommand(request.command, options);
    }
    case "sandboxWaitUntilReady": {
      const sandbox = await getSandbox(request.sandboxId);
      const instance = typeof sandbox.getInstance === "function" ? sandbox.getInstance() : undefined;
      const nativeSandbox = instance?.sandbox ?? instance;
      if (nativeSandbox && typeof nativeSandbox.waitUntilReady === "function") {
        await nativeSandbox.waitUntilReady(request.timeoutMs);
        return null;
      }
      const result = await sandbox.runCommand("true", { timeout: request.timeoutMs });
      if (result.exitCode !== 0) {
        throw new Error(result.stderr || result.stdout || `Readiness command exited ${result.exitCode}`);
      }
      return null;
    }
    case "getInfo":
      return (await getSandbox(request.sandboxId)).getInfo();
    case "getUrl":
      return (await getSandbox(request.sandboxId)).getUrl(request.options);
    case "readFile":
      return (await getSandbox(request.sandboxId)).filesystem.readFile(request.path);
    case "writeFile":
      await (await getSandbox(request.sandboxId)).filesystem.writeFile(request.path, request.content);
      return null;
    case "mkdir":
      await (await getSandbox(request.sandboxId)).filesystem.mkdir(request.path);
      return null;
    case "readdir":
      return (await getSandbox(request.sandboxId)).filesystem.readdir(request.path);
    case "exists":
      return (await getSandbox(request.sandboxId)).filesystem.exists(request.path);
    case "remove":
      await (await getSandbox(request.sandboxId)).filesystem.remove(request.path);
      return null;
    case "createSnapshot":
      return describeSnapshot(await compute.snapshot.create(request.sandboxId, request.options));
    case "listSnapshots":
      return (await compute.snapshot.list()).map(describeSnapshot);
    case "deleteSnapshot":
      await compute.snapshot.delete(request.snapshotId);
      return null;
    case "createTemplate": {
      const manager = getProvider(request.provider).template;
      if (!manager) {
        throw new Error(`Provider "${request.provider}" does not support templates`);
      }
      return manager.create(request.options);
    }
    case "listTemplates": {
      const manager = getProvider(request.provider).template;
      if (!manager) {
        throw new Error(`Provider "${request.provider}" does not support templates`);
      }
      return manager.list(request.options);
    }
    case "deleteTemplate": {
      const manager = getProvider(request.provider).template;
      if (!manager) {
        throw new Error(`Provider "${request.provider}" does not support templates`);
      }
      await manager.delete(request.templateId);
      return null;
    }
    case "daemonSeedScript":
      return computesdk.daemonSeedScript(request.config);
    case "daemonSeedScriptCommand":
      return computesdk.daemonSeedScriptCommand(request.config, request.payload);
    case "parseSeedInvocationOutput":
      return computesdk.parseSeedInvocationOutput(request.output);
    case "close":
      await abortCreateOperations();
      await destroyOwnedSandboxes();
      return null;
    default:
      throw new Error(`Unknown ComputeSDK bridge action "${request.action}"`);
  }
}

/** Process requests concurrently so cancellation and cleanup are never queued behind work. */
async function handleLine(line) {
  let request;
  try {
    request = JSON.parse(line);
    const result = await dispatch(request);
    await send({ id: request.id, result });
    if (request.action === "close") {
      setImmediate(() => process.exit(0));
    }
  } catch (error) {
    const name = error instanceof Error ? error.name : "Error";
    const message = error instanceof Error ? error.message : String(error);
    await send({ id: request?.id, error: { name, message } });
  }
}

redirectConsole();
const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
lines.on("line", (line) => {
  if (line.trim()) {
    void handleLine(line);
  }
});
lines.on("close", () => void shutdown(0));
process.on("SIGINT", () => void shutdown(130));
process.on("SIGTERM", () => void shutdown(143));
process.on("uncaughtException", (error) => {
  console.error(error);
  void shutdown(1);
});
process.on("unhandledRejection", (error) => {
  console.error(error);
  void shutdown(1);
});
