// Optional local PostgreSQL-protocol test server, never imported by the application.
const {createRequire} = require('node:module');
const path = require('node:path');
const fs = require('node:fs');
const runtime = path.resolve(process.env.VECTOR_WASM_RUNTIME_DIR || 'data/vector_test_runtime');
const localRequire = createRequire(path.join(runtime, 'package.json'));
const {PGlite} = localRequire('@electric-sql/pglite');
const {vector} = localRequire('@electric-sql/pglite-pgvector');
const {PGLiteSocketServer} = localRequire('@electric-sql/pglite-socket');
const dataDir = path.resolve(process.env.VECTOR_WASM_DATA_DIR || 'data/vectorstore_wasm/db');
const port = Number(process.env.VECTOR_WASM_PORT || 55439);
if (!Number.isInteger(port) || port < 1024 || port > 65535) throw Error('invalid test port');
fs.mkdirSync(path.dirname(dataDir), {recursive: true});
(async () => {
  const db = new PGlite(dataDir, {extensions: {vector}});
  await db.waitReady;
  const server = new PGLiteSocketServer({db, host:'127.0.0.1', port, maxConnections:1});
  await server.start();
  console.log('VECTOR_WASM_TEST_READY');
  let stopping = false;
  const stop = async () => {
    if (stopping) return;
    stopping = true;
    await server.stop();
    await db.close();
    process.exit(0);
  };
  process.stdin.on('data', data => { if (data.toString().trim() === 'stop') stop(); });
  process.on('SIGINT', stop);
  process.on('SIGTERM', stop);
})().catch(error => { console.error(error.name); process.exit(1); });
