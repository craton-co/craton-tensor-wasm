# Signed kernel registry

_Status: scaffold landed in v0.3.7. Server wire landed in v0.3.7 (B7.5).
Production server-side storage: LANDED in v0.3.7 (T35) — disk-persisted
`DiskRegistry` backed by `tensor-wasm-artifacts::DiskArtifactStore`,
restart-safe, paginated, with an optional publisher allowlist._

Roadmap feature #3 (see [`PATH-TO-V1.md`](PATH-TO-V1.md#post-v036-strategic-features))
gives operators a way to publish vetted PTX kernels — matmul, attention,
conv2d — as first-class artifacts that guests can reference by stable
name, rather than re-emitting the kernel text on every JIT cache miss.
This doc covers the manifest schema, the signing flow, how kernels are
resolved at JIT cache miss time, and the security envelope.

## Motivation

Today every guest module that needs a fused matmul kernel either ships
its own PTX (huge code-size hit, no way to share across tenants) or
asks the auto-offload pipeline to re-derive it from Cranelift IR (works,
but the lowering pipeline is W3-W4 territory and is opt-in). For the
common case — "give me the canonical matmul.f32 for sm_80" — neither
option is right. Design partners need:

1. A way for the runtime operator (not the guest author) to vet a
   kernel once and have every tenant share the binary.
2. Signature verification so a compromised registry server cannot
   inject malicious PTX into the JIT cache.
3. Content-addressing so the same `(name, version, sm)` triple always
   produces bit-identical PTX across every host in the fleet.

The signed kernel registry is that surface.

## Manifest schema

A `KernelManifest` is a Rust struct (defined in
`crates/tensor-wasm-jit/src/registry.rs`) that serialises to JSON for
the v0.4 wire format. The fields:

| Field | Type | Signed? | Notes |
|---|---|---|---|
| `name` | `String` | yes | Stable identifier, e.g. `matmul.f32`. |
| `version` | `String` | yes | SemVer-style, e.g. `1.0.0`. |
| `sm_version` | `u32` | yes | Compute capability, e.g. `80` for sm_80. |
| `digest` | `[u8; 32]` | yes | BLAKE3 of the PTX text. |
| `signature` | `[u8; 32]` | — | HMAC-SHA256 over the envelope below. |
| `published_unix_ms` | `u64` | no (v0.3.7) | Advisory wall-clock metadata. |
| `publisher` | `String` | no (v0.3.7) | Tenant id or signing-key id. |

`published_unix_ms` and `publisher` are advisory in v0.3.7 — the v0.4
wire format will extend the signature envelope to cover them once a
stable canonical encoding is settled. Until then they MUST NOT be
trusted for authorization decisions.

## Signing envelope

The HMAC-SHA256 input is the byte concatenation

```text
name || 0x00 || version || 0x00 || sm_version_le_u32 || digest
```

The `0x00` separators prevent length-extension confusion between
neighbouring fields — without them, `("matmul", "f32-1.0")` and
`("matmul.f32", "-1.0")` would produce the same signed bytes.

Verification uses `subtle::ConstantTimeEq` so a timing oracle cannot
recover bits of the expected MAC through repeated rejected publishes.
This mirrors the constant-time bearer-token comparator in
`tensor-wasm-api/src/middleware.rs` and the snapshot signature check in
`tensor-wasm-snapshot/src/reader.rs`.

## Signing flow

```text
publisher                            registry server (v0.4)
---------                            ----------------------
1. emit PTX text -> ptx.bytes
2. digest = BLAKE3(ptx.bytes)
3. build KernelManifest{name, version, sm, digest, ...}
4. signature = HMAC-SHA256(envelope, hmac_key)
5. POST {server}/kernels { manifest, ptx }
                                     a. verify BLAKE3(ptx) == manifest.digest
                                     b. verify HMAC under trusted key
                                     c. reject duplicate name@version
                                     d. persist (manifest, ptx)
                                     e. 201 Created
```

`tensor-wasm kernel publish` is the CLI that runs steps 1-5. In v0.3.7
the CLI exits 3 (`FEATURE_NOT_EXPOSED`) because step (a)-(e) — the
server-side route — is not deployed yet. Design partners can still wire
the CLI into CI; the contract is stable.

## JIT cache resolution

When the JIT pipeline (`tensor-wasm-jit/src/cache.rs`) encounters a
cache miss for a kernel that the guest references by `(name@version)`,
the resolver:

1. Calls `KernelRegistry::get(name, version)`.
2. The registry returns `Arc<(KernelManifest, String)>` — the
   verified manifest and the PTX text. Verification has already
   happened at publish time, but `get` is allowed to re-verify if the
   backend is on disk (defence-in-depth against tampering at rest).
3. The JIT cache pre-populates an entry keyed by `(blueprint, sm)`
   from the resolved PTX. Subsequent invocations hit the in-memory
   cache directly.

The registry is layered _under_ the JIT cache: a registry hit looks
identical to a normal cache hit from the caller's perspective.

## Resolution flow

The JIT cache is a three-tier structure. `KernelCache::get_with_registry_fallback`
walks the tiers in order on every dispatch:

```text
                 ┌────────────────────────────────────────────────────┐
   dispatch ──▶  │  KernelCache::get_with_registry_fallback(key, res) │
                 └────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
                ┌─────────────────────────────────────┐
                │  L1: in-mem DashMap<CacheKey, …>    │
                │  hit?  ── yes ──▶ return CachedKernel
                └────────────────┬────────────────────┘
                                 │ miss
                                 ▼
                ┌─────────────────────────────────────┐
                │  L2: on-disk DiskCache (V2 .ptxbin) │
                │  HMAC-verify, promote into L1       │
                │  hit?  ── yes ──▶ return CachedKernel
                └────────────────┬────────────────────┘
                                 │ miss
                                 ▼
                ┌─────────────────────────────────────────────────┐
                │  L3: KernelRegistry (Option<Arc<dyn>>)          │
                │  step 1: resolver.resolve(blueprint_fp, sm)     │
                │           → Option<(name, version)>             │
                │  step 2: registry.get(&name, &version)          │
                │           → Arc<(KernelManifest, ptx_text)>     │
                │  step 3: promote into L1 via cache.put          │
                │  hit?  ── yes ──▶ return CachedKernel
                └────────────────┬────────────────────────────────┘
                                 │ miss
                                 ▼
                          return None
                  (caller re-emits via ptx_emit + put)
```

Each tier promotes hits into the tier above it: an L2 hit pre-populates
L1, and an L3 hit pre-populates both L1 and the in-memory promotion
path (L2 writeback is left to the next L1 `put` so the registry path
does not double-pay the HMAC write on a synchronous dispatch).

### v0.3.8 status

`tensor-wasm-jit` v0.3.8 ships the **client-side** cache plumbing:

- `KernelCacheConfig::with_registry` attaches an `Arc<dyn KernelRegistry>`.
- `BlueprintResolver` is the bridge trait — embedders provide the
  `(blueprint, sm) → (name, version)` mapping policy. An
  `InMemoryBlueprintResolver` wraps a `HashMap` for tests.
- `KernelCache::get_with_registry_fallback(key, resolver)` walks the
  three tiers above.

v0.4 ships the **server-side** `/kernels` endpoints (B7.5 is doing
this in parallel): the on-disk store, the `POST /kernels` publish
route, and the `GET /kernels/{name}/{version}` resolver-friendly fetch
that the v0.4 `BlueprintResolver` implementation will call.

## Security notes

### HMAC key rotation

Each registry holds a single 32-byte HMAC-SHA256 signing key, scrubbed
on Drop via `zeroize::Zeroizing`. Rotating the key requires re-signing
every manifest under the new key. The v0.4 server will support a
two-key window (`current_key`, `previous_key`) so publishers can roll
without an atomic flag-day.

### Multi-publisher allowlists

v0.3.7 ships a single-key registry — every publisher signs under the
same key. The v0.4 wire format will introduce a `publisher_keys` map on
the server side so each tenant (or signing-key id, see the `publisher`
manifest field) can sign under its own key. The allowlist is the
operator's responsibility; the registry refuses publishes from
unrecognised keys.

### Content-addressing as a defence

Because `manifest.digest = BLAKE3(ptx_text)` is part of the signed
envelope, an attacker who flips bits in the persisted PTX cannot keep
a valid signature. The `RegistryError::DigestMismatch` branch in
`InMemoryRegistry::publish` catches this at publish time; the v0.4
disk-backed registry repeats the check on every read.

### Why HMAC and not Ed25519?

The snapshot signing path (`tensor-wasm-snapshot`, see
[SNAPSHOT-FORMAT.md](SNAPSHOT-FORMAT.md)) is already HMAC-SHA256. Reusing the same
primitive avoids pulling a second curve implementation into the
default build, and matches the operator threat model: "everyone with
the key can produce signed artifacts" is the right answer for a
single-tenant registry. The v0.4 multi-publisher extension may layer
Ed25519 over HMAC for asymmetric publish-side keys, tracked under
RFC 0001 follow-up.

## CLI surface (v0.3.7 scaffold)

```bash
# Publish a signed PTX kernel (exits 3 in v0.3.7).
tensor-wasm kernel publish matmul.f32 1.0.0 \
    --ptx-file ./matmul.ptx \
    --sm 80 \
    --key-file ~/.tensor-wasm/registry.key \
    --server https://registry.example.com

# List server-side kernels (exits 3 in v0.3.7).
tensor-wasm kernel list --server https://registry.example.com

# Locally verify a manifest blob (exits 3 in v0.3.7).
tensor-wasm kernel verify matmul.f32@1.0.0 \
    --key-file ~/.tensor-wasm/registry.key
```

All three commands exit with code `3` (`FEATURE_NOT_EXPOSED`) and the
documented `"feature not yet exposed"` message in v0.3.7. CI can
distinguish "scaffold not yet wired" from "wrong arguments" by checking
`$? -eq 3` — see `crates/tensor-wasm-cli/src/cmd/kernel.rs` for the
exit-code rationale.

## v0.4 rollout plan

1. `POST /kernels` route on the API server, accepting a JSON manifest
   plus the PTX bytes as `multipart/form-data`. **LANDED (B7.5).**
2. On-disk store. **LANDED (T35)** via
   `tensor-wasm-jit::registry::DiskRegistry`, backed by
   `tensor-wasm-artifacts::DiskArtifactStore`. The store is
   content-addressed (BLAKE3 of the bincode-encoded
   `(KernelManifest, ptx_text)` blob), HMAC-SHA256-signed by the
   artifact store's outer envelope, zstd-compressed, and atomic on
   publish. An in-memory `(name, version, sm_version) → ContentHash`
   keymap sits in front so resolves are O(1) lookups. The registry is
   rebuilt from the on-disk blobs on `DiskRegistry::open`, so manifests
   survive process restarts. The chosen env var is
   `TENSOR_WASM_API_KERNEL_REGISTRY_DIR`; unset leaves the gateway on
   the historical `InMemoryRegistry` (dev mode).
3. `GET /kernels` and `GET /kernels/{name}/{version}` for the CLI's
   `list` and the JIT cache resolver. **LANDED (B7.5).** T35 added
   `?offset=N&limit=M` pagination to the `GET /kernels` route, with
   `limit` clamped to 1000 server-side; the response echoes the
   effective `offset` and `limit` so clients can drive cursor-style
   pagination without local math.
4. CLI flips from exit-3 scaffold to the real flow. Smoke tests in
   `crates/tensor-wasm-cli/tests/cli_smoke.rs` change shape; the
   integration tests in
   `crates/tensor-wasm-jit/tests/kernel_registry_scaffold.rs` stay
   untouched (they exercise the in-memory backend). T35 adds
   `disk_registry_restart.rs`,
   `disk_registry_pagination.rs`, and
   `disk_registry_publisher_allowlist.rs` to exercise the disk path.

### Publisher allowlist (T35)

`DiskRegistry` accepts an optional
`publisher_allowlist: HashSet<String>` chained on after
`DiskRegistry::open(...).with_publisher_allowlist(...)`. When set, the
registry refuses `publish` if `manifest.publisher` is not in the set,
even when the manifest's v2 signature otherwise verifies. This is a
SEPARATE authorization layer over and above the T1 HTTP kernel-publish
scope gate — the scope gate decides who can call `POST /kernels` at
all, the allowlist decides which publisher identity the body can
claim. Together they prevent a single signing-key holder from
impersonating a peer publisher.

When the allowlist is `None` (default) every signed publisher is
accepted, matching today's permissive `InMemoryRegistry` behaviour.

## Related docs

- [PATH-TO-V1.md](PATH-TO-V1.md) — roadmap.
- [SECURITY.md](../SECURITY.md) — threat model.
- [SNAPSHOT-FORMAT.md](SNAPSHOT-FORMAT.md) — prior art on
  HMAC-SHA256-signed artifacts (snapshots v3).
- [CUDA-KERNELS.md](CUDA-KERNELS.md) — kernel authoring guide; the
  registry is the distribution channel for the kernels written using
  the surface documented there.
