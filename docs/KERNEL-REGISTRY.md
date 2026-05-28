# Signed kernel registry

_Status: scaffold landed in v0.3.7. Server wire + on-disk store land in v0.4._

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
   plus the PTX bytes as `multipart/form-data`.
2. On-disk store under `<data-dir>/kernels/<name>/<version>/` with the
   manifest as `manifest.json` and the PTX as `kernel.ptx`. The same
   atomic-rename pattern the snapshot writer uses.
3. `GET /kernels` and `GET /kernels/{name}/{version}` for the CLI's
   `list` and the JIT cache resolver.
4. CLI flips from exit-3 scaffold to the real flow. Smoke tests in
   `crates/tensor-wasm-cli/tests/cli_smoke.rs` change shape; the
   integration tests in
   `crates/tensor-wasm-jit/tests/kernel_registry_scaffold.rs` stay
   untouched (they exercise the in-memory backend).

## Related docs

- [PATH-TO-V1.md](PATH-TO-V1.md) — roadmap.
- [SECURITY.md](../SECURITY.md) — threat model.
- [SNAPSHOT-FORMAT.md](SNAPSHOT-FORMAT.md) — prior art on
  HMAC-SHA256-signed artifacts (snapshots v3).
- [CUDA-KERNELS.md](CUDA-KERNELS.md) — kernel authoring guide; the
  registry is the distribution channel for the kernels written using
  the surface documented there.
