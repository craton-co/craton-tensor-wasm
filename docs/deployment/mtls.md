<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Terminating TLS (and mTLS) in front of Craton TensorWasm

This document is the v0.4 "mTLS support optional but documented" criterion
from [`PATH-TO-V1.md`](../PATH-TO-V1.md). It covers two production
deployment shapes:

- **Architecture A — self-terminated mTLS** at the `tensor-wasm-api`
  process using rustls + `axum-server`.
- **Architecture B — reverse-proxy-terminated mTLS** in front of a
  plaintext `tensor-wasm-api` (nginx, Envoy, Caddy).

It is honest about the v0.4 binary: today `tensor-wasm-api` binds a
plaintext `tokio::net::TcpListener` and serves cleartext HTTP/1.1
(see [`server.rs`](../../crates/tensor-wasm-api/src/server.rs)).
Architecture A is the documented future end-state; until the rustls
feature flag and env-var plumbing below land, the **recommended
production path is Architecture B** — terminate mTLS in a proxy you
already trust and forward plaintext over a private network. Each gap
between this doc and the current binary is marked inline as
**TODO (v0.4):**.

## Contents

1. [Why mTLS](#1-why-mtls) — 2. [Auth-model interaction](#2-how-tls-interacts-with-the-bearer-auth-model) — 3. [Architecture A](#3-architecture-a--self-terminated-mtls) — 4. [Architecture B](#4-architecture-b--reverse-proxy-mtls) — 5. [Kubernetes / Helm](#5-kubernetes-and-helm-integration) — 6. [Rotation / revocation](#6-rotation-and-revocation) — 7. [Troubleshooting](#7-troubleshooting) — 8. [Limitations](#8-limitations-and-what-is-not-in-v04) — 9. [Related](#9-related)

---

## 1. Why mTLS

Bearer tokens alone (`TENSOR_WASM_API_TOKENS`, see
[`API.md`](../../crates/tensor-wasm-api/API.md#authentication)) prove
the caller knows a string. That is enough for many deployments and is
the v0.1–v0.4 default. mTLS adds value in three cases:

1. **Machine-to-machine zero trust.** Inside a service mesh whose
   contract is "every hop authenticates the next hop", a bearer
   header does not satisfy the contract. mTLS proves at the transport
   layer that the peer holds a private key signed by a CA your
   control plane trusts.
2. **Audit and provenance.** A bearer says "someone with this secret
   called us". A client cert says "the subject named in this cert,
   validated against this CA chain, called us". The cert Subject DN
   (or SPIFFE ID) is what an auditor wants — not an opaque token hash.
3. **Defense in depth against token leakage.** A leaked bearer in a
   stray log line is sufficient for an attacker. A leaked bearer plus
   a private key that never leaves the client host is not.

mTLS is **not** a replacement for the v0.1-v0.4 workstreams: not for
[W1.4 rate limiting](../../crates/tensor-wasm-api/API.md#per-token-rate-limiting)
(an authenticated client can still exhaust a node), not for
[W2.1 scoped tokens](../../crates/tensor-wasm-api/API.md#per-tenant-scopes-tensor_wasm_api_tokens-entry-forms)
(scoping authorizes the *action*; mTLS authenticates the *caller*),
and not for the v0.4 audit log (mTLS gives the log a stronger
principal; it does not eliminate the log).

If your threat model is "trusted internal callers on a private VPC",
bearer + scoped tokens is sufficient. If callers reach you from
networks you do not control, or compliance requires transport-layer
caller identity, read on.

---

## 2. How TLS interacts with the bearer auth model

The TensorWasm middleware stack (see `build_router` in
[`server.rs`](../../crates/tensor-wasm-api/src/server.rs#L37)) is
fixed. mTLS slots in **before** the trace layer:

```
TLS handshake (rustls or proxy)
  -> trace -> body_limit -> timeout -> concurrency_limit
  -> bearer_auth  (populates AuthContext { token_id, scope })
  -> tenant_scope (parses X-TensorWasm-Tenant)
  -> rate_limit   (keyed by AuthContext.token_id)
  -> handler      (per-tenant scope check inside the handler)
```

The bearer check is **not** removed when mTLS is enabled. The layers
cooperate:

| Layer            | Authenticates                | Authorizes                            |
|------------------|------------------------------|---------------------------------------|
| TLS handshake    | Client (cert + private key)  | Connection is allowed at all          |
| Bearer header    | Caller (knowledge of secret) | Caller appears in the allowlist       |
| `tenant_scope`   | n/a                          | Token scope covers the target tenant  |
| `rate_limit`     | n/a                          | Caller has not exhausted their bucket |

A request with a valid client cert but no `Authorization` header
still returns `401 unauthorized`. A request with a valid bearer but
no client cert never reaches the gateway — the TLS handshake fails
first.

**Operational consequence.** Turning on mTLS does not turn off
bearer auth — you keep both, and the bearer becomes the tenant-scope
selector for the cert-authenticated host. This keeps the v0.4 scope
and audit machinery usable without a redesign.

**TODO (v0.4):** when self-terminated mTLS lands, the bearer layer
should learn an optional mode
(`TENSOR_WASM_API_TLS_USE_CERT_AS_PRINCIPAL=1`) that keys the
rate-limit bucket by client-cert Subject CN rather than by bearer
string. Until then the bucket is per-bearer — correct but coarser
than the cert principal.

---

## 3. Architecture A — self-terminated mTLS

`tensor-wasm-api` itself owns the TLS handshake. No proxy in front;
the listener is a `tokio_rustls::TlsAcceptor` wrapped around
`axum_server`.

### 3.1 Status today

**TODO (v0.4):** the v0.1 binary does not implement this. The
listener in [`serve()`](../../crates/tensor-wasm-api/src/server.rs#L106)
is a plain `TcpListener` and the crate exposes no `tls` feature.
Architecture A becomes real once the following land:

- A new optional cargo feature `tls` on `tensor-wasm-api` pulling in
  `tokio-rustls`, `rustls`, `rustls-pemfile`, and `axum-server` (with
  `tls-rustls`).
- A new `serve_tls()` path that reads the env vars below, builds a
  `rustls::ServerConfig`, and hands it to
  `axum_server::bind_rustls(addr, cfg).serve(router)`.
- A note in [`API.md`](../../crates/tensor-wasm-api/API.md#authentication)
  that TLS is optional and env-configured, mirroring the bearer
  grammar.

Until those land, **read this section as a design spec, not a
runbook.** If the proposed names change at implementation time,
update this doc in the same commit.

### 3.2 Proposed env vars

| Env var                                       | Default | Required when                | Meaning                                                                  |
|-----------------------------------------------|---------|------------------------------|--------------------------------------------------------------------------|
| `TENSOR_WASM_API_TLS_CERT`                    | unset   | TLS is enabled               | Path to PEM-encoded server certificate (single cert or chain).           |
| `TENSOR_WASM_API_TLS_KEY`                     | unset   | TLS is enabled               | Path to PEM-encoded server private key (PKCS#8 preferred).               |
| `TENSOR_WASM_API_TLS_CLIENT_CA`               | unset   | mTLS is required             | Path to PEM-encoded CA bundle used to validate client certificates.      |
| `TENSOR_WASM_API_TLS_CLIENT_AUTH`             | `none`  | always (when TLS is enabled) | One of `none`, `optional`, `required`. `required` is mTLS.               |
| `TENSOR_WASM_API_TLS_USE_CERT_AS_PRINCIPAL`   | `0`     | optional                     | When `1`, use the client cert Subject CN/SAN as the rate-limit key.      |

Proposed startup rules:

- If `..._TLS_CERT` is set but `..._TLS_KEY` is not (or vice versa),
  the binary **fails to start**. Half-configured TLS is never the
  user's intent.
- If `..._CLIENT_AUTH=required` but `..._CLIENT_CA` is unset, the
  binary **fails to start** — nothing to validate against.
- If `..._CLIENT_AUTH` is unset and `..._CLIENT_CA` is set, the
  binary upgrades to `required` with a startup warning ("you
  probably meant mTLS").
- If neither cert nor key is set, the binary falls back to the
  current plaintext listener. There is no silent TLS upgrade.

### 3.3 Generating a test CA, server cert, and client cert

For dev/staging only — in production, use your existing PKI
(cert-manager, Vault PKI, an internal CA).

```bash
mkdir -p tw-tls && cd tw-tls

# Root CA (10-year self-signed).
openssl genrsa -out ca.key 4096
openssl req -new -x509 -days 3650 -key ca.key -out ca.pem \
  -subj "/CN=TensorWasm Test Root CA/O=Craton TensorWasm Dev"

# Server cert. SAN must include every hostname the API answers on.
cat > server.cnf <<'EOF'
[req]
distinguished_name = dn
req_extensions     = ext
prompt             = no
[dn]
CN = tensor-wasm.example.internal
[ext]
subjectAltName = @alt
[alt]
DNS.1 = tensor-wasm.example.internal
DNS.2 = localhost
IP.1  = 127.0.0.1
EOF
openssl genrsa -out server.key 4096
openssl req -new -key server.key -out server.csr -config server.cnf
openssl x509 -req -in server.csr -CA ca.pem -CAkey ca.key \
  -CAcreateserial -out server.pem -days 365 \
  -extensions ext -extfile server.cnf

# Client cert. The Subject CN becomes the audit-log principal.
openssl genrsa -out client.key 4096
openssl req -new -key client.key -out client.csr \
  -subj "/CN=client-prod/O=Craton TensorWasm Dev Client"
openssl x509 -req -in client.csr -CA ca.pem -CAkey ca.key \
  -CAcreateserial -out client.pem -days 365

# rustls strongly prefers PKCS#8 keys.
openssl pkcs8 -topk8 -nocrypt -in server.key -out server.pk8.pem
```

Sanity-check:

```bash
openssl verify -CAfile ca.pem server.pem
openssl verify -CAfile ca.pem client.pem
openssl x509  -in server.pem -noout -text | grep -A1 "Subject Alternative Name"
```

### 3.4 Server-side configuration

```bash
export TENSOR_WASM_API_TLS_CERT=/etc/tensor-wasm/tls/server.pem
export TENSOR_WASM_API_TLS_KEY=/etc/tensor-wasm/tls/server.pk8.pem
export TENSOR_WASM_API_TLS_CLIENT_CA=/etc/tensor-wasm/tls/ca.pem
export TENSOR_WASM_API_TLS_CLIENT_AUTH=required

# Bearer auth still required; mTLS authenticates the caller, the
# bearer selects the tenant scope.
export TENSOR_WASM_API_TOKENS='client-prod:tenant=*'

tensor-wasm serve --addr 0.0.0.0:8443
```

The bind address moves to a TLS port (8443). The container
`containerPort` in the
[k8s reference Deployment](../../deploy/k8s/20-deployment.yaml#L66-L69)
should update to `8443` and the named port should change from `http`
to `https` when this configuration is used.

### 3.5 Client-side curl example

```bash
curl --cacert ca.pem --cert client.pem --key client.key \
     -H 'Authorization: Bearer client-prod' \
     -H 'X-TensorWasm-Tenant: 7' \
     https://tensor-wasm.example.internal:8443/healthz
```

A missing bearer returns `401 unauthorized` from the bearer layer.
A missing client cert under `TLS_CLIENT_AUTH=required` fails the
handshake (curl prints `SSL_ERROR_BAD_CERT_REQUIRED` or similar) and
no HTTP request is sent.

### 3.6 Build / feature flag

**TODO (v0.4):** Architecture A requires building with the
not-yet-existing `tls` feature:

```bash
cargo build --release -p tensor-wasm-api --features tls
```

The container image in
[`docker/tensor-wasm-api.Dockerfile`](../../docker/tensor-wasm-api.Dockerfile)
should grow a `--features tls` arg flip (or a parallel `-tls` tag) so
operators do not rebuild from source. If `tensor-wasm serve` sees
`TENSOR_WASM_API_TLS_CERT` on a binary built without the feature, it
should fail fast with a cooperative diagnostic naming the missing
build feature.

---

## 4. Architecture B — reverse-proxy mTLS

This is the **recommended production path until Architecture A
lands**. TensorWasm listens on plaintext localhost or a private
cluster network; a battle-tested proxy handles the TLS handshake,
validates the client cert, and forwards a plaintext request with the
client identity in a header.

```
external caller ==HTTPS+mTLS==> reverse proxy --plaintext--> tensor-wasm-api
(curl, SDK, ...)                (nginx/Envoy/Caddy)          (plaintext :8080)
```

### 4.1 nginx

```nginx
server {
    listen 443 ssl http2;
    server_name tensor-wasm.example.com;

    ssl_certificate     /etc/nginx/tls/server.pem;
    ssl_certificate_key /etc/nginx/tls/server.key;
    ssl_protocols TLSv1.3;
    ssl_prefer_server_ciphers on;

    # mTLS. Use `optional` to fall back to bearer-only when no client
    # cert is presented.
    ssl_verify_client       on;
    ssl_client_certificate  /etc/nginx/tls/ca.pem;
    ssl_verify_depth        2;

    # Tell the upstream which client called us. The v0.4 audit-log work
    # (W2.2) reads X-Forwarded-Client-Cert when the trusted-proxy CIDR
    # check passes.
    proxy_set_header X-Forwarded-Client-Cert
        "Subject=\"$ssl_client_s_dn\";Hash=\"$ssl_client_fingerprint\"";
    proxy_set_header X-Client-DN          $ssl_client_s_dn;
    proxy_set_header X-Client-Verify      $ssl_client_verify;
    proxy_set_header X-Client-Fingerprint $ssl_client_fingerprint;

    # IMPORTANT: do NOT strip the Authorization header. The TensorWasm
    # bearer middleware still runs after TLS termination.
    proxy_set_header Authorization        $http_authorization;
    proxy_set_header Host                 $host;
    proxy_set_header X-Forwarded-Proto    https;
    proxy_set_header X-Forwarded-For      $proxy_add_x_forwarded_for;

    location / {
        proxy_pass http://tensor-wasm-api.tensor-wasm.svc.cluster.local:8080;
        proxy_http_version 1.1;
    }
}
```

The XFCC header follows the
[Envoy XFCC format](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_conn_man/headers#x-forwarded-client-cert)
so the same audit-log parser works against nginx and Envoy upstreams.

### 4.2 Envoy

```yaml
static_resources:
  listeners:
    - name: tensor_wasm_listener
      address: { socket_address: { address: 0.0.0.0, port_value: 8443 } }
      filter_chains:
        - filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                stat_prefix: tensor_wasm
                forward_client_cert_details: SANITIZE_SET
                set_current_client_cert_details: { subject: true, uri: true, dns: true }
                route_config:
                  name: local_route
                  virtual_hosts:
                    - name: tensor_wasm
                      domains: ["*"]
                      routes:
                        - match: { prefix: "/" }
                          route: { cluster: tensor_wasm_api }
                http_filters:
                  - name: envoy.filters.http.router
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router
          transport_socket:
            name: envoy.transport_sockets.tls
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.DownstreamTlsContext
              require_client_certificate: true
              common_tls_context:
                tls_certificates:
                  - certificate_chain: { filename: /etc/envoy/tls/server.pem }
                    private_key:       { filename: /etc/envoy/tls/server.key }
                validation_context:
                  trusted_ca: { filename: /etc/envoy/tls/ca.pem }
  clusters:
    - name: tensor_wasm_api
      type: STRICT_DNS
      connect_timeout: 1s
      load_assignment:
        cluster_name: tensor_wasm_api
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address:
                    socket_address:
                      address: tensor-wasm-api.tensor-wasm.svc.cluster.local
                      port_value: 8080
```

Envoy populates `x-forwarded-client-cert` automatically with
`SANITIZE_SET`, which also clobbers any incoming XFCC value from the
edge — see [section 7.4](#74-xfcc-spoofing-via-the-edge).

### 4.3 Caddy

```caddyfile
tensor-wasm.example.com {
    tls /etc/caddy/tls/server.pem /etc/caddy/tls/server.key {
        client_auth {
            mode                 require_and_verify
            trusted_ca_cert_file /etc/caddy/tls/ca.pem
        }
    }
    reverse_proxy http://tensor-wasm-api.tensor-wasm.svc.cluster.local:8080 {
        header_up X-Client-DN          {http.request.tls.client.subject}
        header_up X-Client-Fingerprint {http.request.tls.client.fingerprint}
        header_up X-Forwarded-Client-Cert "Subject=\"{http.request.tls.client.subject}\";Hash=\"{http.request.tls.client.fingerprint}\""
    }
}
```

For a quick test loop, `tls internal` swaps the explicit cert pair
for a Caddy-managed local CA — useful for staging, never for
production.

### 4.4 Passing client identity to the audit log

All three proxies forward either `X-Forwarded-Client-Cert` (Envoy's
de-facto standard, supported by nginx and Caddy via templating) or a
small set of bespoke `X-Client-*` headers. The v0.4 audit-log work
(W2.2) defines the schema the gateway writes per state-mutating call.
The intended interaction is:

- If `X-Forwarded-Client-Cert` is present **and** the connection
  arrived from a configured trusted-proxy CIDR, the audit log
  records the parsed `Subject` and `Hash` alongside the bearer-token
  id.
- Otherwise the audit log records only the bearer-token id.

**TODO (v0.4):** the audit-log middleware (W2.2) is in flight. The
env-var name proposed for the trusted-proxy allowlist is
`TENSOR_WASM_API_TRUSTED_PROXY_CIDRS`, defaulting to empty (do not
trust any XFCC header). The audit-log PR should reference this
section so the two land coherently.

---

## 5. Kubernetes and Helm integration

### 5.1 Where TLS terminates in a cluster

| Cluster shape                                  | Recommended termination point      |
|------------------------------------------------|-------------------------------------|
| Istio / Linkerd service mesh present           | Mesh sidecar (Architecture B)       |
| nginx-ingress / Traefik / AWS ALB present      | Ingress controller (Architecture B) |
| No mesh, no ingress, single-tenant cluster     | `tensor-wasm-api` itself (Arch A)   |
| Multi-tenant cluster with per-tenant cert auth | Mesh + per-tenant policy            |

For the v0.4 Helm chart at
[`deploy/helm/tensor-wasm/values.yaml`](../../deploy/helm/tensor-wasm/values.yaml),
the existing `ingress.tls` field already covers Architecture B (the
ingress controller terminates TLS; the cert lives in a `Secret`).
cert-manager + `Certificate` is the recommended way to populate that
Secret.

### 5.2 Proposed Helm values for Architecture A

**TODO (v0.4):** the chart does not yet expose a top-level `tls:`
block. Proposed shape, mirroring the existing `auth:` and `gpu:`
blocks:

```yaml
# -- TLS termination at the tensor-wasm-api process itself.
# Requires the binary to be built with the `tls` cargo feature.
tls:
  # -- When false (default), listener is plaintext on service.port.
  enabled: false
  # -- Bind address for the HTTPS listener. Conventionally 8443.
  port: 8443
  # -- One of: none, optional, required. `required` is mTLS.
  clientAuth: none
  # -- Existing Secret (release namespace) with keys:
  #   tls.crt -> TENSOR_WASM_API_TLS_CERT
  #   tls.key -> TENSOR_WASM_API_TLS_KEY
  #   ca.crt  -> TENSOR_WASM_API_TLS_CLIENT_CA  (mTLS only)
  # cert-manager's standard kubernetes.io/tls layout matches this.
  existingSecret: ""
  # -- When true, rate-limit buckets keyed by client-cert Subject CN.
  # No effect when clientAuth != required.
  useCertAsPrincipal: false
```

When `tls.enabled`, the chart mutates the PodSpec: adds the env
vars, swaps the container port to 8443, and mounts the Secret at
`/etc/tensor-wasm/tls` (mode `0400`).

### 5.3 cert-manager example

```yaml
apiVersion: cert-manager.io/v1
kind: Issuer
metadata:
  name: tensor-wasm-ca
  namespace: tensor-wasm
spec:
  ca: { secretName: tensor-wasm-root-ca }
---
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: tensor-wasm-server
  namespace: tensor-wasm
spec:
  secretName: tensor-wasm-tls   # consumed by tls.existingSecret
  duration: 720h                # 30 days
  renewBefore: 168h             # rotate 7 days before expiry
  commonName: tensor-wasm.tensor-wasm.svc.cluster.local
  dnsNames: [tensor-wasm.tensor-wasm.svc.cluster.local, tensor-wasm]
  issuerRef: { name: tensor-wasm-ca, kind: Issuer }
```

Install:

```bash
helm upgrade --install tensor-wasm ./deploy/helm/tensor-wasm \
  -n tensor-wasm --create-namespace \
  --set tls.enabled=true --set tls.clientAuth=required \
  --set tls.existingSecret=tensor-wasm-tls
```

---

## 6. Rotation and revocation

### 6.1 Server-cert rotation

With Architecture A as documented, the rustls listener loads cert +
key **once at startup**. Two rotation patterns:

1. **Rolling pod restart.** The k8s Deployment's `Recreate` strategy
   ([reference manifest](../../deploy/k8s/20-deployment.yaml#L26-L29))
   means rotation is a pod restart — short downtime for single-replica
   deployments. For zero-downtime rotation, set `replicaCount: 2` and
   switch to `RollingUpdate`.
2. **Hot reload.** Rustls supports a `cert_resolver` that can swap
   certs at runtime. **TODO (v0.4):** `serve_tls()` should watch the
   cert files via `notify` and rebuild the resolver on change. Until
   then, treat cert rotation as a deploy event.

With Architecture B, rotation is the proxy's responsibility:
`nginx -s reload`, Envoy SDS or hot restart, or Caddy's zero-config
file-watcher.

### 6.2 Client-cert rotation and CA rollover

Client certs rotate independently of server certs; the CA bundle
only changes when the issuing CA itself rotates. Smooth CA rollover:

1. Issue new client certs under the new CA alongside the old.
2. Concatenate both CAs into `ca.pem` and roll the server (or proxy).
3. Cycle each client to its new cert at its own cadence.
4. Drop the old CA from the bundle and roll the server again.

### 6.3 Revocation

**TODO (v0.4):** TensorWasm does not currently consume CRLs or speak
OCSP. Supported paths today: rotate the CA bundle and re-issue every
client (coarse, but works), or rotate the cert and restart the pod.
CRL/OCSP is **deferred to v0.5 at the earliest**. If you need
fine-grained client-cert revocation now, terminate mTLS in a proxy
(Architecture B) — nginx, Envoy, and Caddy all speak CRL/OCSP.
Bearer tokens have a simpler revocation path: remove from
`TENSOR_WASM_API_TOKENS` and restart. The two strategies compose.

---

## 7. Troubleshooting

### 7.1 `SSL_ERROR_BAD_CERT_DOMAIN`

The server cert's SAN list does not include the hostname the client
is connecting to. Modern clients (curl, Go, Python `requests`) ignore
the legacy CN field and only consult SAN.

```bash
openssl x509 -in server.pem -noout -text \
  | grep -A2 "Subject Alternative Name"
```

If the hostname is missing, regenerate the cert with the SAN block
from [section 3.3](#33-generating-a-test-ca-server-cert-and-client-cert)
expanded. For Architecture B, the SAN must include the proxy's
hostname, not the upstream's.

### 7.2 `tls handshake failure`

A generic handshake failure usually means one of:

1. **Client did not present a cert** when `clientAuth=required`. Add
   `--cert client.pem --key client.key` to the curl invocation.
2. **Client cert was not signed by the configured CA.**
   `openssl verify -CAfile ca.pem client.pem` from the server's
   filesystem reveals this immediately.
3. **Cipher mismatch.** rustls accepts only TLS 1.2+ with modern
   suites; very old clients fail to negotiate. This is intentional.
4. **Clock skew.** If the client or server clock is past the cert's
   not-before window in the past, the cert appears unborn. NTP-sync
   both sides.

### 7.3 `400 Bad Request` on the bearer token after mTLS works

Handshake succeeds, `/healthz` returns 200, but state-mutating
endpoints return 400 or 401. The proxy is dropping or rewriting
`Authorization`. Common anti-patterns: an explicit
`proxy_set_header Authorization ""` in nginx, or an `ext_authz` /
`lua` Envoy filter that consumes the bearer before forwarding.

**Fix.** Add an explicit pass-through to the proxy config — the
`proxy_set_header Authorization $http_authorization;` line in the
nginx example above is the canonical form.

### 7.4 XFCC spoofing via the edge

The audit log shows a client identity for a request that should not
have been authenticated by mTLS. A caller behind the trusted-proxy
CIDR is setting `X-Forwarded-Client-Cert` on a plaintext request,
and the gateway trusts it. This is the classic forwarded-header
trust bug, the same shape as `X-Forwarded-For` spoofing.

**Fix.** Never trust XFCC from outside the configured trusted-proxy
CIDRs. The Envoy `forward_client_cert_details: SANITIZE_SET` mode
replaces any incoming XFCC with the one Envoy computes itself; in
nginx, a defensive `proxy_set_header X-Forwarded-Client-Cert ""`
followed by the trusted-write achieves the same.

**TODO (v0.4):** the gateway should default to "no proxies trusted"
and only honour XFCC when the connection's remote address matches
`TENSOR_WASM_API_TRUSTED_PROXY_CIDRS`.

### 7.5 Chain validation failures

`unable to get local issuer certificate` or `certificate verify
failed`. The server is presenting a leaf cert without the
intermediate chain. Browsers paper over this via `AIA` fetching;
rustls and most CLI clients do not.

**Fix.** Concatenate the leaf and every intermediate (leaf -> root
order) into one `server.pem`. Verify with
`openssl crl2pkcs7 -nocrl -certfile server.pem | openssl pkcs7 -print_certs -noout`.
The test PKI in section 3.3 has no intermediates and a single-leaf
file is correct; in a 2- or 3-level production hierarchy, missing
intermediates is the most common omission.

### 7.6 `dev mode` warning fires under mTLS

Startup logs say `TENSOR_WASM_API_TOKENS empty; API accepts all
requests (dev mode)` even though mTLS appears to be working. mTLS
does not satisfy the bearer middleware: with the env unset, every
request that passes the handshake is admitted. Almost never what
you want.

**Fix.** Always set `TENSOR_WASM_API_TOKENS` when mTLS is enabled.
Once `TLS_USE_CERT_AS_PRINCIPAL=1` lands the cert will select the
token; until then a single shared placeholder is acceptable.

---

## 8. Limitations and what is NOT in v0.4

| Limitation                                            | Reason                                                       | Tracked   |
|-------------------------------------------------------|--------------------------------------------------------------|-----------|
| `tensor-wasm-api` does not yet expose a `tls` feature | Architecture A documented but not implemented                | W2.8 f/u  |
| No CRL or OCSP support                                | Out of scope; proxy in front if needed                       | v0.5      |
| No hot-reload of cert files                           | Rotation is a pod restart in v0.4                            | v0.5      |
| Rate-limit bucket keyed by bearer, not by client cert | `TLS_USE_CERT_AS_PRINCIPAL` proposed but unimplemented       | v0.4 W2.8 |
| Audit log does not record the client-cert principal   | Audit log (W2.2) is v0.4 in-flight; XFCC parsing is the hook | v0.4 W2.2 |
| No SPIFFE / SVID native support                       | SPIFFE-aware mesh handles this in Architecture B             | v2        |
| HTTP/2 with TLS not yet validated                     | Plaintext HTTP/1.1 today; rustls cfg will ALPN H1+H2         | v0.4 W2.8 |
| Per-cert revocation list at the gateway               | See CRL/OCSP above                                           | v0.5      |
| Client cert claims surfaced to wasm guests            | WASI-side propagation not designed; guest sees bearer scope  | v2        |

If any of these block your deployment, Architecture B closes the
gap — every one above except WASI-side propagation is already solved
in at least one of nginx, Envoy, or Caddy.

---

## 9. Related

- [`docs/PATH-TO-V1.md`](../PATH-TO-V1.md) — v0.4 "mTLS support
  optional but documented" criterion and Open decision 2 (default
  auth model).
- [`docs/DEPLOYMENT.md`](../DEPLOYMENT.md) — broader production
  guide; section 6 (TLS) recommends proxy-terminated TLS and points
  here.
- [`crates/tensor-wasm-api/API.md`](../../crates/tensor-wasm-api/API.md)
  — bearer-token grammar, scoped-token entry forms, and the full
  middleware stack the TLS layer sits in front of.
- [`crates/tensor-wasm-api/src/server.rs`](../../crates/tensor-wasm-api/src/server.rs)
  — current plaintext listener; the place Architecture A's
  `serve_tls()` will live.
- [`deploy/k8s/20-deployment.yaml`](../../deploy/k8s/20-deployment.yaml)
  — reference Deployment to mutate for Architecture A (8080 -> 8443,
  named port `http` -> `https`).
- [`deploy/helm/tensor-wasm/values.yaml`](../../deploy/helm/tensor-wasm/values.yaml)
  — Helm values to grow the proposed `tls:` block on.
- [`SECURITY.md`](../../SECURITY.md) — disclosure process; file CVEs
  against the TLS surface here once Architecture A ships.

---

_Status: v0.4 design + operator guide. Architecture B sections are
runnable today against the v0.1.0 binary; Architecture A sections
are a design spec for the v0.4 binary work, marked **TODO (v0.4)**
at each gap. When the binary work lands, walk this file end-to-end
and convert every marker into either a finished sentence or a
`TODO (v0.5):` if it slipped a milestone._
