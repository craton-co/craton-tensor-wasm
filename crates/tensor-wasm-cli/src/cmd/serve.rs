// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `tensor-wasm serve` — run the HTTP API gateway in-process.
//!
//! Stands up an axum router built from `tensor_wasm_api::build_router_with_full_config`,
//! binds it to `--addr`, and serves until Ctrl-C. Flags map onto the
//! existing API config surface (auth allowlist, tenant header policy,
//! request body cap); knobs not yet exposed by the api crate fall back to
//! the same `TENSOR_WASM_API_*` environment variables `from_env()` reads.
//!
//! This is the entrypoint the README quickstart and `docs/GETTING-STARTED.md`
//! point at; previously the only path to a running gateway was via embedding
//! the library or running an integration test harness, so any change to the
//! flag set here must keep the quickstart command (`tensor-wasm serve --addr
//! 0.0.0.0:8080`) working with no extra setup.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Args;
use tensor_wasm_api::{
    build_router_with_full_config, AppState, AuthConfig, RateLimitConfig, RateLimiter, TenantConfig,
};

/// Default bind address used when `--addr` is omitted. Matches the loopback
/// quickstart in `docs/GETTING-STARTED.md`; production deployments should
/// pass `0.0.0.0:<port>` (or a private subnet address) explicitly.
pub const DEFAULT_ADDR: &str = "127.0.0.1:8080";

/// Default request-body cap, mirroring the api crate's
/// `MAX_REQUEST_BODY_BYTES` constant. Surfacing the number here (rather than
/// re-exporting the constant) keeps the CLI help text readable while still
/// matching what the gateway actually enforces.
pub const DEFAULT_MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Policy for the `X-TensorWasm-Tenant` request header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum TenantHeaderPolicy {
    /// Header is optional; absence defaults to `TenantId(0)`. Mirrors the
    /// gateway's behaviour when `TENSOR_WASM_API_REQUIRE_TENANT` is unset.
    Optional,
    /// Header is mandatory; requests without it are rejected `400`.
    /// Mirrors `TENSOR_WASM_API_REQUIRE_TENANT=1`.
    Required,
}

impl TenantHeaderPolicy {
    fn into_config(self) -> TenantConfig {
        TenantConfig {
            require_header: matches!(self, TenantHeaderPolicy::Required),
        }
    }
}

/// Arguments to `tensor-wasm serve`.
#[derive(Debug, Args)]
pub struct ServeArgs {
    /// Address to bind the HTTP server to (e.g. `127.0.0.1:8080`, `0.0.0.0:8080`).
    #[arg(long, default_value = DEFAULT_ADDR)]
    pub addr: SocketAddr,

    /// Bearer token accepted by the gateway. Repeat to allowlist multiple
    /// tokens. Each value is treated as a wildcard-scope token (equivalent to
    /// `token:tenant=*` in the env-driven `TENSOR_WASM_API_TOKENS` allowlist).
    /// If omitted entirely, the gateway falls back to reading
    /// `TENSOR_WASM_API_TOKENS` from the environment; empty/unset there =
    /// dev mode (auth disabled with a startup warning).
    #[arg(long = "token", value_name = "TOKEN")]
    pub tokens: Vec<String>,

    /// Policy for the `X-TensorWasm-Tenant` header. `optional` (default)
    /// mirrors `TENSOR_WASM_API_REQUIRE_TENANT` unset; `required` mirrors
    /// `TENSOR_WASM_API_REQUIRE_TENANT=1`.
    #[arg(long, value_enum, default_value_t = TenantHeaderPolicy::Optional)]
    pub tenant_header_policy: TenantHeaderPolicy,

    /// Origin to allow via CORS. Repeat for multiple origins. NOTE: the
    /// api crate does not yet expose a programmatic CORS knob, so this flag
    /// is currently informational — the gateway's CORS surface is wired
    /// via tower-http defaults inside `tensor-wasm-api`. The flag is kept
    /// so the README quickstart command parses cleanly; once
    /// `tensor_wasm_api::CorsConfig` lands (H18-H20 follow-up) this will
    /// thread through to the router builder.
    #[arg(long = "cors-origin", value_name = "ORIGIN")]
    pub cors_origins: Vec<String>,

    /// Maximum inbound request body size, in bytes. Defaults to 64 MiB to
    /// match the api crate's `MAX_REQUEST_BODY_BYTES`. NOTE: the api crate
    /// does not yet expose a programmatic body-cap knob via
    /// `build_router_with_full_config`; passing a non-default value here
    /// emits a warning and the router uses the compiled-in 64 MiB cap.
    /// Wired through once the api crate grows a `BodyLimitConfig`.
    #[arg(long, default_value_t = DEFAULT_MAX_BODY_BYTES)]
    pub max_body_bytes: usize,

    /// Acknowledge that you are knowingly exposing a dev-mode (no auth)
    /// deployment to a non-loopback address. Required when `--addr` resolves
    /// to `0.0.0.0`, `::`, or any non-loopback IP AND no bearer-token
    /// allowlist is configured (no `--token` flags AND empty/unset
    /// `TENSOR_WASM_API_TOKENS`). Without this opt-in, the CLI refuses to
    /// bind such a configuration: a no-auth gateway reachable on a routable
    /// address is a critical misconfiguration that has historically leaked
    /// internal endpoints to the public internet (cli S-35 + api S-35).
    ///
    /// Setting this flag does NOT enable auth — it merely silences the
    /// safety gate. The 60-second recurring "no auth + public bind"
    /// warning continues to fire so the misconfiguration remains visible
    /// in long-running container logs.
    #[arg(
        long,
        env = "TENSOR_WASM_ALLOW_PLAINTEXT_PUBLIC",
        default_value_t = false
    )]
    pub allow_plaintext_public: bool,

    /// Run all argument validation (including the dev-mode bind-safety
    /// gate) and exit before binding the listener or constructing
    /// `AppState`. Used by the `tests/serve_dev_mode_bind_safety.rs`
    /// integration tests so they can assert "this combination is
    /// accepted" without leaving a real HTTP server running. Hidden from
    /// `--help` because it has no production use and might tempt
    /// operators to wire it into health checks where they really want
    /// `/healthz`.
    #[arg(long, hide = true, default_value_t = false)]
    pub check_only: bool,
}

/// How frequently the "dev-mode (no auth) gateway running" warning is
/// re-emitted while the server is up. Kept short enough that the warning
/// is visible in any reasonable log retention window, long enough that it
/// doesn't drown the log in a busy deployment.
const DEV_MODE_WARN_INTERVAL: Duration = Duration::from_secs(60);

/// `true` if `ip` is any non-loopback address, including the wildcard
/// `0.0.0.0` / `::`. Loopback (`127.0.0.0/8`, `::1`) is the only safe
/// bind target for a no-auth gateway.
fn is_non_loopback(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4 == Ipv4Addr::UNSPECIFIED || !v4.is_loopback(),
        IpAddr::V6(v6) => v6 == Ipv6Addr::UNSPECIFIED || !v6.is_loopback(),
    }
}

/// Bind-safety gate enforced before any socket is opened.
///
/// Returns `Err` when the resolved configuration would expose a dev-mode
/// (no allowlisted tokens) gateway on a non-loopback address without an
/// explicit `--allow-plaintext-public` opt-in. The error message names
/// both the address and the opt-out so the operator has a complete fix
/// from a single read.
///
/// Auth-configured deployments (`auth.is_dev_mode() == false`) are
/// always accepted — the bearer-token allowlist is the gateway's primary
/// access-control surface, so exposing it on a routable address is the
/// supported production path.
fn validate_bind_safety(
    addr: SocketAddr,
    auth: &AuthConfig,
    allow_plaintext_public: bool,
) -> Result<()> {
    if !auth.is_dev_mode() {
        return Ok(());
    }
    if !is_non_loopback(addr.ip()) {
        return Ok(());
    }
    if allow_plaintext_public {
        return Ok(());
    }
    anyhow::bail!(
        "refusing to bind {addr}: dev-mode (no auth) deployments must bind to a \
         loopback address (e.g. 127.0.0.1, ::1) or pass \
         --allow-plaintext-public to acknowledge the risk. Configure \
         --token <TOKEN> (or export TENSOR_WASM_API_TOKENS) to enable bearer \
         auth, or rebind to 127.0.0.1 for local-only access."
    )
}

/// Entry point for `tensor-wasm serve`.
pub async fn run(args: ServeArgs) -> Result<()> {
    let addr = args.addr;

    // Auth: if --token was passed, build an explicit allowlist; otherwise
    // fall back to the env-driven loader so operators who already export
    // TENSOR_WASM_API_TOKENS (the documented production knob) don't have to
    // re-declare them on the command line.
    let auth = if args.tokens.is_empty() {
        AuthConfig::from_env()
    } else {
        AuthConfig::from_tokens(args.tokens.iter().map(|s| s.as_str()))
    };

    // Bind-safety gate: must run BEFORE AppState::try_new (which can take
    // tens of milliseconds initialising the wasmtime engine and CUDA
    // backend) so a misconfigured invocation fails fast with a clear,
    // single-line error rather than appearing to "almost start" before
    // erroring on the bind. Also runs before `--check-only` honors its
    // early-exit, so the integration tests can assert the gate fires
    // without standing up the engine.
    validate_bind_safety(addr, &auth, args.allow_plaintext_public)?;

    let dev_mode_on_public = auth.is_dev_mode() && is_non_loopback(addr.ip());

    let tenant = args.tenant_header_policy.into_config();

    // Per-token rate limiter: the api crate currently only exposes
    // RateLimitConfig::from_env / ::disabled. Mirror the env-driven default
    // so `TENSOR_WASM_API_RATE_LIMIT_QPS` keeps working when set; otherwise
    // the limiter is a pass-through.
    let limiter = RateLimiter::new(RateLimitConfig::from_env());

    if args.max_body_bytes != DEFAULT_MAX_BODY_BYTES {
        tracing::warn!(
            target: "tensor_wasm_cli::serve",
            requested = args.max_body_bytes,
            actual = DEFAULT_MAX_BODY_BYTES,
            "--max-body-bytes is currently informational; gateway uses the \
             compiled-in 64 MiB cap until tensor-wasm-api exposes a \
             BodyLimitConfig",
        );
    }
    if !args.cors_origins.is_empty() {
        tracing::warn!(
            target: "tensor_wasm_cli::serve",
            count = args.cors_origins.len(),
            "--cors-origin is currently informational; tensor-wasm-api does \
             not yet expose a CorsConfig (H18-H20 follow-up)",
        );
    }

    // Hidden `--check-only` exit: every gate above has run and accepted
    // the config; bail out before we touch wasmtime or the network. Used
    // by `tests/serve_dev_mode_bind_safety.rs` to assert "this
    // combination parses + passes the bind-safety check" in well under
    // 100ms without leaving a listener around to leak between tests.
    if args.check_only {
        println!("check_only: configuration accepted (addr={addr})");
        return Ok(());
    }

    if dev_mode_on_public {
        // Acknowledged opt-out: surface the risk at startup. The
        // 60-second periodic warning below keeps the misconfiguration
        // visible in long-running deployments where the startup line
        // has long since rolled off the journald buffer.
        tracing::warn!(
            target: "tensor_wasm_cli::serve",
            %addr,
            "DEV-MODE GATEWAY ON PUBLIC ADDRESS: no auth tokens are \
             configured and --allow-plaintext-public was supplied. This \
             deployment accepts every request without authentication. \
             Configure --token / TENSOR_WASM_API_TOKENS before serving \
             real traffic.",
        );
    }

    let state = AppState::try_new().context("initialising AppState (wasmtime engine)")?;
    let router = build_router_with_full_config(state, auth, tenant, limiter);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding TCP listener on {addr}"))?;

    // Print the canonical scheme://host:port line the README quickstart
    // greps for; keep it on a single line so log filters can latch onto it.
    println!("listening on http://{addr}");

    // Spawn a background ticker that re-emits the dev-mode warning every
    // DEV_MODE_WARN_INTERVAL while the server is up. The task is detached
    // because it has no shutdown hook of its own — when the serve future
    // returns, the runtime drops the task with the rest of `main`'s
    // execution context. We deliberately do NOT pipe it through
    // `with_graceful_shutdown` so a stuck ticker can never delay a clean
    // shutdown.
    let _warn_task = if dev_mode_on_public {
        Some(tokio::spawn(async move {
            let mut interval = tokio::time::interval(DEV_MODE_WARN_INTERVAL);
            // The first tick fires immediately; skip it because we already
            // emitted the startup warning above. Subsequent ticks fire
            // every DEV_MODE_WARN_INTERVAL.
            interval.tick().await;
            loop {
                interval.tick().await;
                tracing::warn!(
                    target: "tensor_wasm_cli::serve",
                    %addr,
                    "still running dev-mode (no auth) gateway on public \
                     address; configure --token / TENSOR_WASM_API_TOKENS \
                     to enable bearer auth",
                );
            }
        }))
    } else {
        None
    };

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("axum::serve")?;

    Ok(())
}

/// Resolves when the process should shut down cleanly. On unix we drain on
/// either Ctrl-C (SIGINT) or SIGTERM so containerized `tensor-wasm serve`
/// shuts down gracefully on `docker stop` / kubernetes pod termination. On
/// non-unix targets only Ctrl-C is available.
async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::error!(
                target: "tensor_wasm_cli::serve",
                error = %e,
                "failed to install Ctrl-C handler; serve loop will run until killed",
            );
            // Park forever so the caller's `with_graceful_shutdown` never
            // fires a spurious shutdown on a handler-install error.
            std::future::pending::<()>().await;
        }
    };

    #[cfg(unix)]
    {
        let terminate = async {
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(mut sig) => {
                    sig.recv().await;
                }
                Err(e) => {
                    tracing::error!(
                        target: "tensor_wasm_cli::serve",
                        error = %e,
                        "failed to install SIGTERM handler; relying on Ctrl-C only",
                    );
                    // Park forever so a handler-install error doesn't trip a
                    // spurious shutdown via this arm.
                    std::future::pending::<()>().await;
                }
            }
        };

        tokio::select! {
            _ = ctrl_c => {}
            _ = terminate => {}
        }
    }

    #[cfg(not(unix))]
    {
        ctrl_c.await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Tiny harness to exercise the clap derive without standing up the
    /// whole `Cli` enum.
    #[derive(Parser)]
    struct Probe {
        #[command(flatten)]
        args: ServeArgs,
    }

    #[test]
    fn defaults_parse() {
        // The `--allow-plaintext-public` flag reads
        // TENSOR_WASM_ALLOW_PLAINTEXT_PUBLIC from the env via clap's
        // `env = ...` plumbing. Strip it so a developer's shell config
        // can't silently flip the default to `true` under this test.
        temp_env::with_var("TENSOR_WASM_ALLOW_PLAINTEXT_PUBLIC", None::<&str>, || {
            let p = Probe::try_parse_from(["serve"]).expect("defaults parse");
            assert_eq!(p.args.addr.to_string(), "127.0.0.1:8080");
            assert!(p.args.tokens.is_empty());
            assert_eq!(p.args.tenant_header_policy, TenantHeaderPolicy::Optional);
            assert!(p.args.cors_origins.is_empty());
            assert_eq!(p.args.max_body_bytes, DEFAULT_MAX_BODY_BYTES);
            assert!(
                !p.args.allow_plaintext_public,
                "--allow-plaintext-public must default to false (safety opt-in)"
            );
            assert!(!p.args.check_only);
        });
    }

    #[test]
    fn parses_allow_plaintext_public_flag() {
        temp_env::with_var("TENSOR_WASM_ALLOW_PLAINTEXT_PUBLIC", None::<&str>, || {
            let p = Probe::try_parse_from(["serve", "--allow-plaintext-public"]).expect("parses");
            assert!(p.args.allow_plaintext_public);
        });
    }

    #[test]
    fn parses_addr_override() {
        let p = Probe::try_parse_from(["serve", "--addr", "0.0.0.0:9000"]).expect("parses");
        assert_eq!(p.args.addr.to_string(), "0.0.0.0:9000");
    }

    #[test]
    fn rejects_garbage_addr() {
        assert!(Probe::try_parse_from(["serve", "--addr", "not-an-address"]).is_err());
    }

    #[test]
    fn collects_repeated_tokens() {
        let p = Probe::try_parse_from(["serve", "--token", "a", "--token", "b"]).expect("parses");
        assert_eq!(p.args.tokens, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn collects_repeated_cors_origins() {
        let p = Probe::try_parse_from([
            "serve",
            "--cors-origin",
            "https://app.example",
            "--cors-origin",
            "https://admin.example",
        ])
        .expect("parses");
        assert_eq!(p.args.cors_origins.len(), 2);
    }

    #[test]
    fn parses_tenant_header_policy() {
        let p =
            Probe::try_parse_from(["serve", "--tenant-header-policy", "required"]).expect("parses");
        assert_eq!(p.args.tenant_header_policy, TenantHeaderPolicy::Required);
        let cfg = p.args.tenant_header_policy.into_config();
        assert!(cfg.require_header);
    }

    // --- Bind-safety gate (cli S-35 + api S-35) -----------------------

    fn addr(s: &str) -> SocketAddr {
        s.parse().expect("test addr literal parses")
    }

    fn dev_auth() -> AuthConfig {
        AuthConfig::default()
    }

    fn prod_auth() -> AuthConfig {
        AuthConfig::from_tokens(["secret".to_string()])
    }

    #[test]
    fn is_non_loopback_classifies_known_addresses() {
        // Wildcard binds — always treated as "public" because the kernel
        // will accept connections from any reachable interface.
        assert!(is_non_loopback("0.0.0.0".parse().unwrap()));
        assert!(is_non_loopback("::".parse().unwrap()));
        // Plain public IPs.
        assert!(is_non_loopback("8.8.8.8".parse().unwrap()));
        assert!(is_non_loopback("192.168.1.5".parse().unwrap()));
        assert!(is_non_loopback("2606:4700:4700::1111".parse().unwrap()));
        // Loopback — the only safe bind target without auth.
        assert!(!is_non_loopback("127.0.0.1".parse().unwrap()));
        assert!(!is_non_loopback("127.0.0.5".parse().unwrap()));
        assert!(!is_non_loopback("::1".parse().unwrap()));
    }

    #[test]
    fn validate_rejects_dev_mode_on_wildcard_v4() {
        let err = validate_bind_safety(addr("0.0.0.0:8080"), &dev_auth(), false)
            .expect_err("dev mode + 0.0.0.0 must be rejected");
        let msg = format!("{err}");
        assert!(msg.contains("0.0.0.0:8080"), "missing addr in error: {msg}");
        assert!(
            msg.contains("--allow-plaintext-public"),
            "error must name the opt-out flag: {msg}"
        );
        assert!(
            msg.contains("loopback"),
            "error must mention loopback as the safe alternative: {msg}"
        );
    }

    #[test]
    fn validate_rejects_dev_mode_on_wildcard_v6() {
        assert!(validate_bind_safety(addr("[::]:8080"), &dev_auth(), false).is_err());
    }

    #[test]
    fn validate_rejects_dev_mode_on_routable_v4() {
        assert!(validate_bind_safety(addr("192.168.1.10:8080"), &dev_auth(), false).is_err());
    }

    #[test]
    fn validate_accepts_dev_mode_on_loopback_v4() {
        validate_bind_safety(addr("127.0.0.1:8080"), &dev_auth(), false)
            .expect("loopback bind in dev mode must be accepted");
    }

    #[test]
    fn validate_accepts_dev_mode_on_loopback_v6() {
        validate_bind_safety(addr("[::1]:8080"), &dev_auth(), false)
            .expect("v6 loopback bind in dev mode must be accepted");
    }

    #[test]
    fn validate_accepts_when_allow_plaintext_public_set() {
        validate_bind_safety(addr("0.0.0.0:8080"), &dev_auth(), true)
            .expect("--allow-plaintext-public must override the gate");
    }

    #[test]
    fn validate_accepts_auth_configured_on_public_addr() {
        // The bearer-token allowlist is the gateway's primary
        // access-control surface; binding it to 0.0.0.0 with real auth
        // is the supported production path and MUST NOT trip the gate.
        validate_bind_safety(addr("0.0.0.0:8080"), &prod_auth(), false)
            .expect("auth-configured deployment on 0.0.0.0 must be accepted");
    }

    #[test]
    fn validate_accepts_auth_configured_on_loopback() {
        validate_bind_safety(addr("127.0.0.1:8080"), &prod_auth(), false)
            .expect("auth-configured deployment on loopback must be accepted");
    }
}
