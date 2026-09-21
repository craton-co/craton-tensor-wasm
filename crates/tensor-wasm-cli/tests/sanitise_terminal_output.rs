// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Integration tests for the CLI's terminal-output sanitiser (T18
//! terminal-hijack hardening).
//!
//! `sanitise_terminal_output` is applied at every place where a
//! server-returned string flows to `println!` (snapshot restore id,
//! invoke fallback body, metrics text, kernel-list rows, observe board).
//! The helper itself is a pure string transform — these tests exercise
//! the invariant that ANSI escape bytes and other ASCII controls cannot
//! survive a single round-trip, while legitimate UTF-8 / whitespace
//! does.
//!
//! The helper is `pub` + `#[doc(hidden)]` purely so this integration
//! test can reach it through the lib surface — see `src/cmd/mod.rs`.

use tensor_wasm_cli::cmd::sanitise_terminal_output;

#[test]
fn sanitises_ansi_escape() {
    // ESC `[` `3` `1` `m` is the SGR sequence "set foreground red". A
    // malicious snapshot-restore response could embed this to colour
    // output in a way the operator did not consent to, or worse —
    // chain into a title-bar rewrite (`ESC ] 0 ; title BEL`). The
    // sanitiser must drop the ESC so the bracket and digits become
    // inert text.
    let cleaned = sanitise_terminal_output("hello\x1b[31mworld");
    assert!(
        !cleaned.contains('\x1b'),
        "ESC byte must not survive sanitisation: {cleaned:?}"
    );
    assert!(cleaned.contains("hello"));
    assert!(cleaned.contains("world"));
}

#[test]
fn preserves_unicode_and_whitespace() {
    // Legitimate output regularly contains accented characters, tabs,
    // and embedded newlines (e.g. pretty-printed JSON, Prometheus
    // exposition). The sanitiser must NOT touch any of them.
    let input = "héllo\n\tworld";
    assert_eq!(
        sanitise_terminal_output(input),
        input,
        "newline, tab, and multi-byte UTF-8 must round-trip unchanged"
    );
}

#[test]
fn replaces_del() {
    // 0x7F (DEL) is technically a control byte per Unicode tables. It
    // does not normally appear in legitimate API output and several
    // terminal emulators interpret it as backspace, so we replace it
    // with the visible placeholder `?` rather than letting it through.
    assert_eq!(sanitise_terminal_output("a\x7Fb"), "a?b");
}

#[test]
fn drops_standalone_carriage_return() {
    // sec LOW (mod.rs L3): a bare `\r` is a line-overwrite primitive. A
    // malicious response can print a benign line, emit `\r`, and overwrite
    // it in place so the terminal shows different text than was delivered
    // (e.g. masking a `FAIL` as `ok`). A lone `\r` must not survive.
    let cleaned = sanitise_terminal_output("FAIL\rok  ");
    assert!(
        !cleaned.contains('\r'),
        "standalone CR must not survive: {cleaned:?}"
    );
    assert_eq!(cleaned, "FAILok  ");
}

#[test]
fn normalises_crlf_to_lf() {
    // The Windows newline `\r\n` collapses to a single `\n`, so legitimate
    // CRLF-terminated server output round-trips as clean `\n`-separated text
    // with no spurious placeholder characters introduced.
    assert_eq!(sanitise_terminal_output("a\r\nb"), "a\nb");
}
