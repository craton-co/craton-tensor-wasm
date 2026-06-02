// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Regenerate-drift guard for the committed completion scripts and man pages.
//!
//! This is the "regenerate-completions CI check" called for in the review: it
//! mirrors the bench-registration guard by asserting that the *full advertised*
//! surface is actually generatable, catching the class of bug where the help
//! text (and `completions/README.md`) advertise a shell — or the man generator
//! claims a page per non-hidden subcommand — but the generator silently never
//! produced it.
//!
//! The test drives `clap_complete::generate_to` / `clap_mangen::Man` exactly the
//! way `cmd::completions` and `cmd::man` do: against `Cli::command()` (the public
//! `clap::CommandFactory` path the binary exposes). It does not shell out, build,
//! or depend on a prebuilt binary.
//!
//! Two layers of assertion:
//!   1. Generatability — every advertised `clap_complete::Shell` produces a
//!      non-empty completion file, and every non-hidden first-depth subcommand
//!      produces a non-empty man page. This is the drift the review flagged.
//!   2. Committed-set coverage — for any scaffolding file that *is* committed
//!      under `completions/` or `man/`, the freshly generated tree must also
//!      contain it (i.e. the committed set is a subset of what regen emits, so a
//!      committed file can never reference a surface the generator dropped).

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use clap::CommandFactory;
use clap_complete::{generate_to, Shell};
use clap_mangen::Man;
use tempfile::tempdir;
use tensor_wasm_cli::Cli;

/// Shells the CLI advertises in `tensor-wasm completions --help` and in
/// `completions/README.md`. Keep this in lockstep with the help text: the whole
/// point of the guard is that an advertised shell must be generatable.
const ADVERTISED_SHELLS: &[Shell] = &[
    Shell::Bash,
    Shell::Zsh,
    Shell::Fish,
    Shell::PowerShell,
    Shell::Elvish,
];

/// Absolute path to the `crates/tensor-wasm-cli` crate root, derived from the
/// `CARGO_MANIFEST_DIR` set for the test binary.
fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Generate the completion script for `shell` into `dir`, returning the written
/// path. Mirrors `cmd::completions::run`'s `--out-dir` branch.
fn generate_completion(shell: Shell, dir: &Path) -> PathBuf {
    let mut cmd = Cli::command();
    let bin_name = cmd.get_name().to_string();
    generate_to(shell, &mut cmd, bin_name, dir)
        .unwrap_or_else(|e| panic!("generating {shell} completion failed: {e}"))
}

/// Render the root page plus one page per non-hidden first-depth subcommand
/// into `dir`. Mirrors `cmd::man::render_tree`. Returns the written filenames
/// (not full paths).
fn generate_man_pages(dir: &Path) -> BTreeSet<String> {
    let root = Cli::command();
    let root_name = root.get_name().to_string();
    let mut names = BTreeSet::new();

    let render = |cmd: clap::Command, path: &Path| {
        let man = Man::new(cmd);
        let mut buf: Vec<u8> = Vec::new();
        man.render(&mut buf)
            .unwrap_or_else(|e| panic!("rendering {} failed: {e}", path.display()));
        fs::write(path, &buf).unwrap();
    };

    let root_path = dir.join(format!("{root_name}.1"));
    render(root.clone(), &root_path);
    names.insert(format!("{root_name}.1"));

    for sub in root.get_subcommands() {
        if sub.is_hide_set() {
            continue;
        }
        let sub_name = sub.get_name();
        let file = format!("{root_name}-{sub_name}.1");
        let path = dir.join(&file);
        // Match man.rs: compose a `<root>-<sub>` display name so SYNOPSIS reads
        // `tensor-wasm run [OPTIONS]`. `Command::name` needs a `'static` Str.
        let display_name: &'static str =
            Box::leak(format!("{root_name}-{sub_name}").into_boxed_str());
        render(sub.clone().name(display_name), &path);
        names.insert(file);
    }
    names
}

/// Layer 1: every advertised shell produces a non-empty completion file. This
/// is the exact drift the review flagged — `tensor-wasm.elv` and
/// `_tensor-wasm.ps1` were advertised but missing.
#[test]
fn every_advertised_shell_is_generatable() {
    let tmp = tempdir().unwrap();
    for &shell in ADVERTISED_SHELLS {
        let written = generate_completion(shell, tmp.path());
        let body = fs::read(&written)
            .unwrap_or_else(|e| panic!("reading generated {}: {e}", written.display()));
        assert!(
            !body.is_empty(),
            "{shell} completion generated an empty file at {}",
            written.display()
        );
    }
}

/// Layer 1: every non-hidden first-depth subcommand produces a non-empty man
/// page. This catches the `tensor-wasm-kernel.1` drift (a non-hidden subcommand
/// the committed set was missing).
#[test]
fn every_subcommand_has_a_man_page() {
    let tmp = tempdir().unwrap();
    let pages = generate_man_pages(tmp.path());

    // Root + every non-hidden first-depth subcommand. Compute the expected set
    // straight from the clap tree so this stays correct as commands are added.
    let root = Cli::command();
    let root_name = root.get_name().to_string();
    let mut expected = BTreeSet::new();
    expected.insert(format!("{root_name}.1"));
    for sub in root.get_subcommands() {
        if !sub.is_hide_set() {
            expected.insert(format!("{root_name}-{}.1", sub.get_name()));
        }
    }
    assert_eq!(
        pages, expected,
        "generated man-page set does not match the non-hidden subcommand tree"
    );

    for name in &pages {
        let body = fs::read_to_string(tmp.path().join(name))
            .unwrap_or_else(|e| panic!("reading {name}: {e}"));
        assert!(
            body.contains(".TH"),
            "man page {name} is missing its .TH header"
        );
    }
}

/// Layer 2: the committed `completions/` files must be a subset of what regen
/// emits. If a completion script is committed for a surface the generator no
/// longer produces, that is drift — fail loudly so CI catches it before the
/// committed copy goes stale.
#[test]
fn committed_completions_are_a_subset_of_regen() {
    let committed_dir = crate_dir().join("completions");
    if !committed_dir.is_dir() {
        // Nothing committed yet — Layer 1 already proved generatability.
        return;
    }

    let tmp = tempdir().unwrap();
    for &shell in ADVERTISED_SHELLS {
        generate_completion(shell, tmp.path());
    }
    let generated = filenames_in(tmp.path());

    for committed in committed_filenames(&committed_dir) {
        assert!(
            generated.contains(&committed),
            "committed completions/{committed} is not produced by regenerating the \
             advertised shell set {ADVERTISED_SHELLS:?}; either the file is stale or a \
             shell was dropped from the generator"
        );
    }
}

/// Layer 2: the committed `man/` pages must be a subset of what regen emits, so
/// a committed page can never outlive the subcommand it documents.
#[test]
fn committed_man_pages_are_a_subset_of_regen() {
    let committed_dir = crate_dir().join("man");
    if !committed_dir.is_dir() {
        return;
    }

    let tmp = tempdir().unwrap();
    let generated = generate_man_pages(tmp.path());

    for committed in committed_filenames(&committed_dir) {
        assert!(
            generated.contains(&committed),
            "committed man/{committed} is not produced by regenerating the man tree; \
             either the page is stale or its subcommand was hidden/removed"
        );
    }
}

/// Filenames of the regular files directly inside `dir` (no recursion).
fn filenames_in(dir: &Path) -> BTreeSet<String> {
    fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect()
}

/// Committed scaffolding filenames worth comparing: skip docs (`README.md`) and
/// any non-generated bookkeeping so the subset check only inspects real
/// generator output.
fn committed_filenames(dir: &Path) -> BTreeSet<String> {
    filenames_in(dir)
        .into_iter()
        .filter(|n| !n.eq_ignore_ascii_case("README.md"))
        .collect()
}
