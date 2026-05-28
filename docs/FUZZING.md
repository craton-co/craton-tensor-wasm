# Fuzzing

> `fuzz/` directory holds `cargo-fuzz` targets. Owner: @craton-co/security.

## Targets
TODO: enumerate after reading fuzz/fuzz_targets/.

## Running locally
```
cd fuzz
cargo +nightly fuzz run <target> -- -max_total_time=60
```

## Corpus
Stored in `fuzz/corpus/<target>/`. Public corpora are gitignored beyond seed inputs at `fuzz/corpus_seed/`.

## Cron
- `fuzz.yml`: nightly, 10 minutes per target.
- `fuzz-long.yml`: weekly, 4 hours per target.

## Crash triage
On crash, the workflow uploads the reproducer to `fuzz/artifacts/<target>/`. Open a SEC issue and follow `docs/SECURITY.md` for embargo.
