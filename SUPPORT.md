<!-- SPDX-License-Identifier: Apache-2.0 -->

# Craton TensorWasm — Support

This file is the single entry point for "where do I take my question?".
Pick the row that matches the kind of help you need; each section
points at the appropriate channel and the document that owns the
fine print.

## How to get help

General questions — installation, usage, design discussion, "is this
the right tool for my workload" — belong on
[GitHub Discussions](https://github.com/craton-co/craton-tensor-wasm/discussions).
Search the open and resolved threads before opening a new one; many
recurring questions already have answers there.

Discussions is the right place for:

- "How do I do X with TensorWasm?"
- "Is behavior Y intentional?"
- "Has anyone deployed against hardware Z?"
- Architecture and roadmap conversation that does not fit an RFC.

Response time is best-effort. Maintainers and the commercial sponsor
read Discussions but make no SLO commitment on community-channel
turnaround.

## Reporting a security issue

**Do not open a public GitHub issue for security reports.** Email
[`security@craton.com.ar`](mailto:security@craton.com.ar) with a
reproducer. The mailbox is monitored by the Security Committee and
acknowledged within 72 hours.

The full disclosure contract — triage SLO, embargo handling,
coordinated-disclosure timing, backport policy — is documented in
[`SECURITY.md`](SECURITY.md). Read that file before sending a report
so the message arrives in the expected shape.

## Reporting a bug

For reproducible defects in shipped code, file a
[GitHub issue](https://github.com/craton-co/craton-tensor-wasm/issues/new/choose)
using the bug template. The template asks for the inputs maintainers
need to triage without a follow-up round-trip: TensorWasm version,
toolchain, host platform, CUDA toolkit and driver (if relevant), a
minimal reproducer, and the observed-vs-expected behavior.

If you are reporting a bug you intend to fix yourself, read
[`CONTRIBUTING.md`](CONTRIBUTING.md) first; it covers the development
setup, the DCO sign-off requirement, the PR conventions, and how
reviews are coordinated with the maintainer responsible for the
affected crate area.

Issues that are not reproducible defects — feature requests, design
questions, "how do I" threads — belong in
[Discussions](https://github.com/craton-co/craton-tensor-wasm/discussions),
not the issue tracker.

## Commercial support

Commercial support for Craton TensorWasm is provided by **Craton
Software Company**, the commercial sponsor of the project
(see [`MAINTAINERS.md`](MAINTAINERS.md)). Contact
[`sales@craton.com.ar`](mailto:sales@craton.com.ar) for:

- Production deployment assistance (Kubernetes, Helm, Nomad).
- SLA-backed response times on bugs and feature requests.
- Custom-kernel and integration engineering.
- Training and architecture review for teams adopting TensorWasm.

Commercial support is a separate contract from the open-source
project; the project itself remains Apache-2.0 and governed by
[`GOVERNANCE.md`](GOVERNANCE.md) independent of any commercial
relationship.

## What's out of scope

The project's community channels do **not** provide free consulting.
Specifically:

- Questions about a specific production deployment ("here's our
  Kubernetes config, why is latency high?") belong in commercial
  support, not Discussions or the issue tracker. Maintainers may
  answer such questions when time allows, but no commitment is
  made.
- "Please review my custom kernel" requests are not a community
  service; engage commercial support if you need expert review of
  proprietary code.
- One-off help with non-TensorWasm components (CUDA toolchain,
  Wasmtime upstream, Kubernetes networking) is out of scope. The
  project documentation points at upstream resources where relevant.

Open-source maintenance — fixing defects in shipped code, reviewing
contributed PRs, evolving the project per [`PATH-TO-V1.md`](docs/PATH-TO-V1.md)
— is the maintainers' standing commitment. Everything beyond that
is a commercial conversation.
