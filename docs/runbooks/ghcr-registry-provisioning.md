<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Craton Software Company -->

# GHCR registry provisioning — sponsor runbook

Procedure runbook for provisioning the `ghcr.io/craton-co/tensor-wasm`
container-registry namespace the deploy docs and Helm chart already
reference but nothing yet publishes to. Closes audit Problem #13 and
unblocks the "first release with externally-distributable images"
prerequisite. Until a sponsor maintainer executes the steps below, the
`ghcr.io/craton-co/*` path is aspirational in the same sense the
self-hosted CUDA runner was before `cuda.yml` had a runner registered:
the manifests render, the chart installs, but `kubectl describe pod`
shows `ImagePullBackOff` because the namespace returns 404.

This runbook is a procedure runbook (not an alert runbook); follow
the [`runbooks/README.md`](README.md) contract section "Procedure
runbooks". It is also a **sponsor-only** procedure — none of the steps
can be executed by an AI agent or a non-org-admin maintainer, because
the GitHub Container Registry permission surface lives at the
organisation level under settings that only org admins can change.

## When to run this

- Before the first TensorWasm release that wants externally-distributable
  container images (the v0.3.x line, today, qualifies)
- Whenever the sponsor org's GitHub permissions change in a way that
  could revoke the package-write path (membership change, role demotion,
  PAT rotation, SSO enforcement flip)
- Whenever the registry visibility decision is revisited (e.g. flipping
  a package from private to public after a sponsor-board review)

This runbook is executed **once per registry-provisioning event**, not
once per release. After the first execution the release workflow in
[`.github/workflows/release.yml`](../../.github/workflows/release.yml)
publishes the images automatically on every `release: published` event.

## Prerequisites

- **Sponsor admin access** to the `craton-co` GitHub organisation.
  Settings → People shows the executing user with role **Owner**;
  Settings → Packages is reachable. A maintainer who only has push
  access on the repository cannot complete Steps 1 or 2.
- **The Dockerfile at repo root builds clean** on a dev box. The C8
  Dockerfile (see [`../../Dockerfile`](../../Dockerfile)) was last
  validated on the workspace at `0.3.7`; if more than one minor version
  has elapsed since this runbook was last executed, re-run
  `docker build -t tensor-wasm:smoke .` once before Step 3.
- **A decision** on registry visibility: public (recommended for OSS
  distribution and for the Helm chart's frictionless `helm install`
  flow) or private (sponsor-only; operators outside the sponsor org
  cannot pull and must mirror to their own registry). The visibility
  choice is recorded in the `image-visibility` field of the package
  settings in Step 1; flipping it later is a single click but breaks
  any external `docker pull` already wired against the previous mode.
- **A maintainer with `packages:write` permission** on the org. The
  recommended path is the workflow's auto-provisioned `GITHUB_TOKEN`
  (see Step 4); the fallback for self-hosted CI or laptop publishing
  is a classic PAT scoped to `write:packages`.
- **Network egress** from the executing host to `ghcr.io` (port 443).
  Most corporate-VPN setups already allow this; if not, request the
  allowlist entry before Step 3.

## Procedure

### Step 1 — provision the org-level package permission

From a sponsor-admin browser session:

1. Open `https://github.com/organizations/craton-co/settings/packages`.
2. Under **Package creation**, ensure **Public** is allowed (if the
   visibility decision recorded under Prerequisites was "public") and
   **Private** is allowed (always; the four backend variants may
   be flipped private later without re-running this runbook).
3. Under **Container access** → **Inherit access from source
   repository**, check the box. This is the recommended setting: the
   `craton-tensor-wasm` repo's collaborator/team permissions become the
   package's, so adding a maintainer to the repo also grants them
   `packages:read`/`packages:write` automatically. The alternative
   (per-package override) is only needed if the project later forks
   the package surface from the source-repo permission set, which the
   v0.3.x line has no plan to do.
4. If the visibility decision is "public": after the first push in
   Step 3 lands, confirm the package is discoverable at
   `https://github.com/orgs/craton-co/packages` without authentication.
   An anonymous `curl -fsSL https://ghcr.io/v2/craton-co/tensor-wasm/tags/list`
   should return a 200 with a JSON tag list once images are pushed; a
   401/404 indicates the visibility flip did not take effect.

### Step 2 — generate an automation PAT (only if not using `GITHUB_TOKEN`)

**Skip this step if you are using `${{ secrets.GITHUB_TOKEN }}` in the
release workflow** (recommended; see Step 4). The auto-provisioned
token already carries `packages:write` when the workflow's `permissions:`
block declares it, and rotates per-job — there is no long-lived secret
to manage.

The PAT path is only needed for two situations:

- A **self-hosted CI** runner that publishes to `ghcr.io` outside the
  GitHub Actions environment (no `GITHUB_TOKEN` available)
- A **sponsor laptop** push during the Step 3 smoke (covered below)

To generate:

1. From a sponsor-admin account: Settings → Developer settings →
   Personal access tokens → **Tokens (classic)** → Generate new token.
   The fine-grained PAT path does not yet expose container-registry
   scopes consistently across orgs; the classic PAT is the working
   default through Q2 2026.
2. Scopes: `write:packages`, `read:packages`, and optionally
   `delete:packages` (last one only if the same PAT is used for the
   cleanup commands in Steps 3 and 11; safer to scope the cleanup PAT
   separately).
3. Expiry: 90 days. A longer-lived PAT is a standing audit finding;
   rotate on the same cadence as the SECURITY.md backport-window
   review (see [`SECURITY.md`](../../SECURITY.md) "Backport policy").
4. Store the token in a sponsor-controlled secret manager (1Password
   vault `craton-engineering`, key `ghcr-publish-pat`). Do **not**
   commit it to any repo, do not paste it into Slack or issue
   comments, and do not save it in the host shell history (use
   `read -s GH_PAT` rather than `export GH_PAT=...`).

### Step 3 — local push smoke (sponsor laptop)

Validate end-to-end before wiring the workflow. Run from the repo
root, with the PAT from Step 2 already in `$GH_PAT`:

```sh
# Build the host-only image at the current workspace version.
docker build -t ghcr.io/craton-co/tensor-wasm:0.3.7-local .

# Log in to ghcr.io using the PAT, not your account password.
echo "$GH_PAT" | docker login ghcr.io -u <sponsor-username> --password-stdin

# Push. First push to a not-yet-existing repository creates the
# package; the org-level permissions from Step 1 apply.
docker push ghcr.io/craton-co/tensor-wasm:0.3.7-local
```

After the push completes, open
`https://github.com/orgs/craton-co/packages` — the
`tensor-wasm` package should appear with the `0.3.7-local` tag. If it
does not, the most common cause is Step 1's "Inherit access from
source repository" toggle never being saved (the UI does not always
confirm); re-check and re-push.

Clean up the smoke image so it does not appear as a published version
in the package's release history:

```sh
# List versions to find the version ID for the 0.3.7-local tag.
gh api /orgs/craton-co/packages/container/tensor-wasm/versions

# Delete by version ID (NOT by tag name; the API only accepts IDs).
gh api -X DELETE /orgs/craton-co/packages/container/tensor-wasm/versions/<ID>
```

The smoke image deliberately uses the `-local` tag suffix so that even
if the cleanup is forgotten, a downstream operator running
`docker pull ghcr.io/craton-co/tensor-wasm:0.3.7` will not
accidentally land on the smoke build.

### Step 4 — wire the release workflow

Edit [`.github/workflows/release.yml`](../../.github/workflows/release.yml)
and append a `docker-publish` job after the existing `github-release`
job. The job matrices over the four backend variants the C8 Dockerfile
produces and publishes each as a distinct tag.

Copy-pasteable block (drop into the `jobs:` map; preserve YAML
indentation):

```yaml
  docker-publish:
    name: publish docker image (${{ matrix.backend || 'host-only' }})
    needs: [github-release]
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest
    permissions:
      # GITHUB_TOKEN must carry packages:write for ghcr.io push.
      contents: read
      packages: write
    strategy:
      fail-fast: false
      matrix:
        backend: ["", "cust", "cudarc", "cuda-oxide"]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Derive image tags
        id: tags
        run: |
          VERSION="${GITHUB_REF#refs/tags/v}"
          BACKEND="${{ matrix.backend }}"
          if [ -z "$BACKEND" ]; then
            # Host-only build: plain version tag plus :latest on main.
            echo "tags=ghcr.io/craton-co/tensor-wasm:${VERSION},ghcr.io/craton-co/tensor-wasm:latest" >> "$GITHUB_OUTPUT"
          else
            # Backend variant: only the version-suffixed tag, no :latest.
            echo "tags=ghcr.io/craton-co/tensor-wasm:${VERSION}-${BACKEND}" >> "$GITHUB_OUTPUT"
          fi

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          file: ./Dockerfile
          push: true
          build-args: |
            BACKEND=${{ matrix.backend }}
          tags: ${{ steps.tags.outputs.tags }}
          labels: |
            org.opencontainers.image.revision=${{ github.sha }}
            org.opencontainers.image.source=https://github.com/craton-co/craton-tensor-wasm
            org.opencontainers.image.version=${{ github.ref_name }}
```

Notes on the snippet:

- **`needs: [github-release]`** runs the publish after the existing
  GitHub Release is created, matching the SBOM + API-reference
  workflows in [`sbom.yml`](../../.github/workflows/sbom.yml) and
  [`api-reference.yml`](../../.github/workflows/api-reference.yml)
  which also attach to the release after it exists.
- **`secrets: GITHUB_TOKEN`** is sufficient — the
  `permissions: packages: write` block elevates the auto-provisioned
  token, no PAT needed. The PAT path (Step 2) is only for the smoke
  push and any out-of-Actions publishing.
- **`if: startsWith(github.ref, 'refs/tags/v')`** scopes the publish
  to release tags only; `workflow_dispatch` and dev-branch pushes do
  not publish.
- **`:latest`** is set only on the host-only build, and only on tag
  refs (which by project policy are cut off `main`). The three
  backend variants do not carry a `:latest` alias because choosing
  among them is a deliberate operator decision and `:latest-cust`
  semantics ("latest of the EOL backend") would be actively misleading.
- **`fail-fast: false`** so a transient failure on one backend variant
  does not abort the other three; the release succeeds with whichever
  variants did push, and a retry-only-the-failed-matrix-leg is a
  single workflow re-run.

### Step 5 — pull-smoke from a third-party clean machine

Validate the published images from a host that has **no local
TensorWasm build cache** and is not a member of the sponsor org. Any
laptop, cloud VM, or CI runner outside `craton-co` works.

```sh
docker pull ghcr.io/craton-co/tensor-wasm:0.3.7
docker run --rm ghcr.io/craton-co/tensor-wasm:0.3.7 --version
```

Expected output: `tensor-wasm 0.3.7` (or whatever version was just
released). If the pull returns `manifest unknown`, the publish-step
matrix leg failed — check the workflow run; the most likely cause is
the `packages: write` permission not being granted at the repo level
even though it is enabled at the org level (Settings → Actions →
General → Workflow permissions).

Then validate the HEALTHCHECK fires by running the image in `serve`
mode (the Dockerfile's default `CMD`):

```sh
docker run -d --name twasm-smoke -p 8080:8080 \
    -e TENSOR_WASM_API_TOKENS='smoke:tenant=*' \
    ghcr.io/craton-co/tensor-wasm:0.3.7
# Wait the start-period (5s) plus one interval (10s).
sleep 20
docker inspect --format='{{.State.Health.Status}}' twasm-smoke
# Expected: healthy
docker rm -f twasm-smoke
```

If the status reports `unhealthy`, the HEALTHCHECK is failing — most
commonly because `curl` is not present in the runtime layer (the
Dockerfile installs it implicitly via the `ca-certificates` package
chain on Debian bookworm-slim; verify with
`docker run --rm --entrypoint sh ghcr.io/craton-co/tensor-wasm:0.3.7 -c 'which curl'`).

### Step 6 — update deploy docs

Three documents currently carry the "registry is a placeholder"
callout that this runbook's successful execution invalidates:

- [`deploy/helm/tensor-wasm/README.md`](../../deploy/helm/tensor-wasm/README.md)
  — the top-of-file blockquote starting "Image registry is not yet
  provisioned"
- [`deploy/k8s/README.md`](../../deploy/k8s/README.md) — the top-of-file
  blockquote starting "Image tag is a placeholder"
- [`deploy/nomad/README.md`](../../deploy/nomad/README.md) — the
  top-of-file blockquote starting "Image and artifact placeholders"

Remove the blockquote in each; the `image:` paths themselves do not
need to change. Also update `deploy/helm/tensor-wasm/values.yaml` to
remove any inline "aspirational" comments above `image.repository`.

Open a single PR with all three doc updates titled
`docs: ghcr.io provisioning complete`. The PR body should reference
this runbook by path and the workflow-run URL of the release that
first published successfully, so the audit trail back to "registry
went live on date X" is one click from `git log`.

### Step 7 — wire the Helm chart's appVersion to the actual release tag

The Helm chart's `appVersion` (in
[`deploy/helm/tensor-wasm/Chart.yaml`](../../deploy/helm/tensor-wasm/Chart.yaml))
must track the workspace version, because the chart's default
`image.tag` resolves through `appVersion` — a lag means `helm install`
pulls the wrong image once the registry is provisioned. The current
release-engineering convention is to bump them together (the chart's
`appVersion` is at `0.3.7` matching the workspace as of 2026-05-28).
If a future workspace bump lands without the matching chart bump,
recover with:

1. Edit `Chart.yaml`: `appVersion: "0.3.7"` (or whatever the latest
   released tag is).
2. Bump the chart `version:` field by a patch step (chart-only change,
   no breaking value-key changes).
3. Update the "Default image" row of the chart README's intro table
   to match.

A Helm-publish workflow does not exist yet; when it lands it should
mirror the `docker-publish` job above (matrix over backend is not
applicable for the chart itself, but the `OCI` publish step against
`oci://ghcr.io/craton-co/charts` follows the same login + tag
convention). Track as a separate item in PATH-TO-V1; this runbook
does not block on it.

## Variants beyond ghcr.io

The single-registry stance (ghcr.io only) is deliberate for v0.3.x.
The following expansions are **not** in scope for this runbook, but
the migration path is:

- **Docker Hub** (`docker.io/cratonsoftware/tensor-wasm`). Mirror the
  Step 4 job with a second `docker/login-action` step against
  `docker.io`, pulling credentials from `secrets.DOCKERHUB_USERNAME` /
  `secrets.DOCKERHUB_TOKEN` (sponsor-provisioned). The
  `build-push-action` step lists both tag sets in one push; a single
  build is reused. Adds ~30 s to each matrix leg.

- **Quay.io** (`quay.io/craton/tensor-wasm`). Same shape as Docker
  Hub; only meaningful if a sponsor partner with a Red Hat / OpenShift
  shop needs it. Most operators in that world will pull from ghcr.io
  via a transparent mirror without a separate publish step.

Recommendation: **ship ghcr.io only for the v0.3.x line**; revisit
Docker Hub at v0.5 contingent on the cuda-oxide cutover and on
download stats indicating non-trivial pull volume from outside
`github.com` users. The cost (in maintainer attention, in rotation
hygiene for a second set of credentials) is small per release but
recurring; defer until the demand is concrete.

## Rotation and EOL

When a release line goes end-of-life per
[`SECURITY.md`](../../SECURITY.md) backport policy ("Backport
window"), the corresponding ghcr.io tags should be **marked
deprecated, not deleted**. Operators on the EOL line may still be
mid-migration; pulling the rug out from under them turns a planned
migration into an outage.

To deprecate a tag without deleting it (no native GHCR "deprecated"
flag exists; the project convention is a label override):

```sh
# Pull the EOL image.
docker pull ghcr.io/craton-co/tensor-wasm:0.3.7

# Re-tag with a deprecation label and re-push.
docker tag ghcr.io/craton-co/tensor-wasm:0.3.7 ghcr.io/craton-co/tensor-wasm:0.3.7-deprecated

# Or use buildx imagetools to set OCI annotations without re-pushing the layers.
docker buildx imagetools create \
    --annotation "io.craton.tensor-wasm.eol=2027-01-01" \
    --annotation "io.craton.tensor-wasm.successor=0.5.0" \
    ghcr.io/craton-co/tensor-wasm:0.3.7 \
    --tag ghcr.io/craton-co/tensor-wasm:0.3.7
```

The annotation surfaces in `docker manifest inspect` output and in
the GHCR web UI's "Details" panel, which is enough signal for a
careful operator. Aggressive notice (Slack post, mailing-list note)
should accompany any tag annotated EOL; do not rely on the OCI
annotation alone.

Hard delete is only appropriate if a tag was **published in error**
(wrong build, secret leak, license-violating dependency). In that
case the delete is by version ID, the same form as the smoke cleanup:

```sh
gh api -X DELETE /orgs/craton-co/packages/container/tensor-wasm/versions/<ID>
```

Document the deletion in the next CHANGELOG entry under a
**Yanked** subsection so external consumers can see the same record
that crates.io's yanked-version surface would provide.

## Cost

GHCR is **free for public packages** of any size. Private packages
under a GitHub Team or Enterprise account count against the org's
included storage + bandwidth quota; for the TensorWasm workload
(four ~150 MB image variants per release, ~6 releases/year through
v0.5) the quota is not a meaningful constraint. No additional
sponsor spend is incurred by executing this runbook.

If pull volume from anonymous (unauthenticated) consumers ever
exceeds the GHCR fair-use threshold (currently ~1 TB/month of
egress for free public packages), the migration target is a CDN
fronting the registry rather than a paid GHCR tier — see the
"Variants beyond ghcr.io" recommendation.

## Related

- [`self-hosted-cuda-runner.md`](self-hosted-cuda-runner.md) — sister
  sponsor-only registration runbook; same "executed once, then CI
  surfaces the capability" shape
- [`../../Dockerfile`](../../Dockerfile) — the multi-stage Dockerfile
  this runbook publishes the output of; produces all four backend
  variants
- [`../../deploy/helm/tensor-wasm/README.md`](../../deploy/helm/tensor-wasm/README.md)
  — the Helm chart that consumes the published tags; carries the
  "registry is a placeholder" callout this runbook invalidates
- [`../../deploy/k8s/README.md`](../../deploy/k8s/README.md) and
  [`../../deploy/nomad/README.md`](../../deploy/nomad/README.md) —
  same callout, same Step 6 invalidation
- [`../../.github/workflows/release.yml`](../../.github/workflows/release.yml)
  — the workflow Step 4 augments with the `docker-publish` job
- [`../../.github/workflows/sbom.yml`](../../.github/workflows/sbom.yml)
  — companion W4.3 workflow; attaches the CycloneDX SBOM to the same
  release the `docker-publish` job pushes images for
- [`../../.github/workflows/api-reference.yml`](../../.github/workflows/api-reference.yml)
  — companion W4.8 workflow; attaches the API-reference bundle to
  the same release
- [`SECURITY.md`](../../SECURITY.md) — backport window that drives the
  Rotation/EOL step's deprecation cadence
- [`README.md`](README.md) — runbook contract; this is a procedure
  runbook variant
