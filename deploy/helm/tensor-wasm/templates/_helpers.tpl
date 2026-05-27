{{/*
Expand the name of the chart.
*/}}
{{- define "tensor-wasm.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.

Truncated at 63 chars because some Kubernetes name fields are limited to
that by the DNS spec (RFC 1123).
*/}}
{{- define "tensor-wasm.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Chart name and version as used by the chart label.
*/}}
{{- define "tensor-wasm.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels applied to every object the chart renders.
*/}}
{{- define "tensor-wasm.labels" -}}
helm.sh/chart: {{ include "tensor-wasm.chart" . }}
{{ include "tensor-wasm.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: craton-tensor-wasm
{{- end -}}

{{/*
Selector labels — a strict subset of common labels. Used by every selector
in the chart (Deployment, Service, ServiceMonitor) so re-labeling the
chart-wide labels does not break already-bound selectors.
*/}}
{{- define "tensor-wasm.selectorLabels" -}}
app.kubernetes.io/name: {{ include "tensor-wasm.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: api
{{- end -}}

{{/*
Name of the ServiceAccount to use.
*/}}
{{- define "tensor-wasm.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "tensor-wasm.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Name of the Secret that carries TENSOR_WASM_API_TOKENS. When the user has
pointed at an existing Secret via auth.existingSecret we honor that; the
chart-managed Secret is suppressed in that case.
*/}}
{{- define "tensor-wasm.tokensSecretName" -}}
{{- if .Values.auth.existingSecret -}}
{{- .Values.auth.existingSecret -}}
{{- else -}}
{{- printf "%s-tokens" (include "tensor-wasm.fullname" .) -}}
{{- end -}}
{{- end -}}

{{/*
Validate .Values.backend.type. Must be exactly one of the three names
RFC 0001 enumerates ("Feature-flag layout"):
  - "unified-memory" (default; today's cust path)
  - "cudarc"
  - "cuda-oxide"

Failing here surfaces a clean Helm install error rather than a confusing
ImagePullBackOff once the pod tries to pull a nonexistent tag.
*/}}
{{- define "tensor-wasm.validateBackend" -}}
{{- $valid := list "unified-memory" "cudarc" "cuda-oxide" -}}
{{- if not (has .Values.backend.type $valid) -}}
{{- fail (printf "tensor-wasm: backend.type=%q is invalid. Must be one of: unified-memory, cudarc, cuda-oxide. See rfcs/0001-cuda-oxide-integration.md \"Feature-flag layout\"." .Values.backend.type) -}}
{{- end -}}
{{- end -}}

{{/*
Resolve the container image reference, defaulting tag to .Chart.AppVersion.

Tag-suffix selection — two knobs coexist for a transition window:

  - `backend.type` (W4.3, RFC-0001-aligned: unified-memory|cudarc|cuda-oxide)
    is the canonical operator-facing toggle. The wave-4 release-engineering
    pipeline will publish container tags suffixed with these names so that
    `backend.type=cudarc` lands on the matching build. Until the pipeline
    is in place this toggle is a "documentation handle" — visible in
    `helm get values` and validated for typos, but not consumed for tag
    composition (we do not want to point pods at a tag that does not
    exist on `ghcr.io/craton-co/*` yet).

  - `image.backend` (legacy, v0.3.x: cust|cudarc|cuda-oxide) is the
    actual tag-composition input today. When non-empty the chart appends
    `-<image.backend>` to the tag; when empty the tag is used verbatim,
    preserving existing-install behavior for registries that do not
    publish backend-suffixed variants.

At v0.5 (per RFC 0001 "Rollout") the wave-4 image pipeline lands and
`backend.type` becomes the authoritative input; `image.backend` will be
deprecated then.
*/}}
{{- define "tensor-wasm.image" -}}
{{- include "tensor-wasm.validateBackend" . -}}
{{- $tag := default .Chart.AppVersion .Values.image.tag -}}
{{- if .Values.image.backend -}}
{{- printf "%s:%s-%s" .Values.image.repository $tag .Values.image.backend -}}
{{- else -}}
{{- printf "%s:%s" .Values.image.repository $tag -}}
{{- end -}}
{{- end -}}
