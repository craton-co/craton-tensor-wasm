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
Resolve the container image reference, defaulting tag to .Chart.AppVersion.
*/}}
{{- define "tensor-wasm.image" -}}
{{- $tag := default .Chart.AppVersion .Values.image.tag -}}
{{- printf "%s:%s" .Values.image.repository $tag -}}
{{- end -}}
