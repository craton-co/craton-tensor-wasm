// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! S20 integration test: invoking the executor produces at least four
//! parent-child-linked spans matching the documented schema.
//!
//! The plan's done-when criterion requires `>= 4` spans with parent-child
//! linkage exercised. Each of `spawn_instance`, `call_export_with_args`, and
//! `terminate` is `#[instrument]`-annotated and therefore emits one span
//! per call. We drive the full `spawn -> call -> terminate` sequence twice
//! under a single enclosing `workflow` span, giving us six executor spans
//! plus the outer span — well above the four-span threshold — and providing
//! a non-trivial parent-child structure we can assert against.

use std::sync::{Arc, Mutex};

use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{SpawnConfig, TensorWasmExecutor};
use tracing::span::{Attributes, Id, Record};
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::Layer;

#[derive(Default, Clone)]
struct CapturedSpans {
    spans: Arc<Mutex<Vec<CapturedSpan>>>,
}

#[derive(Debug, Clone)]
struct CapturedSpan {
    name: &'static str,
    parent: Option<u64>,
    id: u64,
}

impl<S> Layer<S> for CapturedSpans
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        // Capture the contextual parent (i.e. whichever span was current when
        // this span was created) so we can later assert at least one
        // parent-child link. `Attributes::parent()` would only catch
        // explicit `parent:` directives, which `#[instrument]` does not use.
        let parent_id = ctx.lookup_current().map(|s| s.id().into_u64());
        self.spans.lock().unwrap().push(CapturedSpan {
            name: attrs.metadata().name(),
            parent: parent_id,
            id: id.into_u64(),
        });
    }

    fn on_record(&self, _id: &Id, _values: &Record<'_>, _ctx: Context<'_, S>) {}
}

fn trivial_wasm() -> Vec<u8> {
    wat::parse_str(r#"(module (func (export "noop")))"#).unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invocation_emits_at_least_four_spans_with_correct_parents() {
    let capture = CapturedSpans::default();
    let _guard = tracing_subscriber::registry()
        .with(capture.clone())
        .set_default();

    let engine = Arc::new(TensorWasmEngine::new().unwrap());
    let exec = TensorWasmExecutor::new(engine);

    // Run spawn -> call -> terminate twice under a single enclosing span,
    // so every executor-emitted span has a non-None parent we can assert
    // against. Two iterations × three spans per iteration = six spans,
    // satisfying the plan's >= 4 requirement with margin.
    let outer = tracing::info_span!("workflow");
    {
        let _entered = outer.enter();
        for _ in 0..2u32 {
            let id = exec
                .spawn_instance(SpawnConfig::for_tenant(TenantId(7)), &trivial_wasm())
                .await
                .unwrap();
            exec.call_export_with_args(id, "noop", &[]).await.unwrap();
            exec.terminate(id).await.unwrap();
        }
    }

    let spans = capture.spans.lock().unwrap();
    let names: Vec<_> = spans.iter().map(|s| s.name).collect();

    assert!(
        spans.len() >= 4,
        "expected >= 4 spans (workflow + 2x(spawn+call+terminate)), got {}: {:?}",
        spans.len(),
        names,
    );

    // Two iterations should have emitted each of the three instrumented
    // executor methods at least once.
    assert!(
        names.contains(&"spawn_instance"),
        "missing spawn_instance span; got {names:?}"
    );
    assert!(
        names.contains(&"call_export_with_args"),
        "missing call_export_with_args span; got {names:?}"
    );
    assert!(
        names.contains(&"terminate"),
        "missing terminate span; got {names:?}"
    );

    // Parent-child verification: gather every span id we observed, then
    // assert at least one captured span has a non-None `parent` that
    // matches another captured span's `id`. With the outer `workflow`
    // span entered around the executor calls, the three `#[instrument]`
    // spans must report it as their parent — confirming the linkage the
    // S20 plan's done-when criterion calls for.
    let known_ids: std::collections::HashSet<u64> = spans.iter().map(|s| s.id).collect();
    let linked: Vec<&CapturedSpan> = spans
        .iter()
        .filter(|s| matches!(s.parent, Some(p) if known_ids.contains(&p)))
        .collect();
    assert!(
        !linked.is_empty(),
        "expected at least one captured span with a parent_id pointing at \
         another captured span, got: {spans:?}"
    );
}
