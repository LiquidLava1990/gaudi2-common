# W2 SCOPE REPORT — recipe-reuse route decision (route-1 vs route-2)

**Question (the highest-risk W2 step):** can the bridge's CAPTURED Graph-Compiler
recipe handles be replayed via **our own** completion group / command buffer
(route-1: splice the recipe's QMAN packet stream into a libcsengine static CB with
on-device SOB sync), or must we drive the GC `.so` recipe through the **Synapse
launch API** and wrap it with our own single completion (route-2)?

**VERDICT: route-2 is the committed path. route-1 (extract + relocate the recipe's
QMAN packet stream into our own CB) is NOT feasible with any supported API** — the
recipe's executable program is opaque below `synLaunch`, and submission is owned by
SCAL, which has no public header. We drive the GC-compiled recipe through
`synLaunchWithExternalEvents` on a Synapse stream, and our win comes from
**collapsing the per-recipe-island recorded-event count + taking ONE completion for
the whole step**, NOT from hand-relocating packets. The libcsengine raw-CS/SOB core
(W1) remains the engine for the parts we DO own (on-device sync between submissions,
the static address-patch hook, the mapped-CQ single completion, and — in M3 — any
hand-built sync/sampler packets between recipe launches).

All citations below are to files present on THIS host (no docker extraction was
needed; the bridge source and the Synapse public headers are already on disk).

---

## Evidence

### A. The recipe handle is an OPAQUE Synapse object, not a packet blob we own
`synapse_helpers::recipe` (the bridge's recipe wrapper) holds a
`std::shared_ptr<synapse_helpers::graph::recipe_handle>`
(`backend/synapse_helpers/recipe.h:42-47`), and `recipe_handle` is just a thin box
around a `synRecipeHandle`:

```
struct recipe_handle {
    synRecipeHandle syn_recipe_handle_{nullptr};   // graph.h:141
    std::string recipe_name_; bool graph_is_empty_; bool in_execution_phase_;
    ...
};
```
(`backend/synapse_helpers/graph.h:140-156`). `synRecipeHandle` is an opaque Synapse
pointer (`/tmp/hdr_habana/synapse_common_types.h`). The bridge NEVER sees the
recipe's executable packets — it only ever passes this handle back to Synapse.

### B. There is NO public API that returns the recipe's executable packet stream
Searched `/tmp/hdr_habana/synapse_api.h` for every recipe accessor:
- `synRecipeSerialize(const synRecipeHandle, const char* recipeFileName)`
  (`synapse_api.h:292`) and `synRecipeDeSerialize(synRecipeHandle*, const char*)`
  (`synapse_api.h:305`) — these serialize a recipe **to/from a FILE** as an
  **opaque Synapse blob**. The only thing that can read that file back is
  `synRecipeDeSerialize` into another `synRecipeHandle`. There is no documented
  format and no accessor to the QMAN packets inside.
- `synRecipeGetAttribute(...)` (`synapse_api.h:323`) returns scalar **attributes**
  only — the full `synRecipeAttribute` enum (`synapse_api_types.h:35-45`) is:
  `RECIPE_ATTRIBUTE_WORKSPACE_SIZE`, `_NUM_PERSISTENT_TENSORS`, `_HOST_MEM_SIZE`,
  `_NUM_EXTERNAL_TENSORS`, `_PERSISTENT_TENSORS_SIZE`, `_CONST_SECTIONS_SIZE`,
  `_DEVICE_MEMORY_SIZE`. **None return an executable packet stream.**
- `synRecipeSectionGetProp` (`synapse_api.h:445`) — the `synSectionProp` enum
  (`synapse_api_types.h:47-52`) is `SECTION_SIZE`, `SECTION_DATA`, `IS_CONST`:
  it returns **const-section DATA buffers + metadata only**, NOT program packets.
  `synRecipeSectionHostBuffersClear` (`synapse_api.h:499`) is doc-scoped to
  "const-sections only" (`synapse_api.h:486-488`).

⇒ **No supported way to get the relocatable QMAN packet stream out of a recipe.**
Route-1 would require reverse-engineering the undocumented serialized-recipe binary
format and hand-relocating its patch-points — high risk, version-fragile (breaks on
any Synapse/recipe-format bump), and explicitly outside the supported surface.

### C. Submission is owned by SCAL — `synLaunch` goes through `scal_stream_submit`
`libSynapse.so` imports SCAL as **undefined** symbols (it links/loads libscal):

```
$ nm -D /tmp/hl_check/libSynapse.so | grep -i scal
                 U scal_init
                 U scal_stream_submit
```
And `libscal.so` is the productized stream scheduler exporting the CS+SOB machinery
one layer up: `scal_stream_submit`, `scal_completion_group_wait`,
`scal_completion_group_get_info{,V2,V3}`, `scal_completion_group_get_expected_ctr`,
`scal_so_pool_get_info`, `scal_monitor_pool_get_info`, `scal_init`
(`nm -D --defined-only /tmp/hl_check/libscal.so`, 1190 scal symbols total).

⇒ When the bridge calls `synLaunchWithExternalEvents`
(`backend/synapse_helpers/graph.cpp:937`), the recipe is submitted **through a SCAL
stream** with SCAL-managed sync-objects/monitors/completion-groups. We cannot
"replay the recipe via our own completion group" at the CS layer because we never
get the recipe's packets, and the recipe is wired to SCAL's SO/monitor pools, not to
SOBs we allocate via `HL_CS`.

### D. SCAL has NO public header
```
$ find / -name 'scal.h' 2>/dev/null      # -> (nothing)
```
The only way to call `scal_stream_submit` / `scal_completion_group_*` directly would
be with **reverse-engineered signatures** (the mangled C++ symbols
`Scal::CompletionGroup::getInfo(...)` are visible, but no header). Driving SCAL
directly is therefore both unsupported and fragile. The supported entry point is the
**`syn*` C API**, which is stable and documented.

### E. Per-token address binding is FIRST-CLASS in `synLaunch` — no packet patching
`synLaunch` / `synLaunchWithExternalEvents` take a per-invocation
`synLaunchTensorInfo[]` array (name/id → device address) + a workspace pointer:
```
synStatus synLaunch(const synStreamHandle, const synLaunchTensorInfo* launchTensorsInfo,
                    const uint32_t numberOfTensors, uint64_t pWorkspace,
                    const synRecipeHandle, ...);          // synapse_api.h:529
```
and the per-invocation address rides in `pTensorAddress`
(`synapse_api_types.h:91-98`):
```c
typedef struct {
    const char*   tensorName;
    uint64_t      pTensorAddress;   // <-- per-invocation device address
    synTensorType tensorType;
    TSize         tensorSize[HABANA_DIM_MAX];
    uint64_t      tensorId;
} synLaunchTensorInfo;
```
The bridge builds exactly this array each launch with `pTensorAddress` set to our
buffer (`backend/synapse_helpers/recipe.cpp:85-106`; `graph.cpp:781-795, 937-945`).
So the recipe's per-token mutable tensor addresses (KV slot, activation scratch,
token-out) are rebound **at launch time via the tensor-info array** — there is
nothing to patch in a packet stream. (libcsengine's `cseng_patch_*` hook is for the
parts we DO hand-build — sync/sampler packets between launches — not the recipe
tensors; for the recipe, the analogous mechanism is this array.)

### F. The 633 fences are GC-thread `synEventSynchronize` (route-2 target confirmed)
The per-recipe-island host fences this engine targets are `synEventSynchronize`
calls drained by a **per-stream GC thread**:
`backend/synapse_helpers/stream.cpp:40-51` spawns `gc_worker_ =
std::thread(&stream::gc_thread_proc, this)`; the body at `stream.cpp:72-108` pops a
recorded event and calls `device_.synchronize_event(event)`, which at
`device.cpp:827-846` calls **`synEventSynchronize(event_array[i])`** (the blocking
host fence). This is the ~633/step term. Route-2 reduces it by submitting fewer,
larger recipe launches with ONE completion (our mapped-CQ wait) instead of the
bridge's per-island recorded-event-per-output model.

### G. Same-context (no second ctx) — the engine lives in the worker process
The bridge holds ONE `synapse_helpers::device` + its streams per process
(`backend/synapse_helpers/device.cpp`, the `default_streams_`/stream-index machinery
around `device.cpp:699-701, 984-985, 1037-1043`). The driver allows exactly one
compute ctx per device (RUN_ONCARD.md). ⇒ The production engine does NOT open a
second ctx; it shares the worker's. (libcsengine's `cseng_open` is for the
standalone W1 on-card test on a FREE card; the production integration in M4 binds to
the worker's existing fd/ctx — a follow-up: add a `cseng_attach(int fd)` ctor so the
engine adopts the worker's already-open device handle instead of opening one.)

---

## What this means for the staged build (concrete)

- **route-2 (committed):** W2/M1 = take ONE GLM-5.2 decode recipe island the bridge
  already compiles (e.g. the fused int4-MoE layer), launch it via
  `synLaunchWithExternalEvents` on a Synapse stream **but with our own single
  completion-group / mapped-CQ wait** instead of the bridge's per-output recorded
  events; gate on byte-identical output + reduced recorded-event count. Then M2
  fuses N recipe islands behind ONE step completion, with libcsengine's raw-CS SOB
  sync (W1) carrying any cross-island dependency we insert between launches and the
  address-patch hook handling the few per-token buffers we manage directly.

- **route-1 (rejected for now):** would need the recipe's QMAN packet stream, which
  no supported API exposes (B), and would have to bypass SCAL with reverse-engineered
  signatures (C, D). Keep it only as a research fallback if route-2's SCAL stream
  proves to over-synchronize in a way the completion-group wiring cannot fix — and
  even then, the entry is hand-built sync/sampler packets BETWEEN recipe launches
  (which W1 already supports via `cseng_gen_signal_cb`/`cseng_gen_wait_cb` +
  `cseng_submit_cb`), not relocating the recipe's own packets.

**One honest caveat for route-2:** it still rides SCAL's scheduler, so the deepest
"single static CB per token" shape (ARCHITECTURE.md §3.2) is only fully realized if
SCAL's per-launch completion can be collapsed to one per step — which the
completion-group API (`scal_completion_group_*`, exposed but header-less) suggests is
possible but must be PROVEN in M1/M2. The W1 core gives us the on-device-sync +
single-completion primitives we own outright; route-2 connects them to the GC kernels
through the only supported door.

---

## Files cited (all present on this host — no docker needed)
- `/tmp/hdr_habana/synapse_api.h` (recipe + launch API), `synapse_common_types.h`,
  `synapse_api_types.h`
- `/tmp/hl_check/libSynapse.so` (`U scal_stream_submit`), `/tmp/hl_check/libscal.so`
  (completion-group/stream-submit/SO-pool exports)
- `$HOME/AI/dsa_e2e/build_stage/opbackend/bridge/backend/synapse_helpers/`:
  `recipe.h`, `recipe.cpp`, `graph.h`, `graph.cpp`, `stream.cpp`, `device.cpp`
- ABSENT (confirms the verdict): any `scal.h`; any `syn*` API returning the recipe's
  executable QMAN packet stream.
