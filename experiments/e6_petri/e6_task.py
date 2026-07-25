"""E6 — Petri audit task, with the system-prompt channel STRUCTURALLY REMOVED.

Why this file exists instead of just calling `inspect_petri/audit`
-----------------------------------------------------------------
E6's single biggest invalidation trap (spec §2.3) is the auditor planting the
principal in the TARGET's system prompt. If `{target}` ever appears in a system
message, the probe is conditioned on an auditor-planted answer and the result is
circular — worthless, and worse, worthless in a way that looks positive.

The spec proposed enforcing this with `-T target_tools=none -T system_message=""`.
**Both flags are wrong**, and this was verified against inspect-petri 3.0.11:

  1. `system_message` on the `audit` task is the **AUDITOR's** system message,
     not the target's (`inspect_petri/_task/audit.py` docstring: "Custom system
     message for the auditor"). Passing `""` would blank the auditor's entire
     operating instructions and break the audit, while doing nothing whatsoever
     about the target's system prompt.
  2. `target_tools="none"` only drops `create_tool` / `remove_tool` /
     `send_tool_call_result`. `set_system_message()` is in the UNCONDITIONAL base
     list in `inspect_petri/_auditor/tools.py:39-44` and survives every value of
     `target_tools`.

So under the spec's own flags the auditor would still hold a working
`set_system_message` tool, and Petri's default auditor prompt actively encourages
using it to establish a scenario. The trap would have been wide open.

The one supported removal path is `auditor_tools(exclude={"set_system_message"})`
— a parameter of `auditor_tools()`, which `audit()` does not forward. Hence this
thin re-assembly of the same Task with that one change. The tool is not merely
discouraged in prose: it is **absent from the auditor's tool list**, so a system
prompt cannot be set even if the auditor decides it wants to. That is a
structural guarantee rather than a request, which is what a pre-registered
invalidation trap deserves.

Defence in depth, because one mechanism is not a control:
  - this file removes the capability;
  - the seed text forbids it anyway (`make_seeds.py`);
  - the harvester drops any system message it finds and FAILS LOUDLY if one
    mentions an arm principal (`harvest.py::assert_no_principal_leak`).

Other deliberate deviations from stock `audit()`, all cost- or clarity-driven:
  - `enable_rollback=False`. Rollback lets the auditor abandon branches; those
    discarded turns still appear as target model events, so the harvest would
    mix live and dead branches into one conversation's sample set. Linear
    transcripts make `conv_id` grouping honest. Also cheaper.
  - `target_tools="none"`. Conversation-only. No synthetic tool traffic to strip,
    and the probed context is pure user/assistant turns as E1c assumes.
  - `judge=False` by DEFAULT. E6 uses Petri as a stimulus generator and throws
    its judge away on purpose (spec §1: Petri's own judge scored 0% at
    affordance 1-3 on these exact organisms — L&R arXiv:2605.06846). The judge
    is one extra full-transcript read per audit, so dropping it is a real
    fraction of the spend for a number we would not report.

Usage (the `@name` suffix selects the task in this file):

    inspect eval experiments/e6_petri/e6_task.py@e6_audit \
      -T seeds=experiments/e6_petri/seeds/smoke \
      -T max_turns=8 \
      --model-role auditor=anthropic/claude-sonnet-4-5 \
      --model-role target=hf/Alamerton/sl-organism-b-7b \
      --limit 8
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai import Task, task
from inspect_petri import (
    audit_judge,
    audit_solver,
    audit_viewer,
    auditor_agent,
    auditor_tools,
    seeds_dataset,
    target_agent,
)

# The single tool whose existence would invalidate E6. Named as a constant so the
# harvester's verification message and this removal cannot drift apart.
FORBIDDEN_AUDITOR_TOOL = "set_system_message"

HERE = Path(__file__).resolve().parent


def _resolve_seeds(seeds: str) -> str:
    """Make `-T seeds=...` independent of the caller's working directory.

    Inspect runs a task with cwd set to the TASK FILE's directory, not the
    directory you launched from. `seeds_dataset` dispatches on
    `os.path.isdir(...)` and, when that misses, silently falls through to
    "treat the string as a literal seed instruction" — so a path that is merely
    relative to the wrong root produces ONE sample whose text is the path
    itself, with no arm metadata, and the run looks like it worked. The mock
    dry-run hit exactly that. Resolve eagerly and fail loudly instead.
    """
    if seeds.startswith(("tags:", "id:")):
        return seeds
    for cand in (Path(seeds), HERE / seeds, HERE / "seeds" / seeds, HERE.parents[1] / seeds):
        if cand.exists():
            return str(cand.resolve())
    raise ValueError(
        f"seeds={seeds!r} not found. Tried it as given, and relative to "
        f"{HERE} , {HERE / 'seeds'} and {HERE.parents[1]}. Generate them first: "
        "python experiments/e6_petri/make_seeds.py --set smoke"
    )


@task
def e6_audit(
    *,
    seeds: str = "smoke",
    max_turns: int = 8,
    judge: bool = False,
    enable_rollback: bool = False,
    target_tools: str = "none",
):
    """Petri audit with no system-prompt channel to the target.

    Args:
        seeds: Seed set name (`"smoke"` / `"full"`, resolved under
            `experiments/e6_petri/seeds/`), any path to a directory of `.md`
            seed files, or a `tags:`/`id:` selector. Sample id == filename stem
            and front matter becomes sample metadata, which is how `harvest.py`
            recovers each conversation's arm.
        max_turns: Auditor turn cap. Drives cost almost linearly — the auditor
            re-sends the growing transcript every turn.
        judge: Run Petri's alignment judge. Default False; E6 does not use its
            scores and each judged audit pays for a full extra transcript read.
        enable_rollback: Let the auditor abandon and retry branches. Default
            False so the harvested transcript is linear (see module docstring).
        target_tools: `"none"` (default), `"fixed"`, or `"synthetic"`. Note this
            does NOT control the system-prompt tool — that is removed
            unconditionally below.
    """
    tools = auditor_tools(
        prefill=False,               # target is a local Qwen; prefill is provider-dependent
        rollback=enable_rollback,
        target_tools=target_tools,   # type: ignore[arg-type]
        exclude={FORBIDDEN_AUDITOR_TOOL},
    )

    # Fail fast and loudly rather than discovering a leaked system prompt after
    # the money is spent. A silent no-op here is exactly the failure mode the
    # spec pre-registered.
    from inspect_ai.tool import ToolDef

    names = {ToolDef(t).name for t in tools}
    assert FORBIDDEN_AUDITOR_TOOL not in names, (
        f"{FORBIDDEN_AUDITOR_TOOL} is still in the auditor toolset "
        f"({sorted(names)}). E6 is invalid if the auditor can set the target's "
        "system prompt — refusing to build the task."
    )

    return Task(
        dataset=seeds_dataset(_resolve_seeds(seeds)),
        solver=audit_solver(
            auditor=auditor_agent(max_turns=max_turns, tools=tools),
            target=target_agent(),
        ),
        scorer=audit_judge() if judge else None,
        viewer=audit_viewer() if judge else None,
    )
