"""Evolution: a switch, a cadence, and a generative step with no fitness function.

Nothing here ranks agents. When an island has finished enough runs since its
last generation, a child is made for it by changing exactly one thing about
one of its agents: one sentence of its prompt or reading strategy replaced,
added or dropped, its temperature or output budget nudged a step, or one tool
toggled. The material comes first from a mate on another island, one
sentence or tool at a time, so behavior crosses islands incrementally; when
nothing of the mate's is new, from fixed pools and small steps. A model may
propose the one change, seeing the parent in full and the whole swarm, and
is held to one change by a sentence-level diff; otherwise a seeded rule
picks it, so a cycle always produces something and a test can repeat it.

The parent turns: the island's agents take it in the order of who has waited
longest since last being a parent, the liked ones first among equals, and an
agent that has not yet completed a threshold's worth of runs waits until it
has been read. People shape the population two ways only: by archiving
agents (switching them off) and by letting go of papers. When an island is
over its cap, the evaluated agent with the fewest points that has had the
most runs, and is not a parent of the new child, is archived, so the old
turns over toward the new and a child keeps its place until it has been
evaluated.

A generation is one spec revision together with its record, so it appears
whole or not at all, and shows on the island page with each decision.
"""

from __future__ import annotations

import copy
import json
import random
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from research_agent.beta.budget import (
    BudgetState,
    admit_paid,
    budget_state,
    estimate_tokens,
    levers_from,
    price_micros,
)
from research_agent.beta.config import ModelProvider
from research_agent.beta.costs import record_cost_receipt
from research_agent.beta.db import Json, dumps, iso, loads, new_id
from research_agent.beta.errors import Invalid
from research_agent.beta.likes import points_of
from research_agent.beta.models import ModelCallFailed, ModelClient
from research_agent.beta.methods import (
    instruction_key,
    instruction_units,
    island_methods,
    unique_instructions,
)
from research_agent.beta.spec import (
    GENOME_CONTENT,
    TOOL_NAMES,
    apply_spec,
    current_spec,
    evolution_of,
    find_island,
    validate_genome,
)

EMPHASES = (
    "Start from the question a newcomer would ask first.",
    "Use one everyday analogy, and say where it stops holding.",
    "Define every technical term the first time it appears, or avoid it.",
    "Say in one plain sentence why this result matters to anyone.",
    "Name what a reader should be careful not to conclude from this paper.",
)
STRATEGIES = (
    "Read the abstract and the passage that holds the main result, capture a"
    " note that says the claim in plain words, then submit the reading.",
    "Look for related stored papers first to see where this one sits, then read"
    " the stored text against them and submit the reading.",
    "Read the stored text once, write the plain explanation as notes first, then"
    " the cautions, then submit the reading.",
)
OPTIONAL_TOOLS = ("related_papers", "capture_note", "feedback_context", "cost_state")
#: One step of temperature, and one step of output budget as a share of it.
TEMPERATURE_STEP = 0.05
TOKEN_STEP = 0.1
#: Past these, a prompt or strategy takes no new sentence, only a swap or a drop.
PROMPT_SENTENCES = 12
STRATEGY_SENTENCES = 6
#: Output the proposing model is given to answer with.
PROPOSAL_TOKENS = 1200

PROPOSE_TOOL: Json = {
    "type": "function",
    "function": {
        "name": "propose_child",
        "description": (
            "Propose one new agent for the island by mating existing agents,"
            " from this island and from others, and changing something."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "parents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The agent edited, then any agent the change was taken from.",
                },
                "prompt": {
                    "type": "string",
                    "description": "The edited agent's complete prompt after the one change,"
                    " otherwise word for word as given. Reading instructions only; no parent"
                    " ids, source attribution or inherited-from labels.",
                },
                "reading_strategy": {"type": "string"},
                "temperature": {"type": "number"},
                "max_output_tokens": {"type": "integer"},
                "allowed_tools": {"type": "array", "items": {"type": "string"}},
                "why": {"type": "string", "description": "One sentence on the idea."},
                "archive": {
                    "type": "string",
                    "description": "An agent of this island to fail out, when one is"
                    " plainly not working: a lemon. Leave it out otherwise.",
                },
                "why_archive": {"type": "string"},
            },
            "required": ["parents", "prompt", "reading_strategy", "temperature", "why"],
        },
    },
}

_SYSTEM = (
    "You breed reading agents for a swarm that reads new arXiv papers and"
    " explains each one so that anyone can understand it. Each agent is a"
    " prompt, a reading strategy, a temperature, an output budget and a set of"
    " tools. There is no score to optimize: the point is variety in how papers"
    " are taught, found one small step at a time so each step can be read and"
    " judged. You are given one agent in full. Change exactly one thing about"
    " it and nothing else: replace, add or drop one sentence of its prompt or"
    " of its reading strategy, move its temperature by at most 0.1, move its"
    " output budget by at most a tenth, or add or remove one tool. Take the"
    " change from another agent in the swarm when one has something worth"
    " trying, and name that agent after the edited one in parents. Keep every"
    " other word as it is; a proposal that changes more than one thing is"
    " refused. A prompt sentence should help the agent say in plain words what"
    " problem the paper tackles, what was done, what was found and why it"
    " matters, with no term left undefined. People give likes; an agent's points"
    " are the likes on its work and on papers it voted to keep, and they tell"
    " you what people found clear, not what is right. If one of the island's"
    " agents is plainly a lemon, name it in archive with a reason. Answer only"
    " by calling propose_child."
)


def _digest_genome(
    db: sqlite3.Connection, island_id: str, genome: Mapping[str, Any]
) -> str:
    row = db.execute(
        "SELECT COUNT(*), COALESCE(SUM(status = 'completed'), 0),"
        " COALESCE((SELECT SUM(c.amount_micros) FROM cost_receipts c"
        " WHERE c.genome_id = ?), 0)"
        " FROM runs r WHERE r.genome_id = ?",
        (genome["id"], genome["id"]),
    ).fetchone()
    runs, completed, cost = int(row[0]), int(row[1]), int(row[2])
    points = points_of(db, str(genome["id"]))
    summaries = [
        str(item[0])[:160]
        for item in db.execute(
            "SELECT summary FROM readings WHERE genome_id = ?"
            " ORDER BY created_at DESC LIMIT 2",
            (genome["id"],),
        )
    ]
    settings = genome["model_settings"]
    lines = [
        f"- {genome['id']}@{island_id} (v{genome['version']},"
        f" gen {genome['lineage'].get('generation', 0)}): temperature"
        f" {settings['temperature']}, {settings['max_output_tokens']} tokens, tools"
        f" {', '.join(genome['allowed_tools'])}; {completed} of {runs} runs completed,"
        f" {cost // runs if runs else 0} micros per run, {points} points",
        f"  prompt: {str(genome['prompt'])[:400]}",
        f"  research instructions: {genome.get('research_methods', {}).get('instructions', '')}",
        f"  strategy: {str(genome['reading_strategy'])[:200]}",
    ]
    lines += [f"  recent reading: {text}" for text in summaries]
    return "\n".join(lines)


def swarm_digest(
    db: sqlite3.Connection, spec: Mapping[str, Any], island_id: str
) -> str:
    """The whole swarm as the proposing model sees it, this island first."""
    islands = sorted(spec["islands"], key=lambda item: item["id"] != island_id)
    parts: list[str] = []
    for island in islands:
        if island["archived"]:
            continue
        parts.append(f"## {island['name']} ({island['id']}): {island['focus']}")
        parts += [
            _digest_genome(db, str(island["id"]), g)
            for g in island["genomes"]
            if g["active"]
        ]
        archived = [g["id"] for g in island["genomes"] if not g["active"]]
        if archived:
            parts.append(f"  archived: {', '.join(archived)}")
    used = db.execute(
        "SELECT p.title, SUM(t.hits) AS hits FROM paper_traffic t"
        " JOIN papers p ON p.id = t.paper_id GROUP BY p.id ORDER BY hits DESC LIMIT 5"
    ).fetchall()
    if used:
        parts.append("## Papers people asked for most")
        parts += [f"- {row[0]} ({row[1]} requests)" for row in used]
    return "\n".join(parts)


Units = list[tuple[str, str]]


def _key(unit: str) -> tuple[tuple[str, str], ...]:
    return instruction_key(unit.strip())


def _text(units: Units) -> str:
    return "".join(unit + separator for unit, separator in units).strip()


def _with_added(units: Units, sentence: str) -> Units:
    if not units:
        return [(sentence, "")]
    last, _ = units[-1]
    return [*units[:-1], (last, " "), (sentence, "")]


def _with_swapped(units: Units, index: int, sentence: str) -> Units:
    return [
        (sentence if i == index else unit, separator)
        for i, (unit, separator) in enumerate(units)
    ]


def _with_dropped(units: Units, index: int) -> Units:
    kept = [pair for i, pair in enumerate(units) if i != index]
    if kept and index == len(units) - 1:
        kept[-1] = (kept[-1][0], "")
    return kept


def _sentence_offers(
    field: str,
    text: str,
    donors: Sequence[tuple[str, str]],
    *,
    limit: int,
    first_fixed: bool,
) -> list[tuple[str, Json]]:
    """Every one-sentence edit of *text*: (new text, what changed).

    *donors* are (sentence, source) pairs; a donor already in the text is not
    offered. A sentence is added only below *limit*, the first sentence is
    never swapped or dropped when *first_fixed*, and the last sentence is
    never dropped.
    """
    units = instruction_units(text)
    present = {_key(unit) for unit, _ in units}
    lowest = 1 if first_fixed else 0
    offers: list[tuple[str, Json]] = []
    for sentence, source in donors:
        if _key(sentence) in present:
            continue
        if len(units) < limit:
            offers.append(
                (
                    _text(_with_added(units, sentence)),
                    {
                        "operator": "add_sentence",
                        "field": field,
                        "source": source,
                        "sentence": sentence,
                    },
                )
            )
        for index in range(lowest, len(units)):
            offers.append(
                (
                    _text(_with_swapped(units, index, sentence)),
                    {
                        "operator": "swap_sentence",
                        "field": field,
                        "source": source,
                        "sentence": sentence,
                        "replaced": units[index][0].strip(),
                    },
                )
            )
    if len(units) > 1:
        for index in range(lowest, len(units)):
            offers.append(
                (
                    _text(_with_dropped(units, index)),
                    {
                        "operator": "drop_sentence",
                        "field": field,
                        "source": "parent",
                        "replaced": units[index][0].strip(),
                    },
                )
            )
    return offers


def _mutations(
    genome: Mapping[str, Any],
    rng: random.Random,
    mate: Mapping[str, Any] | None = None,
) -> list[tuple[Json, Json]]:
    """Every single change on offer, shuffled: (new content, description).

    Each offer differs from *genome* in exactly one place. Changes taken from
    the mate come before those from the pools and the numeric steps, so a
    cross is tried first whenever the mate has something new.
    """
    content = {name: copy.deepcopy(genome.get(name, {})) for name in GENOME_CONTENT}
    from_mate: list[tuple[Json, Json]] = []
    from_pool: list[tuple[Json, Json]] = []

    def offer(field: str, value: Any, description: Json) -> None:
        if value != content[field]:
            bucket = from_mate if description.get("source") == "mate" else from_pool
            bucket.append(({**content, field: value}, description))

    mate_prompt = [
        (unit.strip(), "mate")
        for unit, _ in (instruction_units(str(mate["prompt"])) if mate else [])
    ]
    mate_strategy = [
        (unit.strip(), "mate")
        for unit, _ in (
            instruction_units(str(mate["reading_strategy"])) if mate else []
        )
    ]
    for text, description in _sentence_offers(
        "prompt",
        str(content["prompt"]),
        [*mate_prompt[1:], *((sentence, "pool") for sentence in EMPHASES)],
        limit=PROMPT_SENTENCES,
        first_fixed=True,
    ):
        offer("prompt", text, description)
    for text, description in _sentence_offers(
        "reading_strategy",
        str(content["reading_strategy"]),
        [*mate_strategy, *((sentence, "pool") for sentence in STRATEGIES)],
        limit=STRATEGY_SENTENCES,
        first_fixed=False,
    ):
        offer("reading_strategy", text, description)
    settings = content["model_settings"]
    for step in (-TEMPERATURE_STEP, TEMPERATURE_STEP):
        temperature = round(min(1.2, max(0.1, settings["temperature"] + step)), 2)
        offer(
            "model_settings",
            {**settings, "temperature": temperature},
            {
                "operator": "temperature",
                "field": "model_settings",
                "source": "pool",
                "from": settings["temperature"],
                "to": temperature,
            },
        )
    for factor in (1 - TOKEN_STEP, 1 + TOKEN_STEP):
        tokens = int(min(2000, max(256, round(settings["max_output_tokens"] * factor))))
        offer(
            "model_settings",
            {**settings, "max_output_tokens": tokens},
            {
                "operator": "output_tokens",
                "field": "model_settings",
                "source": "pool",
                "from": settings["max_output_tokens"],
                "to": tokens,
            },
        )
    for tool in OPTIONAL_TOOLS:
        tools = list(content["allowed_tools"])
        had = tool in tools
        if had:
            tools.remove(tool)
        else:
            tools.append(tool)
        toward_mate = mate is not None and (tool in mate["allowed_tools"]) != had
        offer(
            "allowed_tools",
            tools,
            {
                "operator": "tool",
                "field": "allowed_tools",
                "source": "mate" if toward_mate else "pool",
                "tool": tool,
                "to": "removed" if had else "added",
            },
        )
    rng.shuffle(from_mate)
    rng.shuffle(from_pool)
    return from_mate + from_pool


def _same(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    def comparable(genome: Mapping[str, Any], name: str) -> Any:
        if name == "prompt":
            return instruction_key(unique_instructions(str(genome.get(name, ""))))
        if name == "research_methods":
            return instruction_key(
                unique_instructions(genome.get(name, {}).get("instructions", ""))
            )
        return (
            sorted(genome.get(name, {}))
            if name == "allowed_tools"
            else genome.get(name, {})
        )

    return all(comparable(a, name) == comparable(b, name) for name in GENOME_CONTENT)


def changes(parent: Mapping[str, Any], child: Mapping[str, Any]) -> list[Json]:
    """Where a child differs from its parent, one entry per place.

    A sentence added and one removed in the same text count as one place (a
    swap); more than one added or removed count as two. A temperature move
    over 0.1, an output budget move over a tenth, or more than one tool
    toggled also count as two places, so a caller can require subtlety by
    asking for exactly one entry.
    """
    found: list[Json] = []
    for field in ("prompt", "reading_strategy"):
        before = [
            unit.strip() for unit, _ in instruction_units(str(parent.get(field, "")))
        ]
        after = [
            unit.strip() for unit, _ in instruction_units(str(child.get(field, "")))
        ]
        removed = [u for u in before if _key(u) not in {_key(v) for v in after}]
        added = [u for u in after if _key(u) not in {_key(v) for v in before}]
        if not removed and not added:
            continue
        place: Json = {"field": field, "removed": removed, "added": added}
        if len(removed) > 1 or len(added) > 1:
            found.extend([place, {**place, "more": True}])
        else:
            found.append(place)
    old = parent.get("model_settings", {})
    new = child.get("model_settings", {})
    if old.get("temperature") != new.get("temperature"):
        place = {
            "field": "temperature",
            "from": old.get("temperature"),
            "to": new.get("temperature"),
        }
        found.append(place)
        if (
            abs(float(new.get("temperature", 0)) - float(old.get("temperature", 0)))
            > 0.1 + 1e-9
        ):
            found.append({**place, "more": True})
    if old.get("max_output_tokens") != new.get("max_output_tokens"):
        place = {
            "field": "max_output_tokens",
            "from": old.get("max_output_tokens"),
            "to": new.get("max_output_tokens"),
        }
        found.append(place)
        if abs(
            int(new.get("max_output_tokens", 0)) - int(old.get("max_output_tokens", 0))
        ) > (TOKEN_STEP * int(old.get("max_output_tokens", 0)) + 0.5):
            found.append({**place, "more": True})
    toggled = sorted(
        set(parent.get("allowed_tools", [])) ^ set(child.get("allowed_tools", []))
    )
    if toggled:
        place = {"field": "allowed_tools", "toggled": toggled}
        found.append(place)
        if len(toggled) > 1:
            found.append({**place, "more": True})
    if parent.get("research_methods") != child.get("research_methods"):
        found.append({"field": "research_methods"})
    return found


def mutate_genome(
    genome: Mapping[str, Any], existing: Sequence[Mapping[str, Any]], seed: str
) -> tuple[Json, Json] | None:
    """One small change to a genome that no existing agent already is.

    The same seed gives the same child. Returns the child's content and what
    was changed, or ``None`` when every offered change repeats an agent.
    """
    return mate_genomes(genome, None, existing, seed)


def mate_genomes(
    parent: Mapping[str, Any],
    mate: Mapping[str, Any] | None,
    existing: Sequence[Mapping[str, Any]],
    seed: str,
) -> tuple[Json, Json] | None:
    """Change one thing about the parent, taking it from the mate when it can.

    The child is the parent with one sentence of its prompt or strategy
    replaced, added or dropped, one setting nudged a step, or one tool
    toggled; the mate's sentences and tools are tried before the pools, and
    the description says where the change came from (``source`` of ``mate``,
    ``pool`` or ``parent``). The same seed gives the same child; ``None``
    when every offered change repeats an agent already on the island.
    """
    for content, description in _mutations(parent, random.Random(seed), mate):
        if any(_same(content, other) for other in existing):
            continue
        try:
            validate_genome({"id": "child", **content}, "child")
        except Invalid:
            # An offer the validator refuses, such as a sentence that takes a
            # prompt over its limit, is passed over for the next.
            continue
        if mate is not None and description.get("source") == "mate":
            description = {**description, "crossed_with": mate["id"]}
        return content, description
    return None


def _runs_of(db: sqlite3.Connection, genome_id: str) -> int:
    return int(
        db.execute(
            "SELECT COUNT(*) FROM runs WHERE genome_id = ? AND status = 'completed'",
            (genome_id,),
        ).fetchone()[0]
    )


@dataclass(frozen=True)
class Standing:
    """What the island knows about one of its agents when it breeds."""

    runs: int
    points: int
    #: The last generation that bred from it; 0 when it never was a parent.
    last_parent: int
    #: Whether it has completed a threshold's worth of runs: been read.
    evaluated: bool
    #: Its place in the island's list: the older first.
    age: int


def _standings(
    db: sqlite3.Connection,
    island_id: str,
    agents: Sequence[Mapping[str, Any]],
    threshold: int,
) -> dict[str, Standing]:
    last: dict[str, int] = {}
    for number, decisions in db.execute(
        "SELECT number, decisions FROM generations WHERE island_id = ?"
        " AND status = 'committed'",
        (island_id,),
    ):
        for decision in loads(decisions):
            if decision.get("decision") == "parent":
                last[str(decision["genome_id"])] = max(
                    last.get(str(decision["genome_id"]), 0), int(number)
                )
    standings: dict[str, Standing] = {}
    for age, agent in enumerate(agents):
        genome_id = str(agent["id"])
        runs = _runs_of(db, genome_id)
        standings[genome_id] = Standing(
            runs=runs,
            points=points_of(db, genome_id),
            last_parent=last.get(genome_id, 0),
            evaluated=runs >= threshold,
            age=age,
        )
    return standings


def _next_parent(
    active: Sequence[Mapping[str, Any]], standing: Mapping[str, Standing]
) -> Mapping[str, Any]:
    """Whose turn it is: the evaluated agent that has waited longest.

    Among those, the liked come first, then the experienced, then the older.
    When no agent has been evaluated, those with any completed run take the
    turn; when none has, every active agent does.
    """
    evaluated = [g for g in active if standing[str(g["id"])].evaluated]
    read = [g for g in active if standing[str(g["id"])].runs]
    pool = evaluated or read or list(active)
    return min(
        pool,
        key=lambda g: (
            standing[str(g["id"])].last_parent,
            -standing[str(g["id"])].points,
            -standing[str(g["id"])].runs,
            standing[str(g["id"])].age,
        ),
    )


def _since_last(db: sqlite3.Connection, island_id: str) -> tuple[int, int]:
    """Generations so far, and completed runs since the last committed one."""
    last = db.execute(
        "SELECT COUNT(*), COALESCE(MAX(CASE WHEN status = 'committed'"
        " THEN created_at END), '') FROM generations WHERE island_id = ?",
        (island_id,),
    ).fetchone()
    runs = db.execute(
        "SELECT COUNT(*) FROM runs WHERE island_id = ? AND status = 'completed'"
        " AND finished_at > ?",
        (island_id, last[1]),
    ).fetchone()[0]
    return int(last[0]), int(runs)


def _proposal_from_model(
    db: sqlite3.Connection,
    *,
    spec: Mapping[str, Any],
    island: Mapping[str, Any],
    state: BudgetState,
    provider: ModelProvider,
    client: ModelClient,
    generation_id: str,
    now: datetime,
    parent: Mapping[str, Any],
) -> tuple[Json | None, Json]:
    """Ask the model for one change to *parent*. Returns (content or None, what happened)."""
    island_id = str(island["id"])
    settings = parent["model_settings"]
    user = (
        f"Make one new agent for {island['name']} ({island_id}), whose focus is"
        f" {island['focus']}, by changing exactly one thing about {parent['id']}."
        f" Here it is in full.\n\nprompt: {parent['prompt']}\n\nreading_strategy:"
        f" {parent['reading_strategy']}\n\ntemperature: {settings['temperature']}\n"
        f"max_output_tokens: {settings['max_output_tokens']}\nallowed_tools:"
        f" {', '.join(parent['allowed_tools'])}\n\nThe rest of the swarm, to take"
        f" the change from:\n\n{swarm_digest(db, spec, island_id)}"
    )
    estimate = price_micros(provider, estimate_tokens(_SYSTEM + user), PROPOSAL_TOKENS)
    refused = admit_paid(
        state, estimate, state.levers.per_evolution_max_micros, "evolution"
    )
    if refused is not None:
        return None, {"model": "refused", "reason": refused}
    common: dict[str, Any] = {
        "owner_kind": "generation",
        "owner_id": generation_id,
        "parent_kind": "island",
        "parent_id": island_id,
        "island_id": island_id,
        "provider": provider.name,
    }
    try:
        reply = client.complete(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            [PROPOSE_TOOL],
            max_output_tokens=PROPOSAL_TOKENS,
            temperature=0.9,
        )
    except ModelCallFailed as exc:
        record_cost_receipt(
            db,
            action="evolution",
            unit_type="tokens",
            quantity=estimate_tokens(user),
            amount_micros=estimate,
            now=now,
            estimated=True,
            settled=False,
            **common,
        )
        db.commit()
        return None, {"model": "failed", "reason": str(exc)[:200]}
    amount = price_micros(provider, reply.input_tokens, reply.output_tokens)
    record_cost_receipt(
        db,
        action="evolution",
        unit_type="tokens",
        quantity=reply.input_tokens + reply.output_tokens,
        amount_micros=amount,
        now=now,
        estimated=not reply.usage_reported,
        **common,
    )
    db.commit()
    call = next((c for c in reply.tool_calls if c.name == "propose_child"), None)
    if call is None:
        return None, {"model": "no_proposal", "cost_micros": amount}
    arguments = call.arguments
    if arguments is None:
        try:
            arguments = json.loads(call.raw_arguments)
        except (TypeError, ValueError):
            return None, {"model": "malformed_proposal", "cost_micros": amount}
    if not isinstance(arguments, Mapping):
        return None, {"model": "malformed_proposal", "cost_micros": amount}
    known = {g["id"]: g for item in spec["islands"] for g in item["genomes"]}
    parents = [str(parent["id"])] + [
        p
        for p in dict.fromkeys(arguments.get("parents", []))
        if isinstance(p, str) and p in known and p != parent["id"]
    ]
    tools = [
        t
        for t in (arguments.get("allowed_tools") or parent["allowed_tools"])
        if t in TOOL_NAMES
    ]
    if "submit_reading" not in tools:
        tools.append("submit_reading")
    try:
        temperature = float(arguments.get("temperature", settings["temperature"]))
        tokens = int(
            arguments.get("max_output_tokens") or settings["max_output_tokens"]
        )
    except (TypeError, ValueError):
        return None, {"model": "malformed_proposal", "cost_micros": amount}
    content: Json = {
        "prompt": str(arguments.get("prompt") or parent["prompt"]),
        "research_methods": copy.deepcopy(parent.get("research_methods"))
        or island_methods(island),
        "model_settings": {
            "temperature": round(min(1.2, max(0.1, temperature)), 2),
            "max_output_tokens": int(min(2000, max(256, tokens))),
        },
        "allowed_tools": sorted(set(tools)),
        "reading_strategy": str(
            arguments.get("reading_strategy") or parent["reading_strategy"]
        ),
        "scoring_preferences": parent["scoring_preferences"],
    }
    try:
        validate_genome({"id": "child", **content}, "child")
    except Invalid as exc:
        return None, {
            "model": "invalid_proposal",
            "reason": exc.message,
            "cost_micros": amount,
        }
    places = changes(parent, content)
    if len(places) > 1:
        return None, {
            "model": "too_many_changes",
            "reason": ", ".join(dict.fromkeys(str(place["field"]) for place in places)),
            "cost_micros": amount,
        }
    if any(_same(content, g) for g in island["genomes"]):
        return None, {"model": "repeats_an_agent", "cost_micros": amount}
    lemon = arguments.get("archive")
    on_island = {str(g["id"]) for g in island["genomes"] if g["active"]}
    return content, {
        "model": "proposed",
        "change": {**places[0], "source": "model"},
        "parents": parents,
        "why": str(arguments.get("why", ""))[:300],
        "archive": lemon
        if isinstance(lemon, str) and lemon in on_island and lemon not in parents
        else None,
        "why_archive": str(arguments.get("why_archive", ""))[:300],
        "cost_micros": amount,
    }


def _version_of(spec: Mapping[str, Any], genome_id: str) -> int:
    for island in spec["islands"]:
        for genome in island["genomes"]:
            if genome["id"] == genome_id:
                return int(genome["version"])
    return 1


def _unfinished_readers(
    db: sqlite3.Connection, island: Mapping[str, Any], count: int
) -> set[str]:
    protected: set[str] = set()
    fallback = min(count, sum(bool(g["active"]) for g in island["genomes"]))
    papers = db.execute(
        "SELECT a.paper_id FROM assignments a WHERE a.island_id = ? AND a.kept IS NULL"
        " AND NOT EXISTS (SELECT 1 FROM paper_releases r WHERE r.paper_id = a.paper_id)"
        " AND NOT EXISTS (SELECT 1 FROM paper_selections s WHERE s.paper_id = a.paper_id)",
        (island["id"],),
    ).fetchall()
    for paper in papers:
        first = db.execute(
            "SELECT limits FROM runs WHERE paper_id = ? AND island_id = ? ORDER BY rowid LIMIT 1",
            (paper[0], island["id"]),
        ).fetchone()
        size = (
            int(loads(first[0]).get("agents_per_paper", fallback))
            if first
            else fallback
        )
        readers = db.execute(
            "SELECT genome_id FROM runs WHERE paper_id = ? AND island_id = ?"
            " GROUP BY genome_id ORDER BY MIN(rowid) LIMIT ?",
            (paper[0], island["id"], size),
        ).fetchall()
        for reader in readers:
            vote = db.execute(
                "SELECT r.status, (SELECT d.keep FROM readings d WHERE d.run_id = r.id)"
                " FROM runs r WHERE r.paper_id = ? AND r.genome_id = ? ORDER BY r.rowid DESC LIMIT 1",
                (paper[0], reader[0]),
            ).fetchone()
            if vote is None or vote[0] != "completed" or vote[1] is None:
                protected.add(str(reader[0]))
    return protected


def maybe_run_evolution(
    db: sqlite3.Connection,
    island_id: str,
    now: datetime,
    force: bool = False,
    *,
    provider: ModelProvider | None = None,
    client: ModelClient | None = None,
) -> Json | None:
    """Run one evolution cycle for an island when it is due.

    Returns the generation record, committed or skipped with its reason, or
    ``None`` when evolution is off for the island or no threshold is crossed.
    ``force`` runs a cycle now regardless of the threshold, never of the
    switches.
    """
    _, spec = current_spec(db)
    settings = evolution_of(spec)
    island = find_island(spec, island_id)
    if not settings.enabled or not island["evolve"] or island["archived"]:
        return None
    count, runs = _since_last(db, island_id)
    if not (runs >= settings.runs_threshold or force):
        return None

    if not force:
        latest = db.execute(
            "SELECT status, created_at FROM generations WHERE island_id = ?"
            " ORDER BY number DESC LIMIT 1",
            (island_id,),
        ).fetchone()
        if latest is not None and latest[0] == "skipped":
            if latest[1] > iso(now - timedelta(minutes=15)):
                return None

    number = count + 1
    record: Json = {
        "id": new_id("G"),
        "island_id": island_id,
        "number": number,
        "status": "skipped",
        "reason": None,
        "revision": None,
        "decisions": [],
        "cost_micros": 0,
        "created_at": iso(now),
    }

    def close() -> Json:
        db.execute(
            "INSERT INTO generations(id, island_id, number, status, reason, revision,"
            " decisions, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record["id"],
                island_id,
                number,
                record["status"],
                record["reason"],
                record["revision"],
                dumps(record["decisions"]),
                record["created_at"],
            ),
        )
        return record

    def decide(genome_id: str, decision: str, reason: str, **more: Any) -> None:
        record["decisions"].append(
            {"genome_id": genome_id, "decision": decision, "reason": reason, **more}
        )

    active = [g for g in island["genomes"] if g["active"]]
    if not active:
        record["reason"] = "no_active_agent"
        return close()
    standing = _standings(db, island_id, active, settings.runs_threshold)
    parent = _next_parent(active, standing)
    # The mate comes from another island, chosen by the generation's seed so
    # the pairing repeats.
    rng = random.Random(f"{island_id}:{number}")
    elsewhere = [
        (str(other["id"]), g)
        for other in spec["islands"]
        if other["id"] != island_id and not other["archived"]
        for g in other["genomes"]
        if g["active"]
    ]
    mate_island, mate = rng.choice(elsewhere) if elsewhere else (None, None)

    child: Json | None = None
    how: Json = {}
    if island["mutate"]:
        how = {"model": "not_asked"}
        if provider is not None and client is not None:
            state = budget_state(db, spec, now, True)
            child, how = _proposal_from_model(
                db,
                spec=spec,
                island=island,
                state=state,
                provider=provider,
                client=client,
                generation_id=str(record["id"]),
                now=now,
                parent=parent,
            )
            record["cost_micros"] = int(how.get("cost_micros", 0))
        if child is None:
            seed = f"{island_id}:{number}"
            bred = mate_genomes(parent, mate, island["genomes"], seed)
            if bred is None:
                record["reason"] = "no_novel_child"
                record["decisions"].append(
                    {
                        "genome_id": None,
                        "decision": "none",
                        "reason": "no_novel_child",
                        **how,
                    }
                )
                return close()
            child, change = bred
            crossed = [mate["id"]] if mate and change.get("source") == "mate" else []
            how = {**how, "change": change, "parents": [parent["id"], *crossed]}

    if child is not None:
        child["research_methods"] = child.get("research_methods") or island_methods(
            island
        )
        try:
            validate_genome({"id": "child", **child}, "child")
        except Invalid as exc:
            record["reason"] = "invalid_child"
            decide(str(parent["id"]), "kept", "invalid_child", why=exc.message)
            return close()

    parents: list[str] = (
        list(how.get("parents", [parent["id"]])) if child is not None else []
    )
    protected = _unfinished_readers(
        db,
        island,
        levers_from(spec.get("budget", {})).agents_per_paper,
    )
    candidates = [
        g for g in active if g["id"] not in parents and g["id"] not in protected
    ]
    archived: str | None = None
    archive_reason = "population_cap"
    lemon = how.get("archive")
    if isinstance(lemon, str) and any(g["id"] == lemon for g in candidates):
        archived, archive_reason = lemon, "breeder_lemon"
    elif child is not None and len(active) + 1 > settings.max_agents_per_island:
        if not candidates:
            record["reason"] = "pending_reading_cohort"
            return close()
        evaluated = [g for g in candidates if standing[str(g["id"])].evaluated]
        if not evaluated:
            record["reason"] = "population_unevaluated"
            return close()
        # The agent that has had the most chances and earned the fewest likes,
        # the older first among equals.
        victim = min(
            evaluated,
            key=lambda g: (
                standing[str(g["id"])].points,
                -standing[str(g["id"])].runs,
                standing[str(g["id"])].age,
            ),
        )
        archived = str(victim["id"])

    proposed = copy.deepcopy(spec)
    target = find_island(proposed, island_id)
    lineage: dict[str, Json] = {}
    child_id: str | None = None
    if child is not None:
        child_id = f"{island_id}-gen{number}"
        suffix = 1
        taken = {g["id"] for item in proposed["islands"] for g in item["genomes"]}
        while child_id in taken:
            suffix += 1
            child_id = f"{island_id}-gen{number}-{suffix}"
        target["genomes"].append({"id": child_id, "active": True, **child})
        lineage[child_id] = {
            "origin": "mating" if len(parents) > 1 else "mutation",
            "parent": {
                "genome_id": parents[0],
                "version": _version_of(spec, parents[0]),
            },
            "parents": parents,
            "generation": number,
            "mutation": how.get("change"),
            "proposed_by": "model" if how.get("model") == "proposed" else "rule",
            "why": how.get("why"),
        }

    if archived is not None:
        for genome in target["genomes"]:
            if genome["id"] == archived:
                genome["active"] = False

    applied = apply_spec(
        db,
        proposed,
        actor="evolution",
        now=now,
        note=f"generation {number} on {island_id}",
        lineage=lineage,
    )
    for genome in active:
        if genome["id"] == archived:
            decide(
                str(genome["id"]),
                "archived",
                archive_reason,
                why=how.get("why_archive")
                if archive_reason == "breeder_lemon"
                else None,
            )
        elif genome["id"] in parents:
            decide(str(genome["id"]), "parent", "mated")
        else:
            decide(str(genome["id"]), "kept", "no_ranking")
    if mate is not None and mate["id"] in parents:
        decide(str(mate["id"]), "mate", "from_another_island", island_id=mate_island)
    if child_id is not None:
        decide(
            child_id,
            "created",
            "model_mating" if how.get("model") == "proposed" else "rule_mating",
            parents=parents,
            mutation=how.get("change"),
            why=how.get("why"),
            proposal=how.get("model"),
            proposal_reason=how.get("reason"),
        )
    record.update(
        status="committed", revision=applied["revision"] if applied["applied"] else None
    )
    return close()


def build_generation_activity(
    db: sqlite3.Connection, island_id: str, limit: int = 20
) -> list[Json]:
    """An island's recent generations, newest first, with every decision."""
    rows = db.execute(
        "SELECT id, island_id, number, status, reason, revision, decisions, created_at"
        " FROM generations WHERE island_id = ? ORDER BY number DESC LIMIT ?",
        (island_id, limit),
    ).fetchall()
    return [{**dict(row), "decisions": loads(row["decisions"])} for row in rows]
