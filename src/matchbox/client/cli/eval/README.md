# Evaluation CLI architecture

This README summarises the moving parts of the Textual evaluation tool so future
changes stay coherent.

## Module layout

- `app.py` wires the Textual application. It instantiates `EvaluationState`,
  registers the handlers, and coordinates display refreshes.
- `state.py` is the single source of truth. It owns the evaluation queue,
  user metadata, and the observer list used to broadcast state changes.
- `handlers.py` translates Textual key events into state mutations and API
  calls. All actions are defined here to keep the app layer declarative.
- `widgets/` contains dumb view components that subscribe to state updates.

## Observer pattern

`EvaluationState` exposes `add_listener` and invokes `_notify_listeners()` after
every mutation. Widgets register a callback that simply calls `refresh()`. This
keeps rendering logic out of the state layer while ensuring updates remain
synchronous.

```
state.add_listener(widget.refresh)
state.update_status("✓ Ready", "green")  # triggers refresh
```

When queue mutations occur (submissions, new samples) the state also tracks the
set of seen clusters so repeated API calls do not reintroduce entities.

## Event flow

1. `EntityResolutionApp` boots, authenticates the user, and seeds the queue via
   `matchbox.client.eval.get_samples`.
2. Input events land in `EvaluationHandlers.handle_key_input`, which resolves
   to an action method (navigate, assign, submit, etc.).
3. Actions update state; the observer callbacks repaint widgets.
4. Submissions convert each painted `EvaluationItem` into a judgement, send it
   through `_handler.send_eval_judgement`, and backfill the queue.

Keeping the event flow inside handlers makes it easy to unit-test behaviours
and keeps the Textual layer focussed on display concerns.
