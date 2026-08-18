# Technical writing

Every sentence written about this model — in `docs/`, in a code comment, in a commit
message — is a **claim with a shelf life**: true when written, checkable, and standing
over code and measurements that will move under it. Write the claim that survives, and
write it so it fails loudly when it stops being true.

Audience: anyone writing prose about this repository. The GitHub rendering rules live
in [AGENTS.md](../AGENTS.md), enforced by `tools/check_markdown.py`.

## A claim, and what it costs to get wrong

Three sentences about the same line:

```
1.  The block weights are clamped.
2.  `_bounded_weights` renormalises the target to unit geometric mean, then clamps
    the ratio between blocks at `weight_max_ratio`.
3.  `_bounded_weights` renormalises the target to unit geometric mean, then clamps
    the ratio between blocks at `weight_max_ratio`. The scheme it bounds sets
    lambda_k = mean(g)/g_k, so a block whose gradient falls as it is fitted earns an
    ever-larger weight — positive feedback with nothing to stop it. Measured over
    three seeds the weights reached 3.1e5 to 6.2e6 on T_f while the void block pinned
    at 0.451, and the run-to-run spread in the T_f error was 10.4x. Only ratios can
    matter — Adam is scale-invariant to a global factor — so renormalising first
    bounds the spread without touching the balance.
```

The first is accurate and useless. The second is accurate and looks like a nicety
someone can simplify away: a clamp reads as defensive programming until you know what
it is defending against. Only the third stops the next reader removing it.

**Write the sentence someone needs before they change your line**, not the sentence
that describes what it currently does.

## Name the owner, the invariant, and the failure

A claim is complete when it says **which symbol** owns the behaviour, **what must stay
true**, and **what breaks otherwise**. Two of the three is where the defects live.

| incomplete | complete |
|---|---|
| "the RAR reservoir is capped" | "`rar_keep` holds the JAX reservoir at a FIXED count so `jit` never sees a new collocation shape; a growing one would recompile the step every `rar_every`" |
| "the loss is compiled" | "`torch.compile(..., dynamic=False)`: automatic dynamic shapes switch on at the second distinct collocation count and make `torch._make_dual` fail on a symbolic size, so forward-mode AD and dynamic shapes cannot be combined in 2.13" |
| "the model has 49797 parameters" | "the fitting capacity is 17029 at every embedding width — 32768 of the 49797 arrays are the Fourier read-out, whose width follows the embedding rather than what the network can represent, so a per-parameter ratio against 49797 does not move when capacity moves" |

Of the three, the failure clause is the one that turns a description into a reason not
to break something.

## Verify, do not recall

**Grep the number.** Every figure in `docs/` must be locatable in a
`__DEV/studies/*.json` row, and the check is a literal search, not a memory of having
seen it. This rule has caught three things and all three were in prose, because the
enforcement was aimed at studies and prose is not a study:

- a draft quoting a *funded* optimiser bake-off at "0.0296 against 0.0401" when that
  bake-off had never been run and neither number was in any study file — `0.0296` was
  the control arm of six unrelated studies;
- a Richardson-extrapolation uncertainty table with no committed command behind it
  (`tools/axial_study.py verify` now produces it);
- a mesh-sensitivity table with none either (recorded as unreproduced, below).

The claims that survive a careful read and fail verification are always the same
shapes: a count that is one off, a real symbol under a wrong name, a case list that
omits a case, a paraphrase that inverts a condition. None is catchable by a tool. All
are catchable by opening the file.

**Read the configuration a row was run at, not the one its filename implies.**
`optbakeoff.json` looks like output of the `bakeoff` sub-command and is not: every row
carries `adam_iters = 3000, lbfgs_iters = 300`, the starved diagonal that `bakeoff`
exists to replace.

## Show the command

"It converges better" is not a claim. A row of `tools/axial_study.py` output is.

Every published table ships with the committed command that produced it —
`tools/axial_study.py` has one sub-command per study for exactly this reason. A number
measured by an uncommitted script is not reproducible however carefully it was
measured, and the case that proves it is D67, where an uncommitted script hid a default
that produced no boiling front at all, for four milestones.

If you cannot produce a command, you are writing hearsay. Cut the sentence.

## Quote the uncertainty beside the number

A bar without its ruler's uncertainty is not a measurement. Calibration practice
(MIL-STD-45662A, ANSI/NCSL Z540) wants a tolerance at least four times above the
uncertainty of the instrument measuring it; here the temperature result sits at 1.06
and the onset ratios at 0.41 to 0.80. Below one you are measuring the ruler, and the
sentence has to say so rather than report the bar.

The same discipline applies to seeds. **Never write a comparative headline from one
seed**: three seeds with per-seed ranges, or the words "seed N, one sample" in the
sentence that states the result — not in a caveat further down, because a hedge below
a confident headline does not work. And the seed count has to be sufficient on the rung
the conclusion turns on, not on the ladder as a whole: near a threshold a rung is
bistable rather than noisy, and one draw from a bistable rung looks exactly like a
converged result.

## Pin only the numbers a command reproduces

Two classes of figure, and only one of them belongs in prose.

**Numbers a committed command reproduces** are the point of `docs/`. Pin them, with the
command, and re-run it when the code under them moves.

**Numbers the code computes about itself** — parameter counts, array shapes, file sizes
— drift with the next commit that touches their subject, and they drift silently. The
49797 above is the case: it was correct arithmetic over the wrong set, it put the
shipped model at 0.48 when it is 1.41, it inverted the conclusion of a collocation
sweep, and it titled a "capacity ladder" that holds capacity fixed. Where such a figure
earns its place, pin it in a test instead — `tests/axial/test_axial_pinn.py` asserts
the body at 17029 and the read-out at `64 * 2 * fourier_features`, in both backends —
and let the prose name the property: *the read-out of the Fourier encoder*, not
*32768 arrays*.

## Describe a gap as a gap

If something is unimplemented, unchecked or unmeasured, **say so, and say what it
costs**. Framing a hole as a decision is what keeps it alive; nobody fixes a design.

`docs/axial_physics.md` §6.5 is the shape to copy:

> re-scoring fixed models against a 640-node reference was reported to drop temperature
> errors ~2.8x and to move onset the other way — **that is not reproduced by any
> committed command**, because nothing here saves a checkpoint yet. Do not quote it.

The same trap one step further: **never rationalise a defect into a convention.** The
early-time collocation cluster survived because the prose around it read as design —
it was described, its measurement was carried in a report as documentation, and the
sampler went on drawing it. A sentence that makes a strange thing sound intended is
load-bearing for the next reader who might otherwise have fixed it.

## State the limit

A description that omits its own boundary invites over-trust. Say what the thing does
**not** cover, as a property of the thing:

> The long runs are JAX-only, so a comparative headline is a statement about the JAX
> implementation, not about the formulation in general.

That is a limit, and it belongs in the sentence that states the result. "This section
does not cover torch" is not a limit — it is a note about the prose.

An ablation carries the same clause: it is a statement about the formulation it was run
on. Change the formulation and every negative result on the shelf is provisional again.

## What not to write

**No history in `docs/`.** "Used to be", "previously", "this was fixed in" — out of date
the day after, and it tells a reader nothing about the code in front of them. The
before and after belong in the commit message, and the working record belongs in
`__DEV/`.

**No meta.** Prose does not describe itself. No "this page explains", no section listing
what the page does not cover, no summary restating the section above it.

**No padding.** Length is not thoroughness; it is where rot hides. Cut anything that
does not help a reader build, train or verify.

**No images, ever.** Not in `docs/`, not anywhere in the repository. A committed plot is
a number that stops tracking the code that produced it, and a figure cannot be grepped
against a study row the way the rules above require. Draw into the git-ignored
`results/` and look at it locally.

**Pair every prohibition with an alternative.** "Do not use `str(annotation)`" leaves a
reader stuck; "do not use `str(annotation)`, use `typing.get_origin`, because PEP 649
hands back the type object" does not.

**One example beats three paragraphs.**

## The three surfaces

A page, a code comment and a commit message answer to the same rules. They differ in
shelf life.

### Pages

`docs/` is technical: it explains **how to build and train this PINN** — the ansatz, the
hard constraints, the residuals, the schedule, the knobs and what each is for. It is not
the paper in Markdown, and there is no `paper/`. Short tables of convergence metrics
against the reference belong here, each with the command that produced it. Everything
that is a working record — the studies, the retractions, the milestone log — lives in
`__DEV/`.

A deviation from the SAS4A manual is a **contract, not a comment**: it goes in the
`docs/axial_physics.md` register with its equation number. An unregistered deviation is
a bug.

**Change the code, re-read its page in the same commit.** A page is wrong from the
moment the code lands, and nobody knows which claim broke better than the person who
broke it.

### Code comments

**Write only the constraint the code cannot show.** Never restate the next line. If the
line reads plainly, say nothing. Docstrings follow the numpy convention and open in the
imperative — "Return the array module", not "Returns" and not "This function returns".

A comment that carries a measurement is worth more than one that carries an intention,
because the measurement is checkable:

```python
# `foreach=True` explicitly. PyTorch's auto-selection does NOT enable it on CPU --
# `_default_to_fused_or_foreach` returns `(False, False)` there, so leaving it unset
# silently takes the single-tensor for-loop path on the only device this project
# runs on. Measured on this model: 1.50x on the optimiser step alone and 1.061x
# end-to-end.
```

And the one failure mode specific to a two-backend repository: **a contract the code
does not enforce is a comment.** The torch config declared `first_order="ademamix"` a
JAX-only arm, in prose, while the torch loop ran plain Adam under that label. The
declaration was correct and it protected nothing. Raise, or implement it — that one was
implemented, once it turned out a maintained PyTorch AdEMAMix existed outside
`torch.optim`, which the comment had also asserted was not the case.

### Commit messages

The one surface where history is the **subject** rather than the contamination. A commit
message may say what the code used to do, because that is what a commit is.

Subject line is `area: imperative summary`, lower case, no trailing period. The body
says what changed, what it was measured at, and what would have caught it going wrong:

```
training: fuse the first-order loop -- draw and update in one compiled region

The Python `for` loop made three dispatch round-trips per iteration and the cores
idled through the Python between them. Measured on the study configuration: 4.19 ms
drawing against 22.76 ms in the fused step, so 15.6% of wall time sat outside the
compiled region.

<what changed>

Measured after: 38.3 it/s on 8 cores, against 33 it/s unfused on 12.

The correctness risk is that fusing quietly reschedules an event, so that is what the
tests pin. tests/axial/test_fused_loop.py reconstructs the set of iterations at which
Python intervenes and compares it against the cadence, including for odd non-dividing
periods (300 and 70) where a naive fixed block size goes wrong.

Verified: 441 passed, pre-commit green, `backend_smoke.py` passes, ruff and ty clean.
```

**No attribution trailers.** No `Co-Authored-By` for tooling, no "generated with". The
authorship of a commit is its author field; a trailer naming a tool records which
keyboard the text came through, and that is not something a later reader can act on.

Three parts earn their place: **the measurement before**, **the measurement after**, and
**the risk the change carries with what pins it**. A commit that changes a number states
which published numbers move.

Do not defend the change against a baseline that was itself wrong. The loop above is not
compared to its predecessor's output, because its predecessor drew a sampling cluster
that has since been retired as indefensible — reproducing it would be a defect, not a
guarantee.

Commit only when asked, and keep commits logical: one argument each.

## What the gate checks, and what it cannot

```sh
uv run python tools/check_markdown.py     # every tracked .md; also a pre-commit hook
uv run pre-commit run --all-files
```

The scan fails on the inline-math forms GitHub renders wrongly and on dead relative
links. It strips inline code spans first, so a page may quote the broken forms
deliberately — [AGENTS.md](../AGENTS.md) is full of them, and so is this sentence.

**It cannot tell you a sentence is false.** Prose can parse, link, name only real paths,
and still describe code that has moved or a measurement nobody took. The gate buys the
mechanical half. The other half is bought by opening the file and grepping the number.
