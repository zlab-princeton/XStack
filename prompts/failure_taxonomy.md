# Failure taxonomy

The failure judge gives every graded turn one class from the list below, with
one or two sentences of evidence, as a `turn_analysis` row in the ledger
(`$XSTACK_WORK/ledger.jsonl`). It is a record of why a turn came out the way
it did; it never changes a verdict.

| class | meaning |
|---|---|
| `pass` | grading target reached (a pass also needs the judges) |
| `scope_miss` | a different pipeline than the headline one, or only part of the repository |
| `incomplete_transfer` | the right pipeline trained, the headline evaluation never run |
| `fidelity_gap` | the right pipeline and evaluation, the model short of the grading target |
| `scale_deviation` | the recipe run at a different scale than the source (batch, steps, model size, data ratio) |
| `runtime_crash` | the graded run exited non-zero or hung |
| `no_termination` | the migration has no stop rule; the run reached the cap and wrote no metrics |
| `no_tpu` | the graded run never acquired the TPU (CPU fallback) |
| `wrong_fix` | a later turn that changed something other than the defect |
| `unchanged_tree` | the agent returned without editing the deliverable |
| `grader_artifact` | the grade was the harness's error, not the migration's; the class describes the grader |
| `fabricated_result` | the headline number was not produced by running the pipeline (copied, hard-coded, or evaluated on downloaded finished weights) |
| `freeze_violation` | a phase-2 turn changed something other than the entry file |
| `infra` | machine or harness outage; the turn was not consumed |

A turn may carry a secondary class in `also` (for example `fidelity_gap` with
`also: scale_deviation`). A run that reaches the cap but has written metrics is
graded from them and labelled CAPPED, not `no_termination`.
