# SSH Status

Shared dashboard status for other AI tools.

## SSH Health

- Status: `duo_required`
- Message: Please approve Duo for ssh hopper
- Duo required: `True`
- Last SSH success: 2026-03-25 17:52:44 EDT
- Last Duo resolved: --
- Last attempt: 2026-03-25 22:31:04 EDT
- Last error: mux_client_request_session: session request failed: Session open refused by peer
|====================================================================|
| Use of this computer system without authority, or in excess of     |
| granted authority, is prohibited.  This system monitored and       |
| recorded by system personnel.                                      |
|                                                                    |
| Wait for Duo 2FA                                                   |
|====================================================================|

## Reservation Summary

- Running jobs: `0`
- Running GPUs: `0`
- Pending jobs: `6`
- Pending GPUs: `24`

## Running Jobs

- None

## Pending Jobs

- `6616013` | `iter-a100-24h-cq-4` | A100 80GB | 4 GPU(s)
  reservation time: 2026-03-26 15:19:24
- `6616029` | `iter-a100-4g-8h-cq` | A100 80GB | 4 GPU(s)
  reservation time: 2026-03-26 23:30:00
- `6616017` | `iter-a100-8h-cq-4` | A100 80GB | 4 GPU(s)
  reservation time: 2026-03-26 15:19:24
- `6616012` | `iter-a100-24h-gq-4` | A100 80GB | 4 GPU(s)
  reservation time: 2026-03-26 19:38:48
- `6616028` | `iter-a100-4g-8h-gq` | A100 80GB | 4 GPU(s)
  reservation time: 2026-03-27 03:45:00
- `6616016` | `iter-a100-8h-gq-4` | A100 80GB | 4 GPU(s)
  reservation time: 2026-03-26 19:38:48

## Recent Task Records

- `TASK-20260324-203000-hold-gpuq-fallback` | job `6610053` | `Hold gpuq fallback reservation` | 4 GPU(s)
  submitted: 2026-03-24 20:30:00 EDT | description: Keeps the gpuq reservation alive as the fallback while contrib-gpuq is already running.
- `TASK-20260324-180500-train-math-student` | job `6610054` | `Train math student LoRA` | 2 GPU(s)
  submitted: 2026-03-24 18:05:00 EDT | description: Launches a student LoRA training run for the math seed42 teacher outputs.
- `TASK-20260324-191500-eval-math-student` | job `6610054` | `Eval fine-tuned student` | 1 GPU(s)
  submitted: 2026-03-24 19:15:00 EDT | description: Runs a gsm-hard evaluation pass on the latest fine-tuned checkpoint.
