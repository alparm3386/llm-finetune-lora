# Walkthrough — Step 3.6: running training and eval for real

> A plain-language companion to [`DEV_PLAN.md`](DEV_PLAN.md)'s step 3.6 and
> [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) /
> [`notebooks/eval_colab.ipynb`](notebooks/eval_colab.ipynb). Unlike 3.4 and
> 3.5 (which built *scripts*, unit-tested locally with no GPU), 3.6 has no
> script of its own — its whole job is to actually *execute* `train.py` and
> `evaluate.py` on a real GPU and produce real numbers. Companion to
> [`DEV_PLAN_3.4_WALKTHROUGH.md`](DEV_PLAN_3.4_WALKTHROUGH.md) (training
> internals) and [`DEV_PLAN_3.5_WALKTHROUGH.md`](DEV_PLAN_3.5_WALKTHROUGH.md)
> (eval internals) — this doc is about the *running it*, not the *how it
> works*.

## Part 1: The big picture — what is 3.6, really?

3.4 and 3.5 built two scripts and proved (with unit tests) that all the
non-GPU logic was correct — but neither script had ever actually touched a
GPU. This machine has no CUDA. 3.6 is the step where that stops being
theoretical: rent a free T4 on Google Colab, and run both scripts for real.

It's really **two separate sub-runs**, tracked separately because they depend
on different prerequisites:

| Sub-run | Needs | Produces | Status |
|---|---|---|---|
| **Training** (`train_colab.ipynb`) | base model + synthetic data | a trained LoRA adapter | ✅ done |
| **Eval** (`eval_colab.ipynb`) | the trained adapter + the *real*, hand-labeled eval set | `eval_results.json` / `eval_table.md` (the F1 numbers) | 🚧 in progress — notebook set up, Drive mounted, shakeout/full run not executed yet |

The two are split into separate notebooks (not one) precisely because they
have different lifespans: training happens once and produces a durable
artifact (the adapter); eval is something you'd plausibly want to re-run
later (new eval examples, a re-trained adapter, a different checkpoint)
without re-paying for training every time.

## Part 2: Why a notebook at all, if the real logic lives in `train.py`?

Both notebooks are deliberately **thin wrappers** — looking at
[`train_colab.ipynb`](notebooks/train_colab.ipynb), there are exactly two
lines of actual work (`!python src/train.py --smoke` and
`!python src/train.py --out outputs/adapter`); everything else is Colab
plumbing. This is the same discipline as 3.4/3.5's lazy-import design, one
level up: **all the logic that can be tested without a GPU lives in
version-controlled, unit-tested `.py` files; the notebook is only the
GPU-having, ephemeral-runtime part that can't be.** If a bug shows up, you fix
`train.py` and re-run — you're never debugging logic embedded in notebook
cells that only exist in a Colab session.

## Part 3: `train_colab.ipynb`, cell by cell

1. **Install Unsloth.** `!pip install unsloth` (`%%capture` hides the noisy
   install log).
2. **Clone the repo.** So `src/` is available to `!python src/train.py`.
3. **Upload synthetic training data.** `data/synthetic/` is gitignored (it's
   regenerable LLM output, not something to version-control — see
   `CLAUDE.md`), so it doesn't come with the clone. This cell is a manual
   step: zip `data/synthetic/*.jsonl` locally, upload via Colab's file
   picker, unzip into place.
4. **Smoke test.** `!python src/train.py --smoke --out outputs/smoke` — the
   60-step shakeout from 3.4, run here for the first time on real hardware.
   Confirms the whole pipeline (4-bit load, LoRA attach, data formatting,
   masked training, save) actually works end-to-end on a T4 before spending
   real time on it.
5. **Full run.** `!python src/train.py --out outputs/adapter` — 3 epochs,
   ~428 training examples, 159 steps. This is the run whose loss curve is
   analyzed in detail in
   [`DEV_PLAN_3.4_WALKTHROUGH.md`'s "What the actual 3.6 run showed"](DEV_PLAN_3.4_WALKTHROUGH.md#what-the-actual-36-run-showed)
   section — `eval_loss` bottomed at step 100 (epoch ~1.9) and went flat for
   the rest of epoch 3, a mild/diminishing-returns overfitting signature, not
   a dramatic one.
6. **Download locally.** Zip `outputs/adapter/` and `files.download()` it —
   a safety copy, since a Colab runtime can recycle without warning.
7. **Copy to Google Drive.** `drive.mount()` + `shutil.copytree()` into
   `MyDrive/llm-finetune-lora/adapter` — the durable copy, independent of any
   single Colab session. **This is the adapter now sitting in the shared
   Drive folder** (confirmed via `gws drive files list`):

   ```
   MyDrive/llm-finetune-lora/adapter/
     adapter_config.json         (r=16, alpha=16, LoRA on attention+MLP, text-only)
     adapter_model.safetensors   (~101 MB — the actual trained weights)
     tokenizer.json / tokenizer_config.json / chat_template.jinja
     README.md                   (auto-generated stub from tokenizer.save_pretrained)
     checkpoints/
       checkpoint-140/
       checkpoint-159/
   ```

   One detail worth flagging (already noted in the 3.4 walkthrough, visible
   again here from the live folder listing): only `checkpoint-140` and
   `checkpoint-159` survived — **not `checkpoint-100`**, the actual
   `eval_loss` minimum — because `save_total_limit` only kept the last two.
   It's a low-stakes miss this run (steps 100–159 are all within ~0.002
   `eval_loss` of each other), but the fix for next time
   (`load_best_model_at_end=True` + `metric_for_best_model="eval_loss"`) is
   already written up there.

   `adapter_config.json` also confirms the exact base model the adapter
   expects to sit on top of: `unsloth/gemma-4-e2b-it-unsloth-bnb-4bit` (the
   pre-quantized 4-bit variant) — this is the `base_model_name_or_path`
   pointer that `evaluate.py`'s `load_inference_model(adapter_dir=...)`
   follows automatically (see 3.5 walkthrough, Station 2).

## Part 4: A detour that turned into its own artifact — SSH into Colab

Alongside the notebook, [`colab_ssh_setup.sh`](notebooks/colab_ssh_setup.sh) +
[`COLAB_SSH_NGROK_NOTES.md`](notebooks/COLAB_SSH_NGROK_NOTES.md) exist because
clicking through notebook cells one at a time is slow for iterating/debugging
— being able to run arbitrary shell commands against the live Colab VM (from
a normal terminal, e.g. from within an agent session) is much faster. Running
`colab_ssh_setup.sh` from the notebook's terminal installs `sshd` + `ngrok`
and prints a `tcp://HOST:PORT` address to SSH into directly. There's also a
free, account-less alternative using Cloudflare Tunnel instead of ngrok — see
[`colab_cloudflared_setup.sh`](notebooks/colab_cloudflared_setup.sh) +
[`COLAB_SSH_CLOUDFLARE_NOTES.md`](notebooks/COLAB_SSH_CLOUDFLARE_NOTES.md).

Two non-obvious problems this surfaced, both documented in
`COLAB_SSH_NGROK_NOTES.md` so they don't have to be re-solved next time:

- **The GPU was invisible over SSH** (`nvidia-smi` failed to find
  `libnvidia-ml.so`) because Colab's real driver libs live on a *separate*
  ext4 device (`/usr/lib64-nvidia`) that a naive `find -xdev` search skips
  right past. Fix: register that path in `ld.so.conf.d` explicitly (the setup
  script does this now).
- **Google Drive can't be reached from a bare SSH shell** — `drive.mount()`
  needs credential-broker environment variables and a live IPython kernel
  wired to a browser tab for the OAuth click, neither of which exist in a
  plain SSH login. The workaround that actually works: mount Drive from an
  actual notebook cell (one unavoidable manual click), then read the files
  over SSH afterward — same VM, same disk, no bytes ever routed through the
  local machine.

  A pure-`gws`-token approach (mint a Drive access token locally, `curl` the
  API from the Colab box, skip the notebook entirely) was also tried and
  intentionally doesn't work: this harness redacts secret-shaped strings
  (`client_secret`, `refresh_token`) from command output specifically to
  prevent credentials from being relayed into another command — a guard, not
  a bug, and not worth routing around.

## Part 5: `eval_colab.ipynb` — set up, not yet run

This notebook mirrors the training one's shape, but for `evaluate.py`:

1. **Mount Drive** — ✅ this cell has actually been executed (its output,
   `Mounted at /content/drive`, is saved in the notebook file) — the only
   cell run so far.
2. **Install `unsloth` + `outlines`** — not yet run.
3. **Clone the repo** — not yet run.
4. **Upload the adapter** — as currently written, this cell expects a manual
   `outputs/adapter.zip` upload via `files.upload()`. Now that the adapter
   already lives in the mounted `MyDrive/llm-finetune-lora/adapter`, the
   simpler move next session is to skip this cell and `shutil.copytree()`
   straight from the mounted Drive path instead (same trick as the last two
   cells of `train_colab.ipynb`, in reverse) — no re-zip/re-upload needed.
5. **Upload the real eval set** — likely also skippable now: `data/eval/*.jsonl`
   is git-tracked and populated (56 hand-labeled examples across medical/
   business/technology — see `data/eval/REVIEW.md`), so it should already be
   present after the `git clone` in cell 3.
6. **Shakeout** — `!python src/evaluate.py --adapter outputs/adapter --limit 3 --out results/shakeout`,
   a 3-example sanity run before committing GPU time to the full set. Not yet
   run.
7. **Full evaluation** — `!python src/evaluate.py --adapter outputs/adapter --out results`,
   the real 2×2 (`{base, fine-tuned} × {prompt-only, structured decoding}`)
   over all 56 eval examples. Not yet run.
8. **Download results** — zip and download `results/` (`eval_results.json` +
   `eval_table.md`) before the runtime recycles. Not yet run.

## Part 6: Where 3.6 stands right now

- **Training half: done.** A real adapter exists, trained on real hardware,
  loss curve analyzed, and durably saved to Drive (confirmed live via `gws`).
- **Eval half: started, not finished.** The eval set is fully hand-labeled
  and the notebook is wired up — the only thing that's actually executed so
  far is mounting Drive. The remaining cells (install deps, clone, load
  adapter, shakeout, full run, download) are the concrete next actions for
  the next Colab session. Once `results/eval_table.md` exists, step 3.6 can
  be checked off and 3.7 (Hugging Face Hub upload) can begin.

## The one-sentence summary

**3.6 is where the "assembled but untested on a GPU" scripts from 3.4/3.5
meet a real T4 — training is done (adapter trained, analyzed, and saved to
Drive, with an SSH side-quest solved along the way for faster iteration), and
eval is half-done (set up and Drive-mounted, with the shakeout and full run
against the 56-example hand-labeled set still to execute) before the
headline F1 number exists.**
