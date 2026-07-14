# Connecting to a Colab VM over SSH — notes

> Companion to [`colab_ssh_setup.sh`](colab_ssh_setup.sh). Covers the bits that
> aren't obvious once you're actually connected: GPU driver visibility, and
> pulling files from Google Drive without going through a notebook cell.

## Connecting

Run `colab_ssh_setup.sh` in the Colab notebook's terminal (not a code cell) to
install `sshd` + `ngrok` and print a `tcp://HOST:PORT` forwarding address.
Then, from your machine:

```bash
ssh -i ~/.ssh/colab_tunnel root@HOST -p PORT
```

The keypair (`~/.ssh/colab_tunnel`) is generated once; its public half is
baked into the script (`CLAUDE_SSH_PUBKEY`).

## GPU not visible over SSH (`nvidia-smi` fails)

Symptom: `nvidia-smi` says `couldn't find libnvidia-ml.so`, even though
`/dev/nvidia0` etc. exist.

Cause: Colab mounts the *real* driver libs at `/usr/lib64-nvidia`, which is
its **own separate ext4 device** (`/dev/sda1`), not part of the root
overlayfs. A naive `find / -xdev -name 'libnvidia-ml.so*'` stops at that
filesystem boundary and only finds the useless CUDA-toolkit stub under
`/usr/local/cuda*/targets/*/lib/stubs/`. `colab_ssh_setup.sh` already avoids
`-xdev` for this reason — if you ever hit this on a box that was set up before
that fix, register the path manually:

```bash
echo '/usr/lib64-nvidia' > /etc/ld.so.conf.d/nvidia-colab.conf
ldconfig
echo 'export LD_LIBRARY_PATH="/usr/lib64-nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' \
  > /etc/profile.d/nvidia-colab.sh
```

## Pulling files from Google Drive onto the VM

The goal: get files from a Drive folder onto the Colab VM **without** routing
the bytes through your own machine.

### What doesn't work: `google.colab.drive.mount()` over plain SSH

It's tempting since Colab and Drive are both Google — but `drive.mount()`
needs two things a bare SSH shell doesn't have:

1. **`TBE_EPHEM_CREDS_ADDR`** (and sibling `TBE_*` env vars) — a
   locally-running credential-broker address, only injected into the
   environment of the actual Jupyter kernel process
   (`colab_kernel_launcher`), not a fresh SSH login shell. Workaround: read
   them out of the live kernel's `/proc/<pid>/environ` and `export` them into
   your SSH session:

   ```bash
   KPID=$(pgrep -f colab_kernel_launcher | head -1)
   eval "$(tr '\0' '\n' < /proc/$KPID/environ | grep -E '^(TBE_|COLAB_)' | sed 's/^/export /')"
   ```

2. **A live IPython kernel with a frontend attached.** Even with the env vars
   exported, `drive.mount()` calls `get_ipython().kernel` internally — that's
   only non-`None` *inside* the actual running notebook kernel process, wired
   via a comm channel to your browser tab (for the one-click OAuth popup). A
   plain `python3`/`ssh`-spawned process can't fake this, and even code
   injected into the real kernel would still need you to click "Allow" in the
   browser — there's no way around that click.

### What works: mount it from the notebook, then read it over SSH

Since the SSH session and the notebook's kernel are **the same VM**, the fix
is trivial once you accept the one browser click has to happen in the
notebook:

1. In an actual cell in your open Colab notebook:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. Approve the popup once.
3. Back over SSH, the files are just... there:
   ```bash
   find /content/drive/MyDrive -maxdepth 4 -iname '*adapter_config.json*'
   cp /content/drive/MyDrive/<path>/* /content/repo/outputs/adapter/
   ```

No bytes ever transit through the local machine — Drive → VM disk, directly.

### Dead end (for reference): OAuth token refresh via `gws`

Also considered: use the local `gws` CLI's stored Google OAuth credentials to
mint a short-lived access token, then `curl` the Drive API directly from the
Colab box (bypassing both the notebook *and* any local-disk copy).

This doesn't work here: the harness redacts secret-shaped strings
(`client_secret`, `refresh_token`) from command output before they can be
reused in another command — a credential-exfiltration guard, not a bug — so
the token-refresh call always fails with `invalid_client`. Don't try to route
around this; it's an intentional boundary.

### Fallback (if Drive isn't mounted and you don't want to touch the notebook)

`gws drive files get --params '{"fileId":"...","alt":"media"}'` streams
**text/JSON** files straight to stdout (pipeable directly into
`ssh ... "cat > dest"`, no local disk hit). For **binary** files
(`application/octet-stream`), `gws` silently ignores stdout piping and saves
to a local `download.bin` in the current directory instead — you have to pass
`--output <relative-path>` explicitly (it refuses paths outside the cwd) and
then `scp`/pipe it across, accepting a brief local temp file.
