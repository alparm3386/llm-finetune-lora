# Connecting to a Colab VM over SSH via Cloudflare Tunnel — notes

> Companion to [`colab_cloudflared_setup.sh`](colab_cloudflared_setup.sh).
> Free alternative to the ngrok-based setup in
> [`COLAB_SSH_NGROK_NOTES.md`](COLAB_SSH_NGROK_NOTES.md). The GPU-driver-path
> fix and Google Drive notes in that file apply here unchanged — this file
> only covers what's different about the tunnel itself.

## Quick Tunnel (no Cloudflare account, no domain)

Run `colab_cloudflared_setup.sh` in the Colab notebook's terminal. It does the
same sshd setup as the ngrok script, then starts:

```bash
cloudflared tunnel --url ssh://localhost:22
```

This prints a random hostname like `https://some-random-words.trycloudflare.com`
— no login, no domain, works immediately.

**Catch**: Quick Tunnels don't give you a raw `tcp://HOST:PORT` you can feed
straight to `ssh -p`. The connection is wrapped by Cloudflare's edge, so the
client also needs `cloudflared` installed locally, used as an SSH
`ProxyCommand` rather than a direct `ssh -p`:

```bash
# once, on your laptop
brew install cloudflared        # or grab a release binary for your OS
```

```
# ~/.ssh/config
Host colab-cloudflare
    HostName some-random-words.trycloudflare.com
    User root
    IdentityFile ~/.ssh/colab_tunnel
    ProxyCommand cloudflared access ssh --hostname %h
```

Then just `ssh colab-cloudflare`.

**The hostname changes every run** — there's no way around this without a
domain (see Named Tunnel below). Each new Colab session, copy the fresh
`*.trycloudflare.com` line the script prints into `HostName` before
connecting.

## Named Tunnel (free Cloudflare account + a domain you control)

Not yet set up — placeholder for when I try this path. Should give a stable
hostname across Colab sessions (no more copy-pasting a new `HostName` every
time), at the cost of one-time setup:

1. Add a domain to Cloudflare (free plan is enough).
2. `cloudflared tunnel create colab-ssh` → get a tunnel ID + credentials file.
3. Ingress config routing `ssh.yourdomain.com` → `ssh://localhost:22`.
4. `cloudflared tunnel run colab-ssh` on the Colab VM (needs the credentials
   file copied over each session, since Colab VMs are ephemeral — or
   regenerate/pull it from Drive).
5. Client side stays the same `ProxyCommand cloudflared access ssh --hostname %h`
   pattern, just pointed at the stable `ssh.yourdomain.com` hostname instead of
   a random `trycloudflare.com` one.

Will fill this section in properly once actually tried against a real domain.
