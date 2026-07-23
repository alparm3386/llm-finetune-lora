#!/bin/bash
# Run in the Colab Linux terminal (not a notebook cell) to enable SSH access via a
# free Cloudflare Quick Tunnel (no Cloudflare account or domain required).
# Usage: bash colab_cloudflared_setup.sh
set -e

# Default: public half of the ~/.ssh/colab_tunnel keypair (public key, safe to keep in the script).
# Override with CLAUDE_SSH_PUBKEY if you regenerate the keypair.
CLAUDE_SSH_PUBKEY="${CLAUDE_SSH_PUBKEY:-ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIGTaOzyEdomrU0gPeVkB1qY9isvjNUeoM8Ho3dRekwX claude-colab-tunnel}"

echo "==> Registering NVIDIA driver libs for non-interactive shells (SSH logins don't inherit Colab's LD_LIBRARY_PATH)"
# No -xdev: Colab mounts the real driver libs at /usr/lib64-nvidia on their own
# ext4 device (separate from the root overlayfs), so -xdev would skip past it
# and only find the useless CUDA-toolkit stub under /usr/local/cuda*/stubs/.
nvidia_lib_dir="$(dirname "$(find / -name 'libnvidia-ml.so*' 2>/dev/null | grep -v '/stubs/' | head -n1)")"
if [ -n "$nvidia_lib_dir" ] && [ "$nvidia_lib_dir" != "." ]; then
    echo "$nvidia_lib_dir" > /etc/ld.so.conf.d/nvidia-colab.conf
    ldconfig
    echo "export LD_LIBRARY_PATH=\"$nvidia_lib_dir\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\"" > /etc/profile.d/nvidia-colab.sh
    echo "    found libnvidia-ml.so in $nvidia_lib_dir, registered via ldconfig + /etc/profile.d"
else
    echo "    WARNING: libnvidia-ml.so not found; nvidia-smi may fail over SSH. Check runtime has a GPU attached."
fi

echo "==> Installing openssh-server"
apt-get update -qq
apt-get install -y -qq openssh-server

echo "==> Generating host keys"
mkdir -p /var/run/sshd
ssh-keygen -A

echo "==> Configuring sshd"
configure_directive() {
    local key="$1" value="$2" file="/etc/ssh/sshd_config"
    if grep -qE "^${key} " "$file"; then
        sed -i "s/^${key} .*/${key} ${value}/" "$file"
    else
        echo "${key} ${value}" >> "$file"
    fi
}
configure_directive "Port" "22"
configure_directive "ListenAddress" "0.0.0.0"
configure_directive "PermitRootLogin" "yes"
configure_directive "PasswordAuthentication" "yes"

echo "==> Set a root password (you'll use it to log in)"
passwd root

echo "==> Authorizing key-based login"
mkdir -p ~/.ssh
grep -qF "$CLAUDE_SSH_PUBKEY" ~/.ssh/authorized_keys 2>/dev/null || echo "$CLAUDE_SSH_PUBKEY" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys

service ssh restart
echo "==> sshd status:"
service ssh status
ss -tlnp | grep :22

echo "==> Installing cloudflared"
if ! command -v cloudflared &> /dev/null; then
    curl -sSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
    chmod +x /usr/local/bin/cloudflared
fi

echo "==> Starting Quick Tunnel in the background (survives this terminal tab closing)"
CF_LOG=/var/log/cloudflared.log
nohup cloudflared tunnel --url ssh://localhost:22 > "$CF_LOG" 2>&1 < /dev/null &
disown
echo "    cloudflared running as PID $!, logging to $CF_LOG"

echo "==> Waiting for the tunnel hostname..."
CF_HOSTNAME=""
for _ in $(seq 1 30); do
    CF_HOSTNAME="$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' "$CF_LOG" | head -n1)"
    [ -n "$CF_HOSTNAME" ] && break
    sleep 1
done

if [ -z "$CF_HOSTNAME" ]; then
    echo "    Tunnel hostname not found yet after 30s — check $CF_LOG manually (tail -f $CF_LOG)."
else
    echo "    $CF_HOSTNAME"
fi
echo ""
echo "    On your laptop, add to ~/.ssh/config (replace the HostName each time you rerun this script):"
echo "        Host colab-cloudflare"
echo "            HostName ${CF_HOSTNAME#https://}"
echo "            User root"
echo "            IdentityFile ~/.ssh/colab_tunnel"
echo "            ProxyCommand cloudflared access ssh --hostname %h"
echo "    Then: ssh colab-cloudflare"
echo ""
echo "==> Tailing cloudflared log below (Ctrl+C only stops viewing — the tunnel itself keeps running)."
echo "    To actually stop the tunnel later: pkill cloudflared"
tail -f "$CF_LOG"
