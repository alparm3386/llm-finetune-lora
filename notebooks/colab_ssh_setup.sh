#!/bin/bash
# Run in the Colab Linux terminal (not a notebook cell) to enable SSH access via an ngrok tunnel.
# Usage: NGROK_AUTHTOKEN=xxxx bash colab_ssh_setup.sh
set -e

# Default: public half of the ~/.ssh/colab_tunnel keypair (public key, safe to keep in the script).
# Override with CLAUDE_SSH_PUBKEY if you regenerate the keypair.
CLAUDE_SSH_PUBKEY="${CLAUDE_SSH_PUBKEY:-ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIGTaOzyEdomrU0gPeVkB1qY9isvjNUeoM8Ho3dRekwX claude-colab-tunnel}"

if [ -z "$NGROK_AUTHTOKEN" ]; then
    echo "NGROK_AUTHTOKEN is not set."
    read -rp "Paste your ngrok authtoken (from https://dashboard.ngrok.com/get-started/your-authtoken): " NGROK_AUTHTOKEN
fi

echo "==> Registering NVIDIA driver libs for non-interactive shells (SSH logins don't inherit Colab's LD_LIBRARY_PATH)"
nvidia_lib_dir="$(dirname "$(find / -xdev -name 'libnvidia-ml.so*' 2>/dev/null | grep -v '/stubs/' | head -n1)")"
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

echo "==> Installing ngrok"
if ! command -v ngrok &> /dev/null; then
    curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | tee /etc/apt/trusted.gpg.d/ngrok.asc > /dev/null
    echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | tee /etc/apt/sources.list.d/ngrok.list > /dev/null
    apt-get update -qq
    apt-get install -y -qq ngrok
fi
ngrok config add-authtoken "$NGROK_AUTHTOKEN"

echo "==> Starting tunnel (Ctrl+C to stop). Copy the 'Forwarding tcp://...' line for your ssh command."
ngrok tcp 22
