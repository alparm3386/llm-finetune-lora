#!/bin/bash
# Run in the Colab Linux terminal (not a notebook cell) to enable SSH access via an ngrok tunnel.
# Usage: NGROK_AUTHTOKEN=xxxx bash colab_ssh_setup.sh
set -e

if [ -z "$NGROK_AUTHTOKEN" ]; then
    echo "NGROK_AUTHTOKEN is not set."
    read -rp "Paste your ngrok authtoken (from https://dashboard.ngrok.com/get-started/your-authtoken): " NGROK_AUTHTOKEN
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
