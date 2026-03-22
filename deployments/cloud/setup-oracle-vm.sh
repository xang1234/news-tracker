#!/usr/bin/env bash
# =============================================================================
# Oracle Cloud Always Free ARM VM Setup Script
# =============================================================================
# Provisions an Ampere A1 ARM VM (4 OCPUs, 24GB RAM) with Docker and Caddy.
#
# Prerequisites:
#   - Oracle Cloud Always Free account
#   - ARM A1.Flex instance created (Ubuntu 22.04 aarch64)
#   - SSH access configured
#   - Ingress rules open for ports 80, 443, 8001
#
# Usage:
#   ssh ubuntu@<vm-ip> 'bash -s' < setup-oracle-vm.sh
# =============================================================================
set -euo pipefail

echo "=== Updating system packages ==="
sudo apt-get update && sudo apt-get upgrade -y

echo "=== Installing Docker CE (ARM64) ==="
sudo apt-get install -y ca-certificates curl gnupg lsb-release
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "=== Adding user to docker group ==="
sudo usermod -aG docker "$USER"

echo "=== Installing Caddy (reverse proxy + auto HTTPS) ==="
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update
sudo apt-get install -y caddy

echo "=== Configuring iptables (Oracle Cloud requires this) ==="
# Oracle Cloud uses iptables in addition to security lists
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8001 -j ACCEPT
sudo netfilter-persistent save

echo "=== Creating project directory ==="
mkdir -p ~/news-tracker
cd ~/news-tracker

echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Clone your repo:  git clone <repo-url> ~/news-tracker"
echo "  2. Configure env:    cp .env.cloud.example .env && nano .env"
echo "  3. Set up Caddy:     sudo nano /etc/caddy/Caddyfile"
echo "     Example Caddyfile (replace your-domain.com):"
echo "       your-domain.com {"
echo "         reverse_proxy localhost:8001"
echo "       }"
echo "  4. Start Caddy:      sudo systemctl restart caddy"
echo "  5. Init database:    docker compose -f docker-compose.cloud.yml run --rm news-tracker-api init-db"
echo "  6. Start services:   docker compose -f docker-compose.cloud.yml up -d --build"
echo "  7. Check health:     curl http://localhost:8001/health"
echo ""
echo "Useful commands:"
echo "  docker compose -f docker-compose.cloud.yml logs -f        # View logs"
echo "  docker compose -f docker-compose.cloud.yml ps             # Service status"
echo "  docker stats                                               # Resource usage"
echo "  docker compose -f docker-compose.cloud.yml restart <svc>  # Restart service"
