#!/usr/bin/env bash
# TrueScan — Self-Signed SSL Certificate Generator (Linux / macOS / WSL)
# =======================================================================
# Generates a self-signed TLS cert for local HTTPS development.
# Output: /etc/nginx/ssl/truescan.crt  and  /etc/nginx/ssl/truescan.key
#
# Usage:  chmod +x scripts/generate_ssl.sh && sudo ./scripts/generate_ssl.sh

set -euo pipefail

DOMAIN="${1:-truescan.local}"
SSL_DIR="/etc/nginx/ssl"
DAYS=3650  # 10 years for dev

echo "======================================================"
echo " TrueScan — Generating Self-Signed SSL Certificate"
echo " Domain  : $DOMAIN"
echo " Out dir : $SSL_DIR"
echo " Valid   : $DAYS days"
echo "======================================================"

sudo mkdir -p "$SSL_DIR"

sudo openssl req -x509 \
    -newkey rsa:4096 \
    -keyout "$SSL_DIR/truescan.key" \
    -out    "$SSL_DIR/truescan.crt" \
    -sha256 \
    -days   "$DAYS" \
    -nodes \
    -subj "/C=US/ST=Dev/L=Local/O=TrueScan/CN=$DOMAIN" \
    -addext "subjectAltName=DNS:$DOMAIN,DNS:localhost,IP:127.0.0.1"

sudo chmod 600 "$SSL_DIR/truescan.key"
sudo chmod 644 "$SSL_DIR/truescan.crt"

echo ""
echo "✅ Certificate generated!"
echo "   CRT : $SSL_DIR/truescan.crt"
echo "   KEY : $SSL_DIR/truescan.key"
echo ""
echo "Next steps:"
echo "  1. Add to /etc/hosts:  127.0.0.1  $DOMAIN"
echo "  2. Install nginx config: sudo cp nginx/truescan.conf /etc/nginx/sites-available/truescan"
echo "  3. Enable:  sudo ln -s /etc/nginx/sites-available/truescan /etc/nginx/sites-enabled/"
echo "  4. Test:    sudo nginx -t"
echo "  5. Reload:  sudo systemctl reload nginx"
echo ""
echo "To trust the cert in Chrome: chrome://settings/security → Manage certificates"
echo "   (Linux) sudo cp $SSL_DIR/truescan.crt /usr/local/share/ca-certificates/ && sudo update-ca-certificates"
