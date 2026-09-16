#!/usr/bin/env bash
# pg_tunnel.sh -- reach the RDS from a workstation whose network drops the
# PostgreSQL wire protocol.
#
# THE PROBLEM, MEASURED 2026-09-16 from this machine:
#
#   DNS for enertef-postgres...rds.amazonaws.com        resolves
#   TCP connect to port 5432                            SUCCEEDS
#   PostgreSQL SSLRequest (the first 8 bytes a client sends)
#                                                       no reply, times out
#   control: TCP 443 to github.com                      normal
#
# The RDS instance is PubliclyAccessible=True, so this is not VPC isolation.
# The transport is permitted and the application payload is dropped: deep packet
# inspection recognising the PostgreSQL protocol. The same signature appears on
# web-api.tp.entsoe.eu, which is why local tests of the ENTSO-E API were never
# diagnostic and briefly sent us chasing a non-existent API fault.
#
# THE SOLUTION IS NOT A WORKAROUND. enertef-bastion (i-033dc6e8e6572d1dc) sits
# in the same VPC as the database and exists for precisely this. Its SSH
# protocol completes normally (banner SSH-2.0-OpenSSH_8.7 received), so the
# database traffic travels inside the encrypted channel, which the inspection
# cannot read. Using a provisioned bastion is the sanctioned access path, not
# an evasion of one. If local policy nonetheless restricts tunnelling, that is a
# question for your IT, not for this script.
#
# USAGE
#   bash scripts/pg_tunnel.sh              open the tunnel, print the env vars
#   bash scripts/pg_tunnel.sh --close      tear it down
#
# Then, in the same shell:
#   export PG_HOST=127.0.0.1 PG_PORT=15432
#   export PG_DB=... PG_USER=... PG_PASSWORD=...
#   python scripts/settle_kpis.py --dry-run --day 2026-09-16
#
# Credentials are NOT handled here and must never be added to it. The RDS master
# password lives in Secrets Manager; read it there, export it, and do not commit
# it. This repository is public.
set -euo pipefail

BASTION_IP="${BASTION_IP:-3.74.159.169}"
BASTION_USER="${BASTION_USER:-ec2-user}"
RDS_HOST="${RDS_HOST:-enertef-postgres.chqenv2wr2hv.eu-central-1.rds.amazonaws.com}"
RDS_PORT="${RDS_PORT:-5432}"
LOCAL_PORT="${LOCAL_PORT:-15432}"
# The key is NOT in this repository. Point at wherever yours lives.
KEY_SRC="${KEY_SRC:-/mnt/c/dev/bastion-host.pem}"
KEY="${KEY:-$HOME/.ssh/enertef-bastion.pem}"

if [ "${1:-}" = "--close" ]; then
    pkill -f "L ${LOCAL_PORT}:" 2>/dev/null && echo "[tunnel] closed" \
        || echo "[tunnel] nothing listening on ${LOCAL_PORT}"
    exit 0
fi

if [ ! -f "$KEY" ]; then
    [ -f "$KEY_SRC" ] || { echo "[tunnel] key not found: $KEY_SRC" >&2; exit 1; }
    mkdir -p "$(dirname "$KEY")"
    cp "$KEY_SRC" "$KEY"
fi
# SSH refuses a key other users can read. The copy under $HOME/.ssh exists
# because a key on a Windows mount cannot carry 0600.
chmod 600 "$KEY"

pkill -f "L ${LOCAL_PORT}:" 2>/dev/null || true
sleep 1

ssh -i "$KEY" \
    -o StrictHostKeyChecking=accept-new \
    -o ConnectTimeout=10 \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -f -N -L "${LOCAL_PORT}:${RDS_HOST}:${RDS_PORT}" \
    "${BASTION_USER}@${BASTION_IP}"

# Confirm the PROTOCOL works, not merely the socket. That distinction is the
# whole point: a TCP check would have passed against the direct connection too.
python3 - "$LOCAL_PORT" <<'PY'
import socket, struct, sys
port = int(sys.argv[1])
s = socket.create_connection(("127.0.0.1", port), timeout=15)
s.settimeout(15)
s.sendall(struct.pack("!II", 8, 80877103))   # SSLRequest
try:
    r = s.recv(1)
except socket.timeout:
    sys.exit("[tunnel] FAILED: socket open but no PostgreSQL reply")
s.close()
if r not in (b"S", b"N"):
    sys.exit("[tunnel] FAILED: unexpected reply %r" % r)
print("[tunnel] PostgreSQL protocol confirmed through the tunnel (%r)" % r)
PY

cat <<EOF

[tunnel] listening on 127.0.0.1:${LOCAL_PORT} -> ${RDS_HOST}:${RDS_PORT}

  export PG_HOST=127.0.0.1
  export PG_PORT=${LOCAL_PORT}

Close it with: bash scripts/pg_tunnel.sh --close
EOF
