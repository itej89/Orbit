#!/bin/bash
set -x
# Define the full list of cluster IPs
IPADDRS="${IPADDRS:-localhost}"

# Use dynamic ports or defaults
ETCD_CLIENT_PORT="${ETCD_CLIENT_PORT:-2379}"
ETCD_PEER_PORT="${ETCD_PEER_PORT:-2380}"

echo "Starting etcd with ports: client=$ETCD_CLIENT_PORT, peer=$ETCD_PEER_PORT"

# Automatically detect this host's IP (assuming it's the IP on the correct network)
host_ip=$(hostname -I | awk '{print $1}')

# Convert comma-separated IP list into an array
IFS=',' read -ra ADDR <<< "$IPADDRS"

# Determine node name based on position in list
index=0
for ip in "${ADDR[@]}"; do
  if [[ "$ip" == "$host_ip" ]]; then
    break
  fi
  index=$((index + 1))
done
node_name="etcd-$((index+1))"

# Build initial cluster string with dynamic peer port
initial_cluster=""
for i in "${!ADDR[@]}"; do
  peer_name="etcd-$((i+1))"
  initial_cluster+="$peer_name=http://${ADDR[i]}:${ETCD_PEER_PORT}"
  if [[ $i -lt $((${#ADDR[@]} - 1)) ]]; then
    initial_cluster+=","
  fi
done

# Prepare etcd data directory
mkdir -p /var/lib/etcd
rm -rf /var/lib/etcd/*

# Run etcd with dynamic ports
/usr/local/bin/etcd//etcd \
  --name "$node_name" \
  --data-dir /var/lib/etcd \
  --initial-advertise-peer-urls http://$host_ip:${ETCD_PEER_PORT} \
  --listen-peer-urls http://0.0.0.0:${ETCD_PEER_PORT} \
  --listen-client-urls http://0.0.0.0:${ETCD_CLIENT_PORT} \
  --advertise-client-urls http://$host_ip:${ETCD_CLIENT_PORT} \
  --initial-cluster-token etcd-cluster-1 \
  --initial-cluster "$initial_cluster" \
  --initial-cluster-state new \
  2>&1 | tee /run_logs/${SLURM_JOB_ID}/etcd_NODE${NODE_RANK}.log
