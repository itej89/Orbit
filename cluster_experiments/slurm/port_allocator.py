#!/usr/bin/env python3
"""
Dynamic port allocation for disaggregated vLLM.
Finds available ports by incrementing from base if busy.
"""

import socket
import argparse
import json
import os

DEFAULT_PORTS = {
    'barrier': 5000,
    'etcd_client': 2379,
    'etcd_peer': 2380,
    'vllm': 2584,
    'nixl': 14600,
}

def is_port_available(port, host='0.0.0.0'):
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False

def find_available_port(base_port, increment=10, max_attempts=10):
    """Find an available port starting from base_port."""
    port = base_port
    for _ in range(max_attempts):
        if is_port_available(port):
            return port
        port += increment
    raise RuntimeError(f"No available port found starting from {base_port}")

def find_port_offset(increment=10, max_attempts=10):
    """
    Find a single offset that makes ALL ports available.
    This ensures consistency across all services.
    """
    for attempt in range(max_attempts):
        offset = attempt * increment
        all_available = True
        for name, base_port in DEFAULT_PORTS.items():
            if not is_port_available(base_port + offset):
                all_available = False
                break
        if all_available:
            return offset
    raise RuntimeError("Could not find port offset where all ports are available")

def main():
    parser = argparse.ArgumentParser(description='Dynamic port allocator')
    parser.add_argument('--find-offset', action='store_true',
                        help='Find port offset for all services')
    parser.add_argument('--check-port', type=int,
                        help='Check if specific port is available')
    parser.add_argument('--output-env', action='store_true',
                        help='Output as environment variable exports')
    parser.add_argument('--output-file', type=str,
                        help='Write port config to file')
    parser.add_argument('--increment', type=int, default=10,
                        help='Port increment step (default: 10)')
    
    args = parser.parse_args()
    
    if args.check_port:
        available = is_port_available(args.check_port)
        print(f"Port {args.check_port}: {'available' if available else 'in use'}")
        return 0 if available else 1
    
    if args.find_offset:
        try:
            offset = find_port_offset(increment=args.increment)
            ports = {name: base + offset for name, base in DEFAULT_PORTS.items()}
            
            if args.output_env:
                print(f"export PORT_OFFSET={offset}")
                print(f"export BARRIER_PORT={ports['barrier']}")
                print(f"export ETCD_CLIENT_PORT={ports['etcd_client']}")
                print(f"export ETCD_PEER_PORT={ports['etcd_peer']}")
                print(f"export VLLM_PORT={ports['vllm']}")
                print(f"export NIXL_PORT={ports['nixl']}")
            elif args.output_file:
                with open(args.output_file, 'w') as f:
                    json.dump({'offset': offset, 'ports': ports}, f)
                print(f"Port config written to {args.output_file}")
            else:
                print(f"Port offset: {offset}")
                for name, port in ports.items():
                    print(f"  {name}: {port}")
            
            return 0
        except RuntimeError as e:
            print(f"Error: {e}")
            return 1
    
    parser.print_help()
    return 1

if __name__ == '__main__':
    exit(main())
