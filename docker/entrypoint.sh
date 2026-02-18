#!/bin/bash
# Orbit MPC Docker Entrypoint
#
# Usage:
#   docker run orbit_mpc mpc-controller [args]    - Start MPC controller
#   docker run orbit_mpc router [args]            - Start Orbit router
#   docker run orbit_mpc benchmark [args]         - Run benchmark
#   docker run orbit_mpc vllm [args]              - Start vLLM server (default)
#   docker run orbit_mpc bash                     - Interactive shell

set -e

ORBIT_HOME="${ORBIT_HOME:-/opt/orbit}"

case "$1" in
    mpc-controller|mpc)
        shift
        echo "Starting MPC Controller..."
        exec python "$ORBIT_HOME/mpc_controller.py" "$@"
        ;;
    router)
        shift
        echo "Starting Orbit Router..."
        exec python "$ORBIT_HOME/router.py" "$@"
        ;;
    benchmark|bench)
        shift
        echo "Running Benchmark..."
        exec python "$ORBIT_HOME/benchmark.py" "$@"
        ;;
    prefill-server|prefill)
        shift
        echo "Starting Prefill Server..."
        exec python "$ORBIT_HOME/prefill_server.py" "$@"
        ;;
    decode-server|decode)
        shift
        echo "Starting Decode Server..."
        exec python "$ORBIT_HOME/decode_server.py" "$@"
        ;;
    vllm)
        shift
        echo "Starting vLLM Server..."
        exec python -m vllm.entrypoints.openai.api_server "$@"
        ;;
    bash|sh)
        exec /bin/bash
        ;;
    --help|-h)
        cat << EOF
Orbit MPC Docker Image

Usage:
  docker run orbit_mpc <command> [args]

Commands:
  mpc-controller  Start the MPC weight computation sidecar
  router          Start the Orbit routing proxy
  benchmark       Run benchmark client
  prefill-server  Start prefill server simulation
  decode-server   Start decode server simulation
  vllm            Start vLLM OpenAI-compatible server
  bash            Interactive shell

Examples:
  # Start MPC controller
  docker run -p 8090:8090 orbit_mpc mpc-controller --port 8090

  # Start Orbit router with MPC
  docker run -p 8000:8000 orbit_mpc router \\
    --prefiller-hosts localhost --prefiller-ports 8100,8101 \\
    --decoder-hosts localhost --decoder-ports 8200,8201 \\
    --policy po2 --enable-mpc

  # Start vLLM prefill server
  docker run --device=/dev/kfd --device=/dev/dri \\
    -e HIP_VISIBLE_DEVICES=0,1 \\
    orbit_mpc vllm --model meta-llama/Llama-3.1-70B-Instruct \\
    --tensor-parallel-size 2 --port 8100

  # Run benchmark
  docker run orbit_mpc benchmark \\
    --url http://router:8000/v1/chat/completions \\
    --duration 60 --rps 10

EOF
        exit 0
        ;;
    *)
        # Default: pass all args to vllm
        echo "Starting vLLM Server (default mode)..."
        exec python -m vllm.entrypoints.openai.api_server "$@"
        ;;
esac
