## Model Predictive Controller Augmented for Power-of-2 Algorithm
This repository contains verfication code for MPC augmented routing algorithms for VLLM

### Objective
The Goal of this project is to compliment the features of RR and PO2 algorithms with Horizon based controller for Prefill Decode Disaggregated workloads

### Baseline
#### Round Robin
RR algorithm besides being simple, ignores load and doesn't provide gaurantees on the stability

By augmenting with MPC, RR becomes a weighted algorithm that helps shifting the load from the using predicted bottlenecks which reduce congesions and oscillations 

#### PO2
Ideally under steady workloads PO2 is near optimal. But the traditional LLM worklods violates these constraints because of batching, cache affinity and high decode service times.

MPC compliments these drawbacks through future load prediction, per-node-capacity assignment and filtered states using a simplified system model.

### Model Predictive Controller
#### System Model
Below is a simplified system model to estimate the queue length of individual nodes
$$ q_i(k+1)​=q_i(k)+\Delta t(\hat{a​}{p_i}(k)−\mu_i(k))​ $$

Where, for each node i <br>
$q_i(k)$ is the predicted queue length at step k. <br>
$\mu_i(k)$ is the service rate that is continuously estimated <br>
$\hat{a​}$ is the global request arrival rate <br>
$p_i(k)$ is the request arrival rate of node i approximated as $p_i(k)$=$\frac{w_i}{\sum_{j=1}^{n}{w_j}}$ <br>
$\Delta t$ is the controller rate

#### Constraints

$q_{k+1}​≥0$ - queue cannot be negative <br> 
$0.1≤w_k​≤3.0$ - actuation range

#### Initial condition
$q_0$ - current inflight queue.

#### Hyper Parameters
$H$ - Future Queue Legth Horizon <br>
$Target\_q$ -Target queue value that controller should achieve

#### Objective function
$\underbrace{(q_{k+1} - Target_q)^2}_{queue\ tracking} + \underbrace{0.5(w_k-1)^2}_{regularization}$
<br><br>Goals:
<br>1.  Keep the predicted queue $q_{k+1}$​ close to $H$
<br>2.  Keep $w_k$ centered and close to 1

### Controller Output Trajectories
$\{w_0​,w_1​,…,w_{H−1}​\}$ - Node Weight Trajectory 
<br>$\{q_1,q_2​,…,q_H​\}$ - Queue Length Trajectory

### Controller Stability on a simulation of 2P2D with variable decoder server delays
Following graph shows how the controller maintains stable throughput by load balancing across servers for heterogenous workloads. In case of LLMs the heterogenity arrives from the variable promp lengths and generation legths, GPU capacity, KV Cache distribution etc.,.
##### Weighted PO2 with MCP
<img width="2560" height="1332" alt="MPCRouter" src="https://github.com/user-attachments/assets/1e908e82-cf34-4872-936d-042017084e3b" />

##### PO2 without MCP
<img width="2560" height="1332" alt="PO2" src="https://github.com/user-attachments/assets/d69cd047-193a-4bfa-95ad-784c06ff7aef" />


### Instruciton to perform the simulation
```
# Create a container
docker run --rm -it   --network=host   --group-add video   --ipc=host   --cap-add=SYS_PTRACE   --privileged=true   --shm-size=64GB   --device=/dev/kfd   --device=/dev/dri   --cap-add=IPC_LOCK   --ulimit memlock=-1   --device=/dev/infiniband   --device=/dev/infiniband/rdma_cm   -v /sys/class/infiniband:/sys/class/infiniband:ro   -v /sys/class/infiniband_verbs:/sys/class/infiniband_verbs:ro   --security-opt seccomp=unconfined   -v $HOME:$HOME   --name 20251217_rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0_rc1   rocm/7.0:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0_rc1 /bin/bash


# Start all nodes
/Orbit/simulation# python ./prefill_server.py  --port 8100 --delay 0.005
/Orbit/simulation# python prefill_server.py    --port 8101 --delay 0.03
/Orbit/simulation# python ./decode_server.py   --port 8200 --token-delay 0.01
/Orbit/simulation# python ./decode_server.py   --port 8201 --token-delay 0.05

# Start the Router
/orbit# python ./router.py  --prefiller-hosts  127.0.0.1  127.0.0.1  --prefiller-ports 8100 8101   --decoder-hosts 127.0.0.1  127.0.0.1   --decoder-ports 8200 8201

# Benchmark and read the results
/Orbit/simulation# python benchmark.py
```


