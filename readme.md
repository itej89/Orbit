# Model Predictive Controller Augmentation for Power of 2 and Round Robin Routing Algorithms 
This repository contains verfication code for MPC augmented routing algorithms for VLLM

## Objective
The Goal of this project is to compliment the features of RR and PO2 algorithms with Horizon based controller for Prefill Decode Disaggregated workloads

## Baseline
### Round Robin
RR algorithm besides being simple, ignores load and doesn't provide gaurantees on the stability

By augmenting with MPC, RR becomes a weighted algorithm that helps shifting the load from the using predicted bottlenecks which reduce congesions and oscillations 

### PO2
Ideally under steady workloads PO2 is near optimal. But the traditional LLM worklods violates these constraints because of batching, cache affinity and high decode service times.

MPC compliments these drawbacks through future load prediction, per-node-capacity assignment and filtered states using a simplified system model.

## Model Predictive Controller
### System Model
Below is a simplified system model to estimate the queue length of individual nodes
$$ q_i(k+1)​=q_i(k)+\Delta t(\hat{a​}{p_i}(k)−\mu_i(k))​ $$

Where, for each node i <br>
$q_i(k)$ is the predicted queue length at step k. <br>
$\mu_i(k)$ is the service rate that is continuously estimated <br>
$\hat{a​}$ is the global request arrival rate <br>
$p_i(k)$ is the request arrival rate of node i approximated as $p_i(k)$=$\frac{w_i}{\sum_{j=1}^{n}{w_j}}$ <br>
$\Delta t$ is the controller rate

### Constraints

$q_{k+1}​≥0$ - queue cannot be negative <br> 
$0.3≤w_k​≤1.5$ - actuation range

### Initial condition
$q_0$ - current inflight queue.

### Hyper Parameters
$H$ - Future Queue Legth Horizon <br>
$Target\_q$ -Target queue value that controller should achieve

### Objective function
$\underbrace{(q_{k+1} - Target_q)^2}_{queue\ tracking} + \underbrace{0.5(w_k-1)^2}_{regularization}$
<br><br>Goals:
<br>1.  Keep the predicted queue $q_{k+1}$​ close to $H$
<br>2.  Keep $w_k$ centered and close to 1

### Controller Output Trajectories
$\{w_0​,w_1​,…,w_{H−1}​\}$ - Node Weight Trajectory 
<br>$\{q_1,q_2​,…,q_H​\}$ - Queue Length Trajectory

