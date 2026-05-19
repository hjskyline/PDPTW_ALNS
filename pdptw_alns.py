# 让类型提示可以引用当前文件后面才定义的类
from __future__ import annotations
# argparse 用于从命令行读取参数，例如 iterations、seed、filepath 等
import argparse
import copy
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
# EPS 是一个很小的数，用来处理浮点数误差。
EPS = 1e-9
# INF 表示一个非常大的数，通常用于表示“不可行”或者“无限大成本”。
INF = 10**18


# ============================================================
# 0. Data structures#
# 这一部分定义了求解 PDPTW 所需要的基本数据结构
# 没有直接使用很多零散变量，而是把相关信息封装成类
# ============================================================
@dataclass
class Node:
    node_id: int
    x: float
    y: float
    demand: float
    ready: float
    due: float
    service: float
    pickup: int
    delivery: int


@dataclass
class Request:
    """One pickup-delivery request."""
    request_id: int       # we use the pickup node id as the request id
    pickup: int
    delivery: int
    demand: float


@dataclass
class PDPTWInstance:
    vehicle_count: int
    capacity: float
    speed: float
    nodes: Dict[int, Node]
    requests: Dict[int, Request]
    depot_id: int = 0
    distance: List[List[float]] = field(default_factory=list)
    travel_time: List[List[float]] = field(default_factory=list)
    max_distance: float = 0.0
    max_demand: float = 1.0
    max_due: float = 1.0


@dataclass
class RouteStats:
    feasible: bool
    distance: float
    duration: float
    end_time: float
    service_start: Dict[int, float]
    issues: List[str]


@dataclass
class Solution:
    routes: List[List[int]]
    bank: Set[int]

    def copy(self) -> "Solution":
        return Solution(routes=[r[:] for r in self.routes], bank=set(self.bank))


@dataclass
class SolutionStats:
    objective: float
    total_distance: float
    total_duration: float
    vehicles_used: int
    unserved: int
    feasible: bool
    route_stats: List[RouteStats]
    issues: List[str]


@dataclass
class ALNSParams:
    # Objective weights in the paper: distance, vehicle time/duration, request-bank penalty.
    alpha: float = 1.0
    beta: float = 0.0
    gamma: float = 100000.0

    # Optional extra penalty for experiments where vehicle count is first priority.
    # Keep this at 0.0 to match the paper's weighted objective directly.
    vehicle_penalty: float = 0.0

    # Removal size q: paper uses a random number in [4, min(100, phi*n)].
    q_min: int = 4
    q_fraction: float = 0.4
    q_abs_max: int = 100

    # Adaptive weights.
    segment_length: int = 100
    reaction_factor: float = 0.10
    min_weight: float = 0.05
    sigma1: float = 33.0   # new global best
    sigma2: float = 9.0    # new accepted improving solution
    sigma3: float = 13.0   # new accepted worsening solution

    # Simulated annealing.
    start_worse_fraction: float = 0.05  # start T: 5% worse accepted with prob 0.5
    cooling: float = 0.99975

    # Shaw relatedness parameters.
    shaw_distance_weight: float = 9.0
    shaw_time_weight: float = 3.0
    shaw_demand_weight: float = 2.0
    shaw_vehicle_weight: float = 0.0  # Li & Lim fleet is homogeneous, so this term is 0.
    shaw_random_degree: float = 6.0
    worst_random_degree: float = 3.0

    # Noise added to insertion costs.
    noise_rate: float = 0.025

    # Insertion heuristics included in the adaptive roulette wheel.
    insertion_names: Tuple[str, ...] = ("greedy", "regret2", "regret3", "regretm")


@dataclass
class InsertionMove:
    request_id: int
    route_index: int
    pickup_pos: int
    delivery_pos: int
    true_delta: float
    selection_cost: float


# ============================================================
# 1. Read Li & Lim PDPTW data
# ============================================================
def read_li_lim_pdptw(filepath: str) -> PDPTWInstance:
    """
    Read Li & Lim PDPTW instance.

    Format:
        vehicle_count capacity speed
        node_id x y demand ready due service pickup delivery

    In this format:
    - pickup node:   demand > 0, pickup = 0, delivery = paired delivery node
    - delivery node: demand < 0, pickup = paired pickup node, delivery = 0
    - depot:         node 0, pickup = delivery = 0
    """
    # 读取所有非空行
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError("Input file is empty.")
    # 第一行：车辆数、容量、速度
    first = lines[0].split()
    if len(first) < 3:
        raise ValueError("First line must contain: vehicle_count capacity speed")

    vehicle_count = int(float(first[0]))
    capacity = float(first[1])
    speed = float(first[2])
    if speed <= 0:
        raise ValueError("Speed must be positive.")
    # 读取节点信息
    nodes: Dict[int, Node] = {}
    for raw in lines[1:]:
        parts = raw.split()
        if len(parts) < 9:
            raise ValueError(
                "Li & Lim PDPTW node line must have 9 columns: "
                "id x y demand ready due service pickup delivery. "
                f"Bad line: {raw}"
            )
        node_id = int(float(parts[0]))
        nodes[node_id] = Node(
            node_id=node_id,
            x=float(parts[1]),
            y=float(parts[2]),
            demand=float(parts[3]),
            ready=float(parts[4]),
            due=float(parts[5]),
            service=float(parts[6]),
            pickup=int(float(parts[7])),
            delivery=int(float(parts[8])),
        )

    if 0 not in nodes:
        raise ValueError("Depot node 0 is missing.")

    # 根据 pickup-delivery 配对信息构造 requests
    requests: Dict[int, Request] = {}
    for node in nodes.values():
        if node.node_id == 0:
            continue
        # delivery != 0 表示当前节点是 pickup node
        if node.delivery != 0:
            pickup_id = node.node_id
            delivery_id = node.delivery
            if delivery_id not in nodes:
                raise ValueError(f"Pickup node {pickup_id} points to missing delivery node {delivery_id}.")
            delivery_node = nodes[delivery_id]
            # 检查 delivery node 是否反向指回该 pickup
            if delivery_node.pickup != pickup_id:
                raise ValueError(
                    f"Inconsistent pair: pickup {pickup_id} -> delivery {delivery_id}, "
                    f"but delivery points back to {delivery_node.pickup}."
                )
            # pickup demand 应为正，delivery demand 应为对应负数
            if node.demand <= 0:
                raise ValueError(f"Pickup node {pickup_id} should have positive demand.")
            if abs(node.demand + delivery_node.demand) > 1e-6:
                raise ValueError(
                    f"Pickup/delivery demand mismatch for request {pickup_id}: "
                    f"pickup demand {node.demand}, delivery demand {delivery_node.demand}."
                )
            requests[pickup_id] = Request(
                request_id=pickup_id,
                pickup=pickup_id,
                delivery=delivery_id,
                demand=node.demand,
            )

    if not requests:
        raise ValueError("No pickup-delivery requests found. This does not look like Li & Lim PDPTW data.")
    # 构造距离矩阵和行驶时间矩阵
    max_id = max(nodes)
    distance = [[0.0 for _ in range(max_id + 1)] for _ in range(max_id + 1)]
    travel_time = [[0.0 for _ in range(max_id + 1)] for _ in range(max_id + 1)]
    max_distance = 0.0

    for i, ni in nodes.items():
        for j, nj in nodes.items():
            d = math.hypot(ni.x - nj.x, ni.y - nj.y)
            distance[i][j] = d
            travel_time[i][j] = d / speed
            if d > max_distance:
                max_distance = d
    # Shaw removal 中需要用到这些最大值做归一化
    max_demand = max(abs(n.demand) for n in nodes.values()) or 1.0
    max_due = max(n.due for n in nodes.values()) or 1.0

    return PDPTWInstance(
        vehicle_count=vehicle_count,
        capacity=capacity,
        speed=speed,
        nodes=nodes,
        requests=requests,
        depot_id=0,
        distance=distance,
        travel_time=travel_time,
        max_distance=max_distance,
        max_demand=max_demand,
        max_due=max_due,
    )


# ============================================================
# 2.  可行性检查与目标函数
# ============================================================
#判断一个节点是否为 pickup node
def is_pickup_node(inst: PDPTWInstance, node_id: int) -> bool:
    return node_id != inst.depot_id and inst.nodes[node_id].delivery != 0

#判断一个节点是否为 delivery node
def is_delivery_node(inst: PDPTWInstance, node_id: int) -> bool:
    return node_id != inst.depot_id and inst.nodes[node_id].pickup != 0 and inst.nodes[node_id].delivery == 0

#返回某个节点所属的 request_id
def request_of_node(inst: PDPTWInstance, node_id: int) -> Optional[int]:
    if node_id == inst.depot_id:
        return None
    node = inst.nodes[node_id]
    # pickup node 本身就是 request_id
    if node.delivery != 0:
        return node.node_id
    # pickup node 本身就是 request_id
    if node.pickup != 0:
        return node.pickup
    return None


def route_weighted_cost(route: List[int], inst: PDPTWInstance, params: ALNSParams) -> float:
    st = evaluate_route(route, inst)
    if not st.feasible:
        return INF
    return params.alpha * st.distance + params.beta * st.duration


def evaluate_route(route: List[int], inst: PDPTWInstance) -> RouteStats:

    depot = inst.nodes[inst.depot_id]
    # 车辆从 depot 出发，初始时间为 depot 的 ready time
    current_time = depot.ready
    prev = inst.depot_id
    load = 0.0
    # 记录路线总距离和各节点服务开始时间
    distance = 0.0
    service_start: Dict[int, float] = {}
    # 记录已经访问过的 pickup，用于检查 pickup-before-delivery
    seen_pickups: Set[int] = set()
    seen_deliveries: Set[int] = set()
    # 如果路线不可行，把原因记录在 issues 中
    issues: List[str] = []

    # 没有重复的
    if len(set(route)) != len(route):
        issues.append("repeated node in route")

    for node_id in route:
        if node_id == inst.depot_id:
            issues.append("depot should not appear inside a route")
            continue
        if node_id not in inst.nodes:
            issues.append(f"unknown node {node_id}")
            continue

        node = inst.nodes[node_id]
        # 计算从上一个节点到当前节点的距离和到达时间
        distance += inst.distance[prev][node_id]
        arrival = current_time + inst.travel_time[prev][node_id]
        # 如果早到，需要等待到 ready time 才能开始服务
        start_service = max(arrival, node.ready)
        service_start[node_id] = start_service
        # 检查是否超过 due time
        if start_service > node.due + EPS:
            issues.append(
                f"time window violation at node {node_id}: "
                f"start={start_service:.3f}, due={node.due:.3f}"
            )

        # 如果是 delivery node，必须保证对应 pickup 已经出现过
        if is_delivery_node(inst, node_id):
            pickup_id = node.pickup
            if pickup_id not in seen_pickups:
                issues.append(f"precedence violation: delivery {node_id} before pickup {pickup_id}")
            seen_deliveries.add(node_id)

        # 更新车辆载重
        load += node.demand
        if load > inst.capacity + EPS:
            issues.append(
                f"capacity violation after node {node_id}: load={load:.3f}, capacity={inst.capacity:.3f}"
            )
        if load < -EPS:
            issues.append(f"negative load after node {node_id}: load={load:.3f}")

        if is_pickup_node(inst, node_id):
            seen_pickups.add(node_id)
        # 服务完成后，车辆才能离开当前节点
        current_time = start_service + node.service
        prev = node_id

    # 检查每个 request 的 pickup 和 delivery 是否都在同一条 route 中
    route_set = set(route)
    for node_id in route:
        node = inst.nodes[node_id]
        if is_pickup_node(inst, node_id) and node.delivery not in route_set:
            issues.append(f"same-route violation: pickup {node_id} without delivery {node.delivery}")
        if is_delivery_node(inst, node_id) and node.pickup not in route_set:
            issues.append(f"same-route violation: delivery {node_id} without pickup {node.pickup}")

    # 最后车辆需要从最后一个节点返回 depot
    distance += inst.distance[prev][inst.depot_id]
    depot_arrival = current_time + inst.travel_time[prev][inst.depot_id]
    if depot_arrival > depot.due + EPS:
        issues.append(
            f"depot due-time violation: return={depot_arrival:.3f}, due={depot.due:.3f}"
        )

    if abs(load) > EPS:
        issues.append(f"route ends with nonzero load {load:.3f}")

    duration = depot_arrival - depot.ready
    feasible = len(issues) == 0
    return RouteStats(
        feasible=feasible,
        distance=distance,
        duration=duration,
        end_time=depot_arrival,
        service_start=service_start,
        issues=issues,
    )


def planned_requests(solution: Solution, inst: PDPTWInstance) -> Set[int]:
    planned: Set[int] = set()
    for route in solution.routes:
        for node_id in route:
            req_id = request_of_node(inst, node_id)
            if req_id is not None:
                planned.add(req_id)
    return planned

#评估整个 solution 的可行性和目标函数值
def evaluate_solution(solution: Solution, inst: PDPTWInstance, params: ALNSParams) -> SolutionStats:
    route_stats = [evaluate_route(route, inst) for route in solution.routes]
    total_distance = sum(rs.distance for rs in route_stats)
    total_duration = sum(rs.duration for route, rs in zip(solution.routes, route_stats) if route)
    vehicles_used = sum(1 for route in solution.routes if route)
    issues: List[str] = []

    for idx, rs in enumerate(route_stats):
        if not rs.feasible:
            issues.extend([f"route {idx}: {msg}" for msg in rs.issues])

    # 统计每个节点在 solution 中出现了几次
    node_count: Dict[int, int] = {}
    for route in solution.routes:
        for node_id in route:
            node_count[node_id] = node_count.get(node_id, 0) + 1
    # 检查每个 request：
    # 要么 pickup 和 delivery 都在路线中；
    # 要么整个 request 在 bank 中。
    for req_id, req in inst.requests.items():
        p_count = node_count.get(req.pickup, 0)
        d_count = node_count.get(req.delivery, 0)
        in_bank = req_id in solution.bank

        if p_count == 1 and d_count == 1 and not in_bank:
            continue
        if p_count == 0 and d_count == 0 and in_bank:
            continue
        issues.append(
            f"request coverage problem for {req_id}: "
            f"pickup_count={p_count}, delivery_count={d_count}, in_bank={in_bank}"
        )
    # request bank 中未服务 request 数量
    unserved = len(solution.bank)
    infeasible_penalty = 0.0 if not issues else INF / 10
    objective = (
        params.alpha * total_distance
        + params.beta * total_duration
        + params.gamma * unserved
        + params.vehicle_penalty * vehicles_used
        + infeasible_penalty
    )
    feasible = len(issues) == 0

    return SolutionStats(
        objective=objective,
        total_distance=total_distance,
        total_duration=total_duration,
        vehicles_used=vehicles_used,
        unserved=unserved,
        feasible=feasible,
        route_stats=route_stats,
        issues=issues,
    )


# ============================================================
# 3. 插入算子 Insertion heuristics
# ============================================================
def apply_insertion_move(solution: Solution, inst: PDPTWInstance, move: InsertionMove) -> None:
    req = inst.requests[move.request_id]
    route = solution.routes[move.route_index]
    # 先插入 pickup node
    route_after_pickup = route[:move.pickup_pos] + [req.pickup] + route[move.pickup_pos:]
    # 再插入 delivery node
    # delivery_pos 是基于插入 pickup 后的新 route 位置
    route_after_delivery = (
        route_after_pickup[:move.delivery_pos]
        + [req.delivery]
        + route_after_pickup[move.delivery_pos:]
    )
    # 更新该车辆路线
    solution.routes[move.route_index] = route_after_delivery
    # 该 request 已经被服务，从 request bank 中移除
    solution.bank.discard(move.request_id)

# 寻找某个 request 插入某一条 route 的最佳位置
def best_insertion_for_request_in_route(
    solution: Solution,
    inst: PDPTWInstance,
    params: ALNSParams,
    req_id: int,
    route_index: int,
    use_noise: bool,
) -> Optional[InsertionMove]:
    req = inst.requests[req_id]
    route = solution.routes[route_index]
    # 插入前该 route 的成本
    old_cost = route_weighted_cost(route, inst, params)
    if old_cost >= INF:
        return None

    best_move: Optional[InsertionMove] = None
    max_noise = params.noise_rate * inst.max_distance

    # 枚举 pickup 的插入位置
    for p_pos in range(len(route) + 1):
        with_pickup = route[:p_pos] + [req.pickup] + route[p_pos:]

        for d_pos in range(p_pos + 1, len(with_pickup) + 1):
            candidate_route = with_pickup[:d_pos] + [req.delivery] + with_pickup[d_pos:]
            new_stats = evaluate_route(candidate_route, inst)
            if not new_stats.feasible:
                continue
            new_cost = params.alpha * new_stats.distance + params.beta * new_stats.duration
            true_delta = new_cost - old_cost
            selection_cost = true_delta
            if use_noise:
                noise = random.uniform(-max_noise, max_noise)
                selection_cost = max(0.0, true_delta + noise)

            move = InsertionMove(
                request_id=req_id,
                route_index=route_index,
                pickup_pos=p_pos,
                delivery_pos=d_pos,
                true_delta=true_delta,
                selection_cost=selection_cost,
            )
            if best_move is None or move.selection_cost < best_move.selection_cost - EPS:
                best_move = move

    return best_move


def insertion_moves_by_route(
    solution: Solution,
    inst: PDPTWInstance,
    params: ALNSParams,
    req_id: int,
    use_noise: bool,
) -> List[InsertionMove]:
   #对一个 request，寻找它在每一条 route 中的最佳插入方式，返回结果按 selection_cost 从小到大排序
    moves: List[InsertionMove] = []
   # 只需要检查非空路线，以及第一条空路线
   # 因为车辆是同质的，所有空 route 等价
    candidate_route_indices = [idx for idx, route in enumerate(solution.routes) if route]
    first_empty = next((idx for idx, route in enumerate(solution.routes) if not route), None)
    if first_empty is not None:
        candidate_route_indices.append(first_empty)

    for r_idx in candidate_route_indices:
        move = best_insertion_for_request_in_route(solution, inst, params, req_id, r_idx, use_noise)
        if move is not None:
            moves.append(move)
   # 按插入成本从小到大排序，方便 greedy / regret 使用
    moves.sort(key=lambda m: (m.selection_cost, m.true_delta))
    return moves


def select_insertion_move(
    solution: Solution,
    inst: PDPTWInstance,
    params: ALNSParams,
    insertion_name: str,
    use_noise: bool,
) -> Optional[InsertionMove]:
    """
    Select the next request to insert.

    greedy  = regret-1 style: insert the request with smallest best insertion cost.
    regret2 = choose request maximizing second_best - best.
    regret3 = choose request maximizing (second_best - best) + (third_best - best).
    regretm = same idea using all routes.
    """
    if not solution.bank:
        return None

    best_selected: Optional[InsertionMove] = None
    best_score = -INF
    best_tie_cost = INF
    best_feasible_count = INF
    # 根据 insertion_name 决定 regret-k 中的 k
    # greedy 可以理解为 regret-1，只看最好插入位置
    if insertion_name == "greedy":
        k = 1
    elif insertion_name == "regret2":
        k = 2
    elif insertion_name == "regret3":
        k = 3
    elif insertion_name == "regretm":
        k = len(solution.routes)
    else:
        raise ValueError(f"Unknown insertion heuristic: {insertion_name}")
    # 遍历 request bank 中尚未服务的 requests
    for req_id in list(solution.bank):
        # 对当前 request，计算它插入各条 route 的最佳位置
        # moves 已经按 selection_cost 从小到大排序
        moves = insertion_moves_by_route(solution, inst, params, req_id, use_noise)
        # 如果这个 request 没有任何可行插入位置，暂时跳过
        if not moves:
            continue

        best_move = moves[0]
        # feasible_count 表示这个 request 有多少条 route 可以插入
        feasible_count = len(moves)

        if insertion_name == "greedy":
            # 直接选择当前插入成本最低的 request
            score = -best_move.selection_cost
            tie_cost = best_move.selection_cost
        else:
            # 如果一个 request 的最好位置和后续位置差距很大，
            # 说明如果现在不插入，以后可能会很难插或代价很高。
            # 因此 regret score 越大，越应该优先插入。
            # 如果该 request 可行插入路线数量少于 k，
            # 说明它比较难插入，应该提前处理。
            if feasible_count < k:
                score = INF - feasible_count
            else:
                base = moves[0].selection_cost
                score = sum(moves[j].selection_cost - base for j in range(1, k))
            tie_cost = best_move.selection_cost
        # 判断当前 request 是否应该替换目前的 best_selected
        should_update = False
        if insertion_name == "greedy":
            if tie_cost < best_tie_cost - EPS:
                should_update = True
        else:
            if score > best_score + EPS:
                should_update = True
            # 如果 score 基本相同，优先插入可行路线更少的 request
            elif abs(score - best_score) <= EPS:
                if feasible_count < best_feasible_count:
                    should_update = True
                 # 如果可行路线数量也相同，则选择插入成本更小的
                elif feasible_count == best_feasible_count and tie_cost < best_tie_cost - EPS:
                    should_update = True
        # 如果当前 request 更合适，就更新 best_selected
        if should_update:
            best_selected = best_move
            best_score = score
            best_tie_cost = tie_cost
            best_feasible_count = feasible_count

    return best_selected


def repair_solution(
    solution: Solution,
    inst: PDPTWInstance,
    params: ALNSParams,
    insertion_name: str,
    use_noise: bool,
) -> None:
    # 使用指定的 insertion heuristic 修复当前 solution。
    # removal 算子会把一些 requests 删除并放入 request bank。
    # repair_solution 的作用就是：
    # 不断从 bank 中选择 request，
    # 然后用 greedy / regret 插入方法把它重新插回 routes。
    while solution.bank:
        move = select_insertion_move(solution, inst, params, insertion_name, use_noise)
        if move is None:
            break
        # 执行插入动作：
        # 把该 request 的 pickup 和 delivery 插入对应 route，
        # 并从 request bank 中移除该 request
        apply_insertion_move(solution, inst, move)


# ============================================================
# 4. 删除算子 Removal heuristics: random, worst, Shaw
# ============================================================
# ALNS 每次迭代会先从当前解中删除一部分 requests，
# 再用 insertion heuristic 把它们重新插回去。
def remove_request(solution: Solution, inst: PDPTWInstance, req_id: int) -> None:
    # 从路线中删除一个 request，并把它放入 request bank
    req = inst.requests[req_id]
    # 一个 request 包含 pickup 和 delivery 两个节点
    # 所以删除时必须两个节点一起删
    for r_idx, route in enumerate(solution.routes):
        if req.pickup in route or req.delivery in route:
            solution.routes[r_idx] = [n for n in route if n not in (req.pickup, req.delivery)]
    solution.bank.add(req_id)


def request_route_index(solution: Solution, inst: PDPTWInstance, req_id: int) -> Optional[int]:
    # 返回某个 request 当前所在的 route 下标；如果不在任何 route 中，则返回 None
    req = inst.requests[req_id]
    for r_idx, route in enumerate(solution.routes):
        if req.pickup in route or req.delivery in route:
            return r_idx
    return None


def random_removal(solution: Solution, inst: PDPTWInstance, q: int) -> List[int]:
    # 随机删除 q 个已经被服务的 requests
    planned = list(planned_requests(solution, inst))
    # 防止 q 超过当前可删除的 request 数量
    q = min(q, len(planned))
    # 随机选择要删除的 requests
    removed = random.sample(planned, q) if q > 0 else []
    for req_id in removed:
        remove_request(solution, inst, req_id)
    return removed


def request_removal_saving(solution: Solution, inst: PDPTWInstance, params: ALNSParams, req_id: int) -> float:
    """
    计算删除某个 request 后，当前 route 可以节省多少成本。
    这个值越大，说明该 request 当前放置得越“差”，
    worst removal 越倾向于删除它。
    """
    r_idx = request_route_index(solution, inst, req_id)
    if r_idx is None:
        return -INF

    req = inst.requests[req_id]
    old_route = solution.routes[r_idx]
    new_route = [n for n in old_route if n not in (req.pickup, req.delivery)]
    old_cost = route_weighted_cost(old_route, inst, params)
    new_cost = route_weighted_cost(new_route, inst, params)
    # saving 越大，表示删除这个 request 后路线改善越明显
    return old_cost - new_cost


def worst_removal(solution: Solution, inst: PDPTWInstance, params: ALNSParams, q: int) -> List[int]:
    removed: List[int] = []
    q = min(q, len(planned_requests(solution, inst)))

    for _ in range(q):
        # 每轮重新计算候选 request
        # 因为删除一个 request 后，route 成本结构会发生变化
        candidates = list(planned_requests(solution, inst) - set(removed))
        if not candidates:
            break
        # 计算每个 request 的 removal saving
        scored = [(req_id, request_removal_saving(solution, inst, params, req_id)) for req_id in candidates]
        # saving 越大，说明该 request 越“坏”，越优先删除
        scored.sort(key=lambda x: x[1], reverse=True)
        # 加入随机化，避免每次都删除完全相同的 request
        y = random.random()
        idx = int((y ** params.worst_random_degree) * len(scored))
        idx = min(idx, len(scored) - 1)
        req_id = scored[idx][0]
        #删除
        remove_request(solution, inst, req_id)
        removed.append(req_id)

    return removed

#计算当前 solution 中每个节点的实际开始服务时间
def solution_service_start(solution: Solution, inst: PDPTWInstance) -> Dict[int, float]:
    starts: Dict[int, float] = {}
    # Shaw removal 中需要用服务开始时间衡量 request 之间的时间相似性
    for route in solution.routes:
        rs = evaluate_route(route, inst)
        starts.update(rs.service_start)
    return starts


def relatedness(
    solution: Solution,
    inst: PDPTWInstance,
    params: ALNSParams,
    req_i: int,
    req_j: int,
    starts: Dict[int, float],
) -> float:
    """
    计算 Shaw removal 中两个 requests 的相关性。

    返回值越小，说明两个 requests 越相似，
    Shaw removal 越倾向于把它们一起删除。
    """
    ri = inst.requests[req_i]
    rj = inst.requests[req_j]

    p_i, d_i = ri.pickup, ri.delivery
    p_j, d_j = rj.pickup, rj.delivery
    # 距离相似性：pickup 之间近、delivery 之间近，则更相关
    dist_term = (
        inst.distance[p_i][p_j] + inst.distance[d_i][d_j]
    ) / max(inst.max_distance, EPS)

    # 如果某个节点没有 service start，就用 ready time 作为替代
    t_pi = starts.get(p_i, inst.nodes[p_i].ready)
    t_di = starts.get(d_i, inst.nodes[d_i].ready)
    t_pj = starts.get(p_j, inst.nodes[p_j].ready)
    t_dj = starts.get(d_j, inst.nodes[d_j].ready)
    time_term = (abs(t_pi - t_pj) + abs(t_di - t_dj)) / max(inst.max_due, EPS)
    # demand 相似性：货物量越接近，越相关
    demand_term = abs(ri.demand - rj.demand) / max(inst.max_demand, EPS)

    vehicle_term = 0.0  # Li & Lim benchmark 中所有车辆同质，因此车辆兼容性项为 0

    return (
        params.shaw_distance_weight * dist_term
        + params.shaw_time_weight * time_term
        + params.shaw_demand_weight * demand_term
        + params.shaw_vehicle_weight * vehicle_term
    )


def shaw_removal(solution: Solution, inst: PDPTWInstance, params: ALNSParams, q: int) -> List[int]:
    planned = list(planned_requests(solution, inst))
    if not planned:
        return []

    q = min(q, len(planned))
    # 计算当前解中各节点服务开始时间，用于 relatedness 的时间项
    starts = solution_service_start(solution, inst)
    # 先随机选择一个 request 作为起点
    removed: List[int] = [random.choice(planned)]

    while len(removed) < q:
        base = random.choice(removed)
        # remaining 是还没有被选中删除的 request
        remaining = [r for r in planned if r not in removed]
        if not remaining:
            break
        # 按照与 base 的相关性排序：
        # relatedness 越小，越相似，排得越靠前
        remaining.sort(key=lambda r: relatedness(solution, inst, params, base, r, starts))
        # 加入随机化，通常更偏向选择相似 request，
        # 但不是每次都选最相似的那个
        y = random.random()
        idx = int((y ** params.shaw_random_degree) * len(remaining))
        idx = min(idx, len(remaining) - 1)
        removed.append(remaining[idx])
    # 真正从 solution 中删除这些 requests
    for req_id in removed:
        remove_request(solution, inst, req_id)
    return removed


# ============================================================
# 5. 轮盘赌选择与自适应权重更新
# ============================================================
def roulette_select(weights: Dict[str, float]) -> str:
    total = sum(max(0.0, w) for w in weights.values())
    # 如果所有权重都接近 0，则随机选择一个算子
    if total <= EPS:
        return random.choice(list(weights.keys()))
    # 在 [0, total] 中随机取一个数
    r = random.uniform(0.0, total)
    cumulative = 0.0
    for name, weight in weights.items():
        cumulative += max(0.0, weight)
        if r <= cumulative:
            return name
    # 理论上不会走到这里，作为保险返回最后一个
    return list(weights.keys())[-1]


def update_adaptive_weights(
    weights: Dict[str, float],
    scores: Dict[str, float],
    counts: Dict[str, int],
    params: ALNSParams,
) -> None:
    #根据一个 segment 内的表现更新各算子的权重
    for name in weights:
        # 只有被使用过的算子才更新权重
        if counts[name] > 0:
            avg_score = scores[name] / counts[name]
            # 自适应权重更新公式：
            weights[name] = (1.0 - params.reaction_factor) * weights[name] + params.reaction_factor * avg_score
            # 保证权重不会太小，否则某个算子可能永远没有机会再被选中
            weights[name] = max(params.min_weight, weights[name])


def solution_hash(solution: Solution) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[int, ...]]:
    return tuple(tuple(r) for r in solution.routes), tuple(sorted(solution.bank))


# ============================================================
# 6. 初始解构造与 ALNS 主循环
# ============================================================
def construct_initial_solution(inst: PDPTWInstance, params: ALNSParams) -> Solution:
    """
    构造初始解：先把所有 requests 放入 bank，再用 regret-m 插入生成可行初始路线
    """
    solution = Solution(routes=[[] for _ in range(inst.vehicle_count)], bank=set(inst.requests.keys()))
    # 使用 regret-m 插入方法构造初始解
    # 这里不使用 noise，使初始解更稳定
    repair_solution(solution, inst, params, insertion_name="regretm", use_noise=False)
    return solution


def calculate_start_temperature(initial_solution: Solution, inst: PDPTWInstance, params: ALNSParams) -> float:
    """
     根据初始解计算 simulated annealing 的初始温度。
    设定方式：
        让一个比当前解差 start_worse_fraction 比例的解，
        在初始阶段有 50% 的概率被接受
    """
    temp_params = copy.copy(params)
    # 计算初始温度时忽略 request bank penalty
    # 这样可以避免 gamma 太大导致初始温度异常偏高
    temp_params.gamma = 0.0
    # z 是初始解的 modified objective
    z = evaluate_solution(initial_solution, inst, temp_params).objective
    if z <= EPS:
        return 1.0
    return -(params.start_worse_fraction * z) / math.log(0.5)


def choose_q(solution: Solution, inst: PDPTWInstance, params: ALNSParams) -> int:
    # 随机决定本次 ALNS 迭代要删除多少个 requests
    num_planned = len(planned_requests(solution, inst))
    if num_planned == 0:
        return 0
    # 删除数量上限：
    # 不能超过 q_abs_max，也不能超过当前已经服务的 request 数量
    upper = min(params.q_abs_max, max(params.q_min, int(math.ceil(params.q_fraction * len(inst.requests)))))
    upper = min(upper, num_planned)
    lower = min(params.q_min, upper)
    return random.randint(lower, upper)

    """
    ALNS 主循环。

    每次迭代执行：
        1. 复制当前解
        2. 选择 removal / insertion / noise
        3. 删除 q 个 requests
        4. 重新插入 request bank 中的 requests
        5. 评估 candidate solution
        6. 用 simulated annealing 决定是否接受
        7. 更新算子得分和权重
    """
def alns(inst: PDPTWInstance, params: ALNSParams, iterations: int, verbose: bool = True) -> Tuple[Solution, SolutionStats, Dict[str, float], Dict[str, float], Dict[str, float]]:
    # 构造初始解，并计算其目标函数和可行性
    current = construct_initial_solution(inst, params)
    current_stats = evaluate_solution(current, inst, params)
    # best 用于保存搜索过程中找到的全局最好解
    best = current.copy()
    best_stats = current_stats
    # removal 算子的初始权重，开始时三个算子机会相同
    removal_weights = {
        "random": 1.0,
        "worst": 1.0,
        "shaw": 1.0,
    }
    # insertion 算子的初始权重
    insertion_weights = {name: 1.0 for name in params.insertion_names}
    # clean 表示不加 noise，noise 表示插入成本中加入随机扰动
    noise_weights = {
        "clean": 1.0,
        "noise": 1.0,
    }
    # scores 记录当前 segment 内各算子的累计奖励
    # counts 记录当前 segment 内各算子的使用次数
    # 每过 segment_length 次迭代后，会根据它们更新权重
    removal_scores = {k: 0.0 for k in removal_weights}
    insertion_scores = {k: 0.0 for k in insertion_weights}
    noise_scores = {k: 0.0 for k in noise_weights}

    removal_counts = {k: 0 for k in removal_weights}
    insertion_counts = {k: 0 for k in insertion_weights}
    noise_counts = {k: 0 for k in noise_weights}
    # 根据初始解计算模拟退火的初始温度
    temperature = calculate_start_temperature(current, inst, params)
    visited = {solution_hash(current)}

    if verbose:
        print("Initial solution")
        print(f"  objective     : {current_stats.objective:.3f}")
        print(f"  distance      : {current_stats.total_distance:.3f}")
        print(f"  duration      : {current_stats.total_duration:.3f}")
        print(f"  vehicles used : {current_stats.vehicles_used}")
        print(f"  unserved      : {current_stats.unserved}")
        print(f"  feasible      : {current_stats.feasible}")
        print(f"  start temp    : {temperature:.6f}")
    # 开始 ALNS 迭代
    for it in range(1, iterations + 1):
        # 复制当前解，在 candidate 上进行 remove 和 repair
        # 如果 candidate 不被接受，current 不会被破坏
        candidate = current.copy()
        # 随机决定本轮删除多少个 requests
        q = choose_q(candidate, inst, params)
        if q == 0:
            break
        # 根据当前权重，用轮盘赌选择本轮使用的 removal、insertion 和 noise 策略
        rem_name = roulette_select(removal_weights)
        ins_name = roulette_select(insertion_weights)
        noise_name = roulette_select(noise_weights)
        use_noise = noise_name == "noise"
        # 记录本轮使用了哪些算子，后续用于更新权重
        removal_counts[rem_name] += 1
        insertion_counts[ins_name] += 1
        noise_counts[noise_name] += 1
        # 执行 removal：从 candidate 中删除 q 个 requests，并放入 request bank
        if rem_name == "random":
            random_removal(candidate, inst, q)
        elif rem_name == "worst":
            worst_removal(candidate, inst, params, q)
        elif rem_name == "shaw":
            shaw_removal(candidate, inst, params, q)
        else:
            raise ValueError(f"Unknown removal heuristic: {rem_name}")
        # 执行 repair：用选中的 insertion heuristic 尝试把 bank 中的 requests 插回 routes
        repair_solution(candidate, inst, params, insertion_name=ins_name, use_noise=use_noise)
        # 计算 candidate solution 的目标函数和可行性
        candidate_stats = evaluate_solution(candidate, inst, params)
        # 判断 candidate 是否是之前没有访问过的新解
        cand_hash = solution_hash(candidate)
        not_visited = cand_hash not in visited

        reward = 0.0
        accepted = False
        # 如果 candidate 优于历史最好解，则更新 best，并给予最高奖励
        if candidate_stats.objective < best_stats.objective - EPS:
            best = candidate.copy()
            best_stats = candidate_stats
            reward = params.sigma1
        # 如果 candidate 比 current 更好，直接接受
        if candidate_stats.objective < current_stats.objective - EPS:
            current = candidate
            current_stats = candidate_stats
            accepted = True
            if reward == 0.0 and not_visited:
                reward = params.sigma2
        # 如果 candidate 更差，则用 simulated annealing 概率接受
        # 这样可以帮助算法跳出局部最优
        else:
            delta = candidate_stats.objective - current_stats.objective
            prob = math.exp(-delta / temperature) if temperature > EPS and delta < INF / 100 else 0.0
            if random.random() < prob:
                current = candidate
                current_stats = candidate_stats
                accepted = True
                if not_visited:
                    reward = max(reward, params.sigma3)
        # 如果 candidate 被接受，就记录为已访问解
        if accepted:
            visited.add(cand_hash)
        # 把奖励加给本轮使用的 removal、insertion 和 noise 策略
        removal_scores[rem_name] += reward
        insertion_scores[ins_name] += reward
        noise_scores[noise_name] += reward
        # 模拟退火降温，后期会越来越不容易接受较差解
        temperature *= params.cooling
        # 每经过一个 segment，就根据这段时间内的表现更新算子权重
        if it % params.segment_length == 0:
            # 根据 scores 和 counts 更新 removal / insertion / noise 权重
            update_adaptive_weights(removal_weights, removal_scores, removal_counts, params)
            update_adaptive_weights(insertion_weights, insertion_scores, insertion_counts, params)
            update_adaptive_weights(noise_weights, noise_scores, noise_counts, params)
            # 更新完权重后，清空当前 segment 的统计信息
            removal_scores = {k: 0.0 for k in removal_weights}
            insertion_scores = {k: 0.0 for k in insertion_weights}
            noise_scores = {k: 0.0 for k in noise_weights}
            removal_counts = {k: 0 for k in removal_weights}
            insertion_counts = {k: 0 for k in insertion_weights}
            noise_counts = {k: 0 for k in noise_weights}
        # 按一定频率打印当前搜索进度
        if verbose and (it == 1 or it % max(1, iterations // 10) == 0):
            print(
                f"it={it:6d} | best={best_stats.objective:12.3f} "
                f"dist={best_stats.total_distance:9.3f} "
                f"veh={best_stats.vehicles_used:2d} "
                f"unserved={best_stats.unserved:2d} "
                f"T={temperature:.5f}"
            )
    # 返回找到的最好解、最好解的统计信息，以及最终自适应权重
    return best, best_stats, removal_weights, insertion_weights, noise_weights


# ============================================================
# 7. 结果输出
# ============================================================
# 把最终 solution 转换成方便阅读的文本格式
def format_solution(solution: Solution, inst: PDPTWInstance, params: ALNSParams) -> str:
    stats = evaluate_solution(solution, inst, params)
    lines: List[str] = []
    lines.append("=== Li & Lim PDPTW ALNS Solution ===")
    lines.append(f"Objective      : {stats.objective:.6f}")
    lines.append(f"Total distance : {stats.total_distance:.6f}")
    lines.append(f"Total duration : {stats.total_duration:.6f}")
    lines.append(f"Vehicles used  : {stats.vehicles_used}")
    lines.append(f"Unserved       : {stats.unserved}")
    lines.append(f"Feasible       : {stats.feasible}")
    lines.append("")
    # 如果最终解有可行性问题，则输出前 50 条问题
    if stats.issues:
        lines.append("Feasibility issues:")
        for issue in stats.issues[:50]:
            lines.append(f"  - {issue}")
        if len(stats.issues) > 50:
            lines.append(f"  ... {len(stats.issues) - 50} more issues")
        lines.append("")
    # 输出每辆实际使用车辆的路线
    lines.append("Routes:")
    for idx, route in enumerate(solution.routes):
        if not route:
            continue
        rs = stats.route_stats[idx]
        route_str = "0 -> " + " -> ".join(str(n) for n in route) + " -> 0"
        # 统计这条 route 服务了多少个 requests
        reqs = sorted({request_of_node(inst, n) for n in route if request_of_node(inst, n) is not None})
        lines.append(
            f"Vehicle {idx + 1:02d}: distance={rs.distance:.3f}, "
            f"duration={rs.duration:.3f}, requests={len(reqs)}"
        )
        lines.append(f"  {route_str}")
    # 如果还有未服务 requests，则输出 request bank
    if solution.bank:
        lines.append("")
        lines.append("Request bank / unserved requests:")
        lines.append("  " + ", ".join(str(r) for r in sorted(solution.bank)))

    return "\n".join(lines)


def print_instance_summary(inst: PDPTWInstance) -> None:
    print("=== Instance summary ===")
    print(f"Vehicles available : {inst.vehicle_count}")
    print(f"Capacity           : {inst.capacity:g}")
    print(f"Speed              : {inst.speed:g}")
    print(f"Nodes              : {len(inst.nodes)} including depot")
    print(f"Requests           : {len(inst.requests)}")
    depot = inst.nodes[inst.depot_id]
    print(f"Depot              : {inst.depot_id}, ({depot.x:g}, {depot.y:g}), TW=[{depot.ready:g}, {depot.due:g}]")
    print("")


# ============================================================
# 8. Command-line interface
# ============================================================
# 读取命令行参数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Li & Lim PDPTW solved by ALNS following Ropke & Pisinger (2006).")
    # 数据文件路径
    parser.add_argument("filepath", help="Path to Li & Lim PDPTW instance, e.g. lr101.txt")
    # 基本运行参数
    parser.add_argument("--iterations", type=int, default=100, help="Number of ALNS iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="pdptw_solution.txt", help="Output solution file")
    # 目标函数参数
    parser.add_argument("--alpha", type=float, default=1.0, help="Objective weight for total distance")
    parser.add_argument("--beta", type=float, default=0.0, help="Objective weight for total duration")
    parser.add_argument("--gamma", type=float, default=100000.0, help="Penalty for each request in request bank")
    parser.add_argument("--vehicle-penalty", type=float, default=0.0, help="Optional extra penalty per used vehicle")
    # ALNS 参数
    parser.add_argument("--q-min", type=int, default=4, help="Minimum number of removed requests")
    parser.add_argument("--q-fraction", type=float, default=0.15, help="Maximum removed requests as fraction of n")
    parser.add_argument("--segment-length", type=int, default=100, help="Adaptive weight segment length")
    parser.add_argument("--reaction-factor", type=float, default=0.10, help="Adaptive weight reaction factor")
    # 模拟退火和 noise 参数
    parser.add_argument("--cooling", type=float, default=0.99975, help="SA cooling rate")
    parser.add_argument("--start-worse", type=float, default=0.05, help="Initial T control: fraction worse accepted with prob 0.5")
    parser.add_argument("--noise-rate", type=float, default=0.025, help="Noise rate for insertion cost")
    # 是否隐藏中间迭代输出
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return parser.parse_args()

# 程序入口：读取数据、运行 ALNS、保存结果
def main() -> None:
    # 读取命令行参数
    args = parse_args()
    # 固定随机种子，方便复现实验结果
    random.seed(args.seed)
    # 根据命令行参数创建 ALNS 参数对象
    params = ALNSParams(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        vehicle_penalty=args.vehicle_penalty,
        q_min=args.q_min,
        q_fraction=args.q_fraction,
        segment_length=args.segment_length,
        reaction_factor=args.reaction_factor,
        cooling=args.cooling,
        start_worse_fraction=args.start_worse,
        noise_rate=args.noise_rate,
    )
    # 读取 Li & Lim PDPTW 数据
    inst = read_li_lim_pdptw(args.filepath)
    if not args.quiet:
        print_instance_summary(inst)
    # 运行 ALNS 主算法
    best, best_stats, removal_w, insertion_w, noise_w = alns(
        inst=inst,
        params=params,
        iterations=args.iterations,
        verbose=not args.quiet,
    )
    # 保存最终解
    result_text = format_solution(best, inst, params)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result_text)
    # 在终端打印最终结果
    print("\n=== Final result ===")
    print(f"Objective      : {best_stats.objective:.6f}")
    print(f"Total distance : {best_stats.total_distance:.6f}")
    print(f"Total duration : {best_stats.total_duration:.6f}")
    print(f"Vehicles used  : {best_stats.vehicles_used}")
    print(f"Unserved       : {best_stats.unserved}")
    print(f"Feasible       : {best_stats.feasible}")
    print(f"Solution saved : {args.output}")
    # 输出最终自适应权重，用于观察哪些算子表现较好
    print("\nFinal adaptive weights:")
    print("  removal  :", {k: round(v, 4) for k, v in removal_w.items()})
    print("  insertion:", {k: round(v, 4) for k, v in insertion_w.items()})
    print("  noise    :", {k: round(v, 4) for k, v in noise_w.items()})


# 只有直接运行该文件时，才会执行 main()
if __name__ == "__main__":
    main()
