# PDPTW Vehicle Routing Optimization with ALNS

This project implements an Adaptive Large Neighborhood Search (ALNS) heuristic for the Pickup and Delivery Problem with Time Windows (PDPTW).

The project focuses on solving a vehicle routing problem where each request contains a pickup node and a delivery node. The solution must satisfy vehicle capacity constraints, pickup-delivery precedence constraints, and time window constraints.

## Project Overview

The main objectives of this project are:

- Minimize the number of vehicles used
- Minimize the total travel distance
- Maintain feasibility under capacity constraints
- Maintain pickup-before-delivery precedence
- Satisfy time window constraints
- Analyze the impact of vehicle capacity on routing performance

## Problem Description

The Pickup and Delivery Problem with Time Windows includes the following constraints:

- Each pickup request must be paired with its corresponding delivery node
- Pickup must be completed before delivery
- Vehicle capacity must not be exceeded
- Each node must be visited within its time window
- Service time must be considered
- Routes must start and end at the depot

## Methodology

This project uses an Adaptive Large Neighborhood Search framework.

The algorithm includes:

- Initial solution construction
- Request removal and insertion operators
- Feasibility checks for capacity, time windows, and pickup-delivery precedence
- Simulated annealing acceptance criterion
- Two-stage optimization:
  - Stage 1: minimize the number of vehicles
  - Stage 2: minimize total travel distance

## Sensitivity Analysis

A capacity sensitivity analysis was conducted to evaluate how vehicle capacity affects:

- Number of vehicles used
- Total travel distance
- Routing feasibility

The results show that vehicle capacity has a direct impact on fleet size and routing performance.

## Files

```text
pdptw_alns.py
PDPTW_ALNS_Report.pdf
README.md
