using Test
using Plots
using LaTeXStrings
using Printf
using Oceananigans.Advection

using Printf
using Statistics

using Oceananigans
using Oceananigans.Grids

             U = 1
             W = 0.05
            Nx = 16
            Lx = 2.5
             h = Lx / Nx
            Δt = min(0.01 * h / U)
     stop_time = Δt
stop_iteration = 1
     advection = CenteredSecondOrder

c(x, y, z, t, U, W) = exp(-(x - U * t)^2 / W)

domain = (x=(-1, 1.5), y=(0, 1), z=(0, 1))

grid = RegularCartesianGrid(size=(Nx, 1, 1), halo=(3, 3, 3); domain...)

model = IncompressibleModel(architecture = CPU(),
                            timestepper = :RungeKutta3,
                            grid = grid,
                            advection = CenteredSecondOrder(),
                            coriolis = nothing,
                            buoyancy = nothing,
                            tracers = :c,
                            closure = nothing)

set!(model, u = U,
     c = (x, y, z) -> c(x, y, z, 0, U, W),
     v = (x, y, z) -> c(x, y, z, 0, U, W),
     w = (x, y, z) -> c(x, y, z, 0, U, W))

simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, iteration_interval=stop_iteration)

#@info "Running Gaussian advection diffusion test for v and cx with Nx = $Nx and Δt = $Δt ($(typeof(scheme)))..."
run!(simulation)



