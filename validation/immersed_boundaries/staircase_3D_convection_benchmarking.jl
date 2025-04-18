using Oceananigans
using Printf
using JLD2
# using NVTX
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Oceananigans.Solvers: DiagonallyDominantPreconditioner, compute_laplacian!, KrylovPoissonSolver
using Statistics
using CUDA

function initial_conditions!(model)
    h = 0.05
    x₀ = 0.5
    y₀ = 0.5
    z₀ = 0.55
    bᵢ(x, y, z) = - exp(-((x - x₀)^2 + (y - y₀)^2 + (z - z₀)^2) / 2h^2)

    set!(model, b=bᵢ)
end

function setup_grid(N)
    grid = RectilinearGrid(GPU(), Float64,
                        size = (N, N, N), 
                        halo = (6, 6, 6),
                        x = (0, 1),
                        y = (0, 1),
                        z = (0, 1),
                        topology = (Bounded, Bounded, Bounded))

    # slope(x, y) = 1 - (x + y) / 2
    # slope(x, y) = (5 + tanh(40*(x - 1/6)) + tanh(40*(x - 2/6)) + tanh(40*(x - 3/6)) + tanh(40*(x - 4/6)) + tanh(40*(x - 5/6))) / 10 * (0.5y + 0.5)
    slope(x, y) = (5 + tanh(40*(x - 1/6)) + tanh(40*(x - 2/6)) + tanh(40*(x - 3/6)) + tanh(40*(x - 4/6)) + tanh(40*(x - 5/6))) / 20 + 
                  (5 + tanh(40*(y - 1/6)) + tanh(40*(y - 2/6)) + tanh(40*(y - 3/6)) + tanh(40*(y - 4/6)) + tanh(40*(y - 5/6))) / 20

    grid = ImmersedBoundaryGrid(grid, GridFittedBottom(slope))
    return grid
end

function setup_model(grid, pressure_solver)
    model = NonhydrostaticModel(; grid, pressure_solver,
                                  advection = WENO(order=9),
                                  coriolis = FPlane(f=0.1),
                                  tracers = :b,
                                  buoyancy = BuoyancyTracer())

    initial_conditions!(model)
    return model
end

reltol = abstol = 1e-7

function setup_simulation(model, Δt, stop_iteration)
    return Simulation(model, Δt=Δt, stop_iteration=stop_iteration)
end

# Ns = [32, 64, 128, 160, 192, 224, 256]
Ns = [32, 64, 128, 256]

Δt = 2e-2 * 64 / 2 / maximum(Ns)
nsteps = 50

times_FFT = [zeros(nsteps) for _ in Ns]

for (i, N) in enumerate(Ns)
    @info "Benchmarking FFT solver, N = $(N)"
    grid = nothing
    model = nothing
    GC.gc()
    CUDA.reclaim()
    grid = setup_grid(N)
    pressure_solver = nothing
    model = setup_model(grid, pressure_solver)

    for step in 1:3
        time_step!(model, Δt)
    end

    for step in 1:nsteps
        # NVTX.@range "FFT timestep, N $N" begin
        #     time_step!(model, Δt)
        # end
        times_FFT[i][step] = @elapsed time_step!(model, Δt)
    end
end

# cg_softwares = ["Oceananigans", "Krylov.jl"]
cg_softwares = ["Oceananigans", "Krylov.jl"]
preconditioners = ["no", "FFT", "MITgcm"]

times_cg = [zeros(nsteps) for _ in cg_softwares, _ in preconditioners, _ in Ns]
Niters_cg = [zeros(Int, nsteps) for _ in cg_softwares, _ in preconditioners, _ in Ns]

for (i, software) in enumerate(cg_softwares), (j, precond_name) in enumerate(preconditioners), (k, N) in enumerate(Ns)
    @info "Benchmarking N = $(N) $software with $precond_name preconditioner"
    grid = nothing
    model = nothing
    pressure_solver = nothing
    preconditioner = nothing
    GC.gc()
    CUDA.reclaim()

    grid = setup_grid(N)
    if precond_name == "no"
        preconditioner = nothing
    elseif precond_name == "FFT"
        preconditioner = FFTBasedPoissonSolver(grid.underlying_grid)
    elseif precond_name == "MITgcm"
        preconditioner = DiagonallyDominantPreconditioner()
    end
    reltol = abstol = 1e-7
    if software == "Krylov.jl"
        pressure_solver = KrylovPoissonSolver(grid; preconditioner, reltol, abstol, maxiter=10000)
    elseif software == "Oceananigans"
        pressure_solver = ConjugateGradientPoissonSolver(grid, maxiter=10000; reltol, abstol, preconditioner)
    end

    model = setup_model(grid, pressure_solver)

    for step in 1:3
        time_step!(model, Δt)
    end

    # for step in 1:nsteps
    #     NVTX.@range "$software, $precond_name preconditioner N $N" begin
    #         time_step!(model, Δt)
    #     end
    #     if software == "Krylov.jl"
    #         @info "PCG iteration (N = $N, $software, $precond_name preconditioner) = $(model.pressure_solver.krylov_solver.workspace.stats.niter)"
    #         Niters_cg[i, j, k][step] = model.pressure_solver.krylov_solver.workspace.stats.niter
    #     elseif software == "Oceananigans"
    #         @info "PCG iteration (N = $N, $software, $precond_name preconditioner) = $(model.pressure_solver.conjugate_gradient_solver.iteration)"
    #         Niters_cg[i, j, k][step] = model.pressure_solver.conjugate_gradient_solver.iteration
    #     end
    # end

    for step in 1:nsteps
        times_cg[i, j, k][step] = @elapsed time_step!(model, Δt)
        if software == "Krylov.jl"
            @info "PCG iteration (N = $N, $software, $precond_name preconditioner) = $(model.pressure_solver.krylov_solver.workspace.stats.niter)"
            Niters_cg[i, j, k][step] = model.pressure_solver.krylov_solver.workspace.stats.niter
        elseif software == "Oceananigans"
            @info "PCG iteration (N = $N, $software, $precond_name preconditioner) = $(model.pressure_solver.conjugate_gradient_solver.iteration)"
            Niters_cg[i, j, k][step] = model.pressure_solver.conjugate_gradient_solver.iteration
        end
    end
end

jldopen("staircase_3D_convection_benchmarking_results.jld2", "w") do file
    file["times_FFT"] = times_FFT
    file["times_cg"] = times_cg
    file["Niters_cg"] = Niters_cg
    file["Ns"] = Ns
    file["cg_softwares"] = cg_softwares
    file["preconditioners"] = preconditioners
    file["nsteps"] = nsteps
    file["Δt"] = Δt
end

# results = jldopen("complex_domain_convection_benchmarking_results.jld2", "r") do file
#     file["results"]
# end

# results["ImmersedPoissonSolver"]["128"]

# Ns = [32, 64, 128, 256]

# t_median_immersed = [median(results["ImmersedPoissonSolver"]["$N"]).time / 1e9 for N in Ns]
# t_median_FFT = [median(results["FFTBasedPoissonSolver"]["$N"]).time / 1e9 for N in Ns]

# fig = Figure()
# ax = Axis(fig[1, 1], xlabel="N", ylabel="Median time (s)", yscale=log10, xscale=log2, title="Sloped convection, GPU, 3D setup (N³ grid points)")
# lines!(ax, Ns, t_median_immersed, label="Immersed solver")
# lines!(ax, Ns, t_median_FFT, label="FFT solver")
# axislegend(ax, position=:lt)
# display(fig)
# save("sloped_convection_benchmarks.png", fig, px_per_unit=4)