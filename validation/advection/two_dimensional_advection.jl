using Oceananigans
using Oceananigans.Advection: MultiDimensionalScheme
using Oceananigans.Grids: min_Δx, min_Δy, min_Δz
using JLD2
using OffsetArrays
using LinearAlgebra

c₀_2D(x, y, z) = sin(2π*x)*cos(2π*y)
test = :linear

# """
# linear 2D advection test
# """

sol = Vector(undef, 5)

for (idx, N) in enumerate([20, 40, 80, 160, 320])

    @info "runnning N $N"
    grid = RectilinearGrid(size = (N, N), x = (0, 1),  y = (0, 1),  halo = (6, 6), topology = (Periodic, Periodic, Flat))    

    if test == :linear
        U = Field((Face, Center, Center), grid)
        V = Field((Center, Face, Center), grid)

        velU = 1.0
        velV = 2.0
        parent(U) .= velU
        parent(V) .= velV
    else
        U(x, y, z, t) = + sin(π*x)^2 * sin(2π*y) * cos(π * t)
        V(x, y, z, t) = - sin(π*y)^2 * sin(2π*x) * cos(π * t)
    end

    Δt = 0.2 * min_Δy(grid) / 2
    max_iter = Int(ceil(1 / Δt))
    end_time = max_iter * Δt

    @info "running with $Δt, for $max_iter iterations, ending at $end_time seconds"

    solution1d = Vector(undef, 4)
    solution2d = Vector(undef, 4)
    solutionr  = set!(CenterField(grid), c₀_2D) 
    solutionr  = interior(solutionr, :, :, 1)

    for (ord, order) in enumerate([3, 5, 7, 9])
        scheme = WENO(; order)

        model_1d = HydrostaticFreeSurfaceModel(grid = grid,
                                            tracers = :c,
                                   tracer_advection = scheme,
                                         velocities = PrescribedVelocityFields(u=U, v=V), 
                                           coriolis = nothing,
                                            closure = nothing,
                                           buoyancy = nothing)

        model_2d = HydrostaticFreeSurfaceModel(grid = grid,
                                            tracers = :c,
                                   tracer_advection = MultiDimensionalScheme(scheme, order = 6),
                                         velocities = PrescribedVelocityFields(u=U, v=V), 
                                           coriolis = nothing,
                                            closure = nothing,
                                           buoyancy = nothing)

        c_1d_time_series   = Vector(undef, max_iter)
        c_2d_time_series   = Vector(undef, 1000)

        for (model, c_time_series) in zip([model_1d, model_2d], [c_1d_time_series, c_2d_time_series])
            c = model.tracers.c
            set!(model, c=c₀_2D)

            @info "initializing"
            for step in 1:max_iter
                c_time_series[step] = Array(interior(model.tracers.c, :, :, 1))
                time = model.clock.time
                time_step!(model, Δt)
            end       
        end

        solution1d[ord] = c_1d_time_series
        solution2d[ord] = c_2d_time_series
    end

    sol[idx] = deepcopy((solutionr, solution1d, solution2d))
end

EOC = zeros(4, 4)
LIN = zeros(4, 5)
for (n, N) in enumerate([20, 40, 80, 160])
    for idx_ord in [1, 2, 3, 4]
        EOC[idx_ord, n] = log(maximum(abs.(sol[n][1] .- sol[n][2][idx_ord][end])) / maximum(abs.(sol[n+1][1] .- sol[n+1][2][idx_ord][end])))/log(2)
        LIN[idx_ord, n] = norm(sol[n][1] .- sol[n][2][idx_ord][end]) / N
    end
end