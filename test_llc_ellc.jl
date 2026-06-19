using Oceananigans
using Oceananigans.Grids
using Oceananigans.Units
using CairoMakie
using Statistics
using LinearAlgebra

gaussian(λ, φ, λ₀, φ₀, σ) =
    exp(-((λ - λ₀)^2 + (φ - φ₀)^2) / σ^2)

function make_llc_model(grid;
                        λ₀=45,
                        φ₀=45,
                        σ=5,
                       U0=10,
                       V0=10)

    background_fields = (
        u = (λ, φ, z, t) -> U0 * cosd(φ),
        v = (λ, φ, z, t) -> V0,
    )

    model = NonhydrostaticModel(
        grid,
        tracers = :c,
        advection = WENO(order=5),
        coriolis = nothing,
        background_fields = background_fields
    )

    set!(model,
         c = (λ, φ, z) -> gaussian(λ, φ, λ₀, φ₀, σ))

    return model
end

function make_ellc_model(grid;
                         λ₀=45,
                         φ₀=45,
                         σ=5,
                        U0=10,
                        V0=6.1)

    background_fields = (
        u = (λ, φ, z, t) -> U0,
        v = (λ, φ, z, t) -> V0
    )

    model = NonhydrostaticModel(
        grid,
        tracers = :c,
        advection = WENO(order=5),
        coriolis = nothing,
        background_fields = background_fields
    )

    set!(model,
         c = (λ, φ, z) -> gaussian(λ, φ, λ₀, φ₀, σ))

    return model
end

function comparison_plot(
    λ_llc, φ_llc, c_llc,
    λ_ellc, φ_ellc, c_ellc;
    title1="LLC",
    title2="ELLC"
)

    fig = Figure(size=(800, 400))

    ax1 = Axis(
        fig[1,1],
        title=title1,
        xlabel="λ",
        ylabel="φ",
        aspect=DataAspect()
    )

    hm1 = heatmap!(ax1, λ_llc, φ_llc, c_llc)
    Colorbar(fig[1,2], hm1)

    ax2 = Axis(
        fig[1,3],
        title=title2,
        xlabel="φₑ",
        ylabel="λₑ",
        aspect=DataAspect()
    )

    hm2 = heatmap!(ax2, φ_ellc, λ_ellc, c_ellc)
    Colorbar(fig[1,4], hm2)

    display(fig)

    return fig
end

function peak_location(c, λ, φ)

    i, j = Tuple(argmax(c))

    return (
        λ = λ[i],
        φ = φ[j],
        cmax = c[i,j]
    )
end

function centroid(c, λ, φ)

    M = sum(c)

    (
        λ = sum(c .* reshape(λ, :, 1)) / M,
        φ = sum(c .* reshape(φ, 1, :)) / M
    )
end

function index_centroid(c)

    M = sum(c)

    Nx, Ny = size(c)

    i = collect(1:Nx)
    j = collect(1:Ny)

    (
        i = sum(c .* reshape(i, :, 1)) / M,
        j = sum(c .* reshape(j, 1, :)) / M
    )
end

function displacement(c0, cf)

    p0 = index_centroid(c0)
    pf = index_centroid(cf)

    (
        Δi = pf.i - p0.i,
        Δj = pf.j - p0.j
    )
end

function report_case(
    name,
    c0,
    cf,
    λ,
    φ
)

    p0 = peak_location(c0, λ, φ)
    pf = peak_location(cf, λ, φ)

    c0c = centroid(c0, λ, φ)
    cfc = centroid(cf, λ, φ)

    i0 = index_centroid(c0)
    ifc = index_centroid(cf)

    println("\n", "="^40)
    println(name)
    println("="^40)

    println("Peak:")
    println("  initial = ", p0)
    println("  final   = ", pf)

    println("Centroid:")
    println("  initial = ", c0c)
    println("  final   = ", cfc)

    println("Index centroid:")
    println("  initial = ", i0)
    println("  final   = ", ifc)

    println("Displacement:")
    println("  Δi = ", ifc.i - i0.i)
    println("  Δj = ", ifc.j - i0.j)

    println("Field change:")
    println("  max|Δc| = ",
            maximum(abs.(cf .- c0)))
end

Nx, Ny = 256, 256

grid_llc = LatitudeLongitudeGrid(
    size = (Nx, Ny, 1),
    longitude = (15, 75),
    latitude  = (15, 75),
    z = (-1000, 0)
)

grid_ellc = EquatorialLatitudeLongitudeGrid(
    size = (Ny, Nx, 1),
    longitude = (15, 75), 
    latitude  = (15, 75),
    z = (-1000, 0)
)

model_llc = make_llc_model(
    grid_llc;
    λ₀ = 45,
    φ₀ = 45,
    U0 = 10,
    V0 = 10
)

model_ellc = make_ellc_model(
    grid_ellc;
    λ₀ = 45,
    φ₀ = 45,
    U0 = 10,
    V0 = 6.1
)

λ_llc,  φ_llc, _  = nodes(grid_llc,  Center(), Center(), Center())
λ_ellc, φ_ellc, _ = nodes(grid_ellc, Center(), Center(), Center())

sim_llc  = Simulation(model_llc,  Δt = 10minutes, stop_time = 1days)
sim_ellc = Simulation(model_ellc, Δt = 10minutes, stop_time = 1days)

function progress(sim)
    c = interior(sim.model.tracers.c)[:, :, 1]

    println(
        "t = ", sim.model.clock.time / days,
        "  max(c) = ", maximum(c),
        #"  u = ", extrema(interior(sim.model.velocities.u)),
        #"  v = ", extrema(interior(sim.model.velocities.v))
    )
end

sim_llc.callbacks[:progress]  = Callback(progress, IterationInterval(10))
sim_ellc.callbacks[:progress] = Callback(progress, IterationInterval(10))

c_initial_llc  = copy(Array(interior(model_llc.tracers.c))[:, :, 1])
c_initial_ellc = copy(Array(interior(model_ellc.tracers.c))[:, :, 1])

comparison_plot(
    λ_llc,  φ_llc, c_initial_llc,
    λ_ellc, φ_ellc, c_initial_ellc
)

run!(sim_llc)
run!(sim_ellc)

c_final_llc  = Array(interior(model_llc.tracers.c))[:, :, 1]
c_final_ellc = Array(interior(model_ellc.tracers.c))[:, :, 1]

comparison_plot(
    λ_llc, φ_llc, c_final_llc,
    λ_ellc, φ_ellc, c_final_ellc
)

report_case(
    "LLC",
    c_initial_llc,
    c_final_llc,
    λ_llc,
    φ_llc
)

report_case(
    "ELLC",
    c_initial_ellc,
    c_final_ellc,
    λ_ellc,
    φ_ellc
)