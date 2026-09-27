# GPU benchmark: how much faster does a global tripolar BGC model run with split tracer stepping and FFSL?
#
# Runs the same global TripolarGrid model with z-star in several configurations. T and S always use WENO5 on
# every step; only the BGC tracers change.
#   - physics:           no BGC tracers
#   - baseline:          BGC tracers stepped every Δt with WENO5
#   - split-WENO-r:      BGC tracers stepped every r Δt with WENO5
#   - split-FFSL-r:      BGC tracers stepped every r Δt with FluxFormSemiLagrangian
# Every configuration is timed over the same number of simulated days, after a warm-up of one full cycle.
# The BGC fields of every configuration are compared with the baseline.
#
# Usage:
#   julia --project=<env with Oceananigans and CUDA> bgc_speedup.jl <resolution in degrees> <number of BGC tracers> <days> <output directory>
#   e.g. julia --project=bench bgc_speedup.jl 1 24 30 results

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: MutableVerticalDiscretization, inactive_cell
using Oceananigans.BoundaryConditions: fill_halo_regions!
using CUDA
using Printf

CUDA.allowscalar(false)

const repository = joinpath(@__DIR__, "..", "..", "..")
include(joinpath(repository, "test", "setup", "split_tracer_stepping_test_utils.jl")) # MinimalNPZD
include(joinpath(repository, "test", "setup", "volume_integrals.jl"))                 # volume_integral

const resolution = parse(Float64, get(ARGS, 1, "1"))
const Nbgc = parse(Int, get(ARGS, 2, "24"))
const simulated_days = parse(Float64, get(ARGS, 3, "30"))
const output_directory = get(ARGS, 4, "results")
const arch = get(ENV, "BENCHMARK_ARCHITECTURE", "GPU") == "CPU" ? CPU() : GPU() # CPU only for a quick smoke test
mkpath(output_directory)

const Nx = round(Int, 360 / resolution)
const Ny = round(Int, 180 / resolution)
const Nz = 50
const Δt = resolution ≥ 1 ? 10minutes : 10minutes * resolution / 0.5 # 10 min at 1°, 5 min at 1/4°
const warmup = 32                                                      # one full cycle for every r ≤ 32
const steps = 32 * ceil(Int, simulated_days * day / Δt / 32)                     # a whole number of cycles for every r ≤ 32
const sinking_speed = 10 / day

@info "Benchmark" resolution Nx Ny Nz Nbgc Δt steps simulated_days = steps * Δt / day
arch isa GPU && CUDA.versioninfo()

#####
##### Grid, biogeochemistry and model
#####

# Land over the two tripolar north-pole singularities (55°N at 70°E and 250°E), as in the tests
pole_island(λ, φ, λ₀) = exp(-(mod(λ - λ₀ + 180, 360) - 180)^2 / 50 - (φ - 55)^2 / 50)
bottom(λ, φ) = -4000 + 1000 * cosd(2λ) * cosd(φ) + 6000 * (pole_island(λ, φ, 70) + pole_island(λ, φ, 250))

# Every model gets its own grid: a z-star grid stores the free-surface state, so sharing one would start each model
# from the final state of the previous one
function build_grid()
    z = MutableVerticalDiscretization(collect(range(-4000, 0, length = Nz + 1)))
    underlying_grid = TripolarGrid(arch; size = (Nx, Ny, Nz), halo = (7, 7, 4), z)
    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))
end

# A sinking velocity built on the device: -speed on interior faces between two active cells, zero elsewhere
@inline function sinking_w(i, j, k, grid, speed)
    interior_face = (k > 1) & (k ≤ size(grid, 3))
    active = !inactive_cell(i, j, k, grid) & !inactive_cell(i, j, k - 1, grid)
    return ifelse(interior_face & active, - speed, zero(speed))
end

function device_sinking_velocity(grid, speed)
    w = Field(KernelFunctionOperation{Center, Center, Face}(sinking_w, grid, convert(eltype(grid), speed)))
    compute!(w)
    fill_halo_regions!(w)
    return (u = ZeroField(), v = ZeroField(), w = w)
end

function npzd(grid)
    p = MinimalNPZD(grid) # default parameters, no sinking
    return MinimalNPZD(p.growth_rate, p.grazing_rate, p.mortality_rate, p.remineralization_rate,
                       p.half_saturation, p.light_scale, device_sinking_velocity(grid, sinking_speed))
end

const passive_names = Tuple(Symbol(:C, n) for n in 1:max(0, Nbgc - 4))
const bgc_names = (:N, :P, :Z, :D, passive_names...)

# BGC tracers use an adaptive implicit vertical discretization (AVID) under every scheme so the comparison is fair
avid() = AdaptiveVerticallyImplicitDiscretization(cfl = 0.5)
bgc_scheme(scheme) = scheme == :FFSL ? FluxFormSemiLagrangian(vertical_scheme = WENO(order=5, time_discretization = avid())) :
                                       WENO(order=5, time_discretization = avid())

# Idealized westerlies and trades (kinematic stress τ/ρ₀) to spin up gyres; grid-aligned, so only approximately zonal in the Arctic
wind_stress(λ, φ, t) = - 1e-4 * cosd(3φ) * (abs(φ) < 60)
boundary_conditions = (; u = FieldBoundaryConditions(top = FluxBoundaryCondition(wind_stress)))

function build_model(; bgc = true, ratio = nothing, scheme = :WENO)
    grid = build_grid()
    tracers = bgc ? (:T, :S, bgc_names...) : (:T, :S)
    physics_advection = (T = WENO(order=5), S = WENO(order=5))
    tracer_advection = bgc ? merge(physics_advection, NamedTuple{bgc_names}(Tuple(bgc_scheme(scheme) for _ in bgc_names))) :
                             physics_advection

    splitting = isnothing(ratio) ? nothing : TracerTimeStepSplitting(tracers = bgc_names, ratio = ratio, biogeochemistry_substeps = 4)

    model = HydrostaticFreeSurfaceModel(grid; tracers, tracer_advection, boundary_conditions,
                                        biogeochemistry = bgc ? npzd(grid) : nothing,
                                        buoyancy = SeawaterBuoyancy(),
                                        coriolis = HydrostaticSphericalCoriolis(),
                                        closure = RiBasedVerticalDiffusivity(),
                                        momentum_advection = WENOVectorInvariant(),
                                        free_surface = SplitExplicitFreeSurface(grid; substeps = 30),
                                        timestepper = :SplitRungeKutta3,
                                        tracer_time_step_splitting = splitting)

    set!(model.tracers.T, (λ, φ, z) -> 2 + 25 * cosd(φ) * exp(z / 800))
    set!(model.tracers.S, 35)

    if bgc
        set!(model.tracers.N, (λ, φ, z) -> 1 + 20 * (1 - exp(z / 500)))
        set!(model.tracers.P, (λ, φ, z) -> 0.5 * exp(z / 50))
        set!(model.tracers.Z, (λ, φ, z) -> 0.1 * exp(z / 50))
        set!(model.tracers.D, 0.01)
        for name in passive_names
            set!(model.tracers[name], (λ, φ, z) -> exp(-(φ - 20)^2 / 200) * exp(z / 300))
        end
    end

    return model
end

#####
##### Diagnostics
#####

total_nitrogen(model) = sum(volume_integral(model.tracers[name]) for name in (:N, :P, :Z, :D))

@inline wet_cell(i, j, k, grid) = ifelse(inactive_cell(i, j, k, grid), zero(grid), one(grid))
active_cells_field = Field(KernelFunctionOperation{Center, Center, Center}(wet_cell, build_grid()))
compute!(active_cells_field)

relative_l2(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, b))

function long_step_courant_number(model, ratio)
    r = isnothing(ratio) ? 1 : ratio
    u = maximum(abs, interior(model.velocities.u))
    v = maximum(abs, interior(model.velocities.v))
    Δx = minimum_xspacing(model.grid)
    Δy = minimum_yspacing(model.grid)
    return max(u / Δx, v / Δy) * r * Δt
end

function run_timed!(model)
    for _ in 1:warmup
        time_step!(model, Δt)
    end
    arch isa GPU || return @elapsed for _ in 1:steps
        time_step!(model, Δt)
    end
    CUDA.synchronize()
    return CUDA.@elapsed for _ in 1:steps
        time_step!(model, Δt)
    end
end

#####
##### Configurations
#####

configurations = Any[("physics", (; bgc = false)), ("baseline", (;))]
append!(configurations, [("split-WENO-$r", (; ratio = r)) for r in (2, 4, 8, 16, 32)])
append!(configurations, [("split-FFSL-$r", (; ratio = r, scheme = :FFSL)) for r in (4, 8, 16, 32)])

if haskey(ENV, "BENCHMARK_CONFIGURATIONS") # e.g. "physics,baseline,split-FFSL-16"
    selected = split(ENV["BENCHMARK_CONFIGURATIONS"], ",")
    filter!(c -> first(c) in selected, configurations)
end

label_prefix = @sprintf("%sdeg_%dbgc", replace(string(resolution), "." => "p"), Nbgc)
table_path = joinpath(output_directory, "bgc_speedup_$(label_prefix).md")

function write_row(io, cells...)
    println(io, "| ", join(cells, " | "), " |")
    flush(io)
    return nothing
end

open(table_path, "w") do io
    println(io, "# BGC speed-up, $(Nx)×$(Ny)×$(Nz) TripolarGrid ($(resolution)°), $(Nbgc) BGC tracers, Δt = $(prettytime(Δt)), $(steps) steps\n")
    write_row(io, "config", "ms/step", "wall s", "SYPD", "BGC overhead", "BGC speed-up", "L2(N)", "L2(P)", "N drift", "min P", "min D", "long-step C", "status")
    write_row(io, fill("---", 13)...)

    physics_time = NaN
    baseline_time = NaN
    reference = nothing

    for (label, kw) in configurations
        @info "Running $label"
        status = "ok"
        t = NaN
        cells = fill("–", 6)

        try
            model = build_model(; kw...)
            bgc = get(kw, :bgc, true)
            N₀ = bgc ? total_nitrogen(model) : NaN
            t = run_timed!(model)

            C = long_step_courant_number(model, get(kw, :ratio, nothing))
            cells[6] = @sprintf("%.2f", C)

            if bgc
                wet = Array(interior(active_cells_field)) .> 0
                N = Array(interior(model.tracers.N))[wet]
                P = Array(interior(model.tracers.P))[wet]
                D = Array(interior(model.tracers.D))[wet]
                label == "baseline" && (reference = (; N, P))
                if !isnothing(reference)
                    cells[1] = @sprintf("%.2e", relative_l2(N, reference.N))
                    cells[2] = @sprintf("%.2e", relative_l2(P, reference.P))
                end
                cells[3] = @sprintf("%.2e", (total_nitrogen(model) - N₀) / N₀)
                cells[4] = @sprintf("%.2e", minimum(P))
                cells[5] = @sprintf("%.2e", minimum(D))
                all(isfinite, N) || (status = "non-finite")
            end
            model = nothing
        catch err
            status = "failed: " * first(split(sprint(showerror, err), '\n'))
            @warn "$label failed" exception = (err, catch_backtrace())
        end

        GC.gc(true)
        arch isa GPU && CUDA.reclaim()

        label == "physics" && (physics_time = t)
        label == "baseline" && (baseline_time = t)

        SYPD = (steps * Δt / 365days) / (t / day)
        overhead = (t - physics_time) / physics_time
        baseline_overhead = (baseline_time - physics_time) / physics_time
        speedup = label == "physics" ? "–" : @sprintf("%.2f", baseline_overhead / overhead)
        overhead_cell = label == "physics" ? "–" : @sprintf("%.1f%%", 100overhead)

        write_row(io, label, @sprintf("%.2f", 1e3t / steps), @sprintf("%.1f", t), @sprintf("%.2f", SYPD),
                  overhead_cell, speedup, cells..., status)
        @info "$label done" t status
    end
end

println(read(table_path, String))
