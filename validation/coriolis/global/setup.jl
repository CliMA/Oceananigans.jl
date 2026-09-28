# Idealized global ocean on the 1° tripolar grid: ETOPO bathymetry, 4 z⋆ levels, zonal wind stress, SST restoring,
# convective adjustment, WENO momentum and tracer advection, split-explicit free surface, SplitRungeKutta3, Δt = 1 hour.
#
#   CORIOLIS_ARCHITECTURE = CPU | GPU      (default CPU)
#   CORIOLIS_FLOAT_TYPE   = Float64 | Float32   (default Float64)
#   CORIOLIS_OUTPUT       = output directory (default ./output next to this file)

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: φnode
using Oceananigans.Operators: ζ₃ᶠᶠᶜ
using Oceananigans.ImmersedBoundaries: InterfaceImmersedCondition
using Oceananigans.Coriolis: CDScheme, ShearSignedCoriolis
using Oceananigans.Advection: EnergyConserving
using JLD2, Printf

get(ENV, "CORIOLIS_ARCHITECTURE", "CPU") == "GPU" && @eval using CUDA

const OUTPUT = get(ENV, "CORIOLIS_OUTPUT", joinpath(@__DIR__, "output"))
mkpath(OUTPUT)

arch = get(ENV, "CORIOLIS_ARCHITECTURE", "CPU") == "GPU" ? GPU() : CPU()
FT = get(ENV, "CORIOLIS_FLOAT_TYPE", "Float64") == "Float32" ? Float32 : Float64
Oceananigans.defaults.FloatType = FT

Nx, Ny, Nz = 360, 180, 4
z = MutableVerticalDiscretization([-4000, -1500, -500, -100, 0])
underlying_grid = TripolarGrid(arch; size=(Nx, Ny, Nz), z, halo=(5, 5, 5))

# 1° block means of ETOPO 2022 interpolated on this grid, written by bottom_height.jl
bottom_height = jldopen(file -> file["bottom_height"], joinpath(@__DIR__, "bottom_height.jld2"))
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(FT.(bottom_height), InterfaceImmersedCondition());
                            active_cells_map=true)

τ₀ = FT(0.15)
ρ₀ = 1020
zonal_wind_stress(φ, τ₀) = - τ₀ * sin(2 * deg2rad(φ)) * sin(6 * deg2rad(φ))
zonal_momentum_flux(λ, φ, t, p) = - zonal_wind_stress(φ, p.τ₀) / p.ρ₀

restoring_temperature(φ) = 30 * cos(deg2rad(φ))^2

@inline function temperature_flux(i, j, grid, clock, fields, p)
    φ = φnode(i, j, grid.Nz, grid, Center(), Center(), Center())
    return @inbounds p.rate * (fields.T[i, j, grid.Nz] - restoring_temperature(φ))
end

wind_stress = FluxBoundaryCondition(zonal_momentum_flux, parameters=(; τ₀, ρ₀))
drag = BulkDrag(coefficient=FT(2.5e-3))
u_bcs = FieldBoundaryConditions(top=wind_stress, bottom=drag, immersed=ImmersedBoundaryCondition(bottom=drag))
v_bcs = FieldBoundaryConditions(bottom=drag, immersed=ImmersedBoundaryCondition(bottom=drag))
T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(temperature_flux; discrete_form=true,
                                                           parameters=(; rate=FT(100 / 30days))))

Δt = FT(1hour)
free_surface = SplitExplicitFreeSurface(grid; cfl=0.7, fixed_Δt=Δt)
buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion=2e-4), constant_salinity=35)
closure = ConvectiveAdjustmentVerticalDiffusivity(convective_κz=1, convective_νz=1)

function global_simulation(name; coriolis, stop_time, save_interval=10days, checkpoint_interval=365days, pickup=false)
    model = HydrostaticFreeSurfaceModel(grid; coriolis, free_surface, buoyancy, closure,
                                        momentum_advection=WENOVectorInvariant(order=5), tracer_advection=WENO(order=7),
                                        tracers=:T, timestepper=:SplitRungeKutta3, vertical_coordinate=ZStarCoordinate(),
                                        boundary_conditions=(u=u_bcs, v=v_bcs, T=T_bcs))
    set!(model, T=(λ, φ, z) -> 10 * exp(z / 1000))

    simulation = Simulation(model; Δt, stop_time)
    wall_clock = Ref(time_ns())
    function progress(sim)
        @info @sprintf("%s, iter %d, t %s, max|u| %.2f, wall %s", name, iteration(sim), prettytime(sim),
                       maximum(abs, sim.model.velocities.u), prettytime(1e-9 * (time_ns() - wall_clock[])))
        wall_clock[] = time_ns()
    end
    add_callback!(simulation, progress, IterationInterval(500))

    u, v, w = model.velocities
    U = Field(Integral(u, dims=3))
    ψ = Field(CumulativeIntegral(-U, dims=2))
    ζ = Field(KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid, u, v), indices=(:, :, Nz))
    speed = Field(@at((Center, Center, Center), sqrt(u^2 + v^2)), indices=(:, :, Nz))
    surface = (; ψ, ζ, speed, u=view(u, :, :, Nz), v=view(v, :, :, Nz), T=view(model.tracers.T, :, :, Nz))
    below = (; u3=view(u, :, :, Nz-1), v3=view(v, :, :, Nz-1))
    scheme = model.coriolis.scheme
    chirality = scheme isa ShearSignedCoriolis ? (; χ=view(scheme.chirality, :, :, Nz), χ3=view(scheme.chirality, :, :, Nz-1)) : NamedTuple()

    simulation.output_writers[:surface] = JLD2Writer(model, merge(surface, below, chirality); filename=joinpath(OUTPUT, "$name.jld2"),
                                                     schedule=TimeInterval(save_interval), array_type=Array{Float32},
                                                     overwrite_files=!pickup)
    simulation.output_writers[:checkpointer] = Checkpointer(model; schedule=TimeInterval(checkpoint_interval), dir=OUTPUT,
                                                            prefix="$(name)_checkpoint")
    Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)
    return simulation
end
