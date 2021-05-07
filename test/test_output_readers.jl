using Test
using JLD2

using Oceananigans
using Oceananigans.Units

function generate_some_interesting_simulation_data(; architecture=CPU())
    grid = RegularRectilinearGrid(size=(32, 32, 32), extent=(64, 64, 32))

    Qʰ = 200
    ρₒ = 1026
    cᴾ = 3991
    Qᵀ = Qʰ / (ρₒ * cᴾ)
    dTdz = 0.01
    T_bcs = TracerBoundaryConditions(grid, top = FluxBoundaryCondition(Qᵀ), bottom = GradientBoundaryCondition(dTdz))

    u₁₀ = 10
    cᴰ = 2.5e-3
    ρₐ = 1.225
    Qᵘ = - ρₐ / ρₒ * cᴰ * u₁₀ * abs(u₁₀)
    u_bcs = UVelocityBoundaryConditions(grid, top = FluxBoundaryCondition(Qᵘ))

    @inline Qˢ(x, y, t, S, evaporation_rate) = - evaporation_rate * S
    evaporation_rate = 1e-3 / hour
    evaporation_bc = FluxBoundaryCondition(Qˢ, field_dependencies=:S, parameters=evaporation_rate)
    S_bcs = TracerBoundaryConditions(grid, top=evaporation_bc)

    model = IncompressibleModel(
               architecture = architecture,
                       grid = grid,
                   coriolis = FPlane(f=1e-4),
                   buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState()),
                    closure = IsotropicDiffusivity(ν=1e-2, κ=1e-2),
        boundary_conditions = (u=u_bcs, T=T_bcs, S=S_bcs)
    )

    Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz)
    Tᵢ(x, y, z) = 20 + dTdz * z + dTdz * model.grid.Lz * 1e-6 * Ξ(z)
    uᵢ(x, y, z) = sqrt(abs(Qᵘ)) * 1e-3 * Ξ(z)
    set!(model, u=uᵢ, w=uᵢ, T=Tᵢ, S=35)

    wizard = TimeStepWizard(cfl=1.0, Δt=10.0, max_change=1.1, max_Δt=1minute)
    simulation = Simulation(model, Δt=wizard, stop_time=2minutes, iteration_interval=1,
                            progress= sim -> @info "Time $(sim.model.clock.time)")

    # LOTS OF OUTPUT

    u, v, w = model.velocities

    computed_fields = (
        b = BuoyancyField(model),
        ζ = ComputedField(∂x(v) - ∂y(u)),
        ke = ComputedField(√(u^2 + v^2))
    )

    fields_to_output = merge(model.velocities, model.tracers, computed_fields)

    simulation.output_writers[:jld2_3d_with_halos] =
        JLD2OutputWriter(model, fields_to_output,
                  prefix = "test_3d_output_with_halos",
            field_slicer = FieldSlicer(with_halos=true),
                schedule = TimeInterval(30seconds),
                   force = true)

    simulation.output_writers[:jld2_3d_without_halos] =
        JLD2OutputWriter(model, fields_to_output,
              prefix = "test_3d_output_without_halos",
        field_slicer = FieldSlicer(with_halos=false),
            schedule = TimeInterval(30seconds),
               force = true)

    run!(simulation)

    return nothing
end

using OffsetArrays
using JLD2

using DimensionalData: AbstractDimArray, X, Y, Z, Ti

using Oceananigans.Grids: topology, halo_size, all_indices, xnode, ynode, znode

import DimensionalData

struct FieldTimeSeries{X, Y, Z, A, D, Δ, G, FT, B, M} <: AbstractDimArray{FT, 4, Δ, D}
                   data :: D
                   dims :: Δ
           architecture :: A
                   grid :: G
    boundary_conditions :: B
               metadata :: M

    function FieldTimeSeries{X, Y, Z}(data::D, dims::Δ, arch::A, grid::G, bcs::B, metadata::M) where {X, Y, Z, D, Δ, A, G, B, M}
        FT = eltype(grid)
        return new{X, Y, Z, A, D, Δ, G, FT, B, M}(data, dims, arch, grid, bcs, metadata)
    end
end

DimensionalData.data(f::FieldTimeSeries) = f.data.parent
DimensionalData.name(f::FieldTimeSeries) = f.metadata[:name]

# Doesn't seem like it's used by `DimArray` so maybe let's not worry about `refdims`.
# Main loss seems to be plot labels: https://rafaqz.github.io/DimensionalData.jl/dev/developer/#Data-types-and-the-interface
DimensionalData.refdims(f::FieldTimeSeries) = ()

function FieldTimeSeries(filepath, name; architecture=CPU())
    file = jldopen(filepath)

    grid = file["serialized/grid"]
    TX, TY, TZ = topology(grid)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    iterations = parse.(Int, keys(file["timeseries/t"]))
    times = [file["timeseries/t/$i"] for i in iterations]

    LX, LY, LZ = location = file["timeseries/$name/metadata/location"]

    Nt = length(times)
    data_size = size(file["timeseries/$name/0"])
    underlying_data = zeros(data_size..., Nt)
    data = OffsetArray(underlying_data, -Hx, -Hy, -Hz, 0)

    for (n, iter) in enumerate(iterations)
        data.parent[:, :, :, n] .= file["timeseries/$name/$iter"]
    end

    xs = [xnode(LX(), i, grid) for i in all_indices(LX, TX, Nx, Hx)]
    ys = [xnode(LY(), j, grid) for j in all_indices(LY, TY, Ny, Hy)]
    zs = [xnode(LZ(), k, grid) for k in all_indices(LZ, TZ, Nz, Hz)]

    xs_offset = OffsetArray(xs, -Hx)
    ys_offset = OffsetArray(ys, -Hy)
    zs_offset = OffsetArray(zs, -Hz)

    x_dim = X(xs_offset)
    y_dim = Y(ys_offset)
    z_dim = Z(zs_offset)
    t_dim = Ti(times)
    dims = (x_dim, y_dim, z_dim, t_dim)

    bcs = file["timeseries/$name/metadata/boundary_conditions"]
    metadata = Dict(:name => name)

    close(file)

    return FieldTimeSeries{LX, LY, LZ}(data, dims, architecture, grid, bcs, metadata)
end
