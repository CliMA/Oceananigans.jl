include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Grids: immersed_peripheral_node, immersed_inactive_node
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.MultiRegion: yin_yang_interpolation
using Oceananigans.Utils: getregion

const faces = Float32[-4, -2, -1/2, 0]

regional_bottom(grid, region) = on_architecture(CPU(), getregion(grid, region).immersed_boundary.bottom_height)
regional_maximum(field, indices...) = maximum(r -> maximum(abs, interior(getregion(field, r), indices...)), 1:2)
receiver_indices(table, k) = CartesianIndex.(table.receivers.i, table.receivers.j, k)

@testset "Grid-fitted bathymetry and coastal tracer halos [$(typeof(arch)), $s×$s]" for arch in archs, s in (2, 4)
    sphere = YinYangGrid(arch, Float32; size=(48, 16, 3), z=faces, stencil_width=s)
    grid = ImmersedBoundaryGrid(sphere, GridFittedBottom((λ, φ) -> ifelse(φ > 0, 0, -4)))
    tracer = CenterField(grid)
    set!(tracer, 1)
    for region in 1:2
        mask_immersed_field!(getregion(tracer, region))
    end
    fill_halo_regions!(tracer)

    table = on_architecture(CPU(), yin_yang_interpolation(grid).center_center)
    for region in 1:2
        bottom = regional_bottom(grid, region)
        @test extrema(bottom) == (first(faces), last(faces))
        @test all(height -> height ∈ faces, bottom)

        # A uniform wet tracer is interpolated onto the wet receivers to rounding error
        wet = bottom[receiver_indices(table, 1)] .< -1/2
        data = on_architecture(CPU(), getregion(tracer, region).data)
        @test any(wet)
        @test all(value -> isapprox(value, 1; atol=2f-6), data[receiver_indices(table, grid.Nz)[wet]])
    end
end

# A partial bottom cell is at least ϵ times as tall as its full cell
valid_partial_bottom(h, ϵ) = faces[1] ≤ h ≤ faces[end] &&
    all(k -> !(faces[k] ≤ h < faces[k+1]) || h ≤ faces[k+1] - ϵ * (faces[k+1] - faces[k]), 1:length(faces)-1)

@testset "Partial-cell bathymetry halos [$(typeof(arch))]" for arch in archs
    ϵ = 0.2f0
    sphere = YinYangGrid(arch, Float32; size=(48, 16, 3), z=faces)
    bottom = PartialCellBottom((λ, φ) -> ifelse(φ > 0, 0, -1.9f0); minimum_fractional_cell_height=ϵ)
    grid = ImmersedBoundaryGrid(sphere, bottom)
    for region in 1:2
        @test all(h -> valid_partial_bottom(h, ϵ), regional_bottom(grid, region))
    end
end

@testset "Impermeable immersed vector halos [$(typeof(arch)), $LZ]" for arch in archs, LZ in (Center, Nothing)
    sphere = YinYangGrid(arch, Float32; size=(48, 16, 3), z=faces)
    grid = ImmersedBoundaryGrid(sphere, GridFittedBottom((λ, φ) -> 0))
    u, v = Field{Face, Center, LZ}(grid), Field{Center, Face, LZ}(grid)
    set!(u, 1)
    set!(v, 1)
    fill_halo_regions!((u, v))

    tables = on_architecture(CPU(), yin_yang_interpolation(grid))
    for region in 1:2, (field, table) in ((u, tables.face_center), (v, tables.center_face))
        data = on_architecture(CPU(), getregion(field, region).data)
        available = field === u ? table.receivers.i .> 1-grid.Hx : table.receivers.j .> 1-grid.Hy
        @test all(iszero, data[receiver_indices(table, 1)[available]])
        @test maximum(interior(getregion(field, region))) == 1
    end

    boundary = GridFittedBoundary((λ, φ, z) -> z < -2)
    @test_throws "GridFittedBoundary is not supported" ImmersedBoundaryGrid(sphere, boundary)
end

@inline masked_initial_velocity(i, j, k, grid, velocity, loc) =
    ifelse(immersed_peripheral_node(i, j, k, grid, loc...), velocity[i, j, k], zero(grid))

@inline function masked_initial_transport(i, j, k, grid, velocity, loc)
    immersed = true
    for level in 1:grid.Nz
        immersed &= immersed_peripheral_node(i, j, level, grid, loc...)
    end
    return ifelse(immersed, velocity[i, j, 1], zero(grid))
end

@inline wet_constant_tracer_error(i, j, k, grid, tracer) =
    ifelse(immersed_inactive_node(i, j, k, grid, Center(), Center(), Center()), zero(grid), tracer[i, j, k] - 1)

@testset "Immersed multiregion initialization [$(typeof(arch))]" for arch in archs
    Nφ, H, R = 16, 1000, 1e6
    z = MutableVerticalDiscretization((-H, 0))
    sphere = YinYangGrid(arch, Float32; size=(3Nφ, Nφ, 4), z, radius=R, halo=(4, 4, 4))
    bottom(λ, φ) = ifelse((abs(λ) < 15) & (φ > -35), zero(φ), oftype(φ, -1000))
    grid = ImmersedBoundaryGrid(sphere, GridFittedBottom(bottom))
    model = HydrostaticFreeSurfaceModel(grid; tracers=:c, buoyancy=nothing, momentum_advection=nothing,
                                        free_surface=SplitExplicitFreeSurface(grid; substeps=4, extend_halos=false),
                                        timestepper=:SplitRungeKutta3, vertical_coordinate=ZStarCoordinate())
    set!(model, u=(λ, φ, z) -> cosd(φ), v=0, c=1)
    u, v, _ = model.velocities
    U, V = model.free_surface.barotropic_velocities
    for (Lx, Ly, velocity, transport) in ((Face, Center, u, U), (Center, Face, v, V))
        loc = (Lx(), Ly(), Center())
        masked_velocity = Field(KernelFunctionOperation{Lx, Ly, Center}(masked_initial_velocity, grid, velocity, loc))
        masked_transport = Field(KernelFunctionOperation{Lx, Ly, Nothing}(masked_initial_transport, grid, transport, loc))
        for masked in (masked_velocity, masked_transport)
            compute!(masked)
            @test regional_maximum(masked) == 0
        end
    end
    @test regional_maximum(u) > 0.5

    Nt = 10
    for step in 1:Nt
        time_step!(model, 60f0)
    end

    # The uniform tracer and the top transport velocity differ from 1 and 0 by rounding error alone: a few ulps
    # per step, and a few ulps of the vertical velocity H / Δ of the unit horizontal velocity across a cell of width Δ
    error = Field(KernelFunctionOperation{Center, Center, Center}(wet_constant_tracer_error, grid, model.tracers.c))
    compute!(error)
    Δ = R * π / 2 / Nφ
    @test regional_maximum(error) < Nt * eps(Float32)
    @test regional_maximum(model.transport_velocities.w, :, :, grid.Nz+1) < 10 * eps(Float32) * H / Δ
end
