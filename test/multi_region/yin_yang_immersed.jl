include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Grids: immersed_inactive_node
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.MultiRegion: yin_yang_interpolation
using Oceananigans.Utils: getregion

const faces = Float32[-4, -2, -1/2, 0]

regional_bottom(grid, region) = on_architecture(CPU(), getregion(grid, region).immersed_boundary.bottom_height)
regional_maximum(field, indices...) = maximum(r -> maximum(abs, interior(getregion(field, r), indices...)), 1:2)
receiver_indices(table, k) = CartesianIndex.(table.receivers.i, table.receivers.j, k)

# A partial bottom cell is at least ϵ times as tall as its full cell
valid_partial_bottom(h, ϵ) = faces[1] ≤ h ≤ faces[end] &&
    all(k -> !(faces[k] ≤ h < faces[k+1]) || h ≤ faces[k+1] - ϵ * (faces[k+1] - faces[k]), 1:length(faces)-1)

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

    ϵ = 0.2f0
    partial_bottom = PartialCellBottom((λ, φ) -> ifelse(φ > 0, 0, -1.9f0); minimum_fractional_cell_height=ϵ)
    partial_grid = ImmersedBoundaryGrid(sphere, partial_bottom)
    @test all(h -> valid_partial_bottom(h, ϵ), regional_bottom(partial_grid, 1))
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

@inline wet_constant_tracer_error(i, j, k, grid, tracer) =
    ifelse(immersed_inactive_node(i, j, k, grid, Center(), Center(), Center()), zero(grid), tracer[i, j, k] - 1)

@testset "Immersed z-star time stepping [$(typeof(arch))]" for arch in archs
    Nφ, H, R = 16, 1000, 1e6
    z = MutableVerticalDiscretization((-H, 0))
    sphere = YinYangGrid(arch, Float32; size=(3Nφ, Nφ, 4), z, radius=R, halo=(4, 4, 4))
    bottom(λ, φ) = ifelse((abs(λ) < 15) & (φ > -35), zero(φ), oftype(φ, -1000))
    grid = ImmersedBoundaryGrid(sphere, GridFittedBottom(bottom))
    model = HydrostaticFreeSurfaceModel(grid; tracers=:c, buoyancy=nothing, momentum_advection=nothing,
                                        free_surface=SplitExplicitFreeSurface(grid; substeps=4, extend_halos=false),
                                        timestepper=:SplitRungeKutta3, vertical_coordinate=ZStarCoordinate())
    set!(model, u=(λ, φ, z) -> cosd(φ), v=0, c=1)
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
