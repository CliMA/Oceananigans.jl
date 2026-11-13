include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Advection: div_Uc, materialize_advection, update_advection!
using Oceananigans.ImmersedBoundaries: CutCellBottom

function advective_tendency(grid, advection, c, U)
    scheme = materialize_advection(advection, grid)
    update_advection!(scheme, (; grid), c)
    operation = KernelFunctionOperation{Center, Center, Center}(div_Uc, grid, scheme, U, c)
    return Array(interior(compute!(Field(operation))))
end

function constant_velocities(grid; u=0, v=0, w=0)
    U = (u=XFaceField(grid), v=YFaceField(grid), w=ZFaceField(grid))
    set!(U.u, u)
    set!(U.v, v)
    set!(U.w, w)
    foreach(fill_halo_regions!, U)
    return U
end

slope(s) = -1250 - 750tanh((s - 40e3) / 7e3)

# Densest water, in units of the source's, of a dense overflow down a steep slope (Ilicak et al. 2012) at 4 km resolution after 8 hours
function densest_overflow_water(arch, bottom_kind)
    underlying = RectilinearGrid(arch, size=(50, 20), halo=(6, 6), x=(0, 200e3), z=(-2000, 0), topology=(Bounded, Flat, Bounded))
    grid = ImmersedBoundaryGrid(underlying, bottom_kind(slope))
    closure = (HorizontalScalarDiffusivity(ν=100), VerticalScalarDiffusivity(ν=1e-4))
    model = HydrostaticFreeSurfaceModel(grid; tracer_advection=THINCBVD(), momentum_advection=UpwindBiased(order=3), closure,
                                        buoyancy=BuoyancyTracer(), tracers=:b, timestepper=:SplitRungeKutta3,
                                        free_surface=SplitExplicitFreeSurface(grid; substeps=40))
    Δb = 0.0196
    set!(model, b=(x, z) -> -Δb * (x < 20e3))
    densest = 0.0
    for _ in 1:960
        time_step!(model, 30)
        densest = max(densest, -minimum(interior(model.tracers.b)) / Δb)
    end
    return densest
end

@testset "THINC-BVD advection" begin
    for arch in archs
        @testset "Smooth tracer takes WENO's tendency [$(summary(arch)), $steepnesses]" for steepnesses in ((1.6, 2.5), (1.25, 2.5))
            grid = RectilinearGrid(arch, size=(16, 16, 16), halo=(5, 5, 5), extent=(1, 1, 1), topology=(Periodic, Periodic, Periodic))
            c = CenterField(grid)
            set!(c, (x, y, z) -> sin(2π * x) * cos(2π * y) * sin(2π * z))
            fill_halo_regions!(c)
            U = constant_velocities(grid; u=1, v=0.7, w=-0.3)
            thinc = advective_tendency(grid, THINCBVD(; steepnesses), c, U)
            weno = advective_tendency(grid, WENO(order=7), c, U)
            @test maximum(abs, thinc .- weno) ≤ 1e-12 * maximum(abs, weno)
        end

        @testset "y tendency over an immersed slope is the rotated x tendency [$(summary(arch))]" begin
            front(s, z) = -0.0196 * (s < 30e3) * (z < -900) + 1e-4 * sin(s / 3e3 + z / 300)
            gx = ImmersedBoundaryGrid(RectilinearGrid(arch, size=(50, 20), halo=(6, 6), x=(0, 200e3), z=(-2000, 0), topology=(Bounded, Flat, Bounded)), GridFittedBottom(slope))
            gy = ImmersedBoundaryGrid(RectilinearGrid(arch, size=(50, 20), halo=(6, 6), y=(0, 200e3), z=(-2000, 0), topology=(Flat, Bounded, Bounded)), GridFittedBottom(slope))
            cx, cy = CenterField(gx), CenterField(gy)
            set!(cx, front)
            set!(cy, front)
            fill_halo_regions!(cx)
            fill_halo_regions!(cy)
            tx = advective_tendency(gx, THINCBVD(), cx, constant_velocities(gx; u=0.5, w=1e-3))
            ty = advective_tendency(gy, THINCBVD(), cy, constant_velocities(gy; v=0.5, w=1e-3))
            @test maximum(abs, vec(tx) .- vec(ty)) ≤ 1e-12 * maximum(abs, tx)
        end

        @testset "Overflow stays within 10% of its source density [$(summary(arch)), $bottom_kind]" for bottom_kind in (GridFittedBottom, PartialCellBottom, CutCellBottom)
            @test densest_overflow_water(arch, bottom_kind) < 1.1
        end
    end
end
