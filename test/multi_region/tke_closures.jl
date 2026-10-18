include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity, TKEDissipationVerticalDiffusivity
using Oceananigans.Utils: getregion

function stepped_tke(grid, closure, timestepper)
    free_surface = SplitExplicitFreeSurface(substeps=10, extend_halos=!(grid isa YinYangGrid))
    boundary_conditions = (u = FieldBoundaryConditions(top=FluxBoundaryCondition(-1e-4)),
                           T = FieldBoundaryConditions(top=FluxBoundaryCondition(1e-5)))

    model = HydrostaticFreeSurfaceModel(grid; closure, timestepper, free_surface, boundary_conditions,
                                        tracers=(:T, :S), buoyancy=SeawaterBuoyancy())

    set!(model, T=(x, y, z) -> 20 + z / 10, S=35)

    for _ in 1:3
        time_step!(model, 60)
    end

    return model.tracers.e
end

@testset "TKE-based closures on multi-region grids [$(summary(arch))]" for arch in archs
    latitude_longitude_grid = LatitudeLongitudeGrid(arch; size=(32, 8, 4), longitude=(0, 80), latitude=(-20, 20), z=(-100, 0))
    partitioned_grid = MultiRegionGrid(latitude_longitude_grid, partition=XPartition(2))
    yin_yang_grid = YinYangGrid(arch; size=(48, 16, 4), z=(-100, 0))

    for (closure, timestepper) in ((CATKEVerticalDiffusivity(), :SplitRungeKutta3),
                                   (TKEDissipationVerticalDiffusivity(), :QuasiAdamsBashforth2))

        e = on_architecture(CPU(), interior(stepped_tke(latitude_longitude_grid, closure, timestepper)))
        @test maximum(e) > 0

        # Each region of the partitioned grid holds the same TKE as the whole grid
        partitioned_e = stepped_tke(partitioned_grid, closure, timestepper)
        for region in 1:2
            regional_e = on_architecture(CPU(), interior(getregion(partitioned_e, region)))
            @test regional_e ≈ e[16(region - 1) .+ (1:16), :, :]
        end

        # The surface fluxes are uniform, so columns far from walls and seams agree
        if closure isa CATKEVerticalDiffusivity
            yin_yang_e = stepped_tke(yin_yang_grid, closure, timestepper)
            for region in 1:2
                regional_e = on_architecture(CPU(), interior(getregion(yin_yang_e, region)))
                @test regional_e[27, 11, :] ≈ e[16, 4, :]
            end
        end
    end
end
