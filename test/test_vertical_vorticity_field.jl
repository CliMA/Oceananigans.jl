using Oceananigans.Models.HydrostaticFreeSurfaceModels: VerticalVorticityField, VectorInvariant

@testset "VerticalVorticityField with HydrostaticFreeSurfaceModel" begin

    for arch in archs
        @testset "VerticalVorticityField with HydrostaticFreeSurfaceModel [$arch]" begin
            @info "  Testing VerticalVorticityField with HydrostaticFreeSurfaceModel [$arch]..."

            grid = LatitudeLongitudeGrid(size = (3, 3, 3),
                                         longitude = (0, 60),
                                         latitude = (15, 75),
                                         z = (-1, 0))

            model = HydrostaticFreeSurfaceModel(grid = grid,
                                                momentum_advection = VectorInvariant(),
                                                architecture = arch)

            ψᵢ(λ, φ, z) = rand()
            set!(model, u=ψᵢ, v=ψᵢ)

            ζ = VerticalVorticityField(model)

            compute!(ζ)

            @test all(isfinite.(ζ.data))
        end
    end
end
