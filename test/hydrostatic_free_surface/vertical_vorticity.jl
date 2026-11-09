include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Models.HydrostaticFreeSurfaceModels: vertical_vorticity, VectorInvariant

@testset "VerticalVorticityField with HydrostaticFreeSurfaceModel" begin

    @testset "VerticalVorticityField with HydrostaticFreeSurfaceModel [$arch]" for arch in archs
        grid = LatitudeLongitudeGrid(arch, size = (3, 3, 3),
                                     longitude = (0, 60),
                                     latitude = (15, 75),
                                     z = (-1, 0))

        model = HydrostaticFreeSurfaceModel(grid; momentum_advection = VectorInvariant())

        ψᵢ(λ, φ, z) = rand()
        set!(model, u=ψᵢ, v=ψᵢ)

        ζ = vertical_vorticity(model)
        @test ζ isa KernelFunctionOperation
        ζ = Field(ζ)
        compute!(ζ)
        @test all(isfinite.(ζ.data))
    end
end
