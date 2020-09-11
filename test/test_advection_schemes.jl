function time_step_with_advection_scheme(advection_scheme, arch, FT)
    # Use halos of size 2 to accomadate time stepping with AnisotropicBiharmonicDiffusivity.
    grid = RegularCartesianGrid(FT; size=(1, 1, 1), halo=(2, 2, 2), extent=(1, 2, 3))
    model = IncompressibleModel(grid=grid, architecture=arch, advection=advection_scheme,
                                float_type=FT)

    return true
end

advection_schemes = (CenteredSecondOrder(), CenteredFourthOrder(), UpwindThirdOrder())

@testset "Advection schemes" begin
    @info "  Testing time stepping with advection schemes..."
    for scheme in advection_schemes, arch in archs, FT in float_types
        @test time_step_with_advection_scheme(scheme, arch, FT)
    end
end
