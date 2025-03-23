include("reactant_test_utils.jl")

@testset "Reactanigans unit tests" begin
    @info "Performing Reactanigans unit tests..."
    
    arch = ReactantState()
    times = 0:1.0:4
    t = 2.1
    times = Reactant.to_rarray(times)
    @test times isa Reactant.TracedRNumberOverrides.TracedStepRangeLen

    ñ, n₁, n₂ = @jit Oceananigans.OutputReaders.find_time_index(times, t)
    @test ñ ≈ 0.1
    @test n₁ == 3 # eg times = [0 1 2 ⟨⟨2.1⟩⟩ 3]
    @test n₂ == 4 

    grid = RectilinearGrid(arch; size=(4, 4, 4), extent=(1, 1, 1))
    c = CenterField(grid)
    @test parent(c) isa Reactant.ConcreteRArray

    cpu_grid = on_architecture(CPU(), grid)
    @test architecture(cpu_grid) isa CPU

    cpu_c = on_architecture(CPU(), c)
    @test parent(cpu_c) isa Array
    @test architecture(cpu_c.grid) isa CPU

    @info "  Testing field set! with a number..."
    set!(c, 1)
    @test all(c .≈ 1)

    @info "  Testing field set! with a function..."
    set!(c, (x, y, z) -> 1)
    @test all(c .≈ 1)

    @info "  Testing field set! with an array..."
    a = ones(size(c)...)
    set!(c, a)
    @test all(c .≈ 1)

    @info "  Testing simple kernel launch!..."
    add_one!(c)
    @test all(c .≈ 2)

    set!(c, (x, y, z) -> x + y * z)
    x, y, z = nodes(c)

    @allowscalar begin
        @test c[1, 1, 1] == x[1] + y[1] * z[1]
        @test c[1, 2, 1] == x[1] + y[2] * z[1]
        @test c[1, 2, 3] == x[1] + y[2] * z[3]
    end

    @jit fill_halo_regions!(c)

    @allowscalar begin
        @test c[1, 1, 0] == c[1, 1, 1]
    end

    d = CenterField(grid)
    parent(d) .= 2

    cd = Field(c * d)
    compute!(cd)

    @allowscalar begin
        @test cd[1, 1, 1] == 2 * (x[1] + y[1] * z[1])
        @test cd[1, 2, 1] == 2 * (x[1] + y[2] * z[1])
        @test cd[1, 2, 3] == 2 * (x[1] + y[2] * z[3])
    end

    # Deconcretization
    c′ = OceananigansReactantExt.deconcretize(c)
    @test parent(c′) isa Array
    @test architecture(c′) isa ReactantState

    for FT in (Float64, Float32)
        sgrid = RectilinearGrid(arch, FT; size=(4, 4, 4), x=[0, 1, 2, 3, 4], y=(0, 1), z=(0, 1))
        @test architecture(sgrid) isa ReactantState

        @test architecture(sgrid.xᶠᵃᵃ) isa ReactantState
        @test architecture(sgrid.xᶜᵃᵃ) isa ReactantState

        llg = LatitudeLongitudeGrid(arch, FT; size = (4, 4, 4),
                                    longitude = [0, 1, 2, 3, 4],
                                    latitude = [0, 1, 2, 3, 4],
                                    z = (0, 1))

        @test architecture(llg) isa ReactantState

        #= The grid is traced, these tests are broken
        for name in propertynames(llg)
            p = getproperty(llg, name)
            if !(name ∈ (:architecture, :z))
                @test (p isa Number) || (p isa OffsetArray{FT, <:Any, <:Array})
            end
        end
        =#

        ridge(λ, φ) = 0.1 * exp((λ - 2)^2 / 2)
        ibg = ImmersedBoundaryGrid(llg, GridFittedBottom(ridge))
        @test architecture(ibg) isa ReactantState

        @test architecture(ibg.immersed_boundary.bottom_height) isa ReactantState

        rllg = RotatedLatitudeLongitudeGrid(arch, FT; size = (4, 4, 4),
                                            north_pole = (0, 0),
                                            longitude = [0, 1, 2, 3, 4],
                                            latitude = [0, 1, 2, 3, 4],
                                            z = (0, 1))

        @test architecture(rllg) isa ReactantState

        #=
        for name in propertynames(rllg)
            p = getproperty(rllg, name)
            if !(name ∈ (:architecture, :z, :conformal_mapping))
                @test (p isa Number) || (p isa OffsetArray{FT, <:Any, <:Array})
            end
        end
        =#
    end
end

@testset "Reactant RectilinearGrid Simulation Tests" begin
    @info "Performing Reactanigans RectilinearGrid simulation tests..."
    Nx, Ny, Nz = (10, 10, 10) # number of cells
    halo = (7, 7, 7)
    z = (-1, 0)
    rectilinear_kw = (; size=(Nx, Ny, Nz), halo, x=(0, 1), y=(0, 1), z=(0, 1))
    hydrostatic_model_kw = (; free_surface=ExplicitFreeSurface(gravitational_acceleration=1))

    @info "Testing RectilinearGrid + HydrostaticFreeSurfaceModel Reactant correctness"
    test_reactant_model_correctness(RectilinearGrid,
                                    HydrostaticFreeSurfaceModel,
                                    rectilinear_kw,
                                    hydrostatic_model_kw)

    @info "Testing immersed RectilinearGrid + HydrostaticFreeSurfaceModel Reactant correctness"
    test_reactant_model_correctness(RectilinearGrid,
                                    HydrostaticFreeSurfaceModel,
                                    rectilinear_kw,
                                    hydrostatic_model_kw,
                                    immersed_boundary_grid=true)
end

