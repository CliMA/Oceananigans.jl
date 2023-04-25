include("dependencies_for_runtests.jl")

using Oceananigans.Fields: TracerFields

using Oceananigans.BuoyancyModels:
    required_tracers, ρ′, ∂x_b, ∂y_b,
    thermal_expansionᶜᶜᶜ, thermal_expansionᶠᶜᶜ, thermal_expansionᶜᶠᶜ, thermal_expansionᶜᶜᶠ,
    haline_contractionᶜᶜᶜ, haline_contractionᶠᶜᶜ, haline_contractionᶜᶠᶜ, haline_contractionᶜᶜᶠ,
    x_dot_g_bᶠᶜᶜ, y_dot_g_bᶜᶠᶜ, z_dot_g_bᶜᶜᶠ

function instantiate_linear_equation_of_state(FT, α, β)
    eos = LinearEquationOfState(FT, thermal_expansion=α, haline_contraction=β)
    return eos.thermal_expansion == FT(α) && eos.haline_contraction == FT(β)
end

function instantiate_seawater_buoyancy(FT, EquationOfState; kwargs...)
    buoyancy = SeawaterBuoyancy(FT, equation_of_state=EquationOfState(FT); kwargs...)
    return typeof(buoyancy.gravitational_acceleration) == FT
end

function density_perturbation_works(arch, FT, eos)
    grid = RectilinearGrid(arch, FT, size=(3, 3, 3), extent=(1, 1, 1))
    C = TracerFields((:T, :S), grid)
    density_anomaly = CUDA.@allowscalar ρ′(2, 2, 2, grid, eos, C.T, C.S)
    return true
end

function ∂x_b_works(arch, FT, buoyancy)
    grid = RectilinearGrid(arch, FT, size=(3, 3, 3), extent=(1, 1, 1))
    C = TracerFields(required_tracers(buoyancy), grid)
    dbdx = CUDA.@allowscalar ∂x_b(2, 2, 2, grid, buoyancy, C)
    return true
end

function ∂y_b_works(arch, FT, buoyancy)
    grid = RectilinearGrid(arch, FT, size=(3, 3, 3), extent=(1, 1, 1))
    C = TracerFields(required_tracers(buoyancy), grid)
    dbdy = CUDA.@allowscalar ∂y_b(2, 2, 2, grid, buoyancy, C)
    return true
end

function ∂z_b_works(arch, FT, buoyancy)
    grid = RectilinearGrid(arch, FT, size=(3, 3, 3), extent=(1, 1, 1))
    C = TracerFields(required_tracers(buoyancy), grid)
    dbdz = CUDA.@allowscalar ∂z_b(2, 2, 2, grid, buoyancy, C)
    return true
end

function thermal_expansion_works(arch, FT, eos)
    grid = RectilinearGrid(arch, FT, size=(3, 3, 3), extent=(1, 1, 1))
    C = TracerFields((:T, :S), grid)
    α = CUDA.@allowscalar thermal_expansionᶜᶜᶜ(2, 2, 2, grid, eos, C.T, C.S)
    α = CUDA.@allowscalar thermal_expansionᶠᶜᶜ(2, 2, 2, grid, eos, C.T, C.S)
    α = CUDA.@allowscalar thermal_expansionᶜᶠᶜ(2, 2, 2, grid, eos, C.T, C.S)
    α = CUDA.@allowscalar thermal_expansionᶜᶜᶠ(2, 2, 2, grid, eos, C.T, C.S)
    return true
end

function haline_contraction_works(arch, FT, eos)
    grid = RectilinearGrid(arch, FT, size=(3, 3, 3), extent=(1, 1, 1))
    C = TracerFields((:T, :S), grid)
    β = CUDA.@allowscalar haline_contractionᶜᶜᶜ(2, 2, 2, grid, eos, C.T, C.S)
    β = CUDA.@allowscalar haline_contractionᶠᶜᶜ(2, 2, 2, grid, eos, C.T, C.S)
    β = CUDA.@allowscalar haline_contractionᶜᶠᶜ(2, 2, 2, grid, eos, C.T, C.S)
    β = CUDA.@allowscalar haline_contractionᶜᶜᶠ(2, 2, 2, grid, eos, C.T, C.S)
    return true
end


function tilted_gravity_works(arch, FT)
    grid = RectilinearGrid(arch, FT, size=(3, 3, 3), x=(0, 1), y=(0, 1), z=(0, 1),
                           topology=(Periodic, Bounded, Bounded))
    N² = 1e-5
    g̃₁ = (0, 0, 1)
    g̃₂ = (0, 1, 0)
    buoyancy₁ = Buoyancy(model=BuoyancyTracer(), gravity_unit_vector=g̃₁)
    buoyancy₂ = Buoyancy(model=BuoyancyTracer(), gravity_unit_vector=g̃₂)

    BC = GradientBoundaryCondition(N²)
    BC_b1 = FieldBoundaryConditions(bottom=BC, top=BC)
    BC_b2 = FieldBoundaryConditions(south=BC, north=BC)

    model₁ = NonhydrostaticModel(
                       grid = grid,
                   buoyancy = buoyancy₁,
                    tracers = :b,
                    closure = nothing,
        boundary_conditions = (b=BC_b1,)
    )

    model₂ = NonhydrostaticModel(
                       grid = grid,
                   buoyancy = buoyancy₂,
                    tracers = :b,
                    closure = nothing,
        boundary_conditions = (b=BC_b2,)
    )

    b₁(x, y, z) = N² * (y*g̃₁[2] + z*g̃₁[3])
    b₂(x, y, z) = N² * (y*g̃₂[2] + z*g̃₂[3])
    set!(model₁, b=b₁)
    set!(model₂, b=b₂)

    # These have to be taken at the middle point of domain that has an odd-number size
    @test x_dot_g_bᶠᶜᶜ(2, 2, 2, grid, model₁.buoyancy, model₁.tracers) ≈ x_dot_g_bᶠᶜᶜ(2, 2, 2, grid, model₂.buoyancy, model₂.tracers) ≈ 0
    @test y_dot_g_bᶜᶠᶜ(2, 2, 2, grid, model₁.buoyancy, model₁.tracers) ≈ z_dot_g_bᶜᶜᶠ(2, 2, 2, grid, model₂.buoyancy, model₂.tracers) ≈ 0
    @test z_dot_g_bᶜᶜᶠ(2, 2, 2, grid, model₁.buoyancy, model₁.tracers) ≈ y_dot_g_bᶜᶠᶜ(2, 2, 2, grid, model₂.buoyancy, model₂.tracers) ≈ -N² * ynode(2, grid, Face())

    return nothing
end

EquationsOfState = (LinearEquationOfState, SeawaterPolynomials.RoquetEquationOfState, SeawaterPolynomials.TEOS10EquationOfState)
buoyancy_kwargs = (Dict(), Dict(:constant_salinity=>35.0), Dict(:constant_temperature=>20.0))

@testset "BuoyancyModels" begin
    @info "Testing buoyancy..."

#    @testset "Equations of State" begin
#        @info "  Testing equations of state..."
#        for FT in float_types
#            @test instantiate_linear_equation_of_state(FT, 0.1, 0.3)
#
#            for EOS in EquationsOfState
#                for kwargs in buoyancy_kwargs
#                    @test instantiate_seawater_buoyancy(FT, EOS; kwargs...)
#                end
#            end
#
#            for arch in archs
#                @test density_perturbation_works(arch, FT, SeawaterPolynomials.RoquetEquationOfState())
#            end
#
#            buoyancies = (nothing, Buoyancy(model=BuoyancyTracer()), Buoyancy(model=SeawaterBuoyancy(FT)),
#                          (Buoyancy(model=SeawaterBuoyancy(FT, equation_of_state=eos(FT))) for eos in EquationsOfState)...)
#
#            for arch in archs
#                for buoyancy in buoyancies
#                    @test ∂x_b_works(arch, FT, buoyancy)
#                    @test ∂y_b_works(arch, FT, buoyancy)
#                    @test ∂z_b_works(arch, FT, buoyancy)
#                end
#            end
#
#            for arch in archs
#                for EOS in EquationsOfState
#                    @test thermal_expansion_works(arch, FT, EOS())
#                    @test haline_contraction_works(arch, FT, EOS())
#                end
#            end
#        end
#    end

    @testset "Tilted buoyancy" begin
        @info "  Testing tilted buoyancy..."
        for FT in float_types

            # test constructor
            buoyancy_models = (BuoyancyTracer, SeawaterBuoyancy)
            for arch in archs
                for buoyancy_model in buoyancy_models
                    @test Buoyancy(model=buoyancy_model(), gravity_unit_vector=(0, 0, 1)).model isa buoyancy_model
                end
            end

            for arch in archs
                tilted_gravity_works(arch, FT)
            end
        end
    end
end
