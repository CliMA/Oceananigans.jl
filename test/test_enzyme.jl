include("dependencies_for_runtests.jl")

using Enzyme
using EnzymeCore
# Required presently
Enzyme.API.runtimeActivity!(true)
Enzyme.API.looseTypeAnalysis!(true)
Enzyme.API.maxtypeoffset!(2032)

using Oceananigans
using Oceananigans.TurbulenceClosures: with_tracers
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ConstantField
using Oceananigans.Models.HydrostaticFreeSurfaceModels: tracernames
using Oceananigans.Fields: FunctionField
using Oceananigans: architecture
using KernelAbstractions

using Oceananigans.TimeSteppers: update_state!


EnzymeRules.inactive_type(::Type{<:Oceananigans.Clock}) = true

@testset "Enzyme on advection and diffusion WITH flux boundary condition" begin
    Nx = Ny = 64
    Nz = 8

    Lx = Ly = L = 2π
    Lz = 1

    x = y = (-L/2, L/2)
    z = (-Lz/2, Lz/2)
    topology = (Periodic, Periodic, Bounded)

    grid = RectilinearGrid(size=(Nx, Ny, Nz); x, y, z, topology)
    diffusion = VerticalScalarDiffusivity(κ=0.1)

    @inline function tracer_flux(x, y, t, c)
        return c
    end

    top_c_bc = FluxBoundaryCondition(tracer_flux, field_dependencies=:c)
    c_bcs = FieldBoundaryConditions(top=top_c_bc)

    # TODO:
    # 1. Make the velocity fields evolve
    # 2. Add surface fluxes
    # 3. Do a problem where we invert for the tracer fluxes (maybe with CATKE)

    model = HydrostaticFreeSurfaceModel(; grid,
                                        tracers = :c,
                                        buoyancy = nothing,
                                        boundary_conditions = (; c=c_bcs),
                                        closure = diffusion)


    # Now for real
    amplitude = 1.0
    κ = 1.0
    dmodel = Enzyme.make_zero(model)
    
    dc²_dκ = autodiff(Enzyme.Reverse,
                      update_state!,
                      Duplicated(model, dmodel))
    
    #=
    thing1 = Vector{Tuple{Tuple{BoundaryCondition{C, Nothing} where C<:Oceananigans.BoundaryConditions.AbstractBoundaryConditionClassification, BoundaryCondition{C, Nothing} where C<:Oceananigans.BoundaryConditions.AbstractBoundaryConditionClassification, BoundaryCondition{C, Nothing} where C<:Oceananigans.BoundaryConditions.AbstractBoundaryConditionClassification}, Tuple{BoundaryCondition{C, Nothing} where C<:Oceananigans.BoundaryConditions.AbstractBoundaryConditionClassification, BoundaryCondition{C, Nothing} where C<:Oceananigans.BoundaryConditions.AbstractBoundaryConditionClassification, BoundaryCondition}}}
    thing2 = Int64
    thing3 = Vector{Tuple{Tuple{BoundaryCondition{Flux, Nothing}, BoundaryCondition{Flux, Nothing}, BoundaryCondition{Flux, Nothing}}, Tuple{BoundaryCondition{Flux, Nothing}, BoundaryCondition{Flux, Nothing}, BoundaryCondition{Flux, Oceananigans.BoundaryConditions.ContinuousBoundaryFunction{Center, Center, Nothing, Oceananigans.BoundaryConditions.RightBoundary, var"#tracer_flux#23", Nothing, Tuple{Symbol}, Tuple{Int64}, Tuple{typeof(Oceananigans.Operators.identity4)}}}}}}
    thing4 = Int64
    thing5 = Int64

    autodiff_thunk(ReverseSplitWithPrimal,
                Const{typeof(Base._unsafe_copyto!)},
                Active,
                Duplicated{thing1},
                Const{thing2},
                Duplicated{thing3},
                Const{thing4},
                Const{thing5})
    =#
end
