using Oceananigans: AbstractModel, AbstractOutputWriter, AbstractDiagnostic

using Oceananigans.Architectures: AbstractArchitecture
using Oceananigans.Advection: CenteredSecondOrder
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions

using Oceananigans.BoundaryConditions: UVelocityBoundaryConditions,
                                       VVelocityBoundaryConditions,
                                       TracerBoundaryConditions

using Oceananigans.Fields: Field, tracernames, TracerFields, XFaceField, YFaceField, CenterField
using Oceananigans.Forcings: model_forcing
using Oceananigans.Grids: with_halo
using Oceananigans.TimeSteppers: Clock, TimeStepper
using Oceananigans.TurbulenceClosures: ν₀, κ₀, with_tracers, DiffusivityFields, IsotropicDiffusivity
using Oceananigans.Utils: inflate_halo_size, tupleit

struct ConservativeSolution end
struct PrimitiveSolutionLinearizedHeight end

const ConservativeSolutionFields = NamedTuple{(:uh, :vh, :h)}
const PrimitiveSolutionLinearizedHeightFields = NamedTuple{(:u, :v, :η)}

function ShallowWaterSolutionFields(::ConservativeSolution, arch, grid, bcs)

    uh_bcs = :uh ∈ keys(bcs) ? bcs.uh : UVelocityBoundaryConditions(grid)
    vh_bcs = :vh ∈ keys(bcs) ? bcs.vh : VVelocityBoundaryConditions(grid)
    h_bcs  = :h  ∈ keys(bcs) ? bcs.h  : TracerBoundaryConditions(grid)

    uh = XFaceField(arch, grid, uh_bcs)
    vh = YFaceField(arch, grid, vh_bcs)
    h = CenterField(arch, grid, h_bcs)

    return (uh=uh, vh=vh, h=h)
end

function ShallowWaterSolutionFields(::PrimitiveSolutionLinearizedHeight, arch, grid, bcs)
    u_bcs = :u ∈ keys(bcs) ? bcs.u : UVelocityBoundaryConditions(grid)
    v_bcs = :v ∈ keys(bcs) ? bcs.v : VVelocityBoundaryConditions(grid)
    η_bcs  = :η  ∈ keys(bcs) ? bcs.η  : TracerBoundaryConditions(grid)

    u = XFaceField(arch, grid, u_bcs)
    v = YFaceField(arch, grid, v_bcs)
    η = CenterField(arch, grid, η_bcs)

    return (u=u, v=v, η=η)
end

function ShallowWaterTendencyFields(solution, arch, grid, tracer_names)

    solution_tendencies_tuple = (XFaceField(arch, grid, UVelocityBoundaryConditions(grid)),
                                 YFaceField(arch, grid, VVelocityBoundaryConditions(grid)),
                                 CenterField(arch,  grid, TracerBoundaryConditions(grid)))

    solution_names = propertynames(solution)
    solution_tendencies = NamedTuple{solution_names}(solution_tendencies_tuple)

    tracer_tendencies = TracerFields(tracer_names, arch, grid)

    return merge(solution_tendencies, tracer_tendencies)
end

#####
##### ShallowWaterModel
#####

struct ShallowWaterModel{G, A<:AbstractArchitecture, T, V, R, F, E, B, Q, C, K, TS} <: AbstractModel{TS}

                          grid :: G         # Grid of physical points on which `Model` is solved
                  architecture :: A         # Computer `Architecture` on which `Model` is run
                         clock :: Clock{T}  # Tracks iteration number and simulation time of `Model`
    gravitational_acceleration :: T         # Gravitational acceleration, full, or reduced
                     advection :: V         # Advection scheme for velocities _and_ tracers
                      coriolis :: R         # Set of parameters for the background rotation rate of `Model`
                       forcing :: F         # Container for forcing functions defined by the user
                       closure :: E         # Diffusive 'turbulence closure' for all model fields
                    bathymetry :: B         # Bathymetry/Topography for the model
                      solution :: Q         # Container for transports `uh`, `vh`, and height `h`
                       tracers :: C         # Container for tracer fields
                 diffusivities :: K         # Container for turbulent diffusivities
                   timestepper :: TS        # Object containing timestepper fields and parameters

end

function ShallowWaterModel(;
                           grid,
                           gravitational_acceleration,
  architecture::AbstractArchitecture = CPU(),
                               clock = Clock{eltype(grid)}(0, 0, 1),
                           advection = UpwindBiasedFifthOrder(),
                            coriolis = nothing,
                 forcing::NamedTuple = NamedTuple(),
                             closure = nothing,
                          bathymetry = nothing,
                            solution = ConservativeSolution(),
                             tracers = (),
                       diffusivities = nothing,
     boundary_conditions::NamedTuple = NamedTuple(),
                 timestepper::Symbol = :RungeKutta3)

    grid.Nz == 1 || throw(ArgumentError("ShallowWaterModel must be constructed with Nz=1!"))

    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)

    Hx, Hy, Hz = inflate_halo_size(grid.Hx, grid.Hy, grid.Hz, advection)
    grid = with_halo((Hx, Hy, Hz), grid)

    boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, nothing)

    solution = ShallowWaterSolutionFields(solution, architecture, grid, boundary_conditions)
    tracers  = TracerFields(tracers, architecture, grid, boundary_conditions)
    diffusivities = DiffusivityFields(diffusivities, architecture, grid,
                                      tracernames(tracers), boundary_conditions, closure)

    # Instantiate timestepper if not already instantiated
    timestepper = TimeStepper(timestepper, architecture, grid, tracernames(tracers);
                              Gⁿ = ShallowWaterTendencyFields(solution, architecture, grid, tracernames(tracers)),
                              G⁻ = ShallowWaterTendencyFields(solution, architecture, grid, tracernames(tracers)))

    # Regularize forcing and closure for model tracer and velocity fields.
    model_fields = merge(solution, tracers)
    forcing = model_forcing(model_fields; forcing...)
    closure = with_tracers(tracernames(tracers), closure)

    return ShallowWaterModel(grid,
                             architecture,
                             clock,
                             eltype(grid)(gravitational_acceleration),
                             advection,
                             coriolis,
                             forcing,
                             closure,
                             bathymetry,
                             solution,
                             tracers,
                             diffusivities,
                             timestepper)
end
