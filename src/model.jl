mutable struct Model
    metadata::ModelMetadata
    configuration::ModelConfiguration
    boundary_conditions::BoundaryConditions
    constants::PlanetaryConstants
    eos::EquationOfState
    grid::Grid
    velocities::VelocityFields
    tracers::TracerFields
    pressures::PressureFields
    G::SourceTerms
    Gp::SourceTerms
    forcings::ForcingFields
    stepper_tmp::StepperTemporaryFields
    # operator_tmp::OperatorTemporaryFields
    ssp  # ::SpectralSolverParameters (for arch==:cpu only...)
    clock::Clock
    output_writers::Array{OutputWriter,1}
    diagnostics::Array{Diagnostic,1}
end

function Model(N, L, arch=:cpu, float_type=Float64)
    metadata = _ModelMetadata(arch, float_type)
    configuration = _ModelConfiguration(4e-2, 4e-2, 4e-2, 4e-2)
    boundary_conditions = BoundaryConditions(:periodic, :periodic, :rigid_lid, :free_slip)

    constants = Earth()
    eos = LinearEquationOfState()

    grid = RegularCartesianGrid(metadata, N, L)

    velocities  = VelocityFields(metadata, grid)
    tracers = TracerFields(metadata, grid)
    pressures = PressureFields(metadata, grid)
    G  = SourceTerms(metadata, grid)
    Gp = SourceTerms(metadata, grid)
    forcings = ForcingFields(metadata, grid)
    stepper_tmp = StepperTemporaryFields(metadata, grid)
    # operator_tmp = OperatorTemporaryFields(metadata, grid)

    time, time_step, Δt = 0, 0, 0
    clock = Clock(time, time_step, Δt)

    output_writers = OutputWriter[]
    diagnostics = Diagnostic[]

    if metadata.arch == :cpu
        stepper_tmp.fCC1.data .= rand(metadata.float_type, grid.Nx, grid.Ny, grid.Nz)
        ssp = SpectralSolverParameters(grid, stepper_tmp.fCC1, FFTW.PATIENT; verbose=true)
    elseif metadata.arch == :gpu
        ssp = nothing
    end

    # Setting some initial configuration.
    velocities.u.data  .= 0
    velocities.v.data  .= 0
    velocities.w.data  .= 0
    tracers.S.data .= 35
    tracers.T.data .= 283

    pHY_profile = [-eos.ρ₀*constants.g*h for h in grid.zC]

    if metadata.arch == :cpu
        pressures.pHY.data .= repeat(reshape(pHY_profile, 1, 1, grid.Nz), grid.Nx, grid.Ny, 1)
    elseif metadata.arch == :gpu
        pressures.pHY.data .= cu(repeat(reshape(pHY_profile, 1, 1, grid.Nz), grid.Nx, grid.Ny, 1))
    end

    # Calculate initial density based on tracer values.
    ρ!(eos, grid, tracers)

    Model(metadata, configuration, boundary_conditions, constants, eos, grid,
          velocities, tracers, pressures, G, Gp, forcings,
          stepper_tmp, ssp, clock, output_writers, diagnostics)
          # stepper_tmp, operator_tmp, ssp, clock, output_writers, diagnostics)
end
