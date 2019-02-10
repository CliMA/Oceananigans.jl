using Plots, FFTW
using Oceananigans

function rayleigh_benard_convection(Ra_desired, Nx, Ny, Nz, Nt)
    α = 1.43e-7  # Thermal diffusivity [m²/s] of water at 25°C.

    constants = EarthStationary()
    eos = LinearEquationOfState()

    Lx, Ly = 4000, 4000
    Lz = 2000  # Ocean depth [m].
    ΔT = 1  # Temperature difference between top and bottom.
    Pr = 1  # Prandtl number Pr = 𝜈/κ.

    𝜈 = √((α*constants.g *Lz^3 / Ra_desired) * Pr * ΔT)
    Ra = (α*constants.g*Lz^3 / 𝜈^2) * Pr * ΔT

    𝜈h, 𝜈v, κh, κv = 𝜈, 𝜈, 𝜈, 𝜈
    configuration = _ModelConfiguration(𝜈h, 𝜈v, κh, κv)

    println("Rayleigh number: Ra=$Ra")
    println("Prandtl number:  Pr=$Pr")

    # Set up model. Will avoid using Model constructor for now.
    metadata = _ModelMetadata(:cpu, Float64)
    boundary_conditions = BoundaryConditions(:periodic, :periodic, :rigid_lid, :no_slip)

    N, L = (Nx, Ny, Nz), (Lx, Ly, Lz)
    grid = RegularCartesianGrid(metadata, N, L)

    velocities  = VelocityFields(metadata, grid)
    tracers = TracerFields(metadata, grid)
    pressures = PressureFields(metadata, grid)
    G  = SourceTerms(metadata, grid)
    Gp = SourceTerms(metadata, grid)
    forcings = ForcingFields(metadata, grid)
    stepper_tmp = StepperTemporaryFields(metadata, grid)
    operator_tmp = OperatorTemporaryFields(metadata, grid)

    time, time_step = 0, 0
    clock = Clock(time, time_step)

    output_writers = OutputWriter[]

    stepper_tmp.fCC1.data .= rand(metadata.float_type, grid.Nx, grid.Ny, grid.Nz)
    ssp = SpectralSolverParameters(grid, stepper_tmp.fCC1, FFTW.PATIENT; verbose=true)

    velocities.u.data  .= 0
    velocities.v.data  .= 0
    velocities.w.data  .= 0
    tracers.S.data .= 35
    tracers.T.data .= 283

    pHY_profile = [-eos.ρ₀*constants.g*h for h in grid.zC]
    pressures.pHY.data .= repeat(reshape(pHY_profile, 1, 1, grid.Nz), grid.Nx, grid.Ny, 1)

    ρ!(eos, grid, tracers)

    model = Model(metadata, configuration, boundary_conditions, constants, eos, grid,
                  velocities, tracers, pressures, G, Gp, forcings,
                  stepper_tmp, operator_tmp, ssp, clock, output_writers)

    field_writer = FieldWriter(".", "rayleigh_benard", 10, [model.tracers.T], ["T"])
    push!(model.output_writers, field_writer)

    Δt = 20
    for i in 1:Nt
        @. model.tracers.T.data[:, :,   1] = 283 - (ΔT/2) + 0.001*rand()
        @. model.tracers.T.data[:, :, end] = 283 + (ΔT/2) + 0.001*rand()
        time_step!(model; Nt=1, Δt=Δt)
    end

    make_temperature_movie(model, field_writer)
end

function make_temperature_movie(model::Model, fw::FieldWriter)
    n_frames = Int(model.clock.time_step / fw.output_frequency)

    xC, yC, zC = model.grid.xC, model.grid.yC, model.grid.zC
    Δt = 20

    print("Creating temperature movie... ($n_frames frames)\n")

    Plots.gr()
    movie = @animate for tidx in 0:n_frames
        print("\rframe = $tidx / $n_frames   ")
        temperature = read_output(model, fw, "T", tidx*fw.output_frequency*Δt)
        Plots.heatmap(xC, zC, rotl90(temperature[:, Int(model.grid.Ny/2), :]) .- 283, color=:balance,
                      clims=(-0.5, 0.5),
                      title="T @ t=$(tidx*fw.output_frequency*Δt)")
    end

    mp4(movie, "rayleigh_benard_$(round(Int, time())).mp4", fps = 30)
end
