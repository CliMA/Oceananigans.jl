using Plots, PyPlot, FFTW
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

    time, time_step, Δt = 0, 0, 20
    clock = Clock(time, time_step, Δt)

    output_writers = OutputWriter[]
    diagnostics = Diagnostic[]

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
                  stepper_tmp, operator_tmp, ssp, clock, output_writers, diagnostics)

    field_writer = FieldWriter(".", "rayleigh_benard", 10, [model.tracers.T], ["T"])
    push!(model.output_writers, field_writer)

    Nu_wT_diag = Nusselt_wT(1, Float64[], 0)
    push!(model.diagnostics, Nu_wT_diag)

    Nu_Chi_diag = Nusselt_Chi(1, Float64[], 0)
    push!(model.diagnostics, Nu_Chi_diag)

    for i in 1:Nt
        @. model.tracers.T.data[:, :,   1] = 283 - (ΔT/2) + 0.001*rand()
        @. model.tracers.T.data[:, :, end] = 283 + (ΔT/2) + 0.001*rand()
        time_step!(model; Nt=1, Δt=model.clock.Δt)
    end

    make_temperature_movie(model, field_writer)
    plot_Nusselt_number_diagnostics(model, Nu_wT_diag, Nu_Chi_diag)
end

function make_temperature_movie(model::Model, fw::FieldWriter)
    n_frames = Int(model.clock.time_step / fw.output_frequency)

    xC, yC, zC = model.grid.xC, model.grid.yC, model.grid.zC

    print("Creating temperature movie... ($n_frames frames)\n")

    Plots.gr()
    default(dpi=300)
    movie = @animate for tidx in 0:n_frames
        print("\rframe = $tidx / $n_frames   ")
        temperature = read_output(model, fw, "T", tidx*fw.output_frequency*model.clock.Δt)
        Plots.heatmap(xC, zC, rotl90(temperature[:, Int(model.grid.Ny/2), :]) .- 283, color=:balance,
                      clims=(-0.5, 0.5),
                      title="T @ t=$(tidx*fw.output_frequency*model.clock.Δt)")
    end

    mp4(movie, "rayleigh_benard_$(round(Int, time())).mp4", fps = 30)
end

function plot_Nusselt_number_diagnostics(model::Model, Nu_wT_diag::Nusselt_wT, Nu_Chi_diag::Nusselt_Chi)
    println("Plotting Nusselt number diagnostics...")

    t = 0:model.clock.Δt:model.clock.time

    PyPlot.plot(t, Nu_wT_diag.Nu, label="Nu_wT")
    PyPlot.plot(t, Nu_Chi_diag.Nu, label="Nu_Chi")

    PyPlot.title("Rayleigh–Bénard convection (64×64×32) @ Ra=5000")
    PyPlot.xlabel("Time (s)")
    PyPlot.ylabel("Nusselt number Nu")
    PyPlot.legend()
    PyPlot.savefig("rayleigh_benard_nusselt_diag.png", dpi=300, format="png", transparent=false)
end
