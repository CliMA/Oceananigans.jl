using Plots, PyPlot, FFTW
using Oceananigans

function rayleigh_benard_convection(Ra_desired, Nx, Ny, Nz, Lx, Ly, Lz, Nt, Δt)
    model = Model((Nx, Ny, Nz), (Lx, Ly, Lz))

    α = 207e-6  # Volumetric expansion coefficient [K⁻¹] of water at 20°C.
    ΔT = 1      # Temperature difference [K] between top and bottom.
    Pr = 0.7    # Prandtl number Pr = 𝜈/κ.

    # Calculate viscosity needed to get flow with desired Rayleigh number using
    # Eq. (3.5) of Kerr (1996).
    𝜈 = √((α*model.constants.g *Lz^3 / Ra_desired) * Pr * ΔT)
    Ra = (α*model.constants.g*Lz^3 / 𝜈^2) * Pr * ΔT

    println("Aspect ratio:    Γxz=$(Lx/Lz), Γyz=$(Ly/Lz)")
    println("Rayleigh number: Ra=$Ra")
    println("Prandtl number:  Pr=$Pr")

    # Create a new model based on the old with the configuration we want (we
    # just need to change 𝜈 and κ, the boundary conditions, and clock). This is
    # a hack until the Model constructor can be made more flexible.
    𝜈h, 𝜈v, κh, κv = 𝜈, 𝜈, 𝜈/Pr, 𝜈/Pr  # Assuming isotropic 𝜈 and κ.
    configuration = _ModelConfiguration(𝜈h, 𝜈v, κh, κv)
    boundary_conditions = BoundaryConditions(:periodic, :periodic, :rigid_lid, :no_slip)

    time, time_step = 0, 0
    clock = Clock(time, time_step, Δt)

    model = Model(model.metadata, configuration, boundary_conditions,
                  model.constants, model.eos, model.grid,
                  model.velocities, model.tracers, model.pressures,
                  model.G, model.Gp, model.forcings,
                  model.stepper_tmp, model.operator_tmp, model.ssp, clock,
                  model.output_writers, model.diagnostics)

    # Write temperature field to disk every 10 time steps.
    output_dir, output_prefix, output_freq = ".", "rayleigh_benard", 10
    field_writer = FieldWriter(output_dir, output_prefix, output_freq, [model.tracers.T], ["T"])
    push!(model.output_writers, field_writer)

    diag_freq, Nu_running_avg = 1, 0
    Nu_wT_diag = Nusselt_wT(diag_freq, Float64[], Float64[], Nu_running_avg)
    push!(model.diagnostics, Nu_wT_diag)

    Nu_Chi_diag = Nusselt_Chi(diag_freq, Float64[], Float64[], Nu_running_avg)
    push!(model.diagnostics, Nu_Chi_diag)

    # Small random perturbations are added to boundary conditions to ensure instability formation.
    top_T    = 283 .- (ΔT/2) .+ 0.001.*rand(Nx, Ny)
    bottom_T = 283 .+ (ΔT/2) .+ 0.001.*rand(Nx, Ny)

    for i in 1:Nt
        time_step!(model; Nt=1, Δt=model.clock.Δt)
        # Impose constant T boundary conditions at top and bottom every time step.
        @. model.tracers.T.data[:, :,   1] = top_T
        @. model.tracers.T.data[:, :, end] = bottom_T
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
        Plots.heatmap(xC, zC, rotl90(temperature[:, Int(ceil(model.grid.Ny/2)), :]) .- 283, color=:balance,
                      clims=(-0.5, 0.5), aspect_ratio=:equal,
                      title="T @ t=$(tidx*fw.output_frequency*model.clock.Δt)")
    end

    mp4(movie, "rayleigh_benard_$(round(Int, time())).mp4", fps = 30)
end

function plot_Nusselt_number_diagnostics(model::Model, Nu_wT_diag::Nusselt_wT, Nu_Chi_diag::Nusselt_Chi)
    println("Plotting Nusselt number diagnostics...")

    Nx, Ny, Nz = model.grid.Nx, model.grid.Ny, model.grid.Nz
    t = 0:model.clock.Δt:model.clock.time

    PyPlot.plot(t, Nu_wT_diag.Nu, label="Nu_wT")
    PyPlot.plot(t, Nu_wT_diag.Nu_inst, label="Nu_wT_inst")
    PyPlot.plot(t, Nu_Chi_diag.Nu, label="Nu_Chi")
    PyPlot.plot(t, Nu_Chi_diag.Nu_inst, label="Nu_Chi_inst")

    PyPlot.title("Rayleigh–Bénard convection ($Nx×$Ny×$Nz) @ Ra=5000")
    PyPlot.xlabel("Time (s)")
    PyPlot.ylabel("Nusselt number Nu")
    PyPlot.legend()
    PyPlot.savefig("rayleigh_benard_nusselt_diag.png", dpi=300, format="png", transparent=false)
end
