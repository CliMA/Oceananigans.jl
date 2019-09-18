using Statistics, Printf

using Oceananigans
using Oceananigans.TurbulenceClosures

""" Friction velocity. See equation (16) of Vreugdenhil & Taylor (2018). """
function uτ(model, Uavg)
    Nz, Hz, Δz = model.grid.Nz, model.grid.Hz, model.grid.Δz
    U_wall = model.parameters.U_wall
    ν = model.closure.ν

    U = Uavg(model)[1+Hz:end-Hz]  # Exclude average of halo region.

    # Use a finite difference to calculate dU/dz at the top and bottomtom walls.
    # The distance between the center of the cell adjacent to the wall and the
    # wall itself is Δz/2.
    uτ²_top    = ν * abs(U[1] - U_wall)    / (Δz/2)  # Top wall    where u = +U_wall
    uτ²_bottom = ν * abs(-U_wall  - U[Nz]) / (Δz/2)  # Bottom wall where u = -U_wall

    uτ_top, uτ_bottom = √uτ²_top, √uτ²_bottom

    return uτ_top, uτ_bottom
end

""" Heat flux at the wall. See equation (16) of Vreugdenhil & Taylor (2018). """
function q_wall(model, Tavg)
    Nz, Hz, Δz = model.grid.Nz, model.grid.Hz, model.grid.Δz
    Θ_wall = model.parameters.Θ_wall
    κ = model.closure.κ

    Θ = Tavg(model)[1+Hz:end-Hz]  # Exclude average of halo region.

    # Use a finite difference to calculate dθ/dz at the top and bottomtom walls.
    # The distance between the center of the cell adjacent to the wall and the
    # wall itself is Δz/2.
    q_wall_top    = κ * abs(Θ[1] - Θ_wall)   / (Δz/2)  # Top wall    where Θ = +Θ_wall
    q_wall_bottom = κ * abs(-Θ_wall - Θ[Nz]) / (Δz/2)  # Bottom wall where Θ = -Θ_wall

    return q_wall_top, q_wall_bottom
end

struct FrictionReynoldsNumber{H}
    Uavg :: H
end

struct NusseltNumber{H}
    Tavg :: H
end

""" Friction Reynolds number. See equation (20) of Vreugdenhil & Taylor (2018). """
function (Reτ::FrictionReynoldsNumber)(model)
    ν = model.closure.ν
    h = model.grid.Lz / 2
    uτ_top, uτ_bottom = uτ(model, Reτ.Uavg)

    return h * uτ_top / ν, h * uτ_bottom / ν
end

""" Nusselt number. See equation (20) of Vreugdenhil & Taylor (2018). """
function (Nu::NusseltNumber)(model)
    κ = model.closure.κ
    h = model.grid.Lz / 2
    Θ_wall = model.parameters.Θ_wall

    q_wall_top, q_wall_bottom = q_wall(model, Nu.Tavg)

    return (q_wall_top * h)/(κ * Θ_wall), (q_wall_bottom * h)/(κ * Θ_wall)
end

"""
    simulate_stratified_couette_flow(; Nxy, Nz, h, U_wall, Re, Pr, Ri)

Simulate stratified plane Couette flow with `Nxy` grid cells in each horizontal
direction, `Nz` grid cells in the vertical, in a domain of size (4πh, 2πh, 2h),
with wall velocities of `U_wall` at the top and -`U_wall` at the bottom, at a Reynolds
number `Re, Prandtl number `Pr`, and Richardson number `Ri`.
"""
function simulate_stratified_couette_flow(; Nxy, Nz, arch=GPU(), h=1, U_wall=1, Re=4250, Pr=0.7, Ri, end_time=1000)
    ####
    #### Computed parameters
    ####

         ν = U_wall * h / Re    # From Re = U_wall h / ν
    Θ_wall = Ri * U_wall^2 / h  # From Ri = L Θ_wall / U_wall²
         κ = ν / Pr             # From Pr = ν / κ

    parameters = (U_wall=U_wall, Θ_wall=Θ_wall, Re=Re, Pr=Pr, Ri=Ri)

    ####
    #### Impose boundary conditions
    ####

    Tbcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Value,  Θ_wall),
                                    bottom = BoundaryCondition(Value, -Θ_wall))

    ubcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Value,  U_wall),
                                    bottom = BoundaryCondition(Value, -U_wall))

    vbcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Value, 0),
                                    bottom = BoundaryCondition(Value, 0))

    ####
    #### Non-dimensional model setup
    ####

    model = Model(N = (Nxy, Nxy, Nz),
                  L = (4π*h, 2π*h, 2h),
               arch = arch, 
            closure = AnisotropicMinimumDissipation(ν=ν, κ=κ),
                eos = LinearEquationOfState(βT=1, βS=0),
          constants = PlanetaryConstants(f=0, g=1),
                bcs = BoundaryConditions(u=ubcs, v=vbcs, T=Tbcs),
         parameters = parameters)

    ####
    #### Set initial conditions
    ####

    # Add a bit of surface-concentrated noise to the initial condition
    ε(σ, z) = σ * randn() * z/model.grid.Lz * (1 + z/model.grid.Lz)

    # We add a sinusoidal initial condition to u to encourage instability.
    T₀(x, y, z) = 2Θ_wall * (1/2 + z/model.grid.Lz) * (1 + ε(5e-1, z))
    u₀(x, y, z) = 2U_wall * (1/2 + z/model.grid.Lz) * (1 + ε(5e-1, z)) * (1 + 0.5*sin(4π/model.grid.Lx * x))
    v₀(x, y, z) = ε(5e-1, z)
    w₀(x, y, z) = ε(5e-1, z)

    set_ic!(model, u=u₀, v=v₀, w=w₀, T=T₀)

    ####
    #### Print simulation banner
    ####

    @printf(
        """
        Simulating stratified plane Couette flow

                N : %d, %d, %d
                L : %.3g, %.3g, %.3g
               Re : %.3f
               Ri : %.3f
               Pr : %.3f
                ν : %.3g
                κ : %.3g
           U_wall : %.3f
           Θ_wall : %.3f

        """, model.grid.Nx, model.grid.Ny, model.grid.Nz,
             model.grid.Lx, model.grid.Ly, model.grid.Lz,
             Re, Ri, Pr, ν, κ, U_wall, Θ_wall)

    ####
    #### Set up field output writer
    ####

    base_dir = @sprintf("stratified_couette_flow_data_Nxy%d_Nz%d_Ri%.2f", Nxy, Nz, Ri)
    prefix = @sprintf("stratified_couette_flow_Nxy%d_Nz%d_Ri%.2f", Nxy, Nz, Ri)

    function init_save_parameters_and_bcs(file, model)
        file["parameters/reynolds_number"] = Re
        file["parameters/richardson_number"] = Ri
        file["parameters/prandtl_number"] = Pr
        file["parameters/viscosity"] = ν
        file["parameters/diffusivity"] = κ
        file["parameters/wall_velocity"] = U_wall
        file["parameters/wall_temperature"] = Θ_wall
    end

    fields = Dict(
        :u => model -> Array(model.velocities.u.data.parent),
        :v => model -> Array(model.velocities.v.data.parent),
        :w => model -> Array(model.velocities.w.data.parent),
        :T => model -> Array(model.tracers.T.data.parent),
   :kappaT => model -> Array(model.diffusivities.κₑ.T.data.parent),
       :nu => model -> Array(model.diffusivities.νₑ.data.parent))


    field_writer = JLD2OutputWriter(model, fields; dir=base_dir, prefix=prefix * "_fields",
                                    init=init_save_parameters_and_bcs,
                                    interval=10, force=true, verbose=true)

    push!(model.output_writers, field_writer)

    ####
    #### Set up diagnostics
    ####

    push!(model.diagnostics, NaNChecker(model))

    ####
    #### Set up profile output writer
    ####

    Δtₚ = 1 # Time interval for computing and saving profiles.

    Uavg = HorizontalAverage(model, model.velocities.u;       interval=Δtₚ, return_type=Array)
    Vavg = HorizontalAverage(model, model.velocities.v;       interval=Δtₚ, return_type=Array)
    Wavg = HorizontalAverage(model, model.velocities.w;       interval=Δtₚ, return_type=Array)
    Tavg = HorizontalAverage(model, model.tracers.T;          interval=Δtₚ, return_type=Array)
    νavg = HorizontalAverage(model, model.diffusivities.νₑ;   interval=Δtₚ, return_type=Array)
    κavg = HorizontalAverage(model, model.diffusivities.κₑ.T; interval=Δtₚ, return_type=Array)

    profiles = Dict(
         :u => model -> Uavg(model),
         :v => model -> Vavg(model),
         :w => model -> Wavg(model),
         :T => model -> Tavg(model),
        :nu => model -> νavg(model),
    :kappaT => model -> κavg(model))

    profile_writer = JLD2OutputWriter(model, profiles; dir=base_dir, prefix=prefix * "_profiles",
                                      init=init_save_parameters_and_bcs, interval=Δtₚ, force=true, verbose=true)

    push!(model.output_writers, profile_writer)

    ####
    #### Set up statistic output writer
    ####

    Reτ = FrictionReynoldsNumber(Uavg)
     Nu = NusseltNumber(Tavg)

    statistics = Dict(
        :Re_tau => model -> Reτ(model),
        :Nu_tau => model -> Nu(model))

    statistic_writer = JLD2OutputWriter(model, statistics; dir=base_dir, prefix=prefix * "_statistics",
                                     init=init_save_parameters_and_bcs, interval=Δtₚ/2, force=true, verbose=true)

    push!(model.output_writers, statistic_writer)

    ####
    #### Time stepping
    ####

    wizard = TimeStepWizard(cfl=0.02, Δt=0.0001, max_change=1.1, max_Δt=0.02)

    function cell_diffusion_timescale(model)
        Δ = min(model.grid.Δx, model.grid.Δy, model.grid.Δz)
        max_ν = maximum(model.diffusivities.νₑ.data.parent)
        max_κ = max(Tuple(maximum(κₑ.data.parent) for κₑ in model.diffusivities.κₑ)...)
        return min(Δ^2 / max_ν, Δ^2 / max_κ)
    end

    # Take Ni "intermediate" time steps at a time before printing a progress
    # statement and updating the time step.
    Ni = 10

    cfl(t) = min(0.01*t, 0.1)

    while model.clock.time < end_time
        wizard.cfl = cfl(model.clock.time)

        walltime = @elapsed time_step!(model; Nt=Ni, Δt=wizard.Δt)
        progress = 100 * (model.clock.time / end_time)

        umax = maximum(abs, model.velocities.u.data.parent)
        vmax = maximum(abs, model.velocities.v.data.parent)
        wmax = maximum(abs, model.velocities.w.data.parent)
        CFL = wizard.Δt / cell_advection_timescale(model)

        νmax = maximum(model.diffusivities.νₑ.data.parent)
        κmax = maximum(model.diffusivities.κₑ.T.data.parent)

        Δ = min(model.grid.Δx, model.grid.Δy, model.grid.Δz)
        νCFL = wizard.Δt / (Δ^2 / νmax)
        κCFL = wizard.Δt / (Δ^2 / κmax)

        update_Δt!(wizard, model)

        @printf("[%06.2f%%] i: %d, t: %4.2f, umax: (%6.3g, %6.3g, %6.3g) m/s, CFL: %6.4g, νκmax: (%6.3g, %6.3g), νκCFL: (%6.4g, %6.4g), next Δt: %8.5g, ⟨wall time⟩: %s\n",
                progress, model.clock.iteration, model.clock.time,
                umax, vmax, wmax, CFL, νmax, κmax, νCFL, κCFL,
                wizard.Δt, prettytime(walltime / Ni))
    end
end

