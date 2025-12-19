using Oceananigans.Architectures: architecture
using Oceananigans: fields
using Oceananigans.Utils: time_difference_seconds

"""
    FastRungeKutta3TimeStepper{FT, TG, TI} <: AbstractTimeStepper

Hold parameters and tendency fields for a fast, third-order Runge-Kutta
time-stepping scheme described by [Aithal and Ferrante (2020)](@cite AithalFerrante2020).

The key advantage of this scheme is that it solves the Poisson equation for pressure
only once per time step (at the final stage), instead of three times as in standard RK3,
while maintaining third-order accuracy in time for velocity.

This results in a speedup of approximately 4-7× compared to standard RK3 methods.

References
==========
Aithal, A. B. and Ferrante, A. (2020). A fast pressure-correction method for
    incompressible flows over curved walls. Journal of Computational Physics,
    421, 109750.
"""
struct FastRungeKutta3TimeStepper{FT, TG, TP, TI} <: AbstractTimeStepper
                a₁₁ :: FT
                a₂₁ :: FT
                a₃₁ :: FT
                a₃₂ :: FT
                 b₁ :: FT
                 b₂ :: FT
                 b₃ :: FT
                 c₁ :: FT
                 c₂ :: FT
                 c₃ :: FT                
                 G⁰ :: TG  # Storage for tendencies at stage 0
                 G⁻ :: TG  # Storage for previous tendencies
                 Gⁿ :: TG  # Current stage tendencies
                 pⁿ :: TP  # Current pressure (∇φⁿ)
               pⁿ⁻¹ :: TP  # Previous pressure (∇φⁿ⁻¹)
               Δt⁻¹ :: FT  # Previous time step
                RK3 :: RungeKutta3TimeStepper{FT, TG, TI}  # For initialization
    implicit_solver :: TI
end

"""
    FastRungeKutta3TimeStepper(grid, prognostic_fields;
                               implicit_solver = nothing,
                               Gⁿ = map(similar, prognostic_fields),
                               G⁻ = map(similar, prognostic_fields))

Return a fast 3rd-order Runge-Kutta timestepper (`FastRungeKutta3TimeStepper`) on `grid`
and with `prognostic_fields`.

The scheme is described by [Aithal and Ferrante (2020)](@cite AithalFerrante2020).
The key innovation is that the Poisson equation for pressure is solved only once per
timestep at the final RK3 stage, using linear extrapolation of the pressure gradient
from previous time steps at intermediate stages. This reduces computational cost by
approximately 30-60% compared to standard RK3 for the pressure solve, and overall
speedup of 4-7× for the entire flow solver.

The algorithm follows the Sanderse & Koren (2012) RK3 coefficients:
- γ¹ = 1/3, γ² = 2, γ³ = 3/4

**Stage 1:**
```
U*¹ = Uⁿ + Δt * γ¹ * F(Uⁿ)
```

**Stage 2:** (using extrapolated pressure gradient)
```
∇φ₁ = (1 + c₂) * ∇pⁿ - c₂ * ∇pⁿ⁻¹,  where c₂ = 1/3
U*² = Uⁿ + Δt * γ² * F(U*¹ - c₂*Δt*∇φ₁)
```

**Stage 3:** (using extrapolated pressure gradient)
```
∇φ₂ = (1 + c₃) * ∇pⁿ - c₃ * ∇pⁿ⁻¹,  where c₃ = 1
U*³ = Uⁿ + Δt * γ³ * F(U*² - c₃*Δt*∇φ₂)
```

**Final pressure correction:** (only Poisson solve)
```
∇²φⁿ⁺¹ = ρ/Δt * ∇·U*³
Uⁿ⁺¹ = U*³ - Δt * ∇φⁿ⁺¹
```

**Note:** This scheme is non-self-starting and requires pressure gradients from
previous timesteps. For the first timestep, use standard RK3 or initialize with
zero pressure gradients.

References
==========
Aithal, A. B. and Ferrante, A. (2020). A fast pressure-correction method for
    incompressible flows over curved walls. Journal of Computational Physics,
    421, 109750.

Sanderse, B. and Koren, B. (2012). Accuracy analysis of explicit Runge-Kutta
    methods applied to the incompressible Navier-Stokes equations. Journal of
    Computational Physics, 231(8), 3041-3063.
"""
function FastRungeKutta3TimeStepper(grid, prognostic_fields;
                                    implicit_solver::TI = nothing,
                                    G⁰::TG = map(similar, prognostic_fields),
                                    G⁻     = map(similar, prognostic_fields),
                                    Gⁿ     = map(similar, prognostic_fields),
                                    pⁿ::TP = CenterField(grid),
                                    pⁿ⁻¹   =  CenterField(grid)) where {TI, TG, TP}

    !isnothing(implicit_solver) &&
        @warn("Implicit-explicit time-stepping with FastRungeKutta3TimeStepper is not tested. " *
              "\n implicit_solver: $(typeof(implicit_solver))")

    a₂₁ = 1 // 3
    a₃₁ = -1
    a₃₂ = 2
    b₁ = 0
    b₂ = 3 // 4
    b₃ = 1 // 4

    c₁ = 1 // 3
    c₂ = 1
    c₃ = 1

    Δt⁻¹ = 0

    FT = eltype(grid)

    RK3 = RungeKutta3TimeStepper(grid, prognostic_fields; implicit_solver, Gⁿ, G⁻)

    return FastRungeKutta3TimeStepper{FT, TG, TP, TI}(a₃₁, a₃₂, a₂₁, b₁, b₂, b₃, c₁, c₂, c₃, 
                                                      G⁰, G⁻, Gⁿ, pⁿ, pⁿ⁻¹, Δt⁻¹, RK3, implicit_solver)
end
#####
##### Time stepping
#####

"""
    time_step!(model::AbstractModel{<:FastRungeKutta3TimeStepper}, Δt)

Step forward `model` one time step `Δt` with the fast 3rd-order Runge-Kutta method
of [Aithal and Ferrante (2020)](@cite AithalFerrante2020).

The fast RK3 method takes three intermediate substep stages to achieve a single timestep,
but applies a pressure correction step **only at the final stage**, using linear
extrapolation of the pressure gradient from previous time steps at intermediate stages.

This method achieves a 30-60× speedup for the Poisson solve and 4-7× overall speedup
compared to standard RK3, while maintaining third-order accuracy in time.

References
==========
Aithal, A. B. and Ferrante, A. (2020). A fast pressure-correction method for
    incompressible flows over curved walls. Journal of Computational Physics,
    421, 109750.
"""
function time_step!(model::AbstractModel{<:FastRungeKutta3TimeStepper}, Δt; callbacks=[])
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    # For the first two timesteps, use standard RK3 with full pressure corrections
    # to initialize the pressure storage.
    needs_initialization = model.clock.iteration <= 1

    # If needs initialization, zero out the pressure gradient storage
    if needs_initialization
        time_step!(model, model.timestepper.RK3, Δt; callbacks=callbacks)
    end

    if model.clock.iteration == 0
        parent(model.timestepper.pⁿ⁻¹) .= parent(model.pressures.pNHS)
    elseif model.clock.iteration == 1
        parent(model.timestepper.pⁿ) .= parent(model.pressures.pNHS)
    else
        a₁₁ = nothing
        a₂₁ = model.timestepper.a₂₁
        a₃₁ = model.timestepper.a₃₁
        a₃₂ = model.timestepper.a₃₂
        b₁ = model.timestepper.b₁
        b₂ = model.timestepper.b₂
        b₃ = model.timestepper.b₃
        c₁ = model.timestepper.c₁
        c₂ = model.timestepper.c₂
        c₃ = model.timestepper.c₃

        # Compute the next time step a priori to reduce floating point error
        tⁿ⁺¹ = next_time(model.clock, Δt)
        first_stage_Δt  = stage_Δt(Δt, c₁, nothing)      # =  b₁ * Δt
        second_stage_Δt = stage_Δt(Δt, c₂ - c₁, nothing)      # = (b₂ + a₂₁) * Δt
        third_stage_Δt  = stage_Δt(Δt, c₃ - c₂ - c₁, nothing)      # = (b₃ + a₃₁ + a₃₂) * Δt

        #
        # Stage 1:
        #
        compute_flux_bc_tendencies!(model)
        fast_rk3_substep!(model, Δt, a₂₁, a₁₁)

        tick!(model.clock, first_stage_Δt; stage=true)

        cache_previous_tendencies!(model)
        cache_initial_tendencies!(model)
        update_state!(model, callbacks; compute_tendencies = true)
        step_lagrangian_particles!(model, first_stage_Δt)

        #
        # Stage 2: U² = U¹ + Δt * (γ² * G(U¹) + ζ² * G(Uⁿ))
        # With intermediate pressure correction using extrapolation
        #
        extrapolate_pressure!(model.timestepper, model.pressures.pNHS, c₁, Δt)
        premultiply_pressure_with_Δt!(model.pressures.pNHS, c₁ * Δt)
        make_pressure_correction!(model, c₁ * Δt)

        compute_flux_bc_tendencies!(model)
        fast_rk3_substep!(model, Δt, a₃₂, a₃₁ - a₂₁)

        tick!(model.clock, second_stage_Δt; stage=true)

        cache_previous_tendencies!(model)
        update_state!(model, callbacks; compute_tendencies = true)
        step_lagrangian_particles!(model, second_stage_Δt)

        #
        # Stage 3: U³ = U² + Δt * (γ³ * G(U²) + ζ³ * G(U¹))
        # With intermediate pressure correction using extrapolation
        #
        extrapolate_pressure!(model.timestepper, model.pressures.pNHS, c₃, Δt)
        premultiply_pressure_with_Δt!(model.pressures.pNHS, c₃ * Δt)
        make_pressure_correction!(model, c₃ * Δt)

        compute_flux_bc_tendencies!(model)
        fast_rk3_substep!(model, model.timestepper.G⁰, nothing, Δt, -a₃₁, nothing)
        fast_rk3_substep!(model, Δt, b₃, b₂ - a₃₂)

        # # Time adjustment to reduce round-off error
        corrected_Δt = time_difference_seconds(tⁿ⁺¹, model.clock.time)
        tick!(model.clock, corrected_Δt)
        model.clock.last_stage_Δt = corrected_Δt
        model.clock.last_Δt = Δt

        # #
        # # Final pressure correction (ONLY Poisson solve in entire timestep)
        # #
        compute_pressure_correction!(model, Δt)
        make_pressure_correction!(model, Δt)

        cache_pressures!(model)

        update_state!(model, callbacks; compute_tendencies = true)
    end

    return nothing
end

function extrapolate_pressure!(timestepper::FastRungeKutta3TimeStepper, p, c, Δt)
    Δt⁻¹ = timestepper.Δt⁻¹
    p .= (1 + c * Δt / Δt⁻¹) .* timestepper.pⁿ .- (c * Δt / Δt⁻¹) .* timestepper.pⁿ⁻¹
    return nothing
end

function premultiply_pressure_with_Δt!(p, Δt)
    ϵ = eps(eltype(p))
    Δt⁺ = max(ϵ, Δt)
    p .*= Δt⁺
    
    return nothing
end

function cache_initial_tendencies!(model::AbstractModel{<:FastRungeKutta3TimeStepper})
    model.timestepper.G⁰ .= model.timestepper.Gⁿ
    return nothing
end

function cache_pressures!(model::AbstractModel{<:FastRungeKutta3TimeStepper})
    model.timestepper.pⁿ⁻¹ .= model.timestepper.pⁿ
    model.timestepper.pⁿ   .= model.pressures.pNHS
    return nothing
end

#####
##### Substep functions
#####

"""
    fast_rk3_substep!(model, Δt, γⁿ, ζⁿ)

Perform a substep of the fast RK3 method using standard RK3 coefficients:
    Uᵐ⁺¹ = Uᵐ + Δt * (γᵐ * Gᵐ + ζᵐ * Gᵐ⁻¹)

where m denotes the substage, Gᵐ is the current tendency, and Gᵐ⁻¹ is the previous tendency.
"""
function fast_rk3_substep!(model, G¹, G⁰, Δt, γⁿ, ζⁿ)
    grid = model.grid
    arch = architecture(grid)
    model_fields = prognostic_fields(model)

    for (i, field) in enumerate(model_fields)
        kernel_args = (field, Δt, γⁿ, ζⁿ, G¹[i], G⁰[i])
        launch!(arch, grid, :xyz, fast_rk3_substep_field!,
                kernel_args...; exclude_periphery=true)

        tracer_index = Val(i - 3)
        implicit_step!(field,
                       model.timestepper.implicit_solver,
                       model.closure,
                       model.closure_fields,
                       tracer_index,
                       model.clock,
                       fields(model),
                       Δt * γⁿ)
    end

    return nothing
end

fast_rk3_substep!(model, Δt, γⁿ, ζⁿ) = fast_rk3_substep!(model, model.timestepper.Gⁿ, model.timestepper.G⁻, 
                                                         Δt, γⁿ, ζⁿ)

"""
Time step field for FastRK3:
    Uᵐ⁺¹ = Uᵐ + Δt * (γᵐ * Gᵐ + ζᵐ * Gᵐ⁻¹)

where m denotes the substage.
"""
@kernel function fast_rk3_substep_field!(U, Δt, γⁿ::FT, ζⁿ, Gⁿ, G⁻) where FT
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U[i, j, k] += convert(FT, Δt) * (γⁿ * Gⁿ[i, j, k] + ζⁿ * G⁻[i, j, k])
    end
end

@kernel function fast_rk3_substep_field!(U, Δt, γ¹::FT, ::Nothing, G¹, G⁰) where FT
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U[i, j, k] += convert(FT, Δt) * γ¹ * G¹[i, j, k]
    end
end