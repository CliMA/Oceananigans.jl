#####
##### Navier-Stokes and tracer advection equations
#####

""" Store previous value of the source term and calculate current source term. """
function calculate_interior_tendency_contributions!(G, arch, grid, coriolis, buoyancy, surface_waves, closure, 
                                                    U, C, pHY′, K, F, clock)

    workgroup, worksize = work_layout(grid, :xyz)

    calculate_Gu_kernel! = calculate_Gu!(device(arch), workgroup, worksize)
    calculate_Gv_kernel! = calculate_Gv!(device(arch), workgroup, worksize)
    calculate_Gw_kernel! = calculate_Gw!(device(arch), workgroup, worksize)
    calculate_Gc_kernel! = calculate_Gc!(device(arch), workgroup, worksize)

    default_stream = Event(device(arch))

    Gu_event = calculate_Gu_kernel!(G.u, grid, coriolis, surface_waves, closure, U, C, K, F, pHY′, clock, dependencies=default_stream)
    Gv_event = calculate_Gv_kernel!(G.v, grid, coriolis, surface_waves, closure, U, C, K, F, pHY′, clock, dependencies=default_stream)
    Gw_event = calculate_Gw_kernel!(G.w, grid, coriolis, surface_waves, closure, U, C, K, F, clock, dependencies=default_stream)

    events = [Gu_event, Gv_event, Gw_event]

    for tracer_index in 1:length(C)
        @inbounds Gc = G[tracer_index+3]
        @inbounds Fc = F[tracer_index+3]
        @inbounds  c = C[tracer_index]

        Gc_event = calculate_Gc_kernel!(Gc, grid, c, Val(tracer_index), closure, buoyancy, U, C, K, Fc, clock, dependencies=default_stream)

        push!(events, Gc_event)
    end

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

#####
##### Tendency calculators for u, v, w-velocity
#####

""" Calculate the right-hand-side of the u-velocity equation. """
@kernel function calculate_Gu!(Gu, grid, coriolis, surface_waves, closure, U, C, K, F, pHY′, clock)
    i, j, k = @index(Global, NTuple)

    @inbounds Gu[i, j, k] = u_velocity_tendency(i, j, k, grid, coriolis, surface_waves, 
                                                closure, U, C, K, F, pHY′, clock)
end

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function calculate_Gv!(Gv, grid, coriolis, surface_waves, closure, U, C, K, F, pHY′, clock)
    i, j, k = @index(Global, NTuple)

    @inbounds Gv[i, j, k] = v_velocity_tendency(i, j, k, grid, coriolis, surface_waves, 
                                                closure, U, C, K, F, pHY′, clock)
end

""" Calculate the right-hand-side of the w-velocity equation. """
@kernel function calculate_Gw!(Gw, grid, coriolis, surface_waves, closure, U, C, K, F, clock)
    i, j, k = @index(Global, NTuple)

    @inbounds Gw[i, j, k] = w_velocity_tendency(i, j, k, grid, coriolis, surface_waves, 
                                                closure, U, C, K, F, clock)
end

#####
##### Tracer(s)
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function calculate_Gc!(Gc, grid, c, tracer_index, closure, buoyancy, U, C, K, Fc, clock)
    i, j, k = @index(Global, NTuple)

    @inbounds Gc[i, j, k] = tracer_tendency(i, j, k, grid, c, tracer_index,
                                            closure, buoyancy, U, C, K, Fc, clock)
end

#####
##### Boundary contributions to tendencies due to user-prescribed fluxes
#####

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function calculate_boundary_tendency_contributions!(Gⁿ, arch, U, C, clock, state)

    events = []

    # Velocity fields
    for i in 1:3
        x_bcs_event = apply_x_bcs!(Gⁿ[i], U[i], arch, clock, state)
        y_bcs_event = apply_y_bcs!(Gⁿ[i], U[i], arch, clock, state)
        z_bcs_event = apply_z_bcs!(Gⁿ[i], U[i], arch, clock, state)

        push!(events, x_bcs_event, y_bcs_event, z_bcs_event)
    end

    # Tracer fields
    for i in 4:length(Gⁿ)
        x_bcs_event = apply_x_bcs!(Gⁿ[i], C[i-3], arch, clock, state)
        y_bcs_event = apply_y_bcs!(Gⁿ[i], C[i-3], arch, clock, state)
        z_bcs_event = apply_z_bcs!(Gⁿ[i], C[i-3], arch, clock, state)

        push!(events, x_bcs_event, y_bcs_event, z_bcs_event)
    end

    events = filter(e -> typeof(e) <: Base.Event, events)

    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

#####
##### Vertical integrals
#####

"""
Update the hydrostatic pressure perturbation pHY′. This is done by integrating
the `buoyancy_perturbation` downwards:

    `pHY′ = ∫ buoyancy_perturbation dz` from `z=0` down to `z=-Lz`
"""
@kernel function update_hydrostatic_pressure!(pHY′, grid, buoyancy, C)
    i, j = @index(Global, NTuple)

    @inbounds pHY′[i, j, grid.Nz] = - ℑzᵃᵃᶠ(i, j, grid.Nz+1, grid, buoyancy_perturbation, buoyancy, C) * ΔzF(i, j, grid.Nz+1, grid)

    @unroll for k in grid.Nz-1 : -1 : 1
        @inbounds pHY′[i, j, k] =
            pHY′[i, j, k+1] - ℑzᵃᵃᶠ(i, j, k+1, grid, buoyancy_perturbation, buoyancy, C) * ΔzF(i, j, k+1, grid)
    end
end

"""
Compute the vertical velocity w by integrating the continuity equation from the bottom upwards

    `w^{n+1} = -∫ [∂/∂x (u^{n+1}) + ∂/∂y (v^{n+1})] dz`
"""
function compute_w_from_continuity!(model)

    event = launch!(model.architecture, model.grid, :xyz, _compute_w_from_continuity!, datatuple(model.velocities), model.grid,
                    dependencies=Event(device(model.architecture)))

    wait(device(model.architecture), event)

    return nothing
end

@kernel function _compute_w_from_continuity!(U, grid)
    i, j = @index(Global, NTuple)
    # U.w[i, j, 1] = 0 is enforced via halo regions.
    @unroll for k in 2:grid.Nz
        @inbounds U.w[i, j, k] = U.w[i, j, k-1] - ΔzC(i, j, k, grid) * hdivᶜᶜᵃ(i, j, k-1, grid, U.u, U.v)
    end
end
