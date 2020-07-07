#####
##### Navier-Stokes and tracer advection equations
#####

""" Store previous value of the source term and calculate current source term. """
function calculate_interior_tendency_contributions!(G, arch, grid, coriolis, buoyancy, surface_waves, closure, 
                                                    U, C, pHY′, K, F, clock)

    # Manually choose thread-block layout here as it's ~20% faster.
    # See: https://github.com/climate-machine/Oceananigans.jl/pull/308
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    if Nx == 1
        Tx, Ty = 1, min(256, Ny)
        Bx, By, Bz = Tx, floor(Int, Ny/Ty), Nz
    elseif Ny == 1
        Tx, Ty = min(256, Nx), 1
        Bx, By, Bz = floor(Int, Nx/Tx), Ty, Nz
    else
        Tx, Ty = 16, 16
        Bx, By, Bz = floor(Int, Nx/Tx), floor(Int, Ny/Ty), Nz
    end

    @launch(device(arch), threads=(Tx, Ty), blocks=(Bx, By, Bz),
            calculate_Gu!(G.u, grid, coriolis, surface_waves, closure, U, C, K, F, pHY′, clock))

    @launch(device(arch), threads=(Tx, Ty), blocks=(Bx, By, Bz),
            calculate_Gv!(G.v, grid, coriolis, surface_waves, closure, U, C, K, F, pHY′, clock))

    @launch(device(arch), threads=(Tx, Ty), blocks=(Bx, By, Bz),
            calculate_Gw!(G.w, grid, coriolis, surface_waves, closure, U, C, K, F, clock))

    for tracer_index in 1:length(C)
        @inbounds Gc = G[tracer_index+3]
        @inbounds Fc = F[tracer_index+3]
        @inbounds  c = C[tracer_index]

        @launch(device(arch), threads=(Tx, Ty), blocks=(Bx, By, Bz),
                calculate_Gc!(Gc, grid, c, Val(tracer_index), closure, buoyancy, U, C, K, Fc, clock))
    end

    return nothing
end

#####
##### Tendency calculators for u, v, w-velocity
#####

""" Calculate the right-hand-side of the u-velocity equation. """
function calculate_Gu!(Gu, grid, coriolis, surface_waves, closure, U, C, K, F, pHY′, clock)
    @loop_xyz i j k grid begin
        @inbounds Gu[i, j, k] = u_velocity_tendency(i, j, k, grid, coriolis, surface_waves, 
                                                    closure, U, C, K, F, pHY′, clock)
    end
    return nothing
end

""" Calculate the right-hand-side of the v-velocity equation. """
function calculate_Gv!(Gv, grid, coriolis, surface_waves, closure, U, C, K, F, pHY′, clock)
    @loop_xyz i j k grid begin
        @inbounds Gv[i, j, k] = v_velocity_tendency(i, j, k, grid, coriolis, surface_waves, 
                                                    closure, U, C, K, F, pHY′, clock)
    end
    return nothing
end

""" Calculate the right-hand-side of the w-velocity equation. """
function calculate_Gw!(Gw, grid, coriolis, surface_waves, closure, U, C, K, F, clock)
    @loop_xyz i j k grid begin
        @inbounds Gw[i, j, k] = w_velocity_tendency(i, j, k, grid, coriolis, surface_waves, 
                                                    closure, U, C, K, F, clock)
    end
    return nothing
end

#####
##### Tracer(s)
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
function calculate_Gc!(Gc, grid, c, tracer_index, closure, buoyancy, U, C, K, Fc, clock)
    @loop_xyz i j k grid begin
        @inbounds Gc[i, j, k] = tracer_tendency(i, j, k, grid, c, tracer_index,
                                                closure, buoyancy, U, C, K, Fc, clock)
    end
    return nothing
end

#####
##### Boundary contributions to tendencies due to user-prescribed fluxes
#####

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function calculate_boundary_tendency_contributions!(Gⁿ, arch, U, C, clock, state)

    # Velocity fields
    for i in 1:3
        apply_x_bcs!(Gⁿ[i], U[i], arch, clock, state)
        apply_y_bcs!(Gⁿ[i], U[i], arch, clock, state)
        apply_z_bcs!(Gⁿ[i], U[i], arch, clock, state)
    end

    # Tracer fields
    for i in 4:length(Gⁿ)
        apply_x_bcs!(Gⁿ[i], C[i-3], arch, clock, state)
        apply_y_bcs!(Gⁿ[i], C[i-3], arch, clock, state)
        apply_z_bcs!(Gⁿ[i], C[i-3], arch, clock, state)
    end

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
function update_hydrostatic_pressure!(pHY′, grid, buoyancy, C)
    @loop_xy i j grid begin
        @inbounds pHY′[i, j, grid.Nz] = - ℑzᵃᵃᶠ(i, j, grid.Nz+1, grid, buoyancy_perturbation, buoyancy, C) * ΔzF(i, j, grid.Nz+1, grid)
        @unroll for k in grid.Nz-1 : -1 : 1
            @inbounds pHY′[i, j, k] =
                pHY′[i, j, k+1] - ℑzᵃᵃᶠ(i, j, k+1, grid, buoyancy_perturbation, buoyancy, C) * ΔzF(i, j, k+1, grid)
        end
    end
    return nothing
end

"""
Compute the vertical velocity w by integrating the continuity equation from the bottom upwards

    `w^{n+1} = -∫ [∂/∂x (u^{n+1}) + ∂/∂y (v^{n+1})] dz`
"""
function compute_w_from_continuity!(model)
    @launch(device(model.architecture), config=launch_config(model.grid, :xy),
            _compute_w_from_continuity!(datatuple(model.velocities), model.grid))
    return nothing
end

function _compute_w_from_continuity!(U, grid)
    @loop_xy i j grid begin
        # U.w[i, j, 1] = 0 is enforced via halo regions.
        @unroll for k in 2:grid.Nz
            @inbounds U.w[i, j, k] = U.w[i, j, k-1] - ΔzC(i, j, k, grid) * hdivᶜᶜᵃ(i, j, k-1, grid, U.u, U.v)
        end
    end
    return nothing
end
