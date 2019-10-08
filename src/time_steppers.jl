using .TurbulenceClosures: ▶z_aaf

using Oceananigans.Operators

const Tx = 16 # CUDA threads per x-block
const Ty = 16 # CUDA threads per y-block

"""
    time_step!(model, Nt, Δt; init_with_euler=true)

Step forward `model` `Nt` time steps with step size `Δt`.

If `init_with_euler` is set to true, then the first step will be taken using a first-order
forward Euler method.
"""
function time_step!(model, Nt, Δt; init_with_euler=true)

    if model.clock.iteration == 0
        [ run_diagnostic(model, diag) for diag in values(model.diagnostics) ]
        [ write_output(model, out)    for out  in values(model.output_writers) ]
    end

    FT = eltype(model.grid)
    RHS = model.poisson_solver.storage
    U, C, Gⁿ, G⁻, K, p = datatuples(model.velocities, model.tracers, model.timestepper.Gⁿ,
                                    model.timestepper.G⁻, model.diffusivities, model.pressures)

    for n in 1:Nt
        χ = ifelse(init_with_euler && n==1, FT(-0.5), model.timestepper.χ)

        adams_bashforth_time_step!(model, model.architecture, model.grid, model.boundary_conditions, 
                                   U, C, p, K, RHS, Gⁿ,  G⁻, Δt, χ)

        [ time_to_run(model.clock, diag) && run_diagnostic(model, diag) for diag in values(model.diagnostics) ]
        [ time_to_run(model.clock, out) && write_output(model, out) for out in values(model.output_writers) ]
    end

    return nothing
end

time_step!(model; Nt, Δt, kwargs...) = time_step!(model, Nt, Δt; kwargs...)

"""
Step forward one time step with a 2nd-order Adams-Bashforth method and pressure-correction
substep.
"""
function adams_bashforth_time_step!(model, arch, grid, bcs, U, C, p, K, RHS, Gⁿ, G⁻, Δt, χ)

    # Arguments for user-defined boundary condition functions:
    boundary_condition_args = (model.clock.time, model.clock.iteration, U, C, model.parameters)

    # Pre-computations:
    store_previous_source_terms!(G⁻, Gⁿ, arch, grid)
    fill_halo_regions!(merge(U, C), bcs.solution, arch, grid, boundary_condition_args...)

    calc_diffusivities!(K, arch, grid, model.closure, model.buoyancy, U, C)
    fill_halo_regions!(K, bcs.pressure, arch, grid) # diffusivities share bcs with pressure.
    @launch device(arch) config=launch_config(grid, 2) update_hydrostatic_pressure!(p.pHY′, grid, model.buoyancy, C)
    fill_halo_regions!(p.pHY′, bcs.pressure, arch, grid)

    # Calculate tendency terms (minus non-hydrostatic pressure, which is updated in a pressure correction step):
    calculate_interior_source_terms!(Gⁿ, arch, grid, model.coriolis, model.closure, U, C, p.pHY′, K, 
                                     model.forcing, model.parameters, model.clock.time)
    calculate_boundary_source_terms!(Gⁿ, arch, grid, bcs.solution, boundary_condition_args...)

    # Complete explicit substep:
    adams_bashforth_update_source_terms!(Gⁿ, G⁻, arch, grid, χ)

    #@launch device(arch) config=launch_config(grid, 3) 

    # Start pressure correction substep with a pressure solve:
    fill_halo_regions!(Gⁿ[1:3], bcs.tendency[1:3], arch, grid)
    @launch device(arch) config=launch_config(grid, 3) calculate_poisson_right_hand_side!(arch, grid, model.poisson_solver.bcs,
                                                                                          Δt, U, Gⁿ, RHS)
    solve_for_pressure!(arch, model)
    fill_halo_regions!(p.pNHS, bcs.pressure, arch, grid)

    # Complete pressure correction step:
    @launch device(arch) config=launch_config(grid, 3) update_velocities!(U, Gⁿ, p.pNHS, grid, Δt)
    update_tracers!(C, Gⁿ, arch, grid, Δt)

    # Recompute vertical velocity w from continuity equation to ensure incompressibility
    fill_halo_regions!(U, bcs.solution[1:3], arch, grid, boundary_condition_args...)
    @launch device(arch) config=launch_config(grid, 2) compute_w_from_continuity!(U, grid)

    model.clock.time += Δt
    model.clock.iteration += 1

    return nothing
end

function solve_for_pressure!(::CPU, model::Model)
    ϕ = model.poisson_solver.storage

    solve_poisson_3d!(model.poisson_solver, model.grid)
    data(model.pressures.pNHS) .= real.(ϕ)
end

function solve_for_pressure!(::GPU, model::Model)
    ϕ = model.poisson_solver.storage

    solve_poisson_3d!(model.poisson_solver, model.grid)
    @launch device(GPU()) config=launch_config(model.grid, 3) idct_permute!(model.pressures.pNHS.data, model.grid, 
                                                                            model.poisson_solver.bcs, ϕ)
                                                                            
end

""" Store previous source terms before updating them. """
function store_previous_source_terms!(G⁻, Gⁿ, arch, grid)
    @launch device(arch) config=launch_config(grid, 3) store_previous_velocity_source_terms!(G⁻, Gⁿ, grid)
    for i in 4:length(G⁻)
        @inbounds Gc⁻ = G⁻[i]
        @inbounds Gcⁿ = Gⁿ[i]
        @launch device(arch) config=launch_config(grid, 3) store_previous_tracer_source_terms!(Gc⁻, Gcⁿ, grid)
    end
    return nothing
end

function store_previous_velocity_source_terms!(G⁻, Gⁿ, grid)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds G⁻.u[i, j, k] = Gⁿ.u[i, j, k]
                @inbounds G⁻.v[i, j, k] = Gⁿ.v[i, j, k]
                @inbounds G⁻.w[i, j, k] = Gⁿ.w[i, j, k]
            end
        end
    end
    return nothing
end

function store_previous_tracer_source_terms!(Gc⁻, Gcⁿ, grid)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds Gc⁻[i, j, k] = Gcⁿ[i, j, k]
            end
        end
    end
    return nothing
end

"""
Update the hydrostatic pressure perturbation pHY′. This is done by integrating
the `buoyancy_perturbation` downwards:

    `pHY′ = ∫ buoyancy_perturbation dz` from `z=0` down to `z=-Lz`
"""
function update_hydrostatic_pressure!(pHY′, grid, buoyancy, C)
    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
            @inbounds pHY′[i, j, 1] = - ▶z_aaf(i, j, 1, grid, buoyancy_perturbation, buoyancy, C) * grid.Δz
            @unroll for k in 2:grid.Nz
                @inbounds pHY′[i, j, k] = 
                    pHY′[i, j, k-1] - ▶z_aaf(i, j, k, grid, buoyancy_perturbation, buoyancy, C) * grid.Δz
            end
        end
    end
end

""" Calculate the right-hand-side of the u-momentum equation. """
function calculate_Gu!(Gu, grid, coriolis, closure, U, C, K, F, pHY′, parameters, time)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds Gu[i, j, k] = (-u∇u(grid, U.u, U.v, U.w, i, j, k)
                                            - x_f_cross_U(i, j, k, grid, coriolis, U)
                                            - ∂x_p(i, j, k, grid, pHY′)
                                            + ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure, U.u, U.v, U.w, K)
                                            + F.u(i, j, k, grid, time, U, C, parameters))
            end
        end
    end
end

""" Calculate the right-hand-side of the v-momentum equation. """
function calculate_Gv!(Gv, grid, coriolis, closure, U, C, K, F, pHY′, parameters, time)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds Gv[i, j, k] = (-u∇v(grid, U.u, U.v, U.w, i, j, k)
                                            - y_f_cross_U(i, j, k, grid, coriolis, U)
                                            - ∂y_p(i, j, k, grid, pHY′)
                                            + ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure, U.u, U.v, U.w, K)
                                            + F.v(i, j, k, grid, time, U, C, parameters))
            end
        end
    end
end

""" Calculate the right-hand-side of the w-momentum equation. """
function calculate_Gw!(Gw, grid, coriolis, closure, U, C, K, F, parameters, time)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds Gw[i, j, k] = (-u∇w(grid, U.u, U.v, U.w, i, j, k)
                                            - z_f_cross_U(i, j, k, grid, coriolis, U)
                                            + ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure, U.u, U.v, U.w, K)
                                            + F.w(i, j, k, grid, time, U, C, parameters))
            end
        end
    end
end

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
function calculate_Gc!(Gc, grid, closure, tracer_idx, U, C, K, F, parameters, time)
    c = C[tracer_idx]
    Fc = F[tracer_idx]
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds Gc[i, j, k] = (-div_flux(grid, U.u, U.v, U.w, c, i, j, k)
                                            + ∇_κ_∇c(i, j, k, grid, c, tracer_idx, closure, K)
                                            + Fc(i, j, k, grid, time, U, C, parameters))
            end
        end
    end
end

""" Store previous value of the source term and calculate current source term. """
function calculate_interior_source_terms!(G, arch, grid, coriolis, closure, U, C, pHY′, K, F, parameters, time)

    Bx, By, Bz = floor(Int, grid.Nx/Tx), floor(Int, grid.Ny/Ty), grid.Nz  # Blocks in grid

    @launch device(arch) threads=(Tx, Ty) blocks=(Bx, By, Bz) calculate_Gu!(G.u, grid, coriolis, closure, U, C, K, F, 
                                                                            pHY′, parameters, time)

    @launch device(arch) threads=(Tx, Ty) blocks=(Bx, By, Bz) calculate_Gv!(G.v, grid, coriolis, closure, U, C, K, F, 
                                                                            pHY′, parameters, time)

    @launch device(arch) threads=(Tx, Ty) blocks=(Bx, By, Bz) calculate_Gw!(G.w, grid, coriolis, closure, U, C, K, F, 
                                                                            parameters, time)

    for tracer_idx in 1:length(C)
        @inbounds Gc = G[tracer_idx]
        @launch device(arch) threads=(Tx, Ty) blocks=(Bx, By, Bz) calculate_Gc!(Gc, grid, closure, tracer_idx, 
                                                                                U, C, K, F, parameters, time)
                                                                                
    end
end

"""
Evaluate the right-hand-side terms at time step n+½ using a weighted 2nd-order
Adams-Bashforth method

    `G^{n+½} = (3/2 + χ)G^{n} - (1/2 + χ)G^{n-1}`
"""
function adams_bashforth_update_velocity_source_terms!(Gⁿ, G⁻, grid::AbstractGrid{FT}, χ) where FT
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds Gⁿ.u[i, j, k] = (FT(1.5) + χ) * Gⁿ.u[i, j, k] - (FT(0.5) + χ) * G⁻.u[i, j, k]
                @inbounds Gⁿ.v[i, j, k] = (FT(1.5) + χ) * Gⁿ.v[i, j, k] - (FT(0.5) + χ) * G⁻.v[i, j, k]
                @inbounds Gⁿ.w[i, j, k] = (FT(1.5) + χ) * Gⁿ.w[i, j, k] - (FT(0.5) + χ) * G⁻.w[i, j, k]
            end
        end
    end
end

"""
Evaluate the right-hand-side terms at time step n+½ using a weighted 2nd-order
Adams-Bashforth method

    `G^{n+½} = (3/2 + χ)G^{n} - (1/2 + χ)G^{n-1}`
"""
function adams_bashforth_update_tracer_source_term!(Gcⁿ, Gc⁻, grid::AbstractGrid{FT}, χ) where FT
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds Gcⁿ[i, j, k] = (FT(1.5) + χ) * Gcⁿ[i, j, k] - (FT(0.5) + χ) * Gc⁻[i, j, k]
            end
        end
    end
end

function adams_bashforth_update_source_terms!(Gⁿ, G⁻, arch, grid, χ)
    @launch device(arch) config=launch_config(grid, 3) adams_bashforth_update_velocity_source_terms!(Gⁿ, G⁻, grid, χ)
    for i in 4:length(Gⁿ)
        @inbounds Gcⁿ = Gⁿ[i]
        @inbounds Gc⁻ = G⁻[i]
        @launch device(arch) config=launch_config(grid, 3) adams_bashforth_update_tracer_source_term!(Gcⁿ, Gc⁻, grid, χ)
    end
    return nothing
end

"""
Calculate the right-hand-side of the elliptic Poisson equation for the non-hydrostatic
pressure

    `∇²ϕ_{NH}^{n+1} = (∇·u^n)/Δt + ∇·(Gu, Gv, Gw)`
"""
function calculate_poisson_right_hand_side!(::CPU, grid, ::PoissonBCs, Δt, U, G, RHS)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # Calculate divergence of the RHS source terms (Gu, Gv, Gw).
                @inbounds RHS[i, j, k] = div_f2c(grid, U.u, U.v, U.w, i, j, k) / Δt +
                                         div_f2c(grid, G.u, G.v, G.w, i, j, k)
            end
        end
    end
end

"""
Calculate the right-hand-side of the elliptic Poisson equation for the non-hydrostatic
pressure and in the process apply the permutation

    [a, b, c, d, e, f, g, h] -> [a, c, e, g, h, f, d, b]

in the z-direction which is required by the GPU fast cosine transform algorithm for
horizontally periodic model configurations.
"""
function calculate_poisson_right_hand_side!(::GPU, grid, ::PPN, Δt, U, G, RHS)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    @loop for k in (1:Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                if (k & 1) == 1  # isodd(k)
                    k′ = convert(UInt32, CUDAnative.floor(k/2) + 1)
                else
                    k′ = convert(UInt32, Nz - CUDAnative.floor((k-1)/2))
                end
                @inbounds RHS[i, j, k′] = div_f2c(grid, U.u, U.v, U.w, i, j, k) / Δt +
                                          div_f2c(grid, G.u, G.v, G.w, i, j, k)
            end
        end
    end
end

"""
Calculate the right-hand-side of the elliptic Poisson equation for the non-hydrostatic
pressure and in the process apply the permutation

    [a, b, c, d, e, f, g, h] -> [a, c, e, g, h, f, d, b]

in the y- and z-directions which is required by the GPU fast cosine transform algorithm for
reentrant channel model configurations.
"""
function calculate_poisson_right_hand_side!(::GPU, grid, ::PNN, Δt, U, G, RHS)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                if (k & 1) == 1  # isodd(k)
                    k′ = convert(UInt32, CUDAnative.floor(k/2) + 1)
                else
                    k′ = convert(UInt32, Nz - CUDAnative.floor((k-1)/2))
                end

                if (j & 1) == 1  # isodd(j)
                    j′ = convert(UInt32, CUDAnative.floor(j/2) + 1)
                else
                    j′ = convert(UInt32, Ny - CUDAnative.floor((j-1)/2))
                end

                @inbounds RHS[i, j′, k′] = div_f2c(grid, U.u, U.v, U.w, i, j, k) / Δt +
                                           div_f2c(grid, G.u, G.v, G.w, i, j, k)
            end
        end
    end
end

"""
Copy the non-hydrostatic pressure into `pNHS` and undo the permutation

    [a, b, c, d, e, f, g, h] -> [a, c, e, g, h, f, d, b]

along the z-direction.
"""
function idct_permute!(pNHS, grid, ::PPN, ϕ)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    @loop for k in (1:Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                if k <= Nz/2
                    @inbounds pNHS[i, j, 2k-1] = real(ϕ[i, j, k])
                else
                    @inbounds pNHS[i, j, 2(Nz-k+1)] = real(ϕ[i, j, k])
                end
            end
        end
    end
end

"""
Copy the non-hydrostatic pressure into `pNHS` and undo the permutation

    [a, b, c, d, e, f, g, h] -> [a, c, e, g, h, f, d, b]

along the y- and z-direction.
"""
function idct_permute!(pNHS, grid, ::PNN, ϕ)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                if k <= Nz/2
                    k′ = 2k-1
                else
                    k′ = 2(Nz-k+1)
                end

                if j <= Ny/2
                    j′ = 2j-1
                else
                    j′ = 2(Ny-j+1)
                end

                @inbounds pNHS[i, j′, k′] = real(ϕ[i, j, k])
            end
        end
    end
end

"""
Update the horizontal velocities u and v via

    `u^{n+1} = u^n + (Gu^{n+½} - δₓp_{NH} / Δx) Δt`

and the tracers via

    `c^{n+1} = c^n + Gc^{n+½} Δt`

Note that the vertical velocity is not explicitly time stepped.
"""
function update_velocities!(U, G, pNHS, grid, Δt)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds U.u[i, j, k] += (G.u[i, j, k] - ∂x_p(i, j, k, grid, pNHS)) * Δt
                @inbounds U.v[i, j, k] += (G.v[i, j, k] - ∂y_p(i, j, k, grid, pNHS)) * Δt
            end
        end
    end
    return nothing
end

function update_tracer!(c, Gc, grid, Δt)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds c[i, j, k] += Gc[i, j, k] * Δt
            end
        end
    end
    return nothing
end

function update_tracers!(C, G, arch, grid, Δt)
    for i in eachindex(C)
        @inbounds c = C[i]
        @inbounds Gc = G[i+3]
        @launch device(arch) config=launch_config(grid, 3) update_tracer!(c, Gc, grid, Δt)
    end
    return nothing
end

"""
Compute the vertical velocity w by integrating the continuity equation downwards

    `w^{n+1} = -∫ [∂/∂x (u^{n+1}) + ∂/∂y (v^{n+1})] dz`
"""
function compute_w_from_continuity!(U, grid)
    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
            @inbounds U.w[i, j, 1] = 0
            @unroll for k in 2:grid.Nz
                @inbounds U.w[i, j, k] = U.w[i, j, k-1] + grid.Δz * ∇h_u(i, j, k-1, grid, U.u, U.v)
            end
        end
    end
end

""" Apply boundary conditions by adding flux divergences to the right-hand-side. """
function calculate_boundary_source_terms!(Gⁿ, arch, grid, bcs, args...)

    # Velocity fields
    for (i, ubcs) in enumerate(bcs[1:3])
        apply_z_bcs!(Gⁿ[i], arch, grid, ubcs.z.left, ubcs.z.right, args...)
    end

    # Tracer fields
    for (i, cbcs) in enumerate(bcs[4:end])
        apply_z_bcs!(Gⁿ[i+3], arch, grid, cbcs.z.left, cbcs.z.right, args...)
    end

    return nothing
end
