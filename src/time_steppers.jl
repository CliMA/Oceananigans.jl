using Oceananigans.Operators

const Tx = 16 # CUDA threads per x-block
const Ty = 16 # CUDA threads per y-block

"""
    time_step!(model, Nt, Δt)

Step forward `model` `Nt` time steps using a second-order Adams-Bashforth
method with step size `Δt`.
"""
function time_step!(model, Nt, Δt; init_with_euler=true)

    if model.clock.iteration == 0
        [ write_output(model, out)    for out  in model.output_writers ]
        [ run_diagnostic(model, diag) for diag in model.diagnostics ]
    end

    FT = eltype(model.grid)
    RHS = model.poisson_solver.storage
    U, Φ, Gⁿ, G⁻, K, p = datatuples(model.velocities, model.tracers, model.timestepper.Gⁿ,
                                    model.timestepper.G⁻, model.diffusivities, model.pressures)

    for n in 1:Nt
        χ = ifelse(init_with_euler && n==1, FT(-0.5), model.timestepper.χ)

        time_step!(model, model.arch, model.grid, model.constants, model.eos, model.closure,
                   model.forcing, model.boundary_conditions, U, Φ, p, K, RHS, Gⁿ, G⁻, Δt, χ)

        [ time_to_run(model.clock, diag) && run_diagnostic(model, diag) for diag in model.diagnostics ]
        [ time_to_write(model.clock, out) && write_output(model, out) for out in model.output_writers ]
    end

    return nothing
end

"""
    time_step!(args...)

Step forward one time step.
"""
function time_step!(model, arch, grid, constants, eos, closure, forcing, bcs, U, Φ, p, K, RHS, Gⁿ, G⁻, Δt, χ)

    # Pre-computations:
    @launch device(arch) config=launch_config(grid, 3) store_previous_source_terms!(grid, Gⁿ, G⁻)
    fill_halo_regions!(merge(U, Φ), bcs, grid)

    @launch device(arch) config=launch_config(grid, 3) calc_diffusivities!(K, grid, closure, eos, constants.g, U, Φ)
    @launch device(arch) config=launch_config(grid, 2) update_hydrostatic_pressure!(p.pHY′, grid, constants, eos, Φ)
    fill_halo_regions!(p.pHY′, bcs[4], grid)

    # Calc RHS:
    calculate_interior_source_terms!(arch, grid, constants, eos, closure, U, Φ, p.pHY′, Gⁿ, K, forcing)
    calculate_boundary_source_terms!(arch, grid, bcs, model.clock, closure, U, Φ, Gⁿ, K)

    # Complete explicit substep:
    @launch device(arch) config=launch_config(grid, 3) adams_bashforth_update_source_terms!(grid, Gⁿ, G⁻, χ)

    # Start pressure correction substep with a pressure solve:
    fill_halo_regions!(Gⁿ[1:3], bcs[1:3], grid)
    @launch device(arch) config=launch_config(grid, 3) calculate_poisson_right_hand_side!(arch, grid, model.poisson_solver.bcs,
                                                                                          Δt, U, Gⁿ, RHS)
    solve_for_pressure!(arch, model)
    fill_halo_regions!(p.pNHS, bcs[4], grid)

    # Complete pressure correction step:
    @launch device(arch) config=launch_config(grid, 3) update_velocities_and_tracers!(grid, U, Φ, p.pNHS, Gⁿ, Δt)

    # Start pressure correction substep with a pressure solve:
    fill_halo_regions!(U, bcs[1:3], grid)
    @launch device(arch) config=launch_config(grid, 2) compute_w_from_continuity!(grid, U)

    model.clock.time += Δt
    model.clock.iteration += 1

    return nothing
end

time_step!(model; Nt, Δt, kwargs...) = time_step!(model, Nt, Δt; kwargs...)

function solve_for_pressure!(::CPU, model::Model)
    ϕ = model.poisson_solver.storage

    solve_poisson_3d!(model.poisson_solver, model.grid)
    data(model.pressures.pNHS) .= real.(ϕ)
end

function solve_for_pressure!(::GPU, model::Model)
    ϕ = model.poisson_solver.storage

    solve_poisson_3d!(model.poisson_solver, model.grid)
    @launch device(GPU()) config=launch_config(model.grid, 3) idct_permute!(model.grid, model.poisson_solver.bcs, ϕ, model.pressures.pNHS.data)
end

"""Store previous source terms before updating them."""
function store_previous_source_terms!(grid::Grid, Gⁿ, G⁻)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds G⁻.Gu[i, j, k] = Gⁿ.Gu[i, j, k]
                @inbounds G⁻.Gv[i, j, k] = Gⁿ.Gv[i, j, k]
                @inbounds G⁻.Gw[i, j, k] = Gⁿ.Gw[i, j, k]
                @inbounds G⁻.GT[i, j, k] = Gⁿ.GT[i, j, k]
                @inbounds G⁻.GS[i, j, k] = Gⁿ.GS[i, j, k]
            end
        end
    end
end

"Update the hydrostatic pressure perturbation pHY′ and buoyancy δρ."
function update_hydrostatic_pressure!(pHY′, grid, constants, eos, Φ)
    gΔz = constants.g * grid.Δz
    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
            @inbounds pHY′[i, j, 1] = 0.5 * gΔz * δρ(eos, Φ.T, Φ.S, i, j, 1)
            @unroll for k in 2:grid.Nz
                @inbounds pHY′[i, j, k] = pHY′[i, j, k-1] + gΔz * 0.5 * (δρ(eos, Φ.T, Φ.S, i, j, k-1) + δρ(eos, Φ.T, Φ.S, i, j, k))
            end
        end
    end
end

"Store previous value of the source term and calculate current source term."
function calculate_interior_source_terms!(arch, grid, constants, eos, closure, U, Φ, pHY′, G, K, F)

    function calculate_Gu(grid, constants, eos, closure, U, Φ, pHY′, Gu, K, F)
        @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
            @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
                @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                    @inbounds Gu[i, j, k] = (-u∇u(grid, U.u, U.v, U.w, i, j, k)
                                                + fv(grid, U.v, constants.f, i, j, k)
                                                - δx_c2f(grid, pHY′, i, j, k) / (grid.Δx * eos.ρ₀)
                                                + ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, closure, U.u, U.v, U.w, K)
                                                + F.u(grid, U, Φ, i, j, k))
                end
            end
        end
    end

    function calculate_Gv(grid, constants, eos, closure, U, Φ, pHY′, Gv, K, F)
        @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
            @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
                @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                    @inbounds Gv[i, j, k] = (-u∇v(grid, U.u, U.v, U.w, i, j, k)
                                                - fu(grid, U.u, constants.f, i, j, k)
                                                - δy_c2f(grid, pHY′, i, j, k) / (grid.Δy * eos.ρ₀)
                                                + ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, closure, U.u, U.v, U.w, K)
                                                + F.v(grid, U, Φ, i, j, k))
                end
            end
        end
    end

    function calculate_Gw(grid, constants, eos, closure, U, Φ, pHY′, Gw, K, F)
        @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
            @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
                @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                    @inbounds Gw[i, j, k] = (-u∇w(grid, U.u, U.v, U.w, i, j, k)
                                                + ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, closure, U.u, U.v, U.w, K)
                                                + F.w(grid, U, Φ, i, j, k))
                end
            end
        end
    end

    function calculate_GT(grid, constants, eos, closure, U, Φ, pHY′, GT, K, F)
        @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
            @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
                @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                    @inbounds GT[i, j, k] = (-div_flux(grid, U.u, U.v, U.w, Φ.T, i, j, k)
                                                + ∇_κ_∇T(i, j, k, grid, Φ.T, closure, K)
                                                + F.T(grid, U, Φ, i, j, k))
                end
            end
        end
    end

    function calculate_GS(grid, constants, eos, closure, U, Φ, pHY′, GS, K, F)
        @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
            @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
                @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                    @inbounds GS[i, j, k] = (-div_flux(grid, U.u, U.v, U.w, Φ.S, i, j, k)
                                                + ∇_κ_∇S(i, j, k, grid, Φ.S, closure, K)
                                                + F.S(grid, U, Φ, i, j, k))
                end
            end
        end
    end

    Bx, By, Bz = floor(Int, grid.Nx/Tx), floor(Int, grid.Ny/Ty), grid.Nz  # Blocks in grid
    @launch device(arch) threads=(Tx, Ty) blocks=(Bx, By, Bz) calculate_Gu(grid, constants, eos, closure, U, Φ, pHY′, G.Gu, K, F)
    @launch device(arch) threads=(Tx, Ty) blocks=(Bx, By, Bz) calculate_Gv(grid, constants, eos, closure, U, Φ, pHY′, G.Gv, K, F)
    @launch device(arch) threads=(Tx, Ty) blocks=(Bx, By, Bz) calculate_Gw(grid, constants, eos, closure, U, Φ, pHY′, G.Gw, K, F)
    @launch device(arch) threads=(Tx, Ty) blocks=(Bx, By, Bz) calculate_GT(grid, constants, eos, closure, U, Φ, pHY′, G.GT, K, F)
    @launch device(arch) threads=(Tx, Ty) blocks=(Bx, By, Bz) calculate_GS(grid, constants, eos, closure, U, Φ, pHY′, G.GS, K, F)
end

function adams_bashforth_update_source_terms!(grid::Grid{FT}, Gⁿ, G⁻, χ) where FT
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds Gⁿ.Gu[i, j, k] = (FT(1.5) + χ) * Gⁿ.Gu[i, j, k] - (FT(0.5) + χ) * G⁻.Gu[i, j, k]
                @inbounds Gⁿ.Gv[i, j, k] = (FT(1.5) + χ) * Gⁿ.Gv[i, j, k] - (FT(0.5) + χ) * G⁻.Gv[i, j, k]
                @inbounds Gⁿ.Gw[i, j, k] = (FT(1.5) + χ) * Gⁿ.Gw[i, j, k] - (FT(0.5) + χ) * G⁻.Gw[i, j, k]
                @inbounds Gⁿ.GT[i, j, k] = (FT(1.5) + χ) * Gⁿ.GT[i, j, k] - (FT(0.5) + χ) * G⁻.GT[i, j, k]
                @inbounds Gⁿ.GS[i, j, k] = (FT(1.5) + χ) * Gⁿ.GS[i, j, k] - (FT(0.5) + χ) * G⁻.GS[i, j, k]
            end
        end
    end
end

"Store previous value of the source term and calculate current source term."
function calculate_poisson_right_hand_side!(::CPU, grid::Grid, ::PoissonBCs, Δt, U, G, RHS)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # Calculate divergence of the RHS source terms (Gu, Gv, Gw).
                @inbounds RHS[i, j, k] = div_f2c(grid, U.u, U.v, U.w, i, j, k) / Δt + div_f2c(grid, G.Gu, G.Gv, G.Gw, i, j, k)
            end
        end
    end
end

"""
    calculate_poisson_right_hand_side!(::GPU, grid::Grid, ::PPN, Δt, u, v, w, Gu, Gv, Gw, RHS)

Calculate divergence of the RHS source terms (Gu, Gv, Gw) and applying a permutation
which is the first step in the DCT.
"""
function calculate_poisson_right_hand_side!(::GPU, grid::Grid, ::PPN, Δt, U, G, RHS)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    @loop for k in (1:Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                if (k & 1) == 1  # isodd(k)
                    k′ = convert(UInt32, CUDAnative.floor(k/2) + 1)
                else
                    k′ = convert(UInt32, Nz - CUDAnative.floor((k-1)/2))
                end
                @inbounds RHS[i, j, k′] = div_f2c(grid, U.u, U.v, U.w, i, j, k) / Δt + div_f2c(grid, G.Gu, G.Gv, G.Gw, i, j, k)
            end
        end
    end
end

function calculate_poisson_right_hand_side!(::GPU, grid::Grid, ::PNN, Δt, U, G, RHS)
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

                @inbounds RHS[i, j′, k′] = div_f2c(grid, U.u, U.v, U.w, i, j, k) / Δt + div_f2c(grid, G.Gu, G.Gv, G.Gw, i, j, k)
            end
        end
    end
end

function idct_permute!(grid::Grid, ::PPN, ϕ, pNHS)
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

function idct_permute!(grid::Grid, ::PNN, ϕ, pNHS)
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


function update_velocities_and_tracers!(grid::Grid, U, Φ, pNHS, Gⁿ, Δt)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds U.u[i, j, k] = U.u[i, j, k] + (Gⁿ.Gu[i, j, k] - (δx_c2f(grid, pNHS, i, j, k) / grid.Δx)) * Δt
                @inbounds U.v[i, j, k] = U.v[i, j, k] + (Gⁿ.Gv[i, j, k] - (δy_c2f(grid, pNHS, i, j, k) / grid.Δy)) * Δt
                @inbounds Φ.T[i, j, k] = Φ.T[i, j, k] + (Gⁿ.GT[i, j, k] * Δt)
                @inbounds Φ.S[i, j, k] = Φ.S[i, j, k] + (Gⁿ.GS[i, j, k] * Δt)
            end
        end
    end
end

"Compute the vertical velocity w from the continuity equation."
function compute_w_from_continuity!(grid::Grid, U)
    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
            @inbounds U.w[i, j, 1] = 0
            @unroll for k in 2:grid.Nz
                @inbounds U.w[i, j, k] = U.w[i, j, k-1] + grid.Δz * ∇h_u(i, j, k-1, grid, U.u, U.v)
            end
        end
    end
end

get_ν(closure::IsotropicDiffusivity, K) = closure.ν
get_κ(closure::IsotropicDiffusivity, K) = (T=closure.κ, S=closure.κ)

get_ν(closure::ConstantAnisotropicDiffusivity, K) = closure.νv
get_κ(closure::ConstantAnisotropicDiffusivity, K) = (T=closure.κv, S=closure.κv)

"Apply boundary conditions by modifying the source term G."
function calculate_boundary_source_terms!(arch, grid, bcs, clock, closure, U, Φ, Gⁿ, K)
    Bx, By, Bz = floor(Int, grid.Nx/Tx), floor(Int, grid.Ny/Ty), grid.Nz  # Blocks in grid

    # Apply boundary conditions in the vertical direction.
    ν = get_ν(closure, K)
    κ = get_κ(closure, K)

    # Velocity fields
    for (i, ubcs) in enumerate(bcs[1:3])
        apply_bcs!(arch, Val(:z), Bx, By, Bz, ubcs.z.left, ubcs.z.right,
                   grid, U[i], Gⁿ[i], ν, closure, clock.time, clock.iteration, U, Φ)
    end

    # Tracer fields
    for (i, ϕbcs) in enumerate(bcs[4:end])
        apply_bcs!(arch, Val(:z), Bx, By, Bz, ϕbcs.z.left, ϕbcs.z.right,
                   grid, Φ[i], Gⁿ[i+3], κ[i], closure, clock.time, clock.iteration, U, Φ)
    end

    return nothing
end
