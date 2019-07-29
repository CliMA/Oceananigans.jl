@hascuda using CUDAnative, CuArrays

import GPUifyLoops: @launch, @loop, @unroll

using Oceananigans.Operators

const Tx = 16 # CUDA threads per x-block
const Ty = 16 # CUDA threads per y-block

@inline datatuple(obj, flds=propertynames(obj)) = NamedTuple{flds}(getproperty(obj, fld).data for fld in flds)
@inline datatuples(objs...) = (datatuple(obj) for obj in objs)

"""
    time_step!(model, Nt, Δt)

Step forward `model` `Nt` time steps using a second-order Adams-Bashforth
method with step size `Δt`.
"""
function time_step!(model, Nt, Δt)

    if model.clock.iteration == 0
        [ write_output(model, output_writer) for output_writer in model.output_writers ]
        [ run_diagnostic(model, diagnostic) for diagnostic in model.diagnostics ]
    end

    # Unpack model fields
              arch = model.arch
              grid = model.grid
             clock = model.clock
               eos = model.eos
         constants = model.constants
                pr = model.pressures
           forcing = model.forcing
           closure = model.closure
    poisson_solver = model.poisson_solver
     diffusivities = model.diffusivities
               bcs = model.boundary_conditions

    # We can use the same array for the right-hand-side RHS and the solution ϕ.
    RHS = poisson_solver.storage
    U, Φ, Gⁿ, G⁻ = datatuples(model.velocities, model.tracers, model.G, model.Gp)
    Guvw = (Gu=Gⁿ.Gu, Gv=Gⁿ.Gv, Gw=Gⁿ.Gw)
    FT = eltype(grid)

    # Field tuples for fill_halo_regions.
    u_ft = (:u, bcs.u, U.u)
    v_ft = (:v, bcs.v, U.v)
    w_ft = (:w, bcs.w, U.w)
    T_ft = (:T, bcs.T, Φ.T)
    S_ft = (:S, bcs.S, Φ.S)
    Gu_ft = (:u, bcs.u, Gⁿ.Gu)
    Gv_ft = (:v, bcs.v, Gⁿ.Gv)
    Gw_ft = (:w, bcs.w, Gⁿ.Gw)
    pHY′_ft = (:w, bcs.w, pr.pHY′.data)
    pNHS_ft = (:w, bcs.w, pr.pNHS.data)

    uvw_ft = (u_ft, v_ft, w_ft)
    TS_ft = (T_ft, S_ft)
    Guvw_ft = (Gu_ft, Gv_ft, Gw_ft)

    for n in 1:Nt
        χ = ifelse(model.clock.iteration == 0, FT(-0.5), FT(0.125)) # Adams-Bashforth (AB2) parameter.

        @launch device(arch) config=launch_config(grid, 3) store_previous_source_terms!(grid, Gⁿ, G⁻)
        @launch device(arch) config=launch_config(grid, 2) update_hydrostatic_pressure!(grid, constants, eos, Φ..., pr.pHY′.data)

        @launch device(arch) config=launch_config(grid, 3) calc_diffusivities!(diffusivities, grid, closure, eos, constants.g, U, Φ)
                                                           fill_halo_regions!(grid, uvw_ft..., TS_ft..., pHY′_ft)
                                                           calculate_interior_source_terms!(arch, grid, constants, eos, closure, U, Φ,
                                                                                            pr.pHY′.data, Gⁿ, diffusivities, forcing)
                                                           calculate_boundary_source_terms!(model)
        @launch device(arch) config=launch_config(grid, 3) adams_bashforth_update_source_terms!(grid, Gⁿ, G⁻, χ)
                                                           fill_halo_regions!(grid, Guvw_ft...)
        @launch device(arch) config=launch_config(grid, 3) calculate_poisson_right_hand_side!(arch, grid, poisson_solver.bcs, Δt, U..., Guvw..., RHS)
                                                           solve_for_pressure!(arch, model)
                                                           fill_halo_regions!(grid, pNHS_ft)
        @launch device(arch) config=launch_config(grid, 3) update_velocities_and_tracers!(grid, U, Φ, pr.pNHS.data, Gⁿ, Δt)
                                                           fill_halo_regions!(grid, uvw_ft...)
        @launch device(arch) config=launch_config(grid, 2) compute_w_from_continuity!(grid, U...)

        clock.time += Δt
        clock.iteration += 1

        for diagnostic in model.diagnostics
            (clock.iteration % diagnostic.diagnostic_frequency) == 0 && run_diagnostic(model, diagnostic)
        end

        for output_writer in model.output_writers
            (clock.iteration % output_writer.output_frequency) == 0 && write_output(model, output_writer)
        end
    end

    return nothing
end

# dynamic launch configuration
function launch_config(grid, dims)
    return function (kernel)
        fun = kernel.fun
        config = launch_configuration(fun)

        # adapt the suggested config from 1D to the requested grid dimensions
        if dims == 3
            threads = floor(Int, cbrt(config.threads))
            blocks = ceil.(Int, [grid.Nx, grid.Ny, grid.Nz] ./ threads)
            threads = [threads, threads, threads]
        elseif dims == 2
            threads = floor(Int, sqrt(config.threads))
            blocks = ceil.(Int, [grid.Nx, grid.Ny] ./ threads)
            threads = [threads, threads]
        else
            error("unsupported launch configuration")
        end

        return (threads=Tuple(threads), blocks=Tuple(blocks))
    end
end

time_step!(model; Nt, Δt) = time_step!(model, Nt, Δt)

function solve_for_pressure!(::CPU, model::Model)
    RHS, ϕ = model.poisson_solver.storage, model.poisson_solver.storage

    solve_poisson_3d!(model.poisson_solver, model.grid)
    data(model.pressures.pNHS) .= real.(ϕ)
end

function solve_for_pressure!(::GPU, model::Model)
    RHS, ϕ = model.poisson_solver.storage, model.poisson_solver.storage

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
function update_hydrostatic_pressure!(grid::Grid, constants, eos, T, S, pHY′)
    gΔz = constants.g * grid.Δz
    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
            @inbounds pHY′[i, j, 1] = 0.5 * gΔz * δρ(eos, T, S, i, j, 1)
            @unroll for k in 2:grid.Nz
                @inbounds pHY′[i, j, k] = pHY′[i, j, k-1] + gΔz * 0.5 * (δρ(eos, T, S, i, j, k-1) + δρ(eos, T, S, i, j, k))
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
function calculate_poisson_right_hand_side!(::CPU, grid::Grid, ::PoissonBCs, Δt, u, v, w, Gu, Gv, Gw, RHS)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # Calculate divergence of the RHS source terms (Gu, Gv, Gw).
                @inbounds RHS[i, j, k] = div_f2c(grid, u, v, w, i, j, k) / Δt + div_f2c(grid, Gu, Gv, Gw, i, j, k)
            end
        end
    end
end

"""
    calculate_poisson_right_hand_side!(::GPU, grid::Grid, ::PPN, Δt, u, v, w, Gu, Gv, Gw, RHS)

Calculate divergence of the RHS source terms (Gu, Gv, Gw) and applying a permutation
which is the first step in the DCT.
"""
function calculate_poisson_right_hand_side!(::GPU, grid::Grid, ::PPN, Δt, u, v, w, Gu, Gv, Gw, RHS)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    @loop for k in (1:Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                if (k & 1) == 1  # isodd(k)
                    k′ = convert(UInt32, CUDAnative.floor(k/2) + 1)
                else
                    k′ = convert(UInt32, Nz - CUDAnative.floor((k-1)/2))
                end
                @inbounds RHS[i, j, k′] = div_f2c(grid, u, v, w, i, j, k) / Δt + div_f2c(grid, Gu, Gv, Gw, i, j, k)
            end
        end
    end
end

function calculate_poisson_right_hand_side!(::GPU, grid::Grid, ::PNN, Δt, u, v, w, Gu, Gv, Gw, RHS)
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

                @inbounds RHS[i, j′, k′] = div_f2c(grid, u, v, w, i, j, k) / Δt + div_f2c(grid, Gu, Gv, Gw, i, j, k)
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
function compute_w_from_continuity!(grid::Grid, u, v, w)
    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
            @inbounds w[i, j, 1] = 0
            @unroll for k in 2:grid.Nz
                @inbounds w[i, j, k] = w[i, j, k-1] + grid.Δz * ∇h_u(i, j, k-1, grid, u, v)
            end
        end
    end
end

get_ν(closure::IsotropicDiffusivity, K) = closure.ν
get_κ(closure::IsotropicDiffusivity, K) = (T=closure.κ, S=closure.κ)

get_ν(closure::ConstantAnisotropicDiffusivity, K) = closure.νv
get_κ(closure::ConstantAnisotropicDiffusivity, K) = (T=closure.κv, S=closure.κv)

# get_κ looks wrong here because κ = ν / Pr but ConstantSmagorinsky does not compute or store κ, so we pass κ = ν to
# apply_bcs! which will compute κ correctly as it knows both ν and Pr.
get_ν(closure::ConstantSmagorinsky, K) = K.νₑ
get_κ(closure::ConstantSmagorinsky, K) = (T=K.νₑ, S=K.νₑ)

get_ν(closure::AnisotropicMinimumDissipation, K) = K.νₑ
get_κ(closure::AnisotropicMinimumDissipation, K) = (T=K.κₑ.T, S=K.κₑ.S)

"Apply boundary conditions by modifying the source term G."
function calculate_boundary_source_terms!(model)
    arch = model.arch
    grid = model.grid
    clock = model.clock
    eos =  model.eos
    closure = model.closure
    bcs = model.boundary_conditions
    grav = model.constants.g
    t, iteration = clock.time, clock.iteration
    U, Φ, G = datatuples(model.velocities, model.tracers, model.G)

    Bx, By, Bz = floor(Int, model.grid.Nx/Tx), floor(Int, model.grid.Ny/Ty), model.grid.Nz  # Blocks in grid

    coord = :z #for coord in (:x, :y, :z) when we are ready to support more coordinates.

    u_x_bcs = getproperty(bcs.u, coord)
    v_x_bcs = getproperty(bcs.v, coord)
    w_x_bcs = getproperty(bcs.w, coord)
    T_x_bcs = getproperty(bcs.T, coord)
    S_x_bcs = getproperty(bcs.S, coord)

    # Apply boundary conditions in the vertical direction.
    ν = get_ν(closure, model.diffusivities)
    κ = get_κ(closure, model.diffusivities)

    apply_bcs!(arch, Val(coord), Bx, By, Bz, u_x_bcs.left, u_x_bcs.right, grid, U.u, G.Gu, ν,
        closure, eos, grav, t, iteration, U, Φ)

    apply_bcs!(arch, Val(coord), Bx, By, Bz, v_x_bcs.left, v_x_bcs.right, grid, U.v, G.Gv, ν,
        closure, eos, grav, t, iteration, U, Φ)

    apply_bcs!(arch, Val(coord), Bx, By, Bz, T_x_bcs.left, T_x_bcs.right, grid, Φ.T, G.GT, κ.T,
        closure, eos, grav, t, iteration, U, Φ)

    apply_bcs!(arch, Val(coord), Bx, By, Bz, S_x_bcs.left, S_x_bcs.right, grid, Φ.S, G.GS, κ.S,
        closure, eos, grav, t, iteration, U, Φ)

    return nothing
end
