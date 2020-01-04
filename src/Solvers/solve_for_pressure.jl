using Oceananigans.Operators

using Oceananigans: @loop_xyz

#####
##### Pressure-related functions
#####

"Solve the Poisson equation for non-hydrostatic pressure on the CPU."
function solve_for_pressure!(pressure, ::CPU, grid, poisson_solver, ϕ)
    solve_poisson_3d!(poisson_solver, grid)
    view(pressure, 1:grid.Nx, 1:grid.Ny, 1:grid.Nz) .= real.(ϕ)
    return nothing
end

"Solve the Poisson equation for non-hydrostatic pressure on the GPU."
function solve_for_pressure!(pressure, ::GPU, grid, poisson_solver, ϕ)
    solve_poisson_3d!(poisson_solver, grid)
    @launch(device(GPU()), config=launch_config(grid, :xyz),
            idct_permute!(pressure, grid, poisson_solver.bcs, ϕ))
    return nothing
end

"""
Calculate the right-hand-side of the Poisson equation for the non-hydrostatic
pressure

    `∇²ϕ_{NH}^{n+1} = (∇·u^n)/Δt + ∇·(Gu, Gv, Gw)`
"""
function calculate_poisson_right_hand_side!(RHS, ::CPU, grid, ::PoissonBCs, U, G, Δt)
    @loop_xyz i j k grid begin
            @inbounds RHS[i, j, k] = divᶜᶜᶜ(i, j, k, grid, U.u, U.v, U.w) / Δt +
                                     divᶜᶜᶜ(i, j, k, grid, G.u, G.v, G.w)
    end

    return nothing
end

"""
Calculate the right-hand-side of the Poisson equation for the non-hydrostatic
pressure and in the process apply the permutation

    [a, b, c, d, e, f, g, h] -> [a, c, e, g, h, f, d, b]

in the z-direction which is required by the GPU fast cosine transform algorithm for
horizontally periodic model configurations.
"""
function calculate_poisson_right_hand_side!(RHS, ::GPU, grid, ::PPN, U, G, Δt)
    Nz = grid.Nz
    @loop_xyz i j k grid begin
        if (k & 1) == 1  # isodd(k)
            k′ = convert(UInt32, CUDAnative.floor(k/2) + 1)
        else
            k′ = convert(UInt32, Nz - CUDAnative.floor((k-1)/2))
        end

        @inbounds RHS[i, j, k′] = divᶜᶜᶜ(i, j, k, grid, U.u, U.v, U.w) / Δt +
                                  divᶜᶜᶜ(i, j, k, grid, G.u, G.v, G.w)
    end
    return nothing
end

"""
Calculate the right-hand-side of the Poisson equation for the non-hydrostatic
pressure and in the process apply the permutation

    [a, b, c, d, e, f, g, h] -> [a, c, e, g, h, f, d, b]

in the y- and z-directions which is required by the GPU fast cosine transform algorithm for
reentrant channel model configurations.
"""
function calculate_poisson_right_hand_side!(RHS, ::GPU, grid, ::PNN, U, G, Δt)
    Ny, Nz = grid.Ny, grid.Nz
    @loop_xyz i j k grid begin
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

        @inbounds RHS[i, j′, k′] = divᶜᶜᶜ(i, j, k, grid, U.u, U.v, U.w) / Δt +
                                   divᶜᶜᶜ(i, j, k, grid, G.u, G.v, G.w)
    end
    return nothing
end

"""
Copy the non-hydrostatic pressure into `pNHS` and undo the permutation

    [a, b, c, d, e, f, g, h] -> [a, c, e, g, h, f, d, b]

along the z-direction.
"""
function idct_permute!(pNHS, grid, ::PPN, ϕ)
    Nz = grid.Nz
    @loop_xyz i j k grid begin
        if k <= Nz/2
            @inbounds pNHS[i, j, 2k-1] = real(ϕ[i, j, k])
        else
            @inbounds pNHS[i, j, 2(Nz-k+1)] = real(ϕ[i, j, k])
        end
    end
    return nothing
end

"""
Copy the non-hydrostatic pressure into `pNHS` and undo the permutation

    [a, b, c, d, e, f, g, h] -> [a, c, e, g, h, f, d, b]

along the y- and z-direction.
"""
function idct_permute!(pNHS, grid, ::PNN, ϕ)
    Ny, Nz = grid.Ny, grid.Nz
    @loop_xyz i j k grid begin
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
    return nothing
end
