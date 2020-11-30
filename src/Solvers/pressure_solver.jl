struct PressureSolver{A, G, E, S, B, T, C}
    architecture :: A
            grid :: G
     eigenvalues :: E
         storage :: S
          buffer :: B
      transforms :: T
       constants :: C
end

function PressureSolver(arch, grid, planner_flag=FFTW.PATIENT)
    topo = (TX, TY, TZ) =  topology(grid)

    λx = poisson_eigenvalues(grid.Nx, grid.Lx, 1, TX())
    λy = poisson_eigenvalues(grid.Ny, grid.Ly, 2, TY())
    λz = poisson_eigenvalues(grid.Nz, grid.Lz, 3, TZ())

    eigenvalues = (
        λx = arch_array(arch, λx),
        λy = arch_array(arch, λy),
        λz = arch_array(arch, λz)
    )

    constants = nothing
    # kz⁺ = reshape(0:Nz-1,       1, 1, Nz)
    # kz⁻ = reshape(0:-1:-(Nz-1), 1, 1, Nz)

    # ω_4Nz⁺ = ω.(4Nz, kz⁺) |> CuArray
    # ω_4Nz⁻ = ω.(4Nz, kz⁻) |> CuArray

    storage = arch_array(arch, zeros(complex(eltype(grid)), size(grid)...))

    transforms, buffer_needed = plan_transforms(arch, topo, storage, planner_flag)

    buffer = buffer_needed ? similar(storage) : nothing

    return PressureSolver(arch, grid, eigenvalues, storage, buffer, transforms, constants)
end
