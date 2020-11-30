struct PressureSolver{A, G, W, S, B, T, C}
    architecture :: A
            grid :: G
     wavenumbers :: W
         storage :: S
          buffer :: B
      transforms :: T
       constants :: C
end

function PressureSolver(arch, grid, planner_flag=FFTW.PATIENT)
    topo = (TX, TY, TZ) =  topology(grid)

    wavenumbers = (kx² = λx(grid, TX()),
                   ky² = λy(grid, TY()),
                   kz² = λz(grid, TZ()))

    constants = nothing
    # kz⁺ = reshape(0:Nz-1,       1, 1, Nz)
    # kz⁻ = reshape(0:-1:-(Nz-1), 1, 1, Nz)

    # ω_4Nz⁺ = ω.(4Nz, kz⁺) |> CuArray
    # ω_4Nz⁻ = ω.(4Nz, kz⁻) |> CuArray

    storage = arch_array(arch, zeros(complex(eltype(grid)), size(grid)...))

    transforms, buffer_needed = plan_transforms(arch, topo, storage, planner_flag)

    buffer = buffer_needed ? similar(storage) : nothing

    return PressureSolver(arch, grid, wavenumbers, storage, buffer, transforms, constants)
end
