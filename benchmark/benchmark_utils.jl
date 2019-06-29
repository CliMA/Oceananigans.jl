arch_name(::CPU) = "CPU"
arch_name(::GPU) = "GPU"

benchmark_name(N)               = benchmark_name(N, nothing, nothing)
benchmark_name(N, arch::Symbol) = benchmark_name(N, arch, nothing)
benchmark_name(N, ft::DataType) = benchmark_name(N, nothing, ft)

function benchmark_name(N, arch, ft; npad=3)
    Nx, Ny, Nz = N
    print_arch = typeof(arch) <: Architecture ? true : false
    print_ft   = typeof(ft) == DataType && ft <: AbstractFloat ? true : false

    bn = ""
    bn *= lpad(Nx, npad, " ") * "×" * lpad(Ny, npad, " ") * "×" * lpad(Nz, npad, " ")

    if print_arch && print_ft
        arch = arch_name(arch)
        bn *= " ($arch, $ft)"
    elseif print_arch && !print_ft
        arch = arch_name(arch)
        bn *= " ($arch)"
    elseif !print_arch && print_ft
        bn *= " ($ft)"
    end

    return bn
end
