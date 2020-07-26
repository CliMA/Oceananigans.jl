using CUDA
using Oceananigans.Architectures

arch_name(::CPU) = "CPU"
arch_name(::GPU) = "GPU"

benchmark_name(N)               = benchmark_name(N, "", nothing, nothing)
benchmark_name(N, arch::Symbol) = benchmark_name(N, "", arch, nothing)
benchmark_name(N, ft::DataType) = benchmark_name(N, "", nothing, ft)

function benchmark_name(N, id, arch, FT; npad=3)
    Nx, Ny, Nz = N
    print_arch = typeof(arch) <: AbstractArchitecture ? true : false
    print_FT   = typeof(FT) == DataType && FT <: AbstractFloat ? true : false

    bn = ""
    bn *= lpad(Nx, npad, " ") * "×" * lpad(Ny, npad, " ") * "×" * lpad(Nz, npad, " ")
    bn *= " $id"

    if print_arch && print_FT
        arch = arch_name(arch)
        bn *= " [$arch, $FT]"
    elseif print_arch && !print_FT
        arch = arch_name(arch)
        bn *= " [$arch]"
    elseif !print_arch && print_FT
        bn *= " [$FT]"
    end

    return bn
end
