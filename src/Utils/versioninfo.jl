using CUDA
using InteractiveUtils
using Oceananigans.Architectures

function versioninfo_with_gpu()
    s = sprint(versioninfo)
    @hascuda begin
        gpu_name = CuCurrentContext() |> device |> name
        s = s * "  GPU: $gpu_name\n"
    end
    return s
end
