using Oceananigans.Architectures: CPU, GPU

# This should be deprectated. Calls GPU() which is only
# defined when CUDA is loaded and maps to CUDAGPU()
function versioninfo_with_gpu()
    if isdefined(Main, :CUDA)
        try
            return versioninfo_with_gpu(GPU())
        catch
            return "No GPU device found."
        end
    else
        return ""
    end
end

function versioninfo_with_gpu(::CPU)
    return "No GPU device"
end

oceananigans_versioninfo() = "Oceananigans v$(pkgversion(@__MODULE__))"
