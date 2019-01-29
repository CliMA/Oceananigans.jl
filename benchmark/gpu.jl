@inline incmod1(a, n) = a == n ? one(a) : a + 1
@inline decmod1(a, n) = a == 1 ? n : a - 1

function ∇²_ppn!(f, ∇²f, Nx, Ny, Nz, Δx, Δy, Δz)
    for k in 2:(Nz-1), j in 1:Ny, i in 1:Nx
       ∇²f[i, j, k] = (f[incmod1(i, Nx), j, k] - 2*f[i, j, k] + f[decmod1(i, Nx), j, k]) / Δx^2 +
                      (f[i, incmod1(j, Ny), k] - 2*f[i, j, k] + f[i, decmod1(j, Ny), k]) / Δy^2 +
                      (f[i, j, k+1]            - 2*f[i, j, k] + f[i, j, k-1])            / Δz^2
    end
    for j in 1:Ny, i in 1:Nx
        ∇²f[i, j,   1] = (f[i, j, 2] - f[i, j, 1]) / Δz^2 +
                         (f[incmod1(i, Nx), j, 1] - 2*f[i, j, 1] + f[decmod1(i, Nx), j, 1]) / Δx^2 +
                         (f[i, incmod1(j, Ny), 1] - 2*f[i, j, 1] + f[i, decmod1(j, Ny), 1]) / Δy^2
        ∇²f[i, j, end] = (f[i, j, end-1] - f[i, j, end]) / Δz^2 +
                         (f[incmod1(i, Nx), j, end] - 2*f[i, j, end] + f[decmod1(i, Nx), j, end]) / Δx^2 +
                         (f[i, incmod1(j, Ny), end] - 2*f[i, j, end] + f[i, decmod1(j, Ny), end]) / Δy^2
    end
    nothing
end

function δx!(f, δxf, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        @inbounds δxf[i, j, k] =  f[incmod1(i, Nx), j, k] - f[i, j, k]
    end
    nothing
end

function δx!(f, δxf, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny
        for i in 1:(Nx-1)
            @inbounds δxf[i, j, k] =  f[i+1, j, k] - f[i, j, k]
        end
        @inbounds δxf[end, j, k] =  f[1, j, k] - f[end, j, k]
    end
    nothing
end

'''
julia> using BenchmarkTools
julia> Nx, Ny, Nz = 128, 128, 128;
julia> xc = rand(Nx, Ny, Nz); yc = zeros(Nx, Ny, Nz);
julia> @btime δx!(xc, yc, Nx, Ny, Nz)
  1.409 ms (0 allocations: 0 bytes)
'''

'''
~ nvprof --profile-from-start off /home/alir/julia/bin/julia
julia> using CUDAnative, CUDAdrv, CuArrays
julia> Nx, Ny, Nz = 128, 128, 128;
julia> xg = cu(rand(Nx, Ny, Nz)); yg = cu(rand(Nx, Ny, Nz));
julia> CUDAdrv.@profile @cuda threads=128 δx!(xg, yg, Nx, Ny, Nz)
julia> CUDAdrv.@profile @cuda threads=128 δx!(xg, yg, Nx, Ny, Nz)

==54821== Profiling application: /home/alir/julia/bin/julia
==54821== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
100.00%  590.15ms         2  295.08ms  294.44ms  295.72ms  ptxcall__x__1

==54821== API calls:
Time(%)      Time     Calls       Avg       Min       Max  Name
 98.85%  9.3003ms         1  9.3003ms  9.3003ms  9.3003ms  cuModuleLoadDataEx
  0.86%  81.203us         2  40.601us  40.423us  40.780us  cuLaunchKernel
  0.08%  7.6330us         1  7.6330us  7.6330us  7.6330us  cuProfilerStart
  0.08%  7.6270us         4  1.9060us     303ns  4.2190us  cuCtxGetCurrent
  0.06%  5.6430us         1  5.6430us  5.6430us  5.6430us  cuDeviceGetCount
  0.04%  3.7120us         4     928ns     196ns  2.3920us  cuDeviceGetAttribute
  0.01%     863ns         1     863ns     863ns     863ns  cuModuleGetFunction
  0.01%     758ns         2     379ns     314ns     444ns  cuDeviceGet
  0.01%     625ns         1     625ns     625ns     625ns  cuCtxGetDevice
'''

@inline incmod1(a, n) = a == n ? 1 : a+1
δx(f, Nx, i, j, k) = f[incmod1(i, Nx), j, k] - f[i, j, k]

function time_stepping(f, δxf)
    Nx, Ny, Nz = size(f)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        δxf[i, j, k] = δx(f, Nx, i, j, k)
    end
end

'''
julia> using BenchmarkTools
julia> Nx, Ny, Nz = 128, 128, 128;
julia> xc = rand(Nx, Ny, Nz); yc = zeros(Nx, Ny, Nz);
julia> @benchmark time_stepping(xc, yc)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     3.377 ms (0.00% GC)
  median time:      3.387 ms (0.00% GC)
  mean time:        3.449 ms (0.00% GC)
  maximum time:     3.865 ms (0.00% GC)
  --------------
  samples:          1449
  evals/sample:     1
'''

'''
~ nvprof --profile-from-start off /home/alir/julia/bin/julia
julia> using CUDAnative, CUDAdrv, CuArrays
julia> Nx, Ny, Nz = 128, 128, 128;
julia> xg = cu(rand(Float32, Nx, Ny, Nz)); yg = cu(rand(Float32, Nx, Ny, Nz));
julia> CUDAdrv.@profile @cuda threads=128 time_stepping(xg, yg)
julia> CUDAdrv.@profile @cuda threads=128 time_stepping(xg, yg)

==55354== Profiling application: /home/alir/julia/bin/julia
==55354== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
100.00%  2.86728s         2  1.43364s  1.42919s  1.43808s  ptxcall_time_stepping_2

==55354== API calls:
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.82%  98.188ms         1  98.188ms  98.188ms  98.188ms  cuModuleLoadDataEx
  0.14%  138.82us         2  69.411us  51.497us  87.326us  cuLaunchKernel
  0.02%  14.819us         1  14.819us  14.819us  14.819us  cuProfilerStart
  0.01%  11.707us         6  1.9510us     173ns  9.1930us  cuDeviceGetAttribute
  0.01%  7.3640us         6  1.2270us     200ns  2.9550us  cuCtxGetCurrent
  0.00%  4.5960us         1  4.5960us  4.5960us  4.5960us  cuDeviceGetCount
  0.00%     977ns         2     488ns     486ns     491ns  cuCtxGetDevice
  0.00%     930ns         1     930ns     930ns     930ns  cuModuleGetFunction
  0.00%     807ns         2     403ns     288ns     519ns  cuDeviceGet
'''

@inline incmod1(i, n) = 1 + (i % n)

'''
julia> @btime δx!(xc, yc, 100, 100, 100)
  3.715 ms (0 allocations: 0 bytes)

julia> @time δx!(xg, yg, 100, 100, 100)
 19.862191 seconds (15.04 M allocations: 673.130 MiB, 0.25% gc time)
'''
