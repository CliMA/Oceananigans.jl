# Time small-file and NetCDF create/write/close cycles in each directory given on the command line,
# to compare the I/O latency of the partitions the tests could write their output to.
#
# Usage: julia --project=test filesystem_benchmark.jl <dir>...
using NCDatasets
using Printf

function time_cycles(f, dir, n)
    times = Float64[]
    for i in 1:n
        path = joinpath(dir, "fs_benchmark_$(getpid())_$i")
        t = @elapsed f(path)
        push!(times, t)
        rm(path; force=true, recursive=true)
    end
    sort!(times)
    return (median=times[cld(n, 2)], p90=times[ceil(Int, 0.9n)], max=times[end])
end

function small_file(path; fsync=false)
    open(path, "w") do io
        write(io, rand(UInt8, 2^16))
        flush(io)
        fsync && ccall(:fsync, Cint, (Cint,), fd(io))
    end
end

function netcdf_file(path)
    NCDataset(path, "c") do ds
        defDim(ds, "x", 32)
        defDim(ds, "time", Inf)
        v = defVar(ds, "u", Float64, ("x", "time"))
        for n in 1:10
            v[:, n] = rand(32)
        end
    end
end

# Reopen and append, like NetCDFWriter does at every write.
function netcdf_appends(path)
    netcdf_file(path)
    for n in 11:30
        NCDataset(path, "a") do ds
            ds["u"][:, n] = rand(32)
        end
    end
end

netcdf_file(joinpath(mktempdir(), "warmup.nc"))
netcdf_appends(joinpath(mktempdir(), "warmup.nc"))

@printf("%-60s %-28s %10s %10s %10s\n", "directory", "operation", "median ms", "p90 ms", "max ms")
for dir in ARGS
    isdir(dir) || continue
    try
        for (label, f, n) in (("64 KiB write", p -> small_file(p), 200),
                              ("64 KiB write + fsync", p -> small_file(p; fsync=true), 50),
                              ("NetCDF create", netcdf_file, 50),
                              ("NetCDF create + 20 appends", netcdf_appends, 10))
            r = time_cycles(f, dir, n)
            @printf("%-60s %-28s %10.3f %10.3f %10.3f\n", dir, label, 1e3r.median, 1e3r.p90, 1e3r.max)
        end
    catch err
        println(dir, ": ", sprint(showerror, err))
    end
end
