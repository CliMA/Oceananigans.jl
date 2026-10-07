using Profile: Profile
using Serialization: serialize
using LinearAlgebra: BLAS
using FFTW: FFTW

# Write the timing statistics, the process configuration and the profile samples of one test.
function write_profile_record(prefix, name, stats, wall_time)
    mkpath(dirname(prefix))

    open(prefix * ".meta.txt", "w") do io
        println(io, "name                 ", name)
        println(io, "pid                  ", getpid())
        println(io, "wall_time            ", wall_time)
        println(io, "time                 ", stats.time)
        println(io, "compile_time         ", stats.compile_time)
        println(io, "recompile_time       ", stats.recompile_time)
        println(io, "gctime               ", stats.gctime)
        println(io, "bytes                ", stats.bytes)
        println(io, "nthreads_default     ", Threads.nthreads(:default))
        println(io, "nthreads_interactive ", Threads.nthreads(:interactive))
        println(io, "ngcthreads           ", Threads.ngcthreads())
        println(io, "cpu_threads          ", Sys.CPU_THREADS)
        println(io, "blas_threads         ", BLAS.get_num_threads())
        println(io, "fftw_threads         ", FFTW.get_num_threads())
        println(io, "pwd                  ", pwd())
        println(io, "tempdir              ", tempdir())
        println(io, "julia_cmd            ", Base.julia_cmd())
        println(io, "maxrss               ", Sys.maxrss())
        println(io, "free_memory          ", Sys.free_memory())
        println(io, "total_memory         ", Sys.total_memory())
    end

    data, lidict = Profile.retrieve()
    serialize(prefix * ".prof", (data, lidict))

    open(prefix * ".flat.txt", "w") do io
        ctx = IOContext(io, :displaysize => (1000, 400))
        Profile.print(ctx, data, lidict; format=:flat, C=true, groupby=:thread, sortedby=:count, mincount=20)
    end

    open(prefix * ".tree.txt", "w") do io
        ctx = IOContext(io, :displaysize => (1000, 300))
        Profile.print(ctx, data, lidict; C=false, groupby=:thread, mincount=100, maxdepth=80, noisefloor=2)
    end

    return nothing
end
