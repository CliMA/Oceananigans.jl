# Run one test file in this process, outside ParallelTestRunner, and record it like
# `OCEANANIGANS_PROFILE_DIR` does in test/runtests.jl.
#
# Usage: julia --project=test profile_test_file.jl <test name, e.g. simulation/netcdf_writer> <output prefix>
using Test

const test_dir = joinpath(@__DIR__, "..", "test")
include(joinpath(test_dir, "setup", "profile_recording.jl"))

name, prefix = ARGS
file = joinpath(test_dir, name * ".jl")

test_module = Module(:ProfiledTest)
Core.eval(test_module, :(using Test))
Core.eval(test_module, :(include(path) = Base.include($test_module, path)))

Profile.init(n = 5 * 10^7, delay = 0.02)
profile_start = time()
Profile.start_timer()
stats = @timed try
    @testset verbose=true "$name" begin
        Base.include(test_module, file)
    end
finally
    Profile.stop_timer()
end
write_profile_record(prefix, name, stats, time() - profile_start)
