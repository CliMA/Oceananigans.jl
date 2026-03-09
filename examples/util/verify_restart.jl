include(joinpath(@__DIR__, "compare_checkpoints.jl"))
using .compare_checkpoints: compare_all
using Glob
using JLD2
using Oceananigans: CPU, GPU

const arch = CPU()  # Change to GPU() to run on GPU
#const nprocs = 4
const verbose = false

# Substitute CPU()/GPU() in a line to match `arch`.
arch_sub(line) = arch isa GPU ? replace(line, "CPU()" => "GPU()") :
                                replace(line, "GPU()" => "CPU()")

# Stream output from `cmd` (run in `dir`) to both stdout and `log_path`.
function run_simulation(script_path, log_name)
    @info "---------- RUNNING: $script_path ----------"
    casedir, script_name = splitdir(script_path)
    log_path = joinpath(casedir, log_name)
    #launcher = nprocs > 1 ? `julia -p $nprocs $script_path` : `julia $script_path`
    launcher = `$(Base.julia_cmd()) $script_name`
    cmd = Cmd(launcher; dir=casedir)
    open(log_path, "w") do logfile
        out_pipe = Pipe()
        err_pipe = Pipe()
        proc = run(pipeline(cmd, stdout=out_pipe, stderr=err_pipe), wait=false)
        close(out_pipe.in)
        close(err_pipe.in)
        relay(io) = for line in eachline(io)
            println(line)
            println(logfile, line)
            flush(logfile)
        end
        t = @async relay(err_pipe)
        relay(out_pipe)
        wait(t)
        wait(proc)
    end
end

# Write a modified copy of `src_path` to `dest_path`:
#   - Truncate after the first `run!(...)` line
#   - Insert stop conditions and a Checkpointer output writer before `run!`
#   - Append keyword arguments to the `run!` call
#   - If `restarted=true`: comment out `set!` lines and add `pickup` kwarg
#   - Substitute CPU()/GPU() to match `arch`; add "using CUDA" if arch is GPU
function generate_script(src_path, dest_path, prefix; pickup_file=nothing)
    restarted = !isnothing(pickup_file)
    lines = readlines(src_path)

    # If GPU and no CUDA import exists, insert one after the last `using` line
    if arch isa GPU && !any(occursin(r"^#?\s*using CUDA"), lines)
        last_using = findlast(startswith("using "), lines)
        insert!(lines, isnothing(last_using) ? 1 : last_using + 1, "using CUDA")
    end

    # HACK: handle multiple cases run per script
    amend_prefix = (basename(src_path) == "spherical_baroclinic_instability.jl")

    open(dest_path, "w") do out
        for line in lines

            # Comment out set! calls (restarted only, to skip re-initialization)
            if restarted && occursin(r"^\s*set!", line)
                println(out, replace(line, "set!" => "#set!"))
                continue
            end

            # Insert before run!, modify the run! call, then truncate
            if occursin(r"^\s*run!", line)
                idx = findfirst("run!", line).start
                indent = repeat(" ", idx-1)
                @assert !xor(length(indent) > 0, amend_prefix) # spherical_baroclinic_instability.jl example

                println(out, "")
                println(out, "$(indent)simulation.stop_time = 9e99")
                println(out, "$(indent)simulation.stop_iteration = 200")
                if amend_prefix
                    println(out, "$(indent)simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(100), prefix=\"$(prefix)_\"*name)")
                else
                    println(out, "$(indent)simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(100), prefix=\"$prefix\")")
                end
                println(out, "")

                suffix = if restarted && amend_prefix
                    # in the spherical_baroclinic_instability.jl example, multiple cases are run
                    varpickup_file = "\"" * replace(pickup_file, "_" => "_\" * name * \"_") * "\""
                    ", checkpoint_at_end=true, pickup=$varpickup_file)"
                elseif restarted
                    # default
                    ", checkpoint_at_end=true, pickup=\"$pickup_file\")"
                else
                    ", checkpoint_at_end=true)"
                end
                println(out, arch_sub(replace(rstrip(line), r"\)$" => suffix)))

                if amend_prefix
                    continue # and keep writing out remaidner of script
                else
                    break # stop after run!(...)
                end
            end

            println(out, arch_sub(line))
        end
    end
end

function process_example(src_path)
    # create casedir in the current working directory with the same name as the example script
    casename = splitext(basename(src_path))[1]
    casedir = casename
    mkpath(casename)

    norestart_script = joinpath(casedir, "$(casename)_0.jl")
    restarted_script = joinpath(casedir, "$(casename)_1.jl")

    isfile(norestart_script) || generate_script(src_path, norestart_script, "norestart")
    isfile(restarted_script) || generate_script(src_path, restarted_script, "restarted";
                                                pickup_file="norestart_iteration100.jld2")

    # in the spherical_baroclinic_instability.jl example, multiple cases are run
    norestart_chkpts = glob(joinpath(casedir, "norestart*_iteration200.jld2"))
    restarted_chkpts = glob(joinpath(casedir, "restarted*_iteration200.jld2"))

    length(norestart_chkpts) > 0 || run_simulation(norestart_script, "log.run0")
    length(restarted_chkpts) > 0 || run_simulation(restarted_script, "log.run1")

    @info "---------- COMPARING CHECKPOINTS ----------"
    norestart_chkpts = glob(joinpath(casedir, "norestart*_iteration200.jld2"))
    restarted_chkpts = glob(joinpath(casedir, "restarted*_iteration200.jld2"))

    for (norestart_chk, restarted_chk) in zip(norestart_chkpts, restarted_chkpts)
        fname = basename(restarted_chk)
        idx_start = findfirst("_", fname).start + 1
        idx_end = findlast("_", fname).start - 1
        if idx_start == idx_end
            # default path
            log_path = joinpath(casename, "compare_restart.log")
        else
            # workaround for spherical_baroclinic_instability.jl
            name = fname[idx_start:idx_end]
            log_path = joinpath(casename, "compare_restart_$name.log")
        end

        @info "compare $norestart_chk -vs- $restarted_chk ($log_path)"

        chk1 = jldopen(norestart_chk, "r")
        chk2 = jldopen(restarted_chk, "r")

        output = mktemp() do _, io
            redirect_stdout(io) do
                compare_all(chk1["simulation"]["model"], chk2["simulation"]["model"]; verbose=verbose)
            end
            flush(io)
            seek(io, 0)
            read(io, String)
        end
        close(chk1)
        close(chk2)
        print(output)
        write(log_path, output)
    end
end

for script in ARGS
    process_example(abspath(script))
end
