include(joinpath(@__DIR__, "compare_checkpoints.jl"))
using .compare_checkpoints: compare_all
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
    casedir, script_name = splitdir(script_path)
    log_path = joinpath(casedir, log_name)
    #launcher = nprocs > 1 ? `julia -p $nprocs $script_path` : `julia $script_path`
    launcher = `julia $script_name`
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
#   - Comment out / uncomment `using CUDA` and substitute CPU()/GPU() to match `arch`
function generate_script(src_path, dest_path, prefix; restarted=false, pickup_file=nothing)
    lines = readlines(src_path)

    # If GPU and no CUDA import exists, insert one after the last `using` line
    if arch isa GPU && !any(occursin(r"^#?\s*using CUDA", l) for l in lines)
        last_using = findlast(l -> startswith(l, "using "), lines)
        insert!(lines, isnothing(last_using) ? 1 : last_using + 1, "using CUDA")
    end

    open(dest_path, "w") do out
        for line in lines

            # Handle `using CUDA`: uncomment for GPU, comment out for CPU
            if occursin(r"^#?\s*using CUDA", line)
                if arch isa GPU
                    println(out, replace(line, r"^#\s*" => ""))
                else
                    println(out, startswith(line, "#") ? line : "#" * line)
                end
                continue
            end

            # Comment out set! calls (restarted only, to skip re-initialization)
            if restarted && startswith(line, "set!")
                println(out, "#" * line)
                continue
            end

            # Insert before run!, modify the run! call, then truncate
            if startswith(line, "run!")
                println(out, "")
                println(out, "simulation.stop_time = 9e99")
                println(out, "simulation.stop_iteration = 200")
                println(out, "simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(100), prefix=\"$prefix\")")
                println(out, "")
                suffix = restarted ? ", checkpoint_at_end=true; pickup=\"$pickup_file\")" :
                                     ", checkpoint_at_end=true)"
                println(out, arch_sub(replace(rstrip(line), r"\)$" => suffix)))
                break
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
    norestart_chk    = joinpath(casedir, "norestart_iteration200.jld2")
    restarted_chk    = joinpath(casedir, "restarted_iteration200.jld2")

    isfile(norestart_script) || generate_script(src_path, norestart_script, "norestart")
    isfile(restarted_script) || generate_script(src_path, restarted_script, "restarted";
                                                restarted=true,
                                                pickup_file="norestart_iteration100.jld2")

    isfile(norestart_chk) || run_simulation(norestart_script, "log.run0")
    isfile(restarted_chk) || run_simulation(restarted_script, "log.run1")

    log_path = joinpath(casename, "compare_restart.log")
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

for script in ARGS
    process_example(abspath(script))
end
