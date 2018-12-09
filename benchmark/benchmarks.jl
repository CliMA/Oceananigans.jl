using BenchmarkTools, Printf, Statistics

using Oceananigans, Oceananigans.Operators

# Pretty printing functions stolen from BenchmarkTools.jl
prettypercent(p) = string(@sprintf("%.2f", p * 100), "%")

function prettydiff(p)
    diff = p - 1.0
    return string(diff >= 0.0 ? "+" : "", @sprintf("%.2f", diff * 100), "%")
end

function prettytime(t)
    if t < 1e3
        value, units = t, "ns"
    elseif t < 1e6
        value, units = t / 1e3, "μs"
    elseif t < 1e9
        value, units = t / 1e6, "ms"
    else
        value, units = t / 1e9, "s"
    end
    return string(@sprintf("%.3f", value), " ", units)
end

function prettymemory(b)
    if b < 1024
        return string(b, " bytes")
    elseif b < 1024^2
        value, units = b / 1024, "KiB"
    elseif b < 1024^3
        value, units = b / 1024^2, "MiB"
    else
        value, units = b / 1024^3, "GiB"
    end
    return string(@sprintf("%.2f", value), " ", units)
end

function pretty_print_summary(b, func_name)
    print("│",
          lpad(func_name, 14), " │",
          lpad(prettymemory(b.memory), 11), " │",
          lpad(b.allocs, 9), " │",
          lpad(prettytime(minimum(b.times)), 11), " │",
          lpad(prettytime(median(b.times)), 11), " │",
          lpad(prettytime(mean(b.times)), 11), " │",
          lpad(prettytime(maximum(b.times)), 11), " │",
          lpad(b.params.samples, 8), " │",
          lpad(b.params.evals, 6), " │"
          )

    if !(median(b.gctimes) ≈ 0)
        print(" GC min: ", prettypercent(100 * minimum(b.gctimes) / minimum(b.times)), ", ",
              " GC med: ", prettypercent(100 * median(b.gctimes) / median(b.times)), "\n"
              )
    else
        print("\n")
    end
end

function run_benchmarks()
    N = (100, 100, 100)
    L = (1000, 1000, 1000)
    T = Float64

    g  = RegularCartesianGrid(N, L, T)
    ũ  = VelocityFields(g, T)
    tr = TracerFields(g, T)
    t̃  = TemporaryFields(g, T)

    ũ.u.data .= rand(T, size(g))
    ũ.v.data .= rand(T, size(g))
    ũ.w.data .= rand(T, size(g))

    #print("+---------------------------------------------------------------------------------------------------------+\n")
    # print("| ", rpad(" BENCHMARKING OCEANANIGANS: T=$T, (Nx, Ny, Nz)=$N", 103), " |\n")
    # print("+---------------+------------+----------+-----------+-----------+-----------+-----------+---------+-------+\n")
    # print("| function name |   memory   |  allocs  | min. time | med. time | mean time | max. time | samples | evals |\n")

    print("┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐\n")
    print("│ ", rpad(" BENCHMARKING OCEANANIGANS: T=$T (Nx, Ny, Nz) = $N", 107),                                   " │\n")
    print("├───────────────┬────────────┬──────────┬────────────┬────────────┬────────────┬────────────┬─────────┬───────┤\n")
    print("│ function name │   memory   │  allocs  │  min. time │  med. time │  mean time │  max. time │ samples │ evals │\n")

    b = @benchmark δx!($g, $ũ.u, $t̃.fC1); pretty_print_summary(b, "δx! f2c");
    b = @benchmark δx!($g, $t̃.fC1, $ũ.u); pretty_print_summary(b, "δx! c2f");
    b = @benchmark δy!($g, $ũ.v, $t̃.fC2); pretty_print_summary(b, "δy! f2c");
    b = @benchmark δy!($g, $t̃.fC2, $ũ.v); pretty_print_summary(b, "δy! c2f");
    b = @benchmark δz!($g, $ũ.w, $t̃.fC3); pretty_print_summary(b, "δz! f2c");
    b = @benchmark δz!($g, $t̃.fC3, $ũ.w); pretty_print_summary(b, "δz! c2f");

    b = @benchmark avgx!($g, $ũ.u, $t̃.fC1); pretty_print_summary(b, "avgx! f2c");
    b = @benchmark avgx!($g, $t̃.fC1, $ũ.u); pretty_print_summary(b, "avgx! c2f");
    b = @benchmark avgy!($g, $ũ.v, $t̃.fC2); pretty_print_summary(b, "avgy! f2c");
    b = @benchmark avgy!($g, $t̃.fC2, $ũ.v); pretty_print_summary(b, "avgy! c2f");
    b = @benchmark avgz!($g, $ũ.w, $t̃.fC3); pretty_print_summary(b, "avgz! f2c");
    b = @benchmark avgz!($g, $t̃.fC3, $ũ.w); pretty_print_summary(b, "avgz! c2f");

    b = @benchmark div!($g, $ũ.u, $ũ.v, $ũ.w, $t̃.fC1, $t̃); pretty_print_summary(b, "div! f2c");
    b = @benchmark div!($g, $t̃.fC1, $t̃.fC2, $t̃.fC3, $t̃.fFX, $t̃); pretty_print_summary(b, "div! c2f");
    b = @benchmark div_flux!($g, $ũ.u, $ũ.v, $ũ.w, $tr.T, $t̃.fC1, $t̃); pretty_print_summary(b, "div_flux!");

    print("└───────────────┴────────────┴──────────┴────────────┴────────────┴────────────┴────────────┴─────────┴───────┘\n")
    # print("+---------------------------------------------------------------------------------------------------------+\n")
end
