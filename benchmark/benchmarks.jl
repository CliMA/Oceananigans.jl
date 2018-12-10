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
    U  = VelocityFields(g, T)
    tr = TracerFields(g, T)
    tt = TemporaryFields(g, T)

    U.u.data .= rand(T, size(g))
    U.v.data .= rand(T, size(g))
    U.w.data .= rand(T, size(g))

    #print("+---------------------------------------------------------------------------------------------------------+\n")
    # print("| ", rpad(" BENCHMARKING OCEANANIGANS: T=$T, (Nx, Ny, Nz)=$N", 103), " |\n")
    # print("+---------------+------------+----------+-----------+-----------+-----------+-----------+---------+-------+\n")
    # print("| function name |   memory   |  allocs  | min. time | med. time | mean time | max. time | samples | evals |\n")

    print("┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐\n")
    print("│ ", rpad(" BENCHMARKING OCEANANIGANS: T=$T (Nx, Ny, Nz) = $N", 107),                                   " │\n")
    print("├───────────────┬────────────┬──────────┬────────────┬────────────┬────────────┬────────────┬─────────┬───────┤\n")
    print("│ function name │   memory   │  allocs  │  min. time │  med. time │  mean time │  max. time │ samples │ evals │\n")

    b = @benchmark δx!($g, $U.u, $tt.fC1); pretty_print_summary(b, "δx! (f2c)");
    b = @benchmark δx!($g, $tt.fC1, $U.u); pretty_print_summary(b, "δx! (c2f)");
    b = @benchmark δy!($g, $U.v, $tt.fC2); pretty_print_summary(b, "δy! (f2c)");
    b = @benchmark δy!($g, $tt.fC2, $U.v); pretty_print_summary(b, "δy! (c2f)");
    b = @benchmark δz!($g, $U.w, $tt.fC3); pretty_print_summary(b, "δz! (f2c)");
    b = @benchmark δz!($g, $tt.fC3, $U.w); pretty_print_summary(b, "δz! (c2f)");

    b = @benchmark avgx!($g, $U.u, $tt.fC1); pretty_print_summary(b, "avgx! (f2c)");
    b = @benchmark avgx!($g, $tt.fC1, $U.u); pretty_print_summary(b, "avgx! (c2f)");
    b = @benchmark avgy!($g, $U.v, $tt.fC2); pretty_print_summary(b, "avgy! (f2c)");
    b = @benchmark avgy!($g, $tt.fC2, $U.v); pretty_print_summary(b, "avgy! (c2f)");
    b = @benchmark avgz!($g, $U.w, $tt.fC3); pretty_print_summary(b, "avgz! (f2c)");
    b = @benchmark avgz!($g, $tt.fC3, $U.w); pretty_print_summary(b, "avgz! (c2f)");

    b = @benchmark div!($g, $U.u, $U.v, $U.w, $tt.fC1, $tt); pretty_print_summary(b, "div! (f2c)");
    b = @benchmark div!($g, $tt.fC1, $tt.fC2, $tt.fC3, $tt.fFX, $tt); pretty_print_summary(b, "div! (c2f)");
    b = @benchmark div_flux!($g, $U.u, $U.v, $U.w, $tr.T, $tt.fC1, $tt); pretty_print_summary(b, "div_flux!");

    b = @benchmark u∇u!($g, $U, $tmp.fFX, $tmp); pretty_print_summary(b, "u∇u!");

    print("└───────────────┴────────────┴──────────┴────────────┴────────────┴────────────┴────────────┴─────────┴───────┘\n")
    # print("+---------------------------------------------------------------------------------------------------------+\n")
end
