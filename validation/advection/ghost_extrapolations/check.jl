# Unit check of the hierarchical continuation: linear data continue exactly, quadratics exactly on five nodes, a front gives
# constant ghosts, and the copy rule is recovered when one node has data.
using Oceananigans
include(joinpath(@__DIR__, "hierarchical.jl"))
using Oceananigans.Advection: wall_continuation, continued_value
linear = (0.0, 0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0)          # wall at position 2, data from 3 (c₁ = 1, slope 2)
quadratic = (0.0, 0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0)      # c = q² for q = 1, 2, …
front = (0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0)
for (name, S, expected) in (("linear", linear, (-1.0, -3.0)), ("quadratic", quadratic, (0.0, 1.0)), ("front", front, (1.0, 1.0)))
    left = wall_continuation(S, 3, +1, 6)
    ghosts = (continued_value(left, 1), continued_value(left, 2))
    println(rpad(name, 10), ghosts, ghosts == expected ? "  ok" : "  expected $expected")
end
one_node = wall_continuation((0.0, 0.0, 3.0, 0.0), 3, +1, 1)
println(rpad("one node", 10), (continued_value(one_node, 1), continued_value(one_node, 2)))
