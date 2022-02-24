using Oceananigans.Grids: scalar_summary

prettysummary(x) = summary(x)

function prettysummary(f::Function)
    ft = typeof(f)
    mt = ft.name.mt
    name = mt.name
    n = length(methods(f))
    m = n==1 ? "method" : "methods"
    sname = string(name)
    isself = isdefined(ft.name.module, name) && ft == typeof(getfield(ft.name.module, name))
    ns = (isself || '#' in sname) ? sname : string("(::", ft, ")")
    return string(ns, " (", "generic function", " with $n $m)")
end

prettysummary(x::Number) = scalar_summary(σ)

# This is very important
function prettysummary(nt::NamedTuple)
    n = nfields(nt)

    if n == 0
        return "NamedTuple()"
    else
        str = "("
        for i = 1:n
            f = nt[i]
            str = string(str, fieldname(typeof(t), i), " = ", getfield(nt, i))
            if n == 1
                str = string(str, ",")
            elseif i < n
                str = string(str, ",")
            end
        end
    end

    return string(str, ")")
end

