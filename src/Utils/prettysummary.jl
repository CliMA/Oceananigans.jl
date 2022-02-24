using Oceananigans.Grids: scalar_summary

prettysummary(x, args...) = summary(x)

function prettysummary(f::Function, showmethods=true)
    ft = typeof(f)
    mt = ft.name.mt
    name = mt.name
    n = length(methods(f))
    m = n==1 ? "method" : "methods"
    sname = string(name)
    isself = isdefined(ft.name.module, name) && ft == typeof(getfield(ft.name.module, name))
    ns = (isself || '#' in sname) ? sname : string("(::", ft, ")")
    if showmethods
        return string(ns, " (", "generic function", " with $n $m)")
    else
        return string(ns)
    end
end

prettysummary(x::Number, args...) = scalar_summary(x)

# This is very important
function prettysummary(nt::NamedTuple, args...)
    n = nfields(nt)

    if n == 0
        return "NamedTuple()"
    else
        str = "("
        for i = 1:n
            f = nt[i]
            str = string(str, fieldname(typeof(nt), i), "=", prettysummary(getfield(nt, i)))
            if n == 1
                str = string(str, ", ")
            elseif i < n
                str = string(str, ", ")
            end
        end
    end

    return string(str, ")")
end

function prettykeys(t)
    names = collect(keys(t))
    length(names) == 1 && return string(first(names))
    return string("(", (string(n, ", ") for n in names[1:end-1])..., last(names), ")")
end

