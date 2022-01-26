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

prettysummary(x) = summary(x)

