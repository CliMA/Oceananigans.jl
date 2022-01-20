function prettysummary(f::Function)
    ft = typeof(f)
    mt = ft.name.mt
    name = mt.name
    n = length(methods(f))
    m = n==1 ? "method" : "methods"
    sname = string(name)
    ns = (isself || '#' in sname) ? sname : string("(::", ft, ")")
    what = startswith(ns, '@') ? "macro" : "generic function"
    return string(ns, " (", what, " with $n $m)")
end

prettysummary(x) = summary(x)

