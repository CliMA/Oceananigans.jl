"""
    flatten_dissipation_fields(t::VarianceDissipation, tracer_name)

Flattens the dissipation fields of a `VarianceDissipation` object into a named tuple containing:

- The dissipation associated with the advection scheme in fields named `A-tracername-dir`
- The dissipation associated with the closures in fields names `D-tracername-dir`
- The squared gradients (necessary for computing an ``effective diffusivity'') in fields named `G-tracername-dir`
"""
function flatten_dissipation_fields(t::VarianceDissipation) 
    A = t.advective_production
    D = t.diffusive_production
    tracer_name = t.tracer_name

    dirs = (:x, :y, :z)

    prod_names = Tuple(Symbol(:A, tracer_name, dir) for dir in dirs)
    diff_names = Tuple(Symbol(:D, tracer_name, dir) for dir in dirs)

    advective_prod = Tuple(getproperty(A, dir) for dir in dirs)
    diffusive_prod = Tuple(getproperty(D, dir) for dir in dirs)
    
    return NamedTuple{tuple(prod_names..., diff_names...)}(tuple(advective_prod..., diffusive_prod...))
end