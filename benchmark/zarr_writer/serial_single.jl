include("common.jl")

#####
##### Serial, 5 fields, (128, 128, 64), Float32, 50 steps, no compression.
#####

results = []
for kind in (:JLD2, :NetCDF, :Zarr)
    model, tracers = make_model(nfields=2)   # 2 tracers + u/v/w = 5 outputs
    outputs = make_outputs(model, tracers)
    ext = kind === :JLD2 ? ".jld2" : kind === :NetCDF ? ".nc" : ".zarr"
    path = "bench_$(lowercase(string(kind)))$ext"
    @info "Benchmarking $kind..."
    run_s, per_step, bytes = bench_writer(kind, model, outputs, path)
    read_s = bench_read(kind, path, "u")
    push!(results, (string(kind), run_s, per_step, bytes, read_s))
    cleanup(path)
end

println("\n## Serial single-output benchmark (5 fields, $(FIELD_SHAPE), Float32, $(NSTEPS) steps)\n")
print_md_table(stdout, results)
