include("common.jl")

#####
##### Serial, 10 fields, (128, 128, 64), Float32, 50 steps, no compression.
#####

results = []
for kind in (:JLD2, :NetCDF, :Zarr)
    model, tracers = make_model(nfields=7)   # 7 tracers + u/v/w = 10 outputs
    outputs = make_outputs(model, tracers)
    ext = kind === :JLD2 ? ".jld2" : kind === :NetCDF ? ".nc" : ".zarr"
    path = "bench_multi_$(lowercase(string(kind)))$ext"
    @info "Benchmarking $kind..."
    run_s, per_step, bytes = bench_writer(kind, model, outputs, path)
    read_s = bench_read(kind, path, "u")
    push!(results, (string(kind), run_s, per_step, bytes, read_s))
    cleanup(path)
end

println("\n## Serial multi-output benchmark (10 fields, $(FIELD_SHAPE), Float32, $(NSTEPS) steps)\n")
print_md_table(stdout, results)
