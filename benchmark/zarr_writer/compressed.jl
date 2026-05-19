include("common.jl")

#####
##### Same as serial_single, but ZarrWriter uses Blosc compression (clevel=3).
#####

results = []
compressor = nothing
for kind in (:JLD2, :NetCDF, :Zarr_uncompressed, :Zarr_blosc)
    model, tracers = make_model(nfields=2)
    outputs = make_outputs(model, tracers)
    ext = kind === :JLD2 ? ".jld2" :
          kind === :NetCDF ? ".nc" : ".zarr"
    path = "bench_comp_$(lowercase(string(kind)))$ext"
    @info "Benchmarking $kind..."

    if kind === :Zarr_blosc
        compressor = Zarr.BloscCompressor(clevel=3)
        run_s, per_step, bytes = bench_writer(:Zarr, model, outputs, path; compressor)
        read_s = bench_read(:Zarr, path, "u")
    else
        wkind = kind === :Zarr_uncompressed ? :Zarr : kind
        run_s, per_step, bytes = bench_writer(wkind, model, outputs, path)
        read_s = bench_read(wkind, path, "u")
    end
    push!(results, (string(kind), run_s, per_step, bytes, read_s))
    cleanup(path)
end

println("\n## Compressed Zarr benchmark vs uncompressed JLD2/NetCDF (5 fields, $(FIELD_SHAPE), Float32, $(NSTEPS) steps)\n")
print_md_table(stdout, results)
