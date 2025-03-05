using CSV
using DataFrames
using CairoMakie

VERTICAL_FILE_PATH = "./reports/double_gyre_vertical/benchmark_nvtx_sum.csv"
HORIZONTAL_FILE_PATH = "./reports/double_gyre/benchmark_nvtx_sum.csv"

vertical_df = CSV.read(VERTICAL_FILE_PATH, DataFrame)
horizontal_df = CSV.read(HORIZONTAL_FILE_PATH, DataFrame)

vertical_df[!, "Range"]

function extract_main_value(input_string)
    # Find the starting position after "Main:"
    start_pos = findfirst("Main:", input_string)
    
    # Adjust position to skip over the "Main:" prefix
    start_index = last(start_pos) + 1
    
    # Find the position of the next comma
    comma_pos = findnext(',', input_string, start_index)
    
    # Extract and return the substring between "Main:" and the comma
    return strip(input_string[start_index:comma_pos-1])
end

function extract_n_value(input_string, n_type="Nz")
    # Find the starting position of "Nz"
    nz_pos = findfirst(n_type, input_string)
    
    # Adjust position to skip over "Nz"
    start_index = last(nz_pos) + 1
    
    # Extract the rest of the string after "Nz"
    remaining = strip(input_string[start_index:end])
    
    # Extract the numeric part using a regular expression
    m = match(r"^\s*(\d+)", remaining)
    return parse(Int, m.captures[1])
end

vertical_benchmark_type = extract_main_value.(vertical_df[!, "Range"])
horizontal_benchmark_type = extract_main_value.(horizontal_df[!, "Range"])

Nzs = extract_nz_value.(vertical_df[!, "Range"])
Nxys = extract_n_value.(horizontal_df[!, "Range"], "Nxy")

vertical_df.type .= vertical_benchmark_type
vertical_df.Nz .= Nzs

horizontal_df.type .= horizontal_benchmark_type
horizontal_df.Nxy .= Nxys

vertical_NN_time = sort(vertical_df[vertical_df.type .== "NN", :], :Nz)[!, "Med (ns)"] ./ 1e9
vertical_CATKE_time = sort(vertical_df[vertical_df.type .== "CATKE", :], :Nz)[!, "Med (ns)"] ./ 1e9
vertical_k_epsilon_time = sort(vertical_df[vertical_df.type .== "k_epsilon", :], :Nz)[!, "Med (ns)"] ./ 1e9

horizontal_NN_time = sort(horizontal_df[horizontal_df.type .== "NN", :], :Nxy)[!, "Med (ns)"] ./ 1e9
horizontal_CATKE_time = sort(horizontal_df[horizontal_df.type .== "CATKE", :], :Nxy)[!, "Med (ns)"] ./ 1e9
horizontal_k_epsilon_time = sort(horizontal_df[horizontal_df.type .== "k_epsilon", :], :Nxy)[!, "Med (ns)"] ./ 1e9

Nzs_sorted = sort(unique(Nzs))
Nxys_sorted = sort(unique(Nxys))
#%%
fig = Figure(size=(1200, 600), fontsize=25)
axz = Axis(fig[1, 1], xlabel="Vertical grid points", ylabel="Median time per timestep (s)", yscale=log10, xscale=log2, title="128 horizontal grid points")
axx = Axis(fig[1, 2], xlabel="Horizontal grid points", ylabel="Median time per timestep (s)", yscale=log10, xscale=log2, title="192 vertical grid points")

lines!(axz, Nzs_sorted, vertical_NN_time, label="NORi", linewidth=3)
lines!(axz, Nzs_sorted, vertical_CATKE_time, label="CATKE", linewidth=3)
lines!(axz, Nzs_sorted, vertical_k_epsilon_time, label="k-ϵ", linewidth=3)

lines!(axx, Nxys_sorted .^ 2, horizontal_NN_time, label="NORi", linewidth=3)
lines!(axx, Nxys_sorted .^ 2, horizontal_CATKE_time, label="CATKE", linewidth=3)
lines!(axx, Nxys_sorted .^ 2, horizontal_k_epsilon_time, label="k-ϵ", linewidth=3)

Legend(fig[2, :], axz, orientation=:horizontal)

hideydecorations!(axx, ticks=false, grid=false, ticklabels=false)

save("./Output/double_gyre_benchmark.pdf", fig)
display(fig)

#%%