using OrderedCollections: OrderedDict

function ordered_dict_show(dict::OrderedDict, padchar)
    name = "OrderedDict"
    N = length(dict)

    if N == 0
        return "$name with no entries"
    elseif N == 1
        k = dict |> keys |> first
        sum_v = dict |> values |> first |> summary
        return "$name with 1 entry:\n$padchar   └── $k => $sum_v"
    end

    keys_list = dict |> keys |> collect
    summary_vals_list = dict |> values |> collect |> v -> summary(v)
    lines = ["$name with $N entries:"]

    for i in 1:N-1
        k, sum_v  = keys_list[i], summary_vals_list[i]
        push!(lines, "$padchar   ├── $k => $sum_v")
    end

    last_k, last_sum_v = last(keys_list), last(summary_vals_list)
    push!(lines, "$padchar   └── $last_k => $last_sum_v")

    return join(lines, "\n")
end


