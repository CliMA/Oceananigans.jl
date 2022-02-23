using OrderedCollections: OrderedDict

function ordered_dict_show(dict::OrderedDict, padchar)
    if length(dict) == 0
        return "$(typeof(dict).name) with no entries"
    elseif length(dict) == 1
        return "$(typeof(dict).name) with 1 entry:\n" *
               "$padchar   └── $(dict.keys[1]) => $(typeof(dict.vals[1]).name)"
    else
        return string(typeof(dict).name, " with $(length(dict)) entries:\n",
                      Tuple("$padchar   ├── $name => $(typeof(dict[name]).name)\n" for name in dict.keys[1:end-1])...,
                            "$padchar   └── $(dict.keys[end]) => $(typeof(dict.vals[end]).name)")
    end
end
