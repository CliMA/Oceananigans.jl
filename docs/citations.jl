using Documenter
using Documenter.Anchors
using Documenter.Builder
using Documenter.Documents
using Documenter.Selectors
using Documenter.Utilities

using Markdown
using Bibliography
using Bibliography: xnames, xyear

abstract type Citations <: Builder.DocumentPipeline end

Selectors.order(::Type{Citations}) = 3.1  # After cross-references

function Selectors.runner(::Type{Citations}, doc::Documents.Document)
    @info "Citations: building citations."
    expand_citations(doc)
end

function expand_citations(doc::Documents.Document)
    for (src, page) in doc.blueprint.pages
        empty!(page.globals.meta)
        for element in page.elements
            expand_citation(page.mapping[element], page, doc)
        end
    end
end

function expand_citation(elem, page, doc)
    Documents.walk(page.globals.meta, elem) do link
        expand_citation(link, page.globals.meta, page, doc)
    end
end

function expand_citation(link::Markdown.Link, meta, page, doc)
    link.url !== "@cite" && return false

    if length(link.text) === 1 && isa(link.text[1], String)
        citation_name = link.text[1]
        @info "Expanding citation: $citation_name."

        if haskey(BIBLIOGRAPHY, citation_name)
            entry = BIBLIOGRAPHY[citation_name]
            headers = doc.internal.headers
            if Anchors.exists(headers, entry.id)
                if Anchors.isunique(headers, entry.id)
                    # Replace the `@cite` url with a path to the referenced header.
                    anchor   = Anchors.anchor(headers, entry.id)
                    path     = relpath(anchor.file, dirname(page.build))
                    link.text = xnames(entry) * " (" * xyear(entry) * ")"
                    link.url = string(path, Anchors.fragment(anchor))
                    return true
                else
                    push!(doc.internal.errors, :citations)
                    @warn "'$(entry.id)' is not unique in $(Utilities.locrepr(page.source))."
                end
            else
                push!(doc.internal.errors, :citations)
                @warn "reference for '$(entry.id)' could not be found in $(Utilities.locrepr(page.source))."
            end
        else
            error("Citation not found in bibliography: $(citation_name)")
        end
    else
        error("Invalid citation: $(link.text)")
    end

    return false
end

expand_citation(other, meta, page, doc) = true # Continue to `walk` through element `other`.
