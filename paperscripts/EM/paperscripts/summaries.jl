
include("shared.jl")
include("texmatrix.jl")

# Read in the points and data, already maximin-permuted:
if !(@isdefined EX_ESTIMATES)
  const (PTS, DATA_MATRIX) = prepare_dat()
  const R_ESTIMATES   = readdlm("./data/estimates_R_m30.csv",  ',')
  #const EM_ESTIMATES  = readdlm("./data/estimates_em.csv", ',')
  #const EX_ESTIMATES  = readdlm("./data/estimates_mle.csv", ',')[:,1:3]
  #const EXV_ESTIMATES = readdlm("./data/estimates_exactvec.csv", ',')
end

function exact_nll_case(case, j::Int64)
  println("Computing likelihoods for index $j/50...")
  cfg  = maximinconfig(kernel_nonugget, PTS, DATA_MATRIX[:,j], 1, 10)
  if case == :EM
    em_j = EM_ESTIMATES[:,j]
    return EMVecchia2.exact_nll(cfg, em_j)
  elseif case == :SGV
    R_j  = R_ESTIMATES[:,j]
    return EMVecchia2.exact_nll(cfg, R_j)
  else
    throw(error("Options are :EM or :SGV."))
  end
end

function compare_exact_nlls(j::Int64)
  println("Computing likelihoods for index $j/50...")
  cfg  = maximinconfig(kernel_nonugget, PTS, DATA_MATRIX[:,j], 1, 10)
  em_j = EM_ESTIMATES[:,j]
  R_j  = R_ESTIMATES[:,j]
  (em=EMVecchia2.exact_nll(cfg, em_j), R=EMVecchia2.exact_nll(cfg, R_j))
end

function compare_exact_vecchia_nlls(j::Int64)
  cfg  = maximinconfig(kernel_nonugget, PTS, DATA_MATRIX[:,j], 1, 10)
  em_j = EM_ESTIMATES[:,j]
  R_j  = R_ESTIMATES[:,j]
  (em=EMVecchia2.exact_vecchia_nll_noad(cfg, em_j), 
    R=EMVecchia2.exact_vecchia_nll_noad(cfg, R_j))
end

function compare_diffs(j::Int64)
  cfg  = maximinconfig(kernel_nonugget, PTS, DATA_MATRIX[:,j], 1, 10)
  ref  = EX_ESTIMATES[:,j]
  em_j = EM_ESTIMATES[:,j]
  R_j  = R_ESTIMATES[:,j]
  (ref=ref, em_diff=em_j-ref, R_diff=R_j-ref, em=em_j, R=R_j)
end

# TODO (cg 2022/06/17 15:12): make better number bold.
function write_summary_table()
  _m = size(EM_ESTIMATES, 2)
  # Create the header rows:
  top_labels  = ["\\multicolumn{2}{c|}{"*string(j)*"} &" for j in 1:(_m-1)]
  push!(top_labels, "\\multicolumn{2}{c|}{"*string(_m)*"}")
  sec_labels  = reduce(vcat, fill(["EM & SGV &"] ,_m-1))
  push!(sec_labels, "EM & SGV")
  # modify them with the & and \\ and stuff:
  pushfirst!(top_labels, "Trial &")
  pushfirst!(sec_labels, "Method &")
  #writedlm("table_header1.tex", permutedims(top_labels))
  #writedlm("table_header2.tex", permutedims(sec_labels))
  # now make the actual table:
  pdifs = map(1:_m) do j
                   tmp = compare_diffs(j)
                   _em = tmp.em_diff
                   _r  = tmp.R_diff
                   bold_ixs = [(k, ifelse(abs(_em[k]) <= abs(_r[k]), 
                                          2*(j-1)+1, 2*(j-1)+2))
                               for k in 1:4]
                   (hcat(tmp.em_diff, tmp.R_diff), bold_ixs)
                 end
  ldifs = map(1:_m) do j
                   tmp = compare_exact_nlls(j)
                   ix  = (5, (tmp.em <= tmp.R ? 2*(j-1)+1 : 2*(j-1)+2))
                   (hcat(tmp.em, tmp.R), ix)
                 end
  sidelabs = ["\$ \\hat{\\sigma}^2 - \\hat{\\theta}^2_{\\text{MLE}} \$", 
              "\$ \\hat{\\rho} - \\hat{\\rho}_{\\text{MLE}} \$", 
              "\$ \\hat{\\nu} - \\hat{\\nu}_{\\text{MLE}} \$", 
              "\$ \\hat{\\eta}^2 - \\hat{\\eta}^2_{\\text{MLE}} \$", 
              "\$ \\ell(\\bm{\\theta}) \$"]
  pdifs_v  = reduce(hcat, getindex.(pdifs, 1))
  ldifs_v  = reduce(hcat, getindex.(ldifs, 1))
  bold_ixs = vcat(reduce(vcat, vec(getindex.(pdifs, 2))), vec(getindex.(ldifs, 2)))
  texmatrix("table_body_m30.tex", vcat(pdifs_v, ldifs_v), 
            digits=3, pad=true, sidelabels=sidelabs, bolds=bold_ixs)
  nothing
end

if !isinteractive()
  #write_summary_table()
  #const nlls = [compare_exact_nlls(j) for j in 1:50]
  const nlls = [exact_nll_case(:SGV, j) for j in 1:50]
  writedlm("./data/sgvm30_nlls.csv", nlls, ',')
  #serialize("./data/sgvm30_nlls_backup.jls", nlls)
  #writedlm("./data/sgv_compare_exact_nlls.csv", hcat(1:50, map(x->x.em, nlls), map(x->x.R, nlls)), ',')
end

