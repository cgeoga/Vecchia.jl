
hes_reshape(h, len) = [h[i,j] for i=1:len for j=1:i]

function hes_structure!(rows, cols, len)
  idx = 1
  for row in 1:len
    for col in 1:row
      rows[idx] = row
      cols[idx] = col
      idx += 1
    end
  end
  nothing
end

function ipopt_hessian(xarg, rows, cols, obj_factor, lams, values,
                       hessfn, constr_hessv, nconstr)
  isnothing(values) && return hes_structure!(rows, cols, length(xarg))
  @assert length(lams) == nconstr "Disagreement in lengths of lambdas and constraint functions."
  h = hessfn(xarg)
  values .= hes_reshape(h, length(xarg)).*obj_factor
  for (lj, hesconstrj) in zip(lams, constr_hessv)
    constrj_hes = hesconstrj(xarg)
    values .+= lj*hes_reshape(constrj_hes, length(xarg))
  end
end

function jac_structure!(rows, cols, len, nconstr)
  for (j, ix) in enumerate(Iterators.partition(1:(len*nconstr), len))
    rows[ix].=j
    cols[ix].=collect(1:len)
  end
  nothing
end

function ipopt_constr_jac(xarg, rows, cols, values, g_jac, nconstr)
  isnothing(values) && return jac_structure!(rows, cols, length(xarg), nconstr)
  values .= g_jac(xarg)
end

