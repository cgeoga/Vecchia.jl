
module VecchiaGALAHADExt

  using Vecchia, GALAHAD, ForwardDiff
  using Vecchia.Accessors

  struct userdata_trb{Float64}
    p::Float64
  end

  function gen_h_indices(n)
    tril_indices_colmajor = filter(x->x[2]>=x[1], collect(Iterators.product(1:n, 1:n)))
    (getindex.(tril_indices_colmajor, 2), getindex.(tril_indices_colmajor, 1))
  end

  function Vecchia.optimize(obj, init, solver::Vecchia.TRBSolver;
                            box_lower=fill(0.0, length(init)), 
                            box_upper=fill(Inf, length(init)))
    n = length(init)
    n == length(box_lower) == length(box_upper) || throw(error("Dimension disagreement"))

    data    = Ref{Ptr{Cvoid}}()
    control = Ref{trb_control_type{Float64,Int64}}()
    status  = Ref{Int64}(0)
    inform  = Ref{trb_inform_type{Float64, Int64}}()
    trb_initialize(Float64, Int64, data, control, status)

    fun  = (x, f, _) -> begin
      try
        f[] = obj(x)
        return 0
      catch
        return -1
      end
    end

    grad = (x, g, _) -> begin
      try
        ForwardDiff.gradient!(g, obj, x)
        return 0
      catch
        return -1
      end
    end

    hess = let (r, c) = gen_h_indices(n)
      (x, h, _) -> begin
        try
          _h = ForwardDiff.hessian(obj, x)
          for j in eachindex(r, c)
            h[j] = _h[r[j], c[j]] 
          end
          return 0
        catch
          return -1
        end
      end
    end

    prec = (x, u, v, _) -> (u .= v; return 0) # dummy preconditioner (?)

    @reset control[].trs_control.symmetric_linear_solver = galahad_linear_solver("sytr")
    @reset control[].trs_control.definite_linear_solver  = galahad_linear_solver("potr")
    @reset control[].psls_control.definite_linear_solver = galahad_linear_solver("potr")

    @reset control[].print_level       = solver.print_level
    @reset control[].model             = 2
    @reset control[].subproblem_direct = true
    @reset control[].two_norm_tr       = true
    @reset control[].exact_gcp         = true

    trb_import(Float64, Int64, control, data, status, n, box_lower, box_upper,
               "dense", div(n*(n+1), 2), C_NULL, C_NULL, C_NULL)

    eval_status = Ref{Int64}()
    f = Ref{Float64}(0.0)
    g = zeros(n)
    H = zeros(div(n*(n+1), 2))
    u = zeros(n)
    v = zeros(n)
    x = copy(init)

    dummy_data = userdata_trb(NaN)
    terminated = false
    while !terminated
      trb_solve_reverse_with_mat(Float64, Int64, data, status, eval_status, n, x, f[], g,
                                 div(n*(n+1), 2), H, u, v)
      if status[] == 0 # successful termination
        terminated = true
      elseif status[] < 0 # error exit
        terminated = true
      elseif status[] == 2 # evaluate f
        eval_status[] = fun(x, f, dummy_data)
      elseif status[] == 3 # evaluate g
        eval_status[] = grad(x, g, dummy_data)
      elseif status[] == 4 # evaluate H
        eval_status[] = hess(x, H, dummy_data)
      elseif status[] == 6 # evaluate the product with P
        eval_status[] = prec(x, u, v, dummy_data)
      else
        throw(error("GALAHAD/trb returned unexpected status code $(status[])."))
      end
    end

    trb_information(Float64, Int64, data, inform, status)
    iszero(inform[].status[]) || @warn "Optimization returned a non-zero exit status"
    trb_terminate(Float64, Int64, data, control, inform)
    x
  end

end

