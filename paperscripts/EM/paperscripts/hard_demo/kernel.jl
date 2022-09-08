
function get_inv_aniso_norm(xmy, rot, l1, l2)
  rotmat  = @SMatrix [cos(rot) -sin(rot) ; sin(rot) cos(rot)]
  rot_xmy = rotmat*xmy # the compiler will unroll all this for me....right?
  sqrt((rot_xmy[1]/l1)^2 + (rot_xmy[2]/l2)^2)
end

function ns_scale_part(x::SVector{2,Float64}, p2, p3) 
  (2*one(p2) + cos(p2*(x[1]-p3)) + cos(p2*(x[2]-p3+0.15)))
end

function ns_scale(x::SVector{2,Float64}, y::SVector{2,Float64}, p1, p2, p3) 
  sx = ns_scale_part(x, p2, p3)
  sy = ns_scale_part(y, p2, p3)
  p1*sx*sy
end

function matern_ganiso_nonugget(x, y, p)
  (sg2, sg_cos_p2, sg_cos_p3, rot, l1, l2, nu, nug2) = p
  xmy   = x-y
  scale = ns_scale(x, y, sg2, sg_cos_p2, sg_cos_p3)
  all(iszero, xmy) && return scale
  dist = get_inv_aniso_norm(xmy, rot, l1, l2)
  normalizer = scale/((2^(nu-1))*BesselK.gamma(nu))
  normalizer*BesselK.adbesselkxv(nu, dist)
end

function matern_ganiso_withnugget(x, y, p)
  out = matern_ganiso_nonugget(x, y, p)
  x == y ? out + p[end] : out
end

