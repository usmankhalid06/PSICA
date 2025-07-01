function spline_bases = create_spline_dictionary(N, num_splines)
    spline_bases = [];
    knots_coarse = linspace(1, N, 8);
    spline_bases = [spline_bases, create_bspline_basis(N, knots_coarse, 3)];

    knots_medium = linspace(1, N, 16);
    spline_bases = [spline_bases, create_bspline_basis(N, knots_medium, 3)];

    knots_fine = linspace(1, N, 32);
    spline_bases = [spline_bases, create_bspline_basis(N, knots_fine, 3)];

    spline_bases = spline_bases * diag(1./sqrt(sum(spline_bases.^2, 1)));

    spline_bases = spline_bases(:, 1:min(num_splines, size(spline_bases, 2)));
   
end


function spline_bases = create_bspline_basis(N, knots, degree)

knots = knots(:)'; 
extended_knots = [repmat(knots(1), 1, degree), knots, repmat(knots(end), 1, degree)];

num_basis = length(extended_knots) - degree - 1;
spline_bases = zeros(N, num_basis);
t = 1:N;
for i = 1:num_basis
    for j = 1:N
        spline_bases(j, i) = bspline_eval(t(j), i, degree, extended_knots);
    end
end

valid_cols = sum(abs(spline_bases), 1) > 1e-10;
spline_bases = spline_bases(:, valid_cols);

end

function val = bspline_eval(t, i, degree, knots)
% Evaluate B-spline basis function using Cox-de Boor recursion
if degree == 0
    if t >= knots(i) && t < knots(i+1)
        val = 1;
    else
        val = 0;
    end
else
    val = 0;
   
    if knots(i+degree) ~= knots(i)
        val = val + (t - knots(i)) / (knots(i+degree) - knots(i)) * ...
              bspline_eval(t, i, degree-1, knots);
    end
    
    if knots(i+degree+1) ~= knots(i+1)
        val = val + (knots(i+degree+1) - t) / (knots(i+degree+1) - knots(i+1)) * ...
              bspline_eval(t, i+1, degree-1, knots);
    end
end
end