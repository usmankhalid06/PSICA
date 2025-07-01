function [independent_cols,Up_clean] = remove_dependent_qr(Up_final, tol)

[~, R, P] = qr(Up_final, 'vector');

diag_R = abs(diag(R));
rank_est = sum(diag_R > tol * diag_R(1));

independent_cols = sort(P(1:rank_est));
Up_clean = Up_final(:, independent_cols);

end