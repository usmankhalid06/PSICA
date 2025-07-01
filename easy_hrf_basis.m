function Dp = easy_hrf_basis(N, TR, total_bases, focus)
if nargin < 4, focus = 'fast'; end
if nargin < 3, total_bases = 200; end

switch lower(focus)
   case 'fast'
       config = create_fast_config(total_bases);
   case 'medium' 
       config = create_medium_config(total_bases);
   case 'slow'
       config = create_slow_config(total_bases);
   case 'mixed'
       config = create_mixed_config(total_bases);
   case 'custom'
       config = create_custom_config(total_bases);
   otherwise
       warning('Unknown focus: %s, using fast', focus);
       config = create_fast_config(total_bases);
end

Dp = controlled_dense_hrf_basis(N, TR, config);
end

function config = create_fast_config(total_bases)
config.hrf_types = {'canonical', 'fast'};
config.brief = struct('count', round(0.6*total_bases), 'durations', [0.5:0.5:2], 'onset_density', 'dense');
config.short = struct('count', round(0.4*total_bases), 'durations', [2.5:0.5:4], 'onset_density', 'medium');
end

function config = create_medium_config(total_bases)
config.hrf_types = {'canonical', 'fast', 'slow'};
config.short = struct('count', round(0.3*total_bases), 'durations', [1:1:4], 'onset_density', 'medium');
config.medium = struct('count', round(0.5*total_bases), 'durations', [4:1:8], 'onset_density', 'medium');
config.long = struct('count', round(0.2*total_bases), 'durations', [8:2:16], 'onset_density', 'sparse');
end

function config = create_slow_config(total_bases)
config.hrf_types = {'canonical', 'slow'};
config.medium = struct('count', round(0.3*total_bases), 'durations', [4:2:10], 'onset_density', 'sparse');
config.long = struct('count', round(0.4*total_bases), 'durations', [10:2:20], 'onset_density', 'sparse');
config.very_long = struct('count', round(0.3*total_bases), 'durations', [20:5:40], 'onset_density', 'very_sparse');
end

function config = create_mixed_config(total_bases)
config.hrf_types = {'canonical', 'fast', 'slow'};
config.brief = struct('count', round(0.25*total_bases), 'durations', [0.5:0.5:2], 'onset_density', 'dense');
config.short = struct('count', round(0.25*total_bases), 'durations', [2:1:6], 'onset_density', 'medium');
config.medium = struct('count', round(0.25*total_bases), 'durations', [6:2:12], 'onset_density', 'medium');
config.long = struct('count', round(0.25*total_bases), 'durations', [12:3:24], 'onset_density', 'sparse');
end

function config = create_custom_config(total_bases)
choice = input('Enter choice (1-5): ');
hrf_choice = input('Enter choice (1-3): ');

switch choice
   case 1
       durations = [0.5:0.25:2];
       density = 'dense';
   case 2
       durations = [1:0.5:5];
       density = 'dense';
   case 3
       durations = [3:1:10];
       density = 'medium';
   case 4
       durations = [5:2:20];
       density = 'medium';
   case 5
       durations = [10:5:40];
       density = 'sparse';
   otherwise
       durations = [1:1:8];
       density = 'medium';
end

switch hrf_choice
   case 1
       hrf_types = {'canonical'};
   case 2
       hrf_types = {'canonical', 'fast'};
   case 3
       hrf_types = {'canonical', 'fast', 'slow'};
   otherwise
       hrf_types = {'canonical', 'fast'};
end

config.hrf_types = hrf_types;
config.main = struct('count', total_bases, 'durations', durations, 'onset_density', density);
end

function Dp = controlled_dense_hrf_basis(N, TR, config)
   categories = fieldnames(config);
   categories = categories(~strcmp(categories, 'hrf_types'));
   
   Dp = [];
   
   for cat_idx = 1:length(categories)
       cat_name = categories{cat_idx};
       cat_config = config.(cat_name);
       
       cat_basis = generate_dense_category(N, TR, cat_config, config.hrf_types);
       cat_basis = cat_basis(:, 1:min(cat_config.count, size(cat_basis, 2)));
       
       Dp = [Dp, cat_basis];
   end
   
   Dp = Dp * diag(1./sqrt(sum(Dp.*Dp)));
end

function cat_basis = generate_dense_category(N, TR, cat_config, hrf_types)
   durations = cat_config.durations / TR;
   onset_times = get_smart_onsets(N, cat_config.onset_density, max(durations));
   hrf_params = get_hrf_parameters(hrf_types);
   
   cat_basis = [];
   total_combinations = length(durations) * length(onset_times) * size(hrf_params, 1);
   sample_step = max(1, floor(total_combinations / cat_config.count));
   
   combo_count = 0;
   for h = 1:size(hrf_params, 1)
       hrf_kernel = create_hrf_kernel(N, TR, hrf_params(h, :));
       
       for duration = durations
           for onset = onset_times
               combo_count = combo_count + 1;
               
               if mod(combo_count, sample_step) == 1 && size(cat_basis, 2) < cat_config.count
                   stimulus = zeros(N, 1);
                   end_time = min(N, onset + round(duration) - 1);
                   if end_time > onset
                       stimulus(onset:end_time) = 1;
                       convolved = conv(stimulus, hrf_kernel, 'same');
                       cat_basis = [cat_basis, convolved];
                   end
               end
           end
       end
   end
   
   end_region_start = round(0.75 * N);
   num_end_bases = min(25, round(0.2 * cat_config.count));
   
   for i = 1:num_end_bases
       if size(cat_basis, 2) >= cat_config.count, break; end
       
       if i <= num_end_bases/3
           peak_time = end_region_start + round((i-1) * (N - end_region_start) / (num_end_bases/3));
           
           shortest_duration = min(durations);
           canonical_hrf = create_hrf_kernel(N, TR, [6.0, 1.0, 16.0, 1.0, 0.15]);
           
           stimulus = zeros(N, 1);
           stim_start = max(1, peak_time - round(shortest_duration/2));
           stim_end = min(N, stim_start + round(shortest_duration) - 1);
           stimulus(stim_start:stim_end) = 1;
           
           convolved = conv(stimulus, canonical_hrf, 'same');
       elseif i <= 2*num_end_bases/3
           late_onset = round(0.8 * N) + (i - num_end_bases/3 - 1) * 2;
           late_onset = min(late_onset, N-2);
           
           stimulus = zeros(N, 1);
           stimulus(late_onset:N) = 1;
           
           sharp_hrf = create_hrf_kernel(N, TR, [4.0, 0.6, 14.0, 0.7, 0.08]);
           convolved = conv(stimulus, sharp_hrf, 'same');
       else
           end_pos = round(0.9 * N) + (i - 2*num_end_bases/3 - 1) * 2;
           end_pos = min(end_pos, N);
           
           convolved = zeros(N, 1);
           sigma = 5;
           for t = 1:N
               convolved(t) = exp(-0.5 * ((t - end_pos) / sigma)^2);
           end
           convolved = convolved / norm(convolved);
       end
       
       cat_basis = [cat_basis, convolved];
   end
end

function onset_times = get_smart_onsets(N, density_spec, max_duration_trs)
   if isnumeric(density_spec)
       base_step = density_spec;
   else
       switch lower(density_spec)
           case 'dense', base_step = 1;
           case 'medium', base_step = 2;
           case 'sparse', base_step = 4;
           case 'very_sparse', base_step = 8;
           otherwise, base_step = 2;
       end
   end
   
   max_onset = N - 1;
   
   if base_step == 1
       onset_times = 1:max_onset;
   else
       onset_times = [];
       
       primary = 1:base_step:max_onset;
       onset_times = [onset_times, primary];
       
       for offset = 1:(base_step-1)
           secondary = (1+offset):base_step:max_onset;
           onset_times = [onset_times, secondary];
       end
       
       onset_times = unique(onset_times);
       onset_times = onset_times(onset_times >= 1 & onset_times <= max_onset);
   end
end

function hrf_params = get_hrf_parameters(hrf_types)
   all_params = containers.Map();
   all_params('canonical') = [6.0, 1.0, 16.0, 1.0, 0.15];
   all_params('fast') = [5.0, 0.8, 15.0, 0.8, 0.10];
   all_params('slow') = [8.0, 1.2, 17.0, 1.0, 0.20];
   
   hrf_params = [];
   for i = 1:length(hrf_types)
       if all_params.isKey(hrf_types{i})
           hrf_params = [hrf_params; all_params(hrf_types{i})];
       end
   end
   
   if isempty(hrf_params)
       hrf_params = [6.0, 1.0, 16.0, 1.0, 0.15];
   end
end

function hrf_kernel = create_hrf_kernel(N, TR, params)
   hrf_length = min(N, round(30/TR));
   t = (0:hrf_length-1) * TR;
   
   a1 = params(1); b1 = params(2);
   a2 = params(3); b2 = params(4); 
   c = params(5);
   
   hrf_kernel = ((t/a1).^(a1) .* exp(-(t-a1)/b1) - ...
                c * (t/a2).^(a2) .* exp(-(t-a2)/b2));
   
   hrf_kernel = hrf_kernel / max(abs(hrf_kernel));
   hrf_kernel = hrf_kernel(:);
end