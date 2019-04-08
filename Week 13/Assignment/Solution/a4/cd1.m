function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    %Sample pixel values as binaries for input to model
    visible_data = sample_bernoulli(visible_data);
    
    %Sample a binary state for hidden units conditional on data
    hidden_state = sample_bernoulli(visible_state_to_hidden_probabilities(rbm_w, visible_data));
    %Calculate positive direction gradient
    delta_rbm_w_positive = configuration_goodness_gradient(visible_data, hidden_state);
    %Sample a binary state for visible units conditional on hidden units -> Reconstruction
    visible_state = sample_bernoulli(hidden_state_to_visible_probabilities(rbm_w, hidden_state));
    %Sample a binary state for hidden units conditional on visible units
    hidden_state = visible_state_to_hidden_probabilities(rbm_w, visible_state);
    %Calculate negative direction gradient
    delta_rbm_w_negative = configuration_goodness_gradient(visible_state, hidden_state);
    %calculate weight update for CD-1
    ret = delta_rbm_w_positive - delta_rbm_w_negative;
end
