########################
### Huber derivative ###
########################

huber_derivative <- function(r, k){
  # Description :
  #               Compute the derivative of Huber loss function.  
  # Usage : 
  #         huber_derivative(r, k)
  # Arguments : 
  #   r : An numerical value represents the input value of Huber loss derivative 
  #       function. 
  #   k : A numerical value that controls the robustness of the Huber loss 
  #       derivative function.
  # Returns : 
  #   A numerical value represents the derivative of Huber loss function.
  
  pmin(k, pmax(-k, r))
}


