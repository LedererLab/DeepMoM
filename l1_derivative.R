#####################
### L1 derivative ###
#####################


l1_derivative <- function(x){
  # Description :
  #               Compute the derivative of L1 loss function.  
  # Usage : 
  #         l1_derivative(r, k)
  # Arguments : 
  #   x : An numerical value represents the input value of L1 loss derivative 
  #       function. 
  # Returns : 
  #   A numerical value represents the derivative of L1 loss function.
  
  output <- 1 * (x > 0) - 1 * (x <= 0)
  return(output)
}

