##################
### Huber loss ###
##################


huber_loss <- function(r, k){
  # Description :
  #               Compute the value of Huber loss function.  
  # Usage : 
  #         huber_loss(r, k)
  # Arguments : 
  #   r : An numerical value represents the input value of Huber loss function. 
  #   k : A numerical value that controls the robustness of the Huber loss 
  #       function.
  # Returns : 
  #   A numerical value represents the value of Huber loss function.
  
  output <- rep(0, length(r))
  for (i in 1:length(r)){
    if(isTRUE(abs(r[i])<=k)){
      output[i] <- 0.5 * r[i]^2
    }else{
      output[i] <- k * abs(r[i])-0.5 * k ^ 2
    }
  }
  return(output)
}

