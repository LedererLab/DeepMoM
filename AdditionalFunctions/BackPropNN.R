##########################################
### Back Propagation of Neural Network ###
##########################################

BackPropNN <- function(y, X, para=list(), alpha, loss.f=NULL, q=NULL, bias=FALSE
                       , class=FALSE){
  # Description :
  #               Compute the derivative of neural network with respect to each 
  #               weight matrix by back propagation.  
  # Usage : 
  #         BackPropNN(y, X, para=list(), alpha, loss.f=NULL, q=NULL, bias=FALSE
  #                    , class=FALSE)
  # Arguments : 
  #   y : A vector of dimension c represents the output layer of the neural 
  #       network.
  #   X : A matrix of dimension n * p, where each row represents an input of the
  #       neural network.
  #   para : A list contains all weight matrices and all bias terms of the 
  #          neural networks. 
  #   alpha : A numerical value represents the learning rate for the gradient 
  #           descent algorithm.
  #   loss.f : A string represents the loss function for optimizing the neural 
  #            network.
  #   q : A numerical value that controls the robustness of the Huber loss. 
  #   bias : A Boolen value determines whether the neural network contains the 
  #          bias term. 
  #   class : A Boolen value determines whether it is a classification task or 
  #           not.
  # Returns : 
  #   A list contains the updated derivative with respect to each weight 
  #   matrices.
  
  num.obs <- dim(X)[1]
  num.par <- dim(X)[2]

  if(isTRUE(bias==FALSE)){
    for(i in 1:(l+1)){
      para[[i]][[2]] <- rep(0,P[(i+1)])
      para[[i]][[2]] <- matrix(rep(para[[i]][[2]],num.obs),nrow=num.obs,byrow=TRUE)
    }
  }
  
  A <- list()
  Z <- list()
  D <- list()
  l <- length(para)-1
  
  A[[1]] <- X
  for(i in 2:(l+2)){
    Z[[i]] <- A[[(i-1)]]%*%para[[(i-2+1)]][[1]]+para[[(i-2+1)]][[2]]
    if(isTRUE(i==(l+2))){
      A[[i]] <- Z[[i]]
    }else{
      A[[i]] <- pmax(0, Z[[i]])
      A[[i]] <- matrix(A[[i]], nrow=num.obs)
    }
  }
  
  if(isTRUE(class)){
    exp_scores <- exp(A[[(l+2)]])
    probs <- exp_scores/rowSums(exp_scores)
    D[[(l+2)]] <- (probs-y)/num.obs
  }else{
    if(isTRUE(loss.f=="ls")){
      D[[(l+2)]] <- (-2/num.obs)*(y-A[[(l+2)]])
    } else if(isTRUE(loss.f=="l1")){
      D[[(l+2)]] <- (-1/num.obs)*l1_derivative(as.vector(y-A[[(l+2)]]))
      D[[(l+2)]] <- matrix(D[[(l+2)]], nrow=num.obs)
    }else if(isTRUE(loss.f=="huber")){
      D[[(l+2)]] <- (-1/num.obs)*huber_derivative(as.vector(y-A[[(l+2)]]),q)
      D[[(l+2)]] <- matrix(D[[(l+2)]], nrow=num.obs)
    }
  }
  
  for(i in ((l+2)-1):2){
    D[[i]] <- D[[(i+1)]]%*%t(para[[(i-2+1+1)]][[1]])
    D[[i]][Z[[i]]<=0] <- 0
  }
  
  dpara <- list()
  for(i in 1:(l+1)){
    dpara[[i]] <- list()
    dpara[[i]][[1]] <- t(A[[(i-1+2-1)]])%*%D[[(i-1+2)]]
  }
  
  if(isTRUE(bias==TRUE)){
    for(i in 1:(l+1)){
      dpara[[i]][[2]] <- colSums(D[[i-1+2]])
      dpara[[i]][[2]] <- matrix(rep(dpara[[i]][[2]],num.obs),nrow=num.obs,byrow=TRUE)
    }
  }
  
  update1 <- c()
  for(j in 1:(l+1)){
    para[[j]][[1]] <- para[[j]][[1]]-alpha*dpara[[j]][[1]]
    update1 <- c(update1, sum(abs(para[[j]][[1]])^2))
  }
  
  if(isTRUE(bias==TRUE)){
    update2 <- c()
    for(j in 1:(l+1)){
      para[[j]][[2]] <- para[[j]][[2]]-alpha*dpara[[j]][[2]]
      update2 <- c(update2, sum(abs(para[[j]][[2]])^2))
    }
  }else{
    update2 <- NA
  }
  
  if(isTRUE(all(A[[l+1]]==0))){
    update1 <- NA
  }
  
  output <- list()
  output[[1]] <- para
  if(isTRUE(bias)){
    output[[2]] <- sum(update1)+sum(update2)
  }else{
    output[[2]] <- sum(update1)
  }
  
  return(output)
}



