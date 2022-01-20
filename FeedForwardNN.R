#######################################
### Feed Fordward of Neural Network ###
#######################################

FeedForwardNN <- function(X, para=list(), class=FALSE, class.score=FALSE){
  # Description :
  #               Compute the output of the neural netowtk by feed forwardd 
  #               method derivative of neural network with respect to each 
  # Usage : 
  #         FeedForwardNN(X, para=list(), class=FALSE, class.score=FALSE)
  # Arguments : 
  #   X : A matrix of dimension n * p, where each row represents an input of the
  #       neural network.
  #   para : A list contains all weight matrices and all bias terms of the 
  #          neural networks. 
  #   class : A Boolen value determines whether it is a classification task or 
  #           not.
  #   class.score : A Boolen value determines whether to compute the predicted 
  #                 probability of occurs for each class.
  # Returns : 
  #   A value represents the predicted class and the predicted probability of 
  #   occurs for each class. 
  
  num.obs <- dim(X)[1]
  num.par <- dim(X)[2]
  
  for(j in 1:length(para)){
    B.trim <- as.vector(para[[j]][[2]][1,])
    para[[j]][[2]] <- matrix(rep(B.trim,num.obs),nrow=num.obs,byrow=TRUE)
  }
  
  hidden_layer <- list()
  if(isTRUE(length(para)==1)){
    score <- X%*%para[[1]][[1]]+para[[1]][[2]]
  }else if(isTRUE(length(para)==2)){
    hidden_layer <- pmax(0, X%*%para[[1]][[1]]+para[[1]][[2]])
    hidden_layer <- matrix(hidden_layer, nrow=num.obs)
    score <- hidden_layer%*%para[[2]][[1]]+para[[2]][[2]]
  }else{
    hidden_layer <- list()
    hidden_layer[[1]] <- pmax(0, X%*%para[[1]][[1]]+para[[1]][[2]])
    hidden_layer[[1]] <- matrix(hidden_layer[[1]], nrow=num.obs)
    for(i in 2:(length(para)-1)){
      hidden_layer[[i]] <- hidden_layer[[(i-1)]]%*%para[[i]][[1]]+para[[i]][[2]]
      hidden_layer[[i]] <- pmax(0, hidden_layer[[i]])
      hidden_layer[[i]] <- matrix(hidden_layer[[i]], nrow=num.obs)
    }
    score <- hidden_layer[[(length(para)-1)]]%*%para[[length(para)]][[1]]+para[[length(para)]][[2]]
  }
  
  if(isTRUE(class)){
    if(isTRUE(class.score)){
      return(score)
    }else{
      predicted_class <- apply(score, 1, which.max)
      return(predicted_class)
    }
  }else{
    score <- as.vector(score)
    return(score)
  }
}






