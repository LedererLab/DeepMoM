#############################
### Median of Means group ###
#############################


mom <- function(y, X, para=list(), para2=list(), B=NULL, class=FALSE, 
                class.score=FALSE){
  
  # Description :
  #               Find the block of data that achieve the median of empirical 
  #               means. 
  # Usage : 
  #         mom(y, X, para=list(), para2=list(), B=NULL, class=FALSE, 
  #             class.score=FALSE)
  # Arguments : 
  #   y : A vector of dimension c represents the output layer of the neural 
  #       network.
  #   X : A matrix of dimension n * p, where each row represents an input of the
  #       neural network.
  #   para : A list contains all weight matrices and all bias terms of the 
  #          neural networks.
  #   para2 : A list contains the second weight matrices and bias terms of the 
  #          neural networks for increment test.
  #   B : A list contains the partition of the data.
  #   class : A Boolen value determines whether it is a classification task or 
  #           not.
  #   class.score : A Boolen value determines whether to compute the predicted 
  #                 probability of occurs for each class.
  # Returns : 
  #   A list contains the block of data that achieve the median of empèirical 
  #   means. 
  
  num.obs <- dim(X)[1]
  num.par <- dim(X)[2]
  X.original <- X
  y.original <- y

  loss.test <- c()
  loss.first1 <- c()
  loss.first2 <- c()
  for(i in 1:length(B)){
    index <- B[[i]]
    X <- X[index,]
    if(isTRUE(class)){
      y <- y[index,]
    }else{
      y <- y[index]
    }
   
    # First bias parameters 
    for(j in 1:length(para)){
      B.trim <- as.vector(para[[j]][[2]][1,])
      para[[j]][[2]] <- matrix(rep(B.trim,length(index)),nrow=length(index),byrow=TRUE)
    }
    # Second bias parameters
    for(j in 1:length(para2)){
      B.trim <- as.vector(para2[[j]][[2]][1,])
      para2[[j]][[2]] <- matrix(rep(B.trim,length(index)),nrow=length(index),byrow=TRUE)
    }
    # tests
    if(isTRUE(class)){
      score <- FeedForwardNN(X=X,para=para,class=class,class.score=class.score)
      exp_scores <- exp(score)
      probs <- exp_scores / rowSums(exp_scores)
      corect_logprobs <- -log(probs)
      loss1 <- sum(corect_logprobs*y)/num.obs
      ##########################################################################
      score <- FeedForwardNN(X=X,para=para2,class=class,class.score=class.score)
      exp_scores <- exp(score)
      probs <- exp_scores / rowSums(exp_scores)
      corect_logprobs <- -log(probs)
      loss2 <- sum(corect_logprobs*y)/num.obs
    }else{
      loss1 <- sum((y-FeedForwardNN(X=X,para=para,class=class,class.score=class.score))^2)/length(index)
      loss2 <- sum((y-FeedForwardNN(X=X,para=para2,class=class,class.score=class.score))^2)/length(index)
    }
    
    loss.test <- c(loss.test, (loss1-loss2))
    loss.first1 <- c(loss.first1, loss1)
    loss.first2 <- c(loss.first2, loss2)
    
    # Reverse first parameters
    for(j in 1:length(para)){
      B.trim <- matrix(para[[j]][[2]],nrow=length(index),byrow=TRUE)
      B.trim <- as.vector(B.trim[1,])
      para[[j]][[2]] <- matrix(rep(B.trim,num.obs),nrow=num.obs,byrow=TRUE)
    }
    # Reverse second parameters
    for(j in 1:length(para2)){
      B2.trim <- matrix(para2[[j]][[2]],nrow=length(index),byrow=TRUE)
      B2.trim <- as.vector(B2.trim[1,])
      para2[[j]][[2]] <- matrix(rep(B2.trim,num.obs),nrow=num.obs,byrow=TRUE)
    }
    # Reverse dataset
    X <- X.original
    y <- y.original
  }
  order.index <- sort(loss.test,index=TRUE)$ix
  if(isTRUE((length(loss.test)%%2)==0)){
    output.index <- order.index[(length(loss.test)/2)]
  }else{
    output.index <- order.index[(floor(length(loss.test)/2)+1)]
  }
  output <- list()
  output[[1]] <- B[[output.index]]
  output[[2]] <- median(loss.test)
  output[[3]] <- min(min(loss.first1),min(loss.first2))
  return(output)
}






