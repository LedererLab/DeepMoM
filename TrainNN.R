############################
### Train Neural Network ###
############################

TrainNN <- function(y, X, P=P, alpha, iteration, random=TRUE, batch=NULL, 
                    MOM=FALSE, k=NULL, loss.f=NULL, q=NULL, bias=FALSE, 
                    class=FALSE, beta=1, qs=NULL){
  # Description :
  #               Train the neural network under the DeepMoM structure.   
  # Usage : 
  #         TrainNN(y, X, P=P, alpha,iteration, random=TRUE, batch=NULL, 
  #                 stop.i.loss=1e-5, stop.i.update=1e-5, MOM=FALSE, k=NULL, 
  #                 loss.f=NULL, q=NULL, bias=FALSE, class=FALSE, beta=1, 
  #                 qs=NULL)
  # Arguments : 
  #   y : A vector of dimension c represents the output layer of the neural 
  #       network.
  #   X : A matrix of dimension n * p, where each row represents an input of the
  #       neural network.
  #   P : A vector indicates the number of neurons in each layer of the neural 
  #       network.
  #   alpha : A numerical value represents the learning rate for the gradient 
  #           descent algorithm.
  #   iteration : A numerical value represents the limit of gradient updates.
  #   random : A Boolen represents whether to apply the stochastic gradient 
  #            descent.
  #   batch : A numerical value represents the batch size for stochastic 
  #           gradient descent.
  #   MOM : A Boolen value determines whether to apply the DeepMoM structure.
  #   k : A positive intger represents the number of blocks for DeepMoM 
  #       structure.
  #   loss.f : A string represents the loss function for optimizing the neural 
  #            network.
  #   q : A numerical value that controls the robustness of the Huber loss. 
  #   bias : A Boolen value determines whether the neural network contains the 
  #          bias term. 
  #   class : A Boolen value determines whether it is a classification task or 
  #           not.
  #   beta : A numerical value to scale the default initial values. 
  #   qs : A numerical value that represents the robustness parameter of the 
  #        Huber loss. 
  # Returns : 
  #   A list contains the trained weight matrices and bias terms for the neural 
  #   netowrk.
  
  
  num.obs <- dim(X)[1]
  num.par <- dim(X)[2]
  
  if(isTRUE(is.matrix(y))){
    K <- dim(y)[2]
    classIndex <- rep(0,dim(X)[1])
    for(i in 1:dim(X)[1]){
      for(j in 1:K){
        if(isTRUE(y[i,j]==1)){
          classIndex[i] <- j
        }
      }
    }
  }
  
  X.original <- X
  y.original <- y
  if(isTRUE(class)){
    classIndex.original <- classIndex
  }
  alpha.original <- alpha
  
  # Initial values
  para <- list()
  
  Psum <- c()
  for(i in 1:(l+1)){
    para[[i]] <- list()
    para[[i]][[1]] <- beta*matrix(rnorm(P[i]*P[i+1],0,1),nrow=P[i])
    Psum <- c(Psum,sum(abs(para[[i]][[1]])^2))
  }
  
  Psum2 <- c()
  for(i in 1:(l+1)){
    if(isTRUE(bias)){
      para[[i]][[2]] <- rep(0,P[(i+1)])
    }else{
      para[[i]][[2]] <- rep(0,P[(i+1)])
    }
    para[[i]][[2]] <- matrix(rep(para[[i]][[2]],num.obs),nrow=num.obs,byrow=TRUE)
    Psum2 <- c(Psum2,sum(abs(para[[i]][[2]])^2))
  }
  
  if(isTRUE(class==0)){
    M <- max(abs(FeedForwardNN(X=X,para=para,class=class,class.score=FALSE)))
    rescale <- (max(abs(y))/M)^(1/(l+1))
    
    for(i in 1:(l+1)){
      para[[i]][[1]] <- para[[i]][[1]]*rescale
      Psum <- c(Psum,sum(abs(para[[i]][[1]])^2))
    }
  }
  
  
  if(isTRUE(MOM==TRUE)){
    para2 <- list()
    
    P2sum <- c()
    for(i in 1:(l+1)){
      para2[[i]] <- list()
      para2[[i]][[1]] <- beta*matrix(rnorm(P[i]*P[i+1],0,1),nrow=P[i])
      P2sum <- c(P2sum,sum(abs(para2[[i]][[1]])^2))
    }
    
    P2sum2 <- c()
    for(i in 1:(l+1)){
      if(isTRUE(bias)){
        para2[[i]][[2]] <- rep(0,P[(i+1)])
      }else{
        para2[[i]][[2]] <- rep(0,P[(i+1)])
      }
      para2[[i]][[2]] <- matrix(rep(para2[[i]][[2]],num.obs),nrow=num.obs,byrow=TRUE)
      P2sum2 <- c(P2sum2,sum(abs(para2[[i]][[2]])^2))
    }
    
    if(isTRUE(class==0)){
      M2 <- max(abs(FeedForwardNN(X=X,para=para2,class=class,class.score=FALSE)))
      rescale2 <- (max(abs(y))/M2)^(1/(l+1))
      
      for(i in 1:(l+1)){
        para2[[i]][[1]] <- para2[[i]][[1]]*rescale2
        P2sum <- c(P2sum,sum(abs(para2[[i]][[1]])^2))
      }
    }
  }
  
  # Stopping returns
  paraNA <- list()
  
  for(i in 1:(l+1)){
    paraNA[[i]] <- list()
    paraNA[[i]][[1]] <- matrix(rep(NA,P[i]*P[i+1]),nrow=P[i])
  }
  
  for(i in 1:(l+1)){
    paraNA[[i]][[2]] <- rep(NA,P[(i+1)])
    paraNA[[i]][[2]] <- matrix(rep(paraNA[[i]][[2]],num.obs),nrow=num.obs,byrow=TRUE)
  }
  
  # Training process
  L <- c() # Loss
  A <- c()
  Ltest <- c()
  update.par1 <- sum(Psum) + sum(Psum2)
  if(isTRUE(MOM==TRUE)){
    L.mom <- c()
    A.mom <- c()
    IC.mom <- c()
    L2.mom <- c()
    Ltest.mom <- c()
    L2test.mom <- c()
    update1.par1 <- sum(Psum) + sum(Psum2)
    update2.par1 <- sum(P2sum) + sum(P2sum2)
  }
  
  for(i in 1:iteration){
    # Reverse original dataset
    X <- X.original
    y <- y.original
    
    epoch <- floor(dim(X.original)[1]/batch)
    
    if(isTRUE(class)){
      classIndex <- classIndex.original
    }
    
    plot(NULL, xlim=c(0,1), ylim=c(0,1), ylab="y label", xlab="x lablel")
    
    #Stochastic sample
    if(isTRUE(random==TRUE)){
      if(isTRUE(i%%floor(dim(X)[1]/batch)==1)){
        Index <- group(floor(dim(X)[1]/batch), dim(X)[1])
      }
      
      if(isTRUE(MOM==TRUE)){
        if(isTRUE(i<=floor(dim(X)[1]/batch))){
          index <- Index[[i]]
        } else {
          index <- Index[[(i%%floor(dim(X)[1]/batch)+1)]]
        }
        
        if(isTRUE(class)){
          classIndex <- classIndex[index]
        }
        
        X <- X[index,]
        
        if(isTRUE(class)){
          y <- y[index,]
        }else{
          y <- y[index]
        }
        
        for(j in 1:length(para)){
          B.trim <- as.vector(para[[j]][[2]][1,])
          para[[j]][[2]] <- matrix(rep(B.trim,length(index)),nrow=length(index),byrow=TRUE)
        }
        
        for(j in 1:length(para2)){
          B.trim <- as.vector(para2[[j]][[2]][1,])
          para2[[j]][[2]] <- matrix(rep(B.trim,length(index)),nrow=length(index),byrow=TRUE)
        }
      }else{
        if(isTRUE(i<=floor(dim(X)[1]/batch))){
          index <- Index[[i]]
        } else {
          index <- Index[[(i%%floor(dim(X)[1]/batch)+1)]]
        }
        
        if(isTRUE(class)){
          classIndex <- classIndex[index]
        }
        
        X <- X[index,]
        
        if(isTRUE(class)){
          y <- y[index,]
        }else{
          y <- y[index]
        }
        
        for(j in 1:length(para)){
          B.trim <- as.vector(para[[j]][[2]][1,])
          para[[j]][[2]] <- matrix(rep(B.trim,length(index)),nrow=length(index),byrow=TRUE)
        }
      }
    }
    
    X.random <- X
    y.random <- y
    
    #Check Progress
    
    if(isTRUE(MOM==FALSE)){
      # Monitor loss
      if(isTRUE(class)){
        score <- FeedForwardNN(X=X,para=para,class=class,class.score=TRUE)
        pre.class <- FeedForwardNN(X=X,para=para,class=class,class.score=FALSE)
        accuracy <- sum(pre.class==(classIndex))/dim(X.original)[1]
        exp_scores <- exp(score)
        probs <- exp_scores / rowSums(exp_scores)
        corect_logprobs <- -log(probs)
        data_loss <- sum(corect_logprobs*y)/dim(X.original)[1]
        L <- c(L, data_loss)
        A <- c(A, accuracy)
      }else{
        if(isTRUE(loss.f=="ls")){
          L <- c(L, sum((y-FeedForwardNN(X=X,para=para,class=class,class.score=FALSE))^2)/(dim(X.original)[1]))
        }else if(isTRUE(loss.f=="l1")){
          L <- c(L, sum((y-FeedForwardNN(X=X,para=para,class=class,class.score=FALSE))^2)/(dim(X.original)[1]))
        }else if(isTRUE(loss.f=="huber")){
          L <- c(L, sum((y-FeedForwardNN(X=X,para=para,class=class,class.score=FALSE))^2)/(dim(X.original)[1]))
        }
      }
      
      if(isTRUE(L[i]=="NaN")|isTRUE(L[i]==Inf)|isTRUE(L[i]==-Inf)|isTRUE(is.na(L[i]))){
        para <- paraNA
        break
      }
      
      if(isTRUE(is.na(update.par1[i]))){
        para <- paraNA
        break
      }
      
      
      if(isTRUE(class)&isTRUE(i%%epoch==0)){
        print(paste("Epoch", (i/epoch),': loss.softmax', (sum(A[(i+1-epoch):i]))))
      }else{
        if(isTRUE(loss.f=="ls")&isTRUE(i%%epoch==0)){
          print(paste("Epoch", (i/epoch),': loss.ls', sum(L[(i+1-epoch):i])))
        }else if(isTRUE(loss.f=="l1")&isTRUE(i%%epoch==0)){
          print(paste("Epoch", (i/epoch),': loss.l1', sum(L[(i+1-epoch):i])))
        }else if(isTRUE(loss.f=="huber")&isTRUE(i%%epoch==0)){
          print(paste("Epoch", (i/epoch),paste0(": loss.huber", qs), sum(L[(i+1-epoch):i])))
        }
      }
      
      # Stopping criteria
      
      if(isTRUE(class)){
        if(isTRUE(i>epoch)&isTRUE(i%%epoch==0)){
          if(isTRUE(sum(L[(i+1-epoch):i])<=(-log(0.975)))){
            break
          }
        }
      }else{
        if(isTRUE(i>epoch)&isTRUE(i%%epoch==0)){
          if(isTRUE(sum(L[(i+1-epoch):i])<=(1))){
            break
          }
        }
      }
    }else{
      # Monitor loss
    }
    
    # MOM
    if(isTRUE(MOM==TRUE)){
      mom.group <- group(k,dim(X)[1])
      
      # Sample size
      if(isTRUE(random==TRUE)){
        N <- batch
      }else{
        N <- num.obs
      }
      
      # Select median group first time 
      if(isTRUE(class)){
        median.group <- mom(y=y,X=X,para=para,para2=para2,B=mom.group,class=class,class.score=TRUE)
      }else{
        median.group <- mom(y=y,X=X,para=para,para2=para2,B=mom.group,class=class,class.score=FALSE)
      }
      
      if(isTRUE(is.na(median.group[[2]]))){
        para <- paraNA
        break
      }
      
      index <- median.group[[1]]
      X <- X.random[index,]
      
      if(isTRUE(class)){
        y <- y.random[index,]
      }else{
        y <- y.random[index]
      }
      
      if(isTRUE(class)){
        classIndex <- classIndex[index]
      }
      
      # Monitor loss
      
      if(isTRUE(class)){
        score1 <- FeedForwardNN(X=X,para=para,class=class,class.score=TRUE)
        pre.class <- FeedForwardNN(X=X,para=para,class=class,class.score=FALSE)
        accuracy <- sum(pre.class==(classIndex))
        exp_scores <- exp(score1)
        probs <- exp_scores / rowSums(exp_scores)
        corect_logprobs <- -log(probs)
        data_loss <- sum(corect_logprobs*y)
        L.mom <- c(L.mom, data_loss)
        A.mom <- c(A.mom, accuracy)
        IC.mom <- c(IC.mom, length(index))
      }else{
        L.mom1 <- sum((y-FeedForwardNN(X=X,para=para,class=class,class.score=FALSE))^2)
        L.mom <- c(L.mom, L.mom1)
      }
      
      if(isTRUE(L.mom[i]=="NaN")|isTRUE(L.mom[i]==Inf)|isTRUE(L.mom[i]==-Inf)|isTRUE(is.na(L.mom[i]))){
        para <- paraNA
        break
      }
      
      if(isTRUE(is.na(update1.par1[i]))){
        para <- paraNA
        break
      }
      
      if(isTRUE(class)&isTRUE(i%%epoch==0)){
        print(paste("Epoch", (i/epoch), paste0(": loss.mom", k), (sum(A.mom[(i+1-epoch):i])/sum(IC.mom[(i+1-epoch):i]))))
      }else if(isTRUE(i%%epoch==0)){
        print(paste("Epoch", (i/epoch), paste0(": loss.mom", k), (sum(L.mom[(i+1-epoch):i]))))
      }
      
      # First bias parameters 
      for(j in 1:length(para)){
        B.trim <- as.vector(para[[j]][[2]][1,])
        para[[j]][[2]] <- matrix(rep(B.trim,length(index)),nrow=length(index),byrow=TRUE)
      }
      
      # Update first parameters
      back <- BackPropNN(y=y,X=X,para=para, alpha=alpha,loss.f=loss.f,q=q,bias=bias,class=class)
      para <- back[[1]]
      update1.par1 <- c(update1.par1,sum(back[[2]]))
      
      # Reverse first parameters
      for(j in 1:length(para)){
        B.trim <- as.vector(para[[j]][[2]][1,])
        para[[j]][[2]] <- matrix(rep(B.trim,N),nrow=N,byrow=TRUE)
      }
      
      X <- X.random
      y <- y.random
      
      # Select median group second time 
      if(isTRUE(class)){
        median.group <- mom(y=y,X=X,para=para,para2=para2,B=mom.group,class=class,class.score=TRUE)
      }else{
        median.group <- mom(y=y,X=X,para=para,para2=para2,B=mom.group,class=class,class.score=FALSE)
      }
      
      if(isTRUE(is.na(median.group[[2]]))){
        para <- paraNA
        break
      }
      
      index <- median.group[[1]]
      X <- X.random[index,]
      
      if(isTRUE(class)){
        y <- y.random[index,]
      }else{
        y <- y.random[index]
      }
      
      #Monitor loss
      
      if(isTRUE(class)){
        score2 <- FeedForwardNN(X=X,para=para2,class=class,class.score=TRUE)
        exp_scores <- exp(score2)
        probs <- exp_scores / rowSums(exp_scores)
        corect_logprobs <- -log(probs)
        data_loss <- sum(corect_logprobs*y)
        L2.mom <- c(L2.mom, data_loss)
      }else{
        L2.mom2 <- sum((y-FeedForwardNN(X=X,para=para2,class=class,class.score=FALSE))^2)
        L2.mom <- c(L2.mom,(L2.mom2))
      }
      
      
      if(isTRUE(L2.mom[i]=="NaN")|isTRUE(L2.mom[i]==Inf)|isTRUE(L2.mom[i]==-Inf)|isTRUE(is.na(L2.mom[i]))){
        para <- paraNA
        break
      }
      
      if(isTRUE(is.na(update2.par1[i]))){
        para <- paraNA
        break
      }
      
      # Second bias parameters
      for(j in 1:length(para2)){
        B.trim <- as.vector(para2[[j]][[2]][1,])
        para2[[j]][[2]] <- matrix(rep(B.trim,length(index)),nrow=length(index),byrow=TRUE)
      }
      
      # Update second parameters
      back2 <- BackPropNN(y=y,X=X,para=para2, alpha=alpha,loss.f=loss.f,q=q,bias=bias,class=class)
      para2 <- back2[[1]]
      update2.par1 <- c(update2.par1,sum(back2[[2]]))
      
      # Reverse second parameters
      for(j in 1:length(para)){
        B2.trim <- para[[j]][[2]][1,]
        para[[j]][[2]] <- matrix(rep(B2.trim,num.obs),nrow=num.obs,byrow=TRUE)
      }
      
      for(j in 1:length(para2)){
        B2.trim <- para2[[j]][[2]][1,]
        para2[[j]][[2]] <- matrix(rep(B2.trim,num.obs),nrow=num.obs,byrow=TRUE)
      }
      
      if(isTRUE(class)){
        if(isTRUE(i>epoch)&isTRUE(i%%epoch==0)){
          if(isTRUE((sum(L.mom[(i+1-epoch):i])/sum(IC.mom[(i+1-epoch):i]))<=(-log(0.975)))){
            break
          }
        }
        
        if(isTRUE(i>epoch)&isTRUE(i%%epoch==0)){
          if(isTRUE((sum(L2.mom[(i+1-epoch):i])/sum(IC.mom[(i+1-epoch):i]))<=(-log(0.975)))){
            break
          }
        }
      }else{
        if(isTRUE(i>epoch)&isTRUE(i%%epoch==0)){
          if(isTRUE(sum(L.mom[(i+1-epoch):i])<=(1))){
            break
          }
        }
        
        if(isTRUE(i>epoch)&isTRUE(i%%epoch==0)){
          if(isTRUE(sum(L2.mom[(i+1-epoch):i])<=(1))){
            break
          }
        }
      }
    }else{
      # Back propagation
      back <- BackPropNN(y=y,X=X,para=para,alpha=alpha,loss.f=loss.f,q=q,bias=bias,class=class)
      para <- back[[1]]
      update.par1 <- c(update.par1,sum(back[[2]]))
    
      # Reverse second parameters
      for(j in 1:length(para)){
        B <- as.vector(para[[j]][[2]][1,])
        para[[j]][[2]] <- matrix(rep(B,num.obs),nrow=num.obs,byrow=TRUE)
      }
    }
  }
  
  output <- list()
  output[[1]] <- para
  if(isTRUE(MOM==TRUE)){
    output[[2]] <- L.mom
  }else{
    output[[2]] <- L
  }
  
  return(output)
}



