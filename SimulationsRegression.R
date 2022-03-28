########################
### Simulation Study ###
########################

# Settings

## A positive integer indicates the number of samples.
num.obs <- 800 

## A positive integer indicates the dimension of input vector. 
num.par <- 50 

## A positive integer indicates the number of classes.
K <- 1 

## A positive integer indicates the number of layers for the neural network.
l <- 10 

## A positive integer indicates the number of neurons in each hidden layer of 
## the neural network.
num.neurons <- 20 

## A positive value within [0,1] indicates the propostion of informative 
## samples.
prop <- 1 

## A positive value within [0,1] indicates the batch size for stochastic 
## gradient descent algorithm.
BatchSize <- 1

## A Boolean value indicates whether to corrupt the input vectors. 
NoiseX <- FALSE 

## A Boolean value indicates whether to generate noise from t-distribution.
NoiseT <- FALSE 

## A positive integer indicates the degree of freedom of t-distribution
DF <- NULL 

## A positive integer indicates the number of simulations.
num.sim <- 1

## A positive integer indicates the number of testing samples.
num.test <- num.obs 

## A positive value to scale the initial values for updating the gradients of 
## the neural network.
scale <- 1 

## A positive integer indicates the number of epoch to update the gradient of 
## the neural network.
E <- 2000 

## A positive value indicates the learning rate for the stochastic gradient 
## descent algorithm with softmax loss. 
alphaSoftmax <- 1e-2

## A positive value indicates the learning rate for the stochastic gradient 
## descent algorithm with DeepMoM structure. 
alphaMoM <- 1e-2

## A positive value indicates the learning rate for the stochastic gradient 
## descent algorithm with softmax loss. 
alphaL1 <- 1e-2

## A positive value indicates the learning rate for the stochastic gradient 
## descent algorithm with softmax loss. 
alphaHuber <- 1e-2

## A vector of positive integers indicates the number of blocks for DeepMoM.
Blocks <- c(21, 41, 61, 81, 101, 121)

# Loading required functions

source("./AdditionalFunctions/FeedForwardNN.R")
source("./AdditionalFunctions/BackPropNN.R")
source("./AdditionalFunctions/TrainNN.R")
source("./AdditionalFunctions/GroupK.R")
source("./AdditionalFunctions/Mom.R")
source("./AdditionalFunctions/HuberLoss.R")
source("./AdditionalFunctions/HuberDerivative.R")
source("./AdditionalFunctions/L1Derivative.R")

set.seed(2022314) # Set the random seed for reproducible research

# Generate simulated data

data_all <- list()
for(s in 1:num.sim){
  X <- matrix(rnorm(num.obs*num.par, 0, 10), nrow=num.obs)
  X.test <- matrix(rnorm(num.test*num.par, 0, 10), nrow=num.test)
  
  P <- rep(num.neurons, l)
  P <- c(num.par, P, K)
  
  #Para <- list()
  
  #for(i in 1:(l+1)){
  #  Para[[i]] <- list()
  #  Para[[i]][[1]] <- matrix(runif(P[i]*P[i+1], -1, 1), nrow=P[i])
  #}
  
  #for(i in 1:(l+1)){
  #  Para[[i]][[2]] <- rep(runif(1, -1, 1), P[(i+1)])
  #  Para[[i]][[2]] <- matrix(rep(Para[[i]][[2]], num.obs), nrow=num.obs, 
  #                           byrow=TRUE)
  #}
  
  #signal <- FeedForwardNN(X, para=Para, class=FALSE, class.score=FALSE) 
  #signal.test <- FeedForwardNN(X.test, para=Para, class=FALSE, 
  #                             class.score=FALSE)
  
  beta <- rnorm(num.par, 0, 10)
  signal <- 20*sin(0.001*X %*% beta)
  signal.test <- 20*sin(0.001*X.test %*% beta)
  
  Data <- list()
  Data[[1]] <- X
  Data[[2]] <- X.test
  Data[[3]] <- signal
  Data[[4]] <- signal.test
  Data[[5]] <- beta
  
  data_all[[s]] <- Data 
}

error.ls <- list()
error.mom <- list()
error.l1 <- list()
error.huber <- list()

error.ls[[1]] <- list()
error.mom[[1]] <- list()
error.l1[[1]] <- list()
error.huber[[1]] <- list()

error.ls[[2]] <- list()
error.mom[[2]] <- list()
error.l1[[2]] <- list()
error.huber[[2]] <- list()

error.ls[[3]] <- list()
error.mom[[3]] <- list()
error.l1[[3]] <- list()
error.huber[[3]] <- list()

for(s in 1:num.sim){
  D <- data_all[[s]]
  #S <- c(D[[3]],D[[4]])
  
  signal <- D[[3]]
  signal.test <- D[[4]]
  beta <- D[[5]]
  
  X <- D[[1]]
  X.test <- D[[2]]
  
  y <- rep(NA,num.obs)
  
  if (isTRUE(NoiseT)){
    y <- signal+rt(num.obs, DF, ncp=0)
  }else if(isTRUE(NoiseX)){
    noise <- sample(c(1:num.obs), floor((1-prop)*num.obs), replace = FALSE)
    X[noise, ] <- matrix(rnorm(num.par*length(noise), 5, 1), 
                         nrow=length(noise), ncol=num.par)
    y <- signal + rnorm(dim(X)[1], 0, 1)
  }else{
    good <- sample(c(1:dim(X)[1]), floor(dim(X)[1]*prop), replace=FALSE)
    u.good <- rnorm(length(good), 0, 1)
    u.bad <- runif((dim(X)[1]-length(good)), (3*max(abs(signal))), 
                   (5*max(abs(signal))))
    y[good] <- signal[good] + u.good
    y[-good] <- signal[-good] + u.bad
  }
  
  b <- floor(BatchSize*num.obs)
  num.i <- floor(num.obs/b)*E
  set.seed(NULL)
  
  plot(X%*%beta, y)
  
  # Training models
  
  # Ls  
  
  alpha <- alphaSoftmax
  repeat{
    Pre_Para.ls <- TrainNN(y, X, P=P, para=NULL, alpha=alpha, iteration=num.i, random=FALSE, 
                           batch=b, MOM=FALSE, k=1, loss.f="ls", q=NULL, 
                           bias=TRUE, class=FALSE, beta=scale)
    train.test <- sum((signal - FeedForwardNN(X, para=Pre_Para.ls[[1]], 
                                              class=FALSE, 
                                              class.score=FALSE))^2)/num.obs
    
    if(isTRUE(train.test=="NaN")|isTRUE(is.na(train.test))|
       isTRUE(train.test==Inf)|isTRUE(train.test==-Inf)){
      alpha <- alpha/10
      next
    }else{
      break
    }
  }
  
  Prediction.ls <- sum((signal.test-FeedForwardNN(X.test, para=Pre_Para.ls[[1]], 
                                                  class=FALSE, 
                                                  class.score=FALSE))^2)/num.obs
  alpha.ls <- alpha
  LossTracking.ls <- Pre_Para.ls[[2]]
  
  error.ls[[1]][[s]] <- Prediction.ls
  error.ls[[2]][[s]] <-  LossTracking.ls 
  error.ls[[3]][[s]] <- alpha.ls
  
  #save.image(paste0(paste0(num.obs, (prop*100)), "LS.RData"))
  #save.image(paste0(paste0(num.obs, (prop*100)), "XLS.RData"))
  #save.image(paste0(paste0(num.obs, DF), "TLS.RData"))
  
  # MOM 
  
  Prediction.mom <- c()
  alpha.mom <- c() 
  LossTracking.mom <- list()
  for(i in 1:length(Blocks)){
    alpha <- alphaMoM
    repeat{
      Pre_Para.mom <- TrainNN(y, X, P=P, alpha=alpha, iteration=num.i, 
                              random=FALSE, batch=b, MOM=TRUE, k=Blocks[i], 
                              loss.f="ls", q=NULL, bias=TRUE, class=FALSE, 
                              beta=scale, para=NULL)
      
      train.test <- sum((signal-FeedForwardNN(X, para=Pre_Para.mom[[1]], 
                                              class=FALSE, 
                                              class.score=FALSE))^2)/num.obs
      
      if(isTRUE(train.test=="NaN")|isTRUE(is.na(train.test))|
         isTRUE(train.test==Inf)|isTRUE(train.test==-Inf)|
         isTRUE(is.na(train.test))){
        alpha <- alpha/10
        next
      }else{
        break
      }
    }
    
    Prediction <- sum((signal.test-FeedForwardNN(X.test, para=Pre_Para.mom[[1]], 
                                                 class=FALSE, 
                                                 class.score=FALSE))^2)/num.obs
    LossTracking.mom[[i]] <- Pre_Para.mom[[2]]
    Prediction.mom <- c(Prediction.mom, Prediction)
    alpha.mom <- c(alpha.mom, alpha)
    
    error.mom[[2]][[s]] <- list()
    error.mom[[2]][[s]][[i]] <-  LossTracking.mom[[i]]
    
    #save.image(paste0(paste0(paste0(num.obs, (prop*100)), Blocks[i]), "MOM.RData"))
    #save.image(paste0(paste0(paste0(num.obs, (prop*100)), Blocks[i]), "XMOM.RData"))
    #save.image(paste0(paste0(paste0(num.obs, DF), Blocks[i]), "TMOM.RData"))
  }
  
  error.mom[[1]][[s]] <- Prediction.mom
  error.mom[[3]][[s]] <- alpha.mom
  
  
  # Huber 
  
  Q <- seq(0.75, 1.00, by=0.05)
  Prediction.huber <- c()
  alpha.huber <- c()
  LossTracking.huber <- list()
  for(i in 1:length(Q)){
    alpha <- alphaHuber
    qt <- as.numeric(quantile(abs(y), prob=Q[i]))
    repeat{
      Pre_Para.huber <- TrainNN(y, X, P=P, alpha=alpha, iteration=num.i, 
                                random=FALSE, batch=b, MOM=FALSE, k=NULL, 
                                loss.f="huber", q=qt, bias=TRUE, class=FALSE, 
                                beta=scale, para=NULL)
      train.test <- sum((signal-FeedForwardNN(X, para=Pre_Para.huber[[1]], 
                                              class=FALSE, 
                                              class.score=FALSE))^2)/num.obs
      
      if(isTRUE(train.test=="NaN")|isTRUE(is.na(train.test))|
         isTRUE(train.test==Inf)|isTRUE(train.test==-Inf)){
        alpha <- alpha/10
        next
      }else{
        break
      }
    }
    
    Prediction <- sum((signal.test-FeedForwardNN(X.test, 
                                                 para=Pre_Para.huber[[1]], 
                                                 class=FALSE, 
                                                 class.score=FALSE))^2)/num.obs
    LossTracking.huber[[i]] <- Pre_Para.huber[[2]]
    Prediction.huber <- c(Prediction.huber, Prediction)
    alpha.huber <- c(alpha.huber, alpha)
    
    error.huber[[2]][[s]] <- list()
    error.huber[[2]][[s]][[i]] <-  LossTracking.huber[[i]]
    
    #save.image(paste0(paste0(paste0(num.obs, (prop*100)), (Q[i]*100)), "HUBER.RData"))
    #save.image(paste0(paste0(paste0(num.obs, (prop*100)), (Q[i]*100)), "XHUBER.RData"))
    #save.image(paste0(paste0(paste0(num.obs, DF), (Q[i]*100)), "THUBER.RData"))
  }
  
  error.huber[[1]][[s]] <- Prediction.huber
  error.huber[[3]][[s]] <- alpha.huber
  
  # L1 
  
  alpha <- alphaL1
  repeat{
    Pre_Para.l1 <- TrainNN(y, X, P=P, alpha=alpha, iteration=num.i, random=FALSE, 
                           batch=b, MOM=FALSE, k=NULL, loss.f="l1", q=NULL, 
                           bias=TRUE, class=FALSE, beta=scale, para=NULL)
    train.test <- sum((signal-FeedForwardNN(X,para=Pre_Para.l1[[1]], 
                                            class=FALSE, 
                                            class.score=FALSE))^2)/num.obs
    
    if(isTRUE(train.test=="NaN")|isTRUE(is.na(train.test))|
       isTRUE(train.test==Inf)|isTRUE(train.test==-Inf)){
      alpha <- alpha/10
      next
    }else{
      break
    }
  }
  
  Prediction.l1 <- sum((signal.test-FeedForwardNN(X.test, para=Pre_Para.l1[[1]], 
                                                  class=FALSE, 
                                                  class.score=FALSE))^2)/num.obs
  alpha.l1 <- alpha
  LossTracking.l1 <- Pre_Para.l1[[2]]
  
  error.l1[[1]][[s]] <- Prediction.l1
  error.l1[[2]][[s]] <- LossTracking.l1  
  error.l1[[3]][[s]] <- alpha.l1
  
  #save.image(paste0(paste0(num.obs, (prop*100)), "L1.RData"))
  #save.image(paste0(paste0(num.obs, (prop*100)), "XL1.RData"))
  #save.image(paste0(paste0(num.obs, DF), "TL1.RData"))
  
}

E.mom <- c()
for(s in 1:num.sim){
  E.mom <- c(E.mom,min(error.mom[[1]][[s]]))
}

E.ls <- c()
for(s in 1:num.sim){
  E.ls <- c(E.ls,error.ls[[1]][[s]])
}

E.l1 <- c()
for(s in 1:num.sim){
  E.l1 <- c(E.l1,error.l1[[1]][[s]])
}

E.huber <- c()
for(s in 1:num.sim){
  E.huber <- c(E.huber,min(error.huber[[1]][[s]]))
}

# Predicted Results

mean(E.ls)/mean(E.mom)
mean(E.mom)/mean(E.mom)
mean(E.l1)/mean(E.mom)
mean(E.huber)/mean(E.mom)


#save.image("800100.RData")


