###############################
#### Tcga Data Application ####
###############################

# Setting

## A positive integer K for applying K-fold cross validation to select the 
## number of blocks in the DeepMoM.
K <- 10 

## A positive integer indicates the number of neurons in each hidden layer of 
## the neural network.
num.neurons <- 150

## A positive integer indicates the number of layers for the neural network.
l <- 2

## A positive integer indicates the number of epoch to update the gradient of 
## the neural network.
E <- 20000

## A positive value between (0,1) indicates the batch size for stochastic 
## gradient descent algorithm.
BatchSize <- 0.15

## A positive value indicates the learning rate for the stochastic gradient 
## descent algorithm with softmax loss. 
alphaSoftmax <- 1e-2

## A positive value indicates the learning rate for the stochastic gradient 
## descent algorithm with DeepMoM structure. 
alphaMoM <- 1e-2

## A vector of positive integers indicates the number of blocks for DeepMoM.
Blocks <- c(3, 5, 7, 9, 11)

## A positive value to scale the initial values for updating the gradients of 
## the neural network.
scale <- 1

## A positive value to set the random seed for shuffling the data for 
## reproducible research. 
seed1 <- 202101

## A positive value to set the random seed for shuffling the data for 
## reproducible research. 
seed2 <- 202102

# Loading required functions
source("./AdditionalFunctions/FeedForwardNN.R")
source("./AdditionalFunctions/BackPropNN.R")
source("./AdditionalFunctions/TrainNN.R")
source("./AdditionalFunctions/GroupK.R")
source("./AdditionalFunctions/Mom.R")
source("./AdditionalFunctions/HuberLoss.R")
source("./AdditionalFunctions/HuberDerivative.R")
source("./AdditionalFunctions/L1Derivative.R")

# Loading Tcga Data
load("./TcgaData/TcgaData.RData")

# Initialization

A.soft <- rep(0, K)
A.mom <- rep(0, K)

X_ov <- as.matrix(TCGA.data$OV$RC)
y_ov <- rep(1, dim(X_ov)[1])
data_ov <- cbind(y_ov, X_ov)

X_sarc <- as.matrix(TCGA.data$SARC$RC)
y_sarc <- rep(2, dim(X_sarc)[1])
data_sarc <- cbind(y_sarc, X_sarc)

X_kirc <- as.matrix(TCGA.data$KIRC$RC)
y_kirc <- rep(3, dim(X_kirc)[1])
data_kirc <- cbind(y_kirc, X_kirc)

X_luad <- as.matrix(TCGA.data$LUAD$RC)
y_luad <- rep(4, dim(X_luad)[1])
data_luad <- cbind(y_luad, X_luad)

X_skcm <- as.matrix(TCGA.data$SKCM$RC)
y_skcm <- rep(5, dim(X_skcm)[1])
data_skcm <- cbind(y_skcm, X_skcm)

X_esca <- as.matrix(TCGA.data$ESCA$RC)
y_esca <- rep(6, dim(X_esca)[1])
data_esca <- cbind(y_esca, X_esca)

X_laml <- as.matrix(TCGA.data$LAML$RC)
y_laml <- rep(7, dim(X_laml)[1])
data_laml <- cbind(y_laml, X_laml)

data <- rbind(data_ov,
              data_sarc,
              data_kirc,
              data_luad,
              data_skcm,
              data_esca,
              data_laml)

set.seed(seed1)
data <- data[sample(c(1:dim(data)[1]), dim(data)[1], replace=FALSE),]
set.seed(seed2)

set.seed(202102)
GG <- group(K, dim(data)[1])
set.seed(NULL)

for (sim in 1:K){
  X <- as.matrix(data[,-1])
  
  for (i in 1:dim(X)[2]){
    s <- sum(abs(X[,i]))
    if(isTRUE(s!=0)){
      X[,i] <- as.vector(X[,i])/s*1e+6
    }
  }
  
  X <- X/max(abs(X))
  y <- as.vector(data[,1])
  
  num.obs <- dim(X)[1]
  
  IndexT <- GG[[sim]]
  
  X.original <- X[-IndexT,]
  y.original <- y[-IndexT]
  
  X.test <- X[IndexT,]
  y.test <- y[IndexT]
  
  ################################################################################
  C <- y.original
  CT <- y.test
  
  X <- X.original
  
  K <- length(unique(y))
  
  Y <- matrix(rep(0, dim(X)[1]*K), ncol=K)
  
  for (i in 1:(dim(X)[1])){
    Y[i, C[i]] <- 1
  }
  
  num.obs <- dim(X)[1]
  num.par <- dim(X)[2]
  
  b <- floor(num.obs*BatchSize)
  
  num.i <- floor(num.obs/b)*E
  
  P <- rep(num.neurons,l)
  P <- c(num.par,P,K)
  
  par(mfrow=c(1,1))
  ################################################################################
  set.seed(NULL)
  # Softmax
  alpha <- alphaSoftmax
  repeat{
    Pre_Para.softmax <- TrainNN(y=Y,X,P=P,alpha=alpha,iteration=num.i,random=TRUE,batch=b,MOM=FALSE,k=3,loss.f="ls",q=NULL,bias=TRUE,class=TRUE,beta=scale,qs=NULL)
    train.test <- FeedForwardNN(X,para=Pre_Para.softmax[[1]],class=TRUE,class.score=TRUE)
    
    if(any(is.na(train.test))|isTRUE(train.test=="NaN")|isTRUE(is.na(train.test))|isTRUE(train.test==Inf)|isTRUE(train.test==-Inf)|isTRUE(identical(train.test,integer(0)))){
      alpha <- alpha/2
      next
    }else{
      break
    }
  }
  
  Prediction.class <- FeedForwardNN(X.test,para=Pre_Para.softmax[[1]],class=TRUE,class.score=FALSE)
  alpha.softmax <- alpha
  LossTracking.ls <- Pre_Para.softmax[[2]]
  accuracy.softmax <- mean(Prediction.class==(CT))
  
  # MOM 
  b <- floor(num.obs*BatchSize)
  num.i <- floor(num.obs/b)*E
  
  Prediction.mom <- c()
  alpha.mom <- c() 
  LossTracking.mom <- list()
  for(i in 1:length(Blocks)){
    alpha <- alphaMoM
    repeat{
      Pre_Para.mom <- TrainNN(y=Y,X=X,P=P,alpha=alpha,iteration=num.i,random=TRUE,batch=b,MOM=TRUE,k=Blocks[i],loss.f="ls",q=NULL,bias=TRUE,class=TRUE,beta=0.1*scale,qs=NULL)
      train.test <- FeedForwardNN(X,para=Pre_Para.mom[[1]],class=TRUE,class.score=TRUE)
      
      if(any(is.na(train.test))|isTRUE(train.test=="NaN")|isTRUE(is.na(train.test))|isTRUE(train.test==Inf)|isTRUE(train.test==-Inf)|isTRUE(identical(train.test,integer(0)))){
        alpha <- alpha/2
        next
      }else{
        break
      }
    }
    
    Prediction.class.mom <- FeedForwardNN(X.test,para=Pre_Para.mom[[1]],class=TRUE,class.score=FALSE)
    LossTracking.mom[[i]] <- Pre_Para.mom[[2]]
    accuracy.mom <- mean(Prediction.class.mom==(CT))
    Prediction.mom <- c(Prediction.mom,accuracy.mom)
    alpha.mom <- c(alpha.mom,alpha)
  }
  
  A.soft[sim] <- accuracy.softmax
  A.mom[sim] <- max(Prediction.mom)
}

mean(A.soft)
mean(A.mom)
