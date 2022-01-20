########################
### Simulation Study ###
########################

#setwd("~/Desktop/TCGA/Code")

# Settings
num.obs <- 1000 #number of observations
num.par <- 50 #number of parameers
K <- 1 #number of classes
l <- 5 #number of layers (number of weight matrices-1)
num.neurons <- 50
min.neurons <- 5
prop <- 0.95
b <- floor(0.15*num.obs)
NoiseX <- FALSE
NoiseT <- FALSE
DF <- NULL
num.sim <- 1
check <- FALSE
num.i <- 1e+6

c.loss.mom <- 1e-100
c.update.mom <- 1e-3

c.loss.ls <- 1e-100
c.update.ls <- 1e-100

c.loss.l1 <- 1e-100
c.update.l1 <- 1e-100

c.loss.huber <- 1e-100
c.update.huber <- 1e-100

num.test <- num.obs
scale <- 1

E <- 20000
num.i <- floor(num.obs/b)*E

# Loading functions
source("FeedForwardNN.R")
source("BackPropN.R")
source("TrainNN.R")
source("Group.R")
source("mom.R")
source("huber_loss.R")
source("huber_derivative.R")
source("l1_derivative.R")

set.seed(2021313)
data_all <- list()

for(s in 1:num.sim){
  # Generating data and parameters
  #input <- matrix(rnorm(num.obs*num.par,0,1),nrow=num.obs)
  #index.train <- sample(c(1:num.obs),floor(num.obs*0.8),replace=FALSE)
  X <- matrix(runif(num.obs*num.par,-1,1),nrow=num.obs)
  X.test <- matrix(runif(num.test*num.par,-1,1),nrow=num.test)
  
  P <- rep(num.neurons,l)
  #P <- floor(seq(num.par,(num.par*0.1),length.out=l))
  #P <- seq(10,10*l,length.out = l)
  #P <- P[l:1]
  P <- c(num.par,P,K)
  
  Para <- list()
  
  for(i in 1:(l+1)){
    Para[[i]] <- list()
    Para[[i]][[1]] <- matrix(runif(P[i]*P[i+1],-1,1),nrow=P[i])
  }
  
  for(i in 1:(l+1)){
    Para[[i]][[2]] <- rep(runif(1,-1,1),P[(i+1)])
    Para[[i]][[2]] <- matrix(rep(Para[[i]][[2]],num.obs),nrow=num.obs,byrow=TRUE)
  }
  
  signal <- FeedForwardNN(X, para=Para,class=FALSE,class.score=FALSE) 
  signal.test <- FeedForwardNN(X.test, para=Para,class=FALSE,class.score=FALSE)
  
  #beta <- rnorm(num.par,0,1)
  #signal <- (X^2)%*%beta
  #signal.test <- (X.test^2)%*%beta
  
  Data <- list()
  Data[[1]] <- X
  Data[[2]] <- X.test
  Data[[3]] <- signal
  Data[[4]] <- signal.test
  
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
  S <- c(D[[3]],D[[4]])
  
  #var(D[[3]])
  #plot(D[[3]])
  
  X <- D[[1]]
  X.test <- D[[2]]
  
  y <- rep(NA,num.obs)
  
  if (isTRUE(NoiseT)){
    y <- signal+rt(num.obs, DF, ncp=0)
  }else if(isTRUE(NoiseX)){
    noise <- sample(c(1:num.obs), floor((1-prop)*num.obs), replace = FALSE)
    X[noise, ] <- X[noise, ] + matrix(rnorm(num.par*length(noise),0,1), nrow=length(noise), ncol=num.par)
    #sx.noise <- FeedForwardNN(X, para=Para,class=FALSE,class.score=FALSE) 
    y <- signal + rnorm(dim(X)[1],0,1)
  }else{
    good <- sample(c(1:dim(X)[1]), floor(dim(X)[1]*prop), replace=FALSE)
    u.good <- rnorm(length(good),0,1)
    u.bad <- runif((dim(X)[1]-length(good)),(3*max(abs(signal))),(5*max(abs(signal))))
    #sign <- sample(c(1,-1),(dim(X)[1]-length(good)),replace=TRUE)
    #u.bad <- u.bad*sign
    
    y[good] <- signal[good]+u.good
    y[-good] <- signal[-good]+u.bad
  }
  
  #plot(sx.noise)
  #plot(y)
  
  set.seed(NULL)
  
  # Training models
  
  # Ls  
  
  alpha <- 1e-2
  repeat{
    Pre_Para.ls <- TrainNN(y,X,P=P,alpha=alpha,iteration=num.i,random=TRUE,batch=b,stop.i.loss=c.loss.ls,stop.i.update=c.update.ls,MOM=FALSE,k=1,loss.f="ls",q=NULL,bias=TRUE,class=FALSE,beta=scale)
    train.test <- sum((signal-FeedForwardNN(X,para=Pre_Para.ls[[1]],class=FALSE,class.score=FALSE))^2)/num.obs
    
    if(isTRUE(train.test=="NaN")|isTRUE(is.na(train.test))|isTRUE(train.test==Inf)|isTRUE(train.test==-Inf)){
      alpha <- alpha/10
      next
    }else{
      break
    }
  }
  
  Prediction.ls <- sum((signal.test-FeedForwardNN(X.test,para=Pre_Para.ls[[1]],class=FALSE,class.score=FALSE))^2)/num.obs
  alpha.ls <- alpha
  LossTracking.ls <- Pre_Para.ls[[2]]
  
  error.ls[[1]][[s]] <- Prediction.ls
  error.ls[[2]][[s]] <-  LossTracking.ls 
  error.ls[[3]][[s]] <- alpha.ls
  
  # MOM 
  Blocks <- c(3,5,7,9,11)
  Prediction.mom <- c()
  alpha.mom <- c() 
  LossTracking.mom <- list()
  for(i in 1:length(Blocks)){
    alpha <- 1e-2
    repeat{
      Pre_Para.mom <- TrainNN(y,X,P=P,alpha=alpha,iteration=num.i,random=TRUE,batch=b,stop.i.loss=c.loss.mom,stop.i.update=c.update.mom,MOM=TRUE,k=Blocks[i],loss.f="ls",q=NULL,bias=TRUE,class=FALSE,beta=scale)
      train.test <- sum((signal-FeedForwardNN(X,para=Pre_Para.mom[[1]],class=FALSE,class.score=FALSE))^2)/num.obs
      
      if(isTRUE(train.test=="NaN")|isTRUE(is.na(train.test))|isTRUE(train.test==Inf)|isTRUE(train.test==-Inf)|isTRUE(is.na(train.test))){
        alpha <- alpha/10
        next
      }else{
        break
      }
    }
    
    Prediction <- sum((signal.test-FeedForwardNN(X.test,para=Pre_Para.mom[[1]],class=FALSE,class.score=FALSE))^2)/num.obs
    LossTracking.mom[[i]] <- Pre_Para.mom[[2]]
    Prediction.mom <- c(Prediction.mom,Prediction)
    alpha.mom <- c(alpha.mom,alpha)
    
    error.mom[[2]][[s]] <- list()
    error.mom[[2]][[s]][[i]] <-  LossTracking.mom[[i]]
  }
  
  error.mom[[1]][[s]] <- Prediction.mom
  error.mom[[3]][[s]] <- alpha.mom
  
  # L1  
  alpha <- 1e-2
  repeat{
    Pre_Para.l1 <- TrainNN(y,X,P=P,alpha=alpha,iteration=num.i,random=TRUE,batch=b,stop.i.loss=c.loss.l1,stop.i.update=c.update.l1,MOM=FALSE,k=NULL,loss.f="l1",q=NULL,bias=TRUE,class=FALSE,beta=scale)
    train.test <- sum((signal-FeedForwardNN(X,para=Pre_Para.l1[[1]],class=FALSE,class.score=FALSE))^2)/num.obs
    
    if(isTRUE(train.test=="NaN")|isTRUE(is.na(train.test))|isTRUE(train.test==Inf)|isTRUE(train.test==-Inf)){
      alpha <- alpha/10
      next
    }else{
      break
    }
  }
  
  Prediction.l1 <- sum((signal.test-FeedForwardNN(X.test,para=Pre_Para.l1[[1]],class=FALSE,class.score=FALSE))^2)/num.obs
  alpha.l1 <- alpha
  LossTracking.l1 <- Pre_Para.l1[[2]]
  
  error.l1[[1]][[s]] <- Prediction.l1
  error.l1[[2]][[s]] <- LossTracking.l1  
  error.l1[[3]][[s]] <- alpha.l1
  
  # Huber 
  Q <- seq(0.75, 1.00, by=0.05)
  Prediction.huber <- c()
  alpha.huber <- c()
  LossTracking.huber <- list()
  for(i in 1:length(Q)){
    alpha <- 1e-2
    qt <- as.numeric(quantile(abs(y), prob=Q[i]))
    repeat{
      Pre_Para.huber <- TrainNN(y,X,P=P,alpha=alpha,iteration=num.i,random=TRUE,batch=b,stop.i.loss=c.loss.huber,stop.i.update=c.update.huber,MOM=FALSE,k=NULL,loss.f="huber",q=qt,bias=TRUE,class=FALSE,beta=scale)
      train.test <- sum((signal-FeedForwardNN(X,para=Pre_Para.huber[[1]],class=FALSE,class.score=FALSE))^2)/num.obs
      
      if(isTRUE(train.test=="NaN")|isTRUE(is.na(train.test))|isTRUE(train.test==Inf)|isTRUE(train.test==-Inf)){
        alpha <- alpha/10
        next
      }else{
        break
      }
    }
    
    Prediction <- sum((signal.test-FeedForwardNN(X.test,para=Pre_Para.huber[[1]],class=FALSE,class.score=FALSE))^2)/num.obs
    LossTracking.huber[[i]] <- Pre_Para.huber[[2]]
    Prediction.huber <- c(Prediction.huber,Prediction)
    alpha.huber <- c(alpha.huber,alpha)
    
    
    error.huber[[2]][[s]] <- list()
    error.huber[[2]][[s]][[i]] <-  LossTracking.huber[[i]]
  }
  
  error.huber[[1]][[s]] <- Prediction.huber
  error.huber[[3]][[s]] <- alpha.huber
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


if(isTRUE(check)){
  y=y
  X=X
  P=P
  alpha=1e-5
  iteration=10000
  random=TRUE
  batch=b
  stop.i.loss=c.loss.mom
  stop.i.update=c.update.mom
  MOM=TRUE
  k=3
  loss.f="ls"
  q=NULL
  bias=TRUE
  class=FALSE
  beta=scale
}

#compTime.LS
#compTime.MOM
#compTime.L1
#compTime.HUBER

#100%
20.5703728199005/1000*242399/60/60+
1732.64452028275/1000*(31+595+936+813+942+1106+38)/60/60+
20.4389324188232/1000*43005/60/60+
123.85004401207/1000*(2498+2982+3222+1957+2476+4406)/60/60

#95%
19.8364479541779/1000*115345/60/60+
1663.45031070709/1000*(301+598+109+306+159+91+1142)/60/60+
19.5515949726105/1000*44001/60/60+
120.698728561401/1000*(1278+2218+1489+1291+3004+1960)/60/60

#85%
20.226419210434/1000*35409/60/60+
1660.21042132378/1000*(558+221+1055+829+412+279+261)/60/60+
20.0333120822906/1000*44002/60/60+
123.586020708084/1000*(1278+2218+1489+1291+4360+6013)/60/60

#75%
20.5138430595398/1000*116209/60/60+
1719.66549563408/1000*(463+496+31+188+7+239+127)/60/60+
20.0420942306519/1000*44023/60/60+
123.986365556717/1000*(4008+4217+4250+4671+4685+5360)/60/60


#T1
20.38174700737/1000*268252/60/60+
1649.89619421959/1000*(640+749+280+568+715+1838+88)/60/60+
19.6864948272705/1000*43049/60/60+
120.105998516083/1000*(2437+2614+2355+2319+4444+2529)/60/60

#T10
20.0091071128845/1000*211152/60/60+
1648.99000072479/1000*(868+25+241+447+554+639+1118)/60/60+
19.5056688785553/1000*43034/60/60+
119.920931816101/1000*(2097+2086+1829+2269+2553+3109)/60/60



#X95%
19.847348690033/1000*245791/60/60+
1732.60032343864/1000*(548+803+310+1131+511+313+843)/60/60+
20.1141107082367/1000*43024/60/60+
123.310090780258/1000*(1278+2218+1489+1291+3004+7698)/60/60


#X85%
21.1621582508087/1000*220908/60/60+
1726.15628886223/1000*(253+1355+476+1426+684+353+1258)/60/60+
19.6037344932556/1000*43028/60/60+
119.865998029709/1000*(1837+3229+2892+1817+2328+2591)/60/60


#X75%
20.0990536212921/1000*192948/60/60+
1745.14250826836/1000*(618+1197+1577+230+495+28+351)/60/60+
20.0847613811493/1000*43022/60/60+
123.036078453064/1000*(1837+3229+2892+1817+6259+8568)/60/60

sd(error.Mom)/sqrt(20)
sd(error.L1)/sqrt(20)
sd(error.Huber)/sqrt(20)
sd(error.Ls)/sqrt(20)




