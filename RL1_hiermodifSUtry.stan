data {
  int<lower=1> ntr;
  int nsub;
  int resp[ntr,nsub];
  int outc[ntr,nsub];
}

// The 'parameters' block defines the parameter that we want to fit
parameters {
// Group
  real<lower=0,upper=1> alpha_mu;
  real<lower=0> alpha_sd;
  real<lower=0> beta_mu;
  real<lower=0> beta_sd;

  // Indiv Sub
  real<lower=0,upper=1> alphaT[nsub];
  real<lower=0> betaT[nsub];
}

transformed parameters{
  real alpha[nsub];
  real beta[nsub];
for (is in 1:nsub){
  alpha[is] = alpha_mu + alphaT[is]*alpha_sd;
  beta[is] = beta_mu + betaT[is]*beta_sd;
}
}

// This block runs the actual model
model {
  real da [ntr,nsub];
  real preds[(ntr+1),nsub,2];
  real predsTr[(ntr+1),nsub,2];

  // Priors
    alpha_mu  ~ normal(0,2);
    alpha_sd ~ normal(0,2);
    beta_mu  ~ normal(0,2);
    beta_sd  ~ normal(0,2);

    alphaT[:] ~ normal(0,1);
    betaT[:]  ~ normal(0,1);

  for (is in 1:nsub){
    // Learning
    preds[1,is,1] = 0.5;
    preds[1,is,2] = 0.5;
    for (it in 1:ntr){
      if (outc[it,is]!=999){
      da[it,is] = outc[it,is]-preds[it,is,(resp[it,is]+1)];
      preds[it+1,is,(resp[it,is]+1)] = preds[it,is,(resp[it,is]+1)] + alpha[is]*da[it,is];
      preds[it+1,is,(2-resp[it,is])] = preds[it,is,(2-resp[it,is])];
      }else{
      preds[it+1,is,1]= preds[it,is,1];
      preds[it+1,is,2]= preds[it,is,2];
      da[it,is] = 999;
    }
    }
        // Decision
        for (it in 1:ntr){
          if (resp[it,is]!=999){
            // Transform prediction
          predsTr[it,is,(resp[it,is]+1)] = fmax(fmin(preds[it,is,(resp[it,is]+1)],1),0);
          predsTr[it,is,(2-resp[it,is])] = fmax(fmin(preds[it,is,(2-resp[it,is])],1),0);
            // Compare to choice
           resp[it,is] ~ bernoulli_logit(beta[is]*(predsTr[it,is,1]-predsTr[it,is,2]));
  // running without predsTr
  //      resp[it,is] ~ bernoulli_logit(beta[is]*(preds[it,is,1]-preds[it,is,2]));
          }
        }
      }
}
