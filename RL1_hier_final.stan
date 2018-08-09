  // testing: learning rates.
// implemented options: 2 learning rates; 2 beta based on outcome; SU-DU. 2=SU, 1=DU.
// does it work or get confused if i say alpha[nsub,1]?

data {
  int<lower=1> ntr;
  int nsub;
  int resp[ntr,nsub];
  int outc[ntr,nsub];
  int a;
  int b;
//  int up[updating];
}


// The 'parameters' block defines the parameter that we want to fit
parameters {
// Group
  real<lower=0,upper=1> alpha_mu[a]
  real<lower=0> alpha_sd;
  real<lower=0> beta_mu[b];
  real<lower=0> beta_sd;

  // Indiv Sub
  real<lower=0,upper=1> alphaT[nsub,a];
  real<lower=0> betaT[nsub,b];
}

transformed parameters{
  real alpha[nsub,a];
  real beta[nsub,b];
for (is in 1:nsub){
  alpha[is,:] = alpha_mu + alphaT[is,:]*alpha_sd;
  beta[is,:] = beta_mu + betaT[is,:]*beta_sd;
}
}

// This block runs the actual model
model {
  real da [ntr,nsub];
  real preds[(ntr+1),nsub,2]; //where up is either 1(DU) or 2(SU)
  real predsTr[(ntr+1),nsub,2];

  // Priors
    alpha_mu  ~ normal(0,2);
    alpha_sd ~ normal(0,2);
    beta_mu  ~ normal(0,2);
    beta_sd  ~ normal(0,2);

    alphaT[:] ~ normal(0,1);
    betaT[:]  ~ normal(0,1);
// test whether we can instead do alphaT ~ normal(0,1);

  for (is in 1:nsub){
    // Learning
    preds[1,is,1] = 0.5;
    preds[1,is,2] = 0.5;

    for (it in 1:ntr){
      if (outc[it,is]!=999){
      da[it,is] = outc[it,is]-preds[it,is,(resp[it,is]+1)];
      preds[it+1,is,(resp[it,is]+1)] = preds[it,is,(resp[it,is]+1)] + alpha[is,(resp[it,is]+1)]*da[it,is];
      preds[it+1,is,(2-resp[it,is])] = preds[it,is,(2-resp[it,is])];
      }else{
      da[it,is] = 999;
      preds[it+1,is,1]= preds[it,is,1];
      preds[it+1,is,2]= preds[it,is,2];
    }
    }
        // Decision
        for (it in 1:ntr){
          if (resp[it,is]!=999){
            // Transform prediction
            predsTr[it,is,(resp[it,is]+1)] = fmax(fmin(preds[it,is,(resp[it,is]+1)],1),0);
            predsTr[it,is,(2-resp[it,is])] = fmax(fmin(preds[it,is,(2-resp[it,is])],1),0);
              // Compare to choice
             resp[it,is] ~ bernoulli_logit(beta[is,(resp[it,is]+1)]*(predsTr[it,is,1]-predsTr[it,is,2])); // how to modify beta here - based on response?
            }
          }
        }
      }
