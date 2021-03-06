data {
  int<lower=1> ntr;
  int nsub;
  int resp[ntr,nsub];
  int outc[ntr,nsub];
  int a;
  int ka;
  int b;
  int <lower = 0, upper = 1> run_estimation;
}

// The 'parameters' block defines the parameter that we want to fit
parameters {
  // Group
  real<lower=0,upper=1> alpha_mu[a];
  real<lower=0> alpha_sd;
  real<lower=0> beta_mu[b];
  real<lower=0> beta_sd;
  real<lower=0> kappa_mu;
  real<lower=0> kappa_sd;
  // individual
  real <lower=0, upper=1> alpha [nsub,a];
  real <lower=0> beta [nsub,b];
  real <lower=0, upper=1> kappa [nsub];
}


// This block runs the actual model
model {
real da [ntr,nsub,2]; // can be vectorised for speed when 2 dimensional: row_vector[nsub] da[ntr];
real preds[(ntr+1),nsub,2];

  // Priors
    alpha_mu  ~ normal(0.5,0.5); // changed alpha and kappa priors!
    alpha_sd ~ normal(0.5,0.5);
    beta_mu  ~ normal(0,10);
    beta_sd  ~ normal(0,10);
    kappa_mu  ~ normal(0.5,0.5);
    kappa_sd  ~ normal(0.5,0.5);

  for (is in 1:nsub){
    for (lr in 1:a){
 alpha[is,a] ~ normal(alpha_mu,alpha_sd);
 }
 for (be in 1:b){
     beta[is,b] ~ normal(beta_mu,beta_sd);
 }
 if (ka==2){
  kappa[is] ~ normal(kappa_mu, kappa_sd); // is it okay if kappa doesn't get estimated (where ka != 2?)
 }
}

// run estimation
if (run_estimation==1){
  for (is in 1:nsub){
    // Learning
    preds[1,is,1] = 0.5; // predicted value for stimulus A
    preds[1,is,2] = 0.5; // predicted value for stimulus B

    for (it in 1:ntr){
      if (outc[it,is]!=999){
      da[it,is,1] = outc[it,is]-preds[it,is,(resp[it,is]+1)]; // da[x,y,z] z is 1 or 2, 1=selected stimulus 2=not-selected stimulus
      da[it,is,2] = 1-outc[it,is]-preds[it,is,(2-resp[it,is])];

      if (a==2){
        if (ka==2){
        preds[it+1,is,(resp[it,is]+1)] = preds[it,is,(resp[it,is]+1)] + alpha[is,(outc[it,is]+1)]*da[it,is,1];
        preds[it+1,is,(2-resp[it,is])] = preds[it,is,(2-resp[it,is])]+ kappa[is]*alpha[is,(outc[it,is]+1)]*da[it,is,2];
        }else if (ka!=2){
        preds[it+1,is,(resp[it,is]+1)] = preds[it,is,(resp[it,is]+1)] + alpha[is,(outc[it,is]+1)]*da[it,is,1];
        preds[it+1,is,(2-resp[it,is])] = preds[it,is,(2-resp[it,is])]+ka*alpha[is,(outc[it,is]+1)]*da[it,is,2];
        }
    } else if (a==1){
        if (ka==2){
        preds[it+1,is,(resp[it,is]+1)] = preds[it,is,(resp[it,is]+1)] + alpha[is,1]*da[it,is,1];
        preds[it+1,is,(2-resp[it,is])] = preds[it,is,(2-resp[it,is])]+ kappa[is]*alpha[is,1]*da[it,is,2];
        }else if (ka!=2){
        preds[it+1,is,(resp[it,is]+1)] = preds[it,is,(resp[it,is]+1)] + alpha[is,1]*da[it,is,1];
        preds[it+1,is,(2-resp[it,is])] = preds[it,is,(2-resp[it,is])]+ka*alpha[is,1]*da[it,is,2];
        }
      }
      }else if (outc[it,is]==999){
      da[it,is,1] = 999;
      da[it,is,2] = 999;
      preds[it+1,is,1]= preds[it,is,1];
      preds[it+1,is,2]= preds[it,is,2];
    }
  } // ntr

        // Decision
        for (it in 1:ntr){
          if (resp[it,is]!=999){
                // Compare to choice
    //       resp[it,is] ~ bernoulli_logit(beta[is,1]*(preds[it,is,2]-preds[it,is,1]));
          real prob = exp(beta[is,b]*preds[it,is,2])/(exp(beta[is,b]*preds[it,is,2])+ exp(beta[is,b]*preds[it,is,1])); // softmax
          real prob1 = log(prob/(1-prob));
          resp[it,is] ~ bernoulli_logit(prob1); // target += bernoulli_logit(resp[it,is] | prob1);
          } // resp != 999
          } // ntr
        } // sub
      } // run estimation
} // model



generated quantities {
  real prob;
  real prob1;
  real resp_sim [ntr,nsub] ;
  real part;
  real logp [ntr,nsub];

    // simulating values
      for (is in 1:nsub){
      // Learning
      real da [ntr,nsub,2];
      real preds[(ntr+1),nsub,2];
      preds[1,is,1] = 0.5; // predicted value for stimulus A
      preds[1,is,2] = 0.5; // predicted value for stimulus B

      for (it in 1:ntr){
        if (outc[it,is]!=999){
        da[it,is,1] = outc[it,is]-preds[it,is,(resp[it,is]+1)]; // da[x,y,z] z is 1 or 2, 1=selected stimulus 2=not-selected stimulus
        da[it,is,2] = 1-outc[it,is]-preds[it,is,(2-resp[it,is])];

        if (a==2){
          if (ka==2){
          preds[it+1,is,(resp[it,is]+1)] = preds[it,is,(resp[it,is]+1)] + alpha[is,(outc[it,is]+1)]*da[it,is,1];
          preds[it+1,is,(2-resp[it,is])] = preds[it,is,(2-resp[it,is])]+ kappa[is]*alpha[is,(outc[it,is]+1)]*da[it,is,2];
          }else if (ka!=2){
          preds[it+1,is,(resp[it,is]+1)] = preds[it,is,(resp[it,is]+1)] + alpha[is,(outc[it,is]+1)]*da[it,is,1];
          preds[it+1,is,(2-resp[it,is])] = preds[it,is,(2-resp[it,is])]+ka*alpha[is,(outc[it,is]+1)]*da[it,is,2];
          }
      } else if (a==1){
          if (ka==2){
          preds[it+1,is,(resp[it,is]+1)] = preds[it,is,(resp[it,is]+1)] + alpha[is,1]*da[it,is,1];
          preds[it+1,is,(2-resp[it,is])] = preds[it,is,(2-resp[it,is])]+ kappa[is]*alpha[is,1]*da[it,is,2];
          }else if (ka!=2){
          preds[it+1,is,(resp[it,is]+1)] = preds[it,is,(resp[it,is]+1)] + alpha[is,1]*da[it,is,1];
          preds[it+1,is,(2-resp[it,is])] = preds[it,is,(2-resp[it,is])]+ka*alpha[is,1]*da[it,is,2];
          }
        }

        // generate a log posterior based on tapas script
        part = exp(beta[is,b]*preds[it,is,(resp[it,is]+1)]);
        logp[it,is] = log(part)/(part + exp(beta[is,b]*preds[it,is,(2-resp[it,is])])); // only difference to our softmax is taht this is about chosen vs unchosen value, not stim1 and stim2
        // adapted from: logp(t) = log(exp(beta_t(t-1)*v(t,a(t)))/(exp(beta_t(t-1)*v(t,a(t))) + exp(beta_t(t-1)*v(t, 3-a(t)))));
        }else if (outc[it,is]==999){
        da[it,is,1] = 999;
        da[it,is,2] = 999;
        preds[it+1,is,1]= preds[it,is,1];
        preds[it+1,is,2]= preds[it,is,2];
      }
      prob = exp(beta[is,b]*preds[it,is,2])/(exp(beta[is,b]*preds[it,is,2])+ exp(beta[is,b]*preds[it,is,1])); // softmax
      prob1 = log(prob/(1-prob));
      resp_sim[it,is] = bernoulli_logit_rng(prob1); // simulating values

      // calculate LME


    }
}
}



  //    generated quantities {
  //      real log_lik[ntr,nsub];
  //      for (is in 1:nsub){ // need to predict for all parameter in one go! could also to multiple logliks if we want to disentangle the prediction for different parameters
  //          for (it in 1:ntr) {
  //          if (resp[it,is]!=999){
  //          log_lik[it,is] = bernoulli_logit_lpmf(resp[it,is]|beta[is,b]*outc[it,is]); // not sure about outc here
         // how can we get the log likelihood for alpha ?
  //       }else{
  //       log_lik[it,is] = 999;
  //    }
  //  }
  //  }
  //  }
