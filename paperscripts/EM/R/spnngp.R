
# This is just a quick test to inspect the output of the spNNGP package as well,
# as requested by the AE during review.

library("spNNGP")

# Read in the data. If you are just trying to run this script after cloning the
# repo, note that you'll need to use Julia to generate the data file first. But
# it should be reproducible, because the code to generate it uses a fixed seed
# and RNG software version. 
locs_data = read.csv("../data/data.csv", header=FALSE)
locs = as.matrix(locs_data[,1:2])
data = locs_data[,3:dim(locs_data)[2]]

# Read in the inits, although of course there are many ways to parameterize the
# matern and I will need to be careful to make sure that I convert whatever
# estiamtes come out back to this scale (which directly parameterizes sigma^2
# and tau^2, because that's what GPVecchia has as their hard-coded model.)
ini = c(read.csv("../data/initparameters.csv", header=FALSE))$V1

# Starting at literally the true values to be as generous as possible.
starting <- list("phi"=1/0.025, "sigma.sq"=0.25, "tau.sq"=10.0, "nu"=2.25)

# I don't really understand what this argument does, so I'm just using the
# default values shown in the docs.
tuning   <- list("phi"=0.5, "sigma.sq"=0.5, "tau.sq"=0.5, "nu"=0.5)

# Again picked to be pretty generous: They're all centered in almost exactly the
# right place, and I put an upper bound on nu so that the sampler won't go too
# far out in big positive values.
nsamples = 2000
priors <- list("phi.Unif"=c(10.0, 75.0), "sigma.sq.IG"=c(9.0, 1.1),
               "tau.sq.IG"=c(0.25, 1.1), "nu.Unif"=c(0.25, 5.0))

# From looking at the print output of the spNNGP function, it looks like the
# first ~100 or so trials have a decent acceptance rate (10-20%), but then the
# sampler seems to pretty reliably get trapped. This makes sense considering
# that we know the log-likelihood surface is weirdly shaped in this asymptotic
# regime, and it is why optimizers have a hard time as well (which is why in
# this paper and several prior ones we make such a big deal about optimizing
# with gradients and Hessians).
data_as_df = data.frame(y=data[,1], x=rep(0, 15000))
t_start  = Sys.time()
res <- spNNGP(y~1, data=data_as_df, coords=locs, starting=starting, 
              method="response", n.neighbors=10, tuning=tuning, priors=priors,
              cov.model="matern", n.samples=nsamples, n.omp.threads=4)
t_stop = Sys.time()

# I'm not taking the time too seriously here because I'm not an expert Bayesian.
# But with that said, I have specified the initializer and priors to be pretty
# generous (to my eye), and if several minutes in the end gives you 30-40
# accepted steps, that certainly seems like evidence that this Bayesian approach
# doesn't blow any optimization-based approach out of the water.
#
# More importantly, looking at these results, it seems like more evidence that
# the sampler hits a portion of the likelihood surface with a decent
# log-likelihood and then really struggles to travel along the very narrow
# banana-shaped trench to other values with good log-likelihoods. I am not very
# knowledgeable about Bayesian methodology and so I wouldn't assert that as a
# fact, but if you re-run this script a handful of times you'll see that the
# estimated percentiles you get out here are very narrow, and when you run it
# again you get disjoint intervals in general. The terminal log-likelihood of
# these parameters is not terrible---I would bet within 100 of the true
# (Vecchia-approximated) MLE---but considering how few samples you get and how
# close all the samples are, this again seems at the very least like evidence
# that this is not an obvious win for spNNGP.
summary(res)

