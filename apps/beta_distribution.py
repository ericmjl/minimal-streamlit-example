import streamlit as st
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
import hvplot.pandas  # noqa: F401
import holoviews as hv

hv.extension("bokeh")

st.header("Beta Distribution Tutorial")

st.write(
    """
The beta distribution describes a probability distribution
over values from the range (0, 1).

In our make believe protein engineering project,
we use the beta distribution to help us
with the estimation of a mutant's actual activity,
defined as a fraction from 0 to 1,
aggregated over multiple biological replicates (multiple colonies per mutant)
and multiple technical replicates
(replicate measurement of individual colonies).

Before we go on to what Bayesian estimation is,
please go ahead and change the values of the alpha and beta parameters
using the sliders on the left sidebar
(click the arrow on the left if it is closed).
"""
)

# These are going to be "globally" defined, because I intend to use them
# across multiple plots.
st.sidebar.markdown(
    """
# Control Panel
"""
)
alpha_slider = st.sidebar.slider(
    "Value of alpha parameter",
    min_value=0.1,
    max_value=100.0,
    step=1.0,
    value=2.0,
)
beta_slider = st.sidebar.slider(
    "Value of beta parameter",
    min_value=0.1,
    max_value=100.0,
    step=1.0,
    value=12.0,
)


def plot_dist(alpha_value: float, beta_value: float, data: np.ndarray = None):
    beta_dist = beta(alpha_value, beta_value)

    xs = np.linspace(0, 1, 1000)
    ys = beta_dist.pdf(xs)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(xs, ys)
    ax.set_xlim(0, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("P(x)")

    if data is not None:
        likelihoods = beta_dist.pdf(data)
        sum_log_likelihoods = np.sum(beta_dist.logpdf(data))
        ax.vlines(data, ymin=0, ymax=likelihoods)
        ax.scatter(data, likelihoods, color="black")
        st.write(
            f"""
_Under your alpha={alpha_slider:.2f} and beta={beta_slider:.2f},
the sum of log likelihoods is {sum_log_likelihoods:.2f}_
"""
        )
    st.pyplot(fig)


plot_dist(alpha_slider, beta_slider)

st.subheader("What are the alpha and beta parameters?")
st.write(
    """
The alpha and beta control the shape of the beta distribution.
Their colloquial interpretation corresponds to
the coin flip, roughly "number of successes" (alpha)
and "number of failures" (beta).
The only difference here is that alpha and beta can be non-integer values,
i.e. having decimal places.
"""
)

st.subheader("How do we use it?")

st.write(
    """
The beta distribution is useful for modelling quantities
that can only take on values between 0 and 1.
Our conversion ratio is an example of this.
"""
)

st.header("Bayesian Estimation of Fractions using Beta-Distribution")

st.write(
    """
The core activity that we are trying to do
is to obtain the parameters of the alpha and beta distribution
that best explain the data that we observe.

Let us assume that we made three observations of activity for a mutant:
"""
)

data = [0.83, 0.86, 0.91]
st.write(data)

st.write(
    """
Can you find beta distribution parameters
that best explain these three data points?

The best ratio of alpha to beta is probably around 1:6.
However, is it 1:6, or is it 6:36, or is it 15:90?

Play around with different ratios
to see which one maximizes the log likelihood.
"""
)

plot_dist(alpha_slider, beta_slider, data)

st.write(
    """
Let's try this out with another dataset.

Which parameter values should best explain these values?
(Hint, it's approximately a ratio of alpha:beta ~ 1:1)
"""
)

data = [0.53, 0.56, 0.51]
st.write([0.53, 0.56, 0.51])

plot_dist(alpha_slider, beta_slider, data)

st.write(
    """
As you might have noticed,
the greater the magnitude of the value of your alpha and beta parameters,
the tighter the distribution of the beta distribution.

Let's take a look at it in the following chart.
"""
)

alpha_value = st.radio(label="Select a value for alpha", options=(2, 5))
beta_value = st.radio(label="Select a value for beta", options=(2, 5))

beta_dist_1 = beta(alpha_value, beta_value)
beta_dist_2 = beta(alpha_value * 5, beta_value * 5)
beta_dist_3 = beta(alpha_value * 10, beta_value * 10)

xs = np.linspace(0, 1, 1000)
ys_1 = beta_dist_1.pdf(xs)
ys_2 = beta_dist_2.pdf(xs)
ys_3 = beta_dist_3.pdf(xs)
fig4 = plt.figure(figsize=(7, 3))
plt.plot(xs, ys_1, label=f"alpha={alpha_value}, beta={beta_value}")
plt.plot(xs, ys_2, label=f"alpha={alpha_value * 5}, beta={beta_value * 5}")
plt.plot(xs, ys_3, label=f"alpha={alpha_value * 10}, beta={beta_value * 10}")
plt.ylabel("P(x)")
plt.legend()
st.pyplot(fig4)

st.write(
    """
As you can see, the width of the curve decreases
as the scale of the alpha and beta parameters increases.
_We become less uncertain._
"""
)


st.write(
    """
Now, if we focus our attention on
the variance of the beta distribution when operating in the same scale
(i.e. fix the sum of alpha and beta),
let's see what happens to the variance of the likelihood distribution.

Let's assume that alpha + beta = 50,
and we will have you adjust the value of alpha.
"""
)
max_val = 50
alpha_value = st.slider("alpha", 1, max_val - 1)
beta_value = max_val - alpha_value

beta_dist = beta(alpha_value, beta_value)
xs = np.linspace(0, 1, 1000)
ys = beta_dist.pdf(xs)
fig = plt.figure(figsize=(7, 3))
plt.plot(xs, ys, label=f"alpha={alpha_value}, beta={beta_value}")
plt.title(f"StDev: {beta_dist.std():.3f}")
plt.ylabel("P(x)")
plt.legend()
st.pyplot(fig)


st.write(
    """
As you can see, as we go more to the extremes, the variance decreases.

A more comprehensive look at variance as a function of alpha and beta,
keeping the "scale" (i.e. sum of alpha and beta) the same is below.
"""
)

alpha_values = np.arange(1, 50)
beta_dists = beta(alpha_values, 50 - alpha_values)
ys = beta_dists.std()
fig = plt.figure(figsize=(7, 4))
plt.scatter(alpha_values, ys)
plt.xlabel("alpha")
plt.title("standard deviation of the beta distribution as a function of alpha")
plt.ylabel("std")
st.pyplot()

st.header("Conclusions")

st.write(
    """
What we have found thus far is that
according to the math of the beta distribution,
at a given scale, variance is always going to be highest in the middle,
where the peak variance consistently is close to the middle
of the beta distribution likelihood.

The more noisy the calculated ratios are,
the smaller the alpha + beta scale needed to maximize their likelihoods,
and hence the greater their estimated likelihood distribution variance will be.

When we do Bayesian estimation,
we are estimating the parameters alpha and beta
that explain both the central tendency measures
of our measured ratios (i.e. mean)
and the variance in the measured ratios.
Ratios that are more variably distributed, such as `[0.81, 0.56, 0.93]`,
will naturally get smaller estimated alphas and betas
to explain the high variance in the ratios.
(They will also not tend to be centered around the extremes,
which also makes sense: to be generate data centered around the extremes,
we need extremely tightly-distributed data.)
Ratios that are more tightly distributed, such as `[0.51, 0.49, 0.48]`,
will naturally get higher estimated alphas and betas.
This is the spirit of what we are doing in the Bayesian estimation model.

In short, there's both math + measurement contributing to the "wave" phenomena.
"""
)

st.header("Try with your own data!")
st.write(
    """
I'd like to invite you to upload your own data.

Type in numbers into the text box below, separated by commas.
Example data have been provided for you.
"""
)

data = st.text_input("Your data", value="0.993, 0.91, 0.779")


def process_data(data):
    # One thing nice about streamlit is it allows us to surface informative
    # errors to our end-users using built-in Python error handling.
    # This is done by capturing the output of stdout and surfacing it to HTML.
    try:
        data = np.array([float(i) for i in data.split(", ")])
    except ValueError:
        raise ValueError("The data that you input must be castable as floats!")
    if not (np.all(data > 0) and np.all(data < 1)):
        raise ValueError("Your input data must be 0 < x < 1.")
    return data


plot_dist(alpha_slider, beta_slider, process_data(data))


st.header("Congratulations!")
st.write("Click the button below once you're done with this tutorial.")
if st.button("CLICK ME!"):
    st.balloons()


st.sidebar.markdown(
    """
# Thank you!
Did you like this mini-tutorial?

If you did, please give it a star on [GitHub](https://github.com/ericmjl/minimal-streamlit-example).

This was hand-crafted using streamlit in under 3 hours,
2.5 of which were spent crafting prose.

Created by [Eric J. Ma](https://ericmjl.github.io).
"""
)
