import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch.distributions
np.random.seed(5)
# Define generative process function
def generative_process(x, w, noise_var=1):
    y = w[4] + w[0]*x + w[1]*x**2 + w[2]*np.sin(x) + w[3]*np.log(x)
    noise = np.random.normal(loc=0, scale=noise_var, size=len(x))

    y=y+noise

    return y

# Define prior distribution function
def prior_distribution(mean, cov, x):
    prior = np.random.multivariate_normal(mean, cov)
    y = generative_process(x, prior)
    return y

# Define likelihood function
def likelihood(x, y, mean, cov):
    likelihood = np.random.multivariate_normal(mean, cov)
    y_pred = generative_process(x, likelihood)
    likelihood_val = np.exp(-0.5*(y-y_pred)**2/cov[0,0])
    return likelihood_val

# Define posterior distribution function
def posterior_distribution(mean_prior, cov_prior, x, y, mean_likelihood, cov_likelihood):
    cov_posterior = np.linalg.inv(np.linalg.inv(cov_prior) + 1/cov_likelihood[0,0]*np.outer(x,x))
    mean_posterior = cov_posterior @ (np.linalg.inv(cov_prior) @ mean_prior + y/cov_likelihood[0,0]*x)
    posterior = np.random.multivariate_normal(mean_posterior, cov_posterior)
    y_posterior = generative_process(x, posterior)
    return y_posterior

# Define posterior predictive distribution function
def posterior_predictive_distribution(mean_prior, cov_prior, x, mean_likelihood, cov_likelihood):
    cov_posterior_predictive = cov_prior + cov_likelihood
    mean_posterior_predictive = mean_prior + mean_likelihood
    posterior_predictive = np.random.multivariate_normal(mean_posterior_predictive, cov_posterior_predictive)
    y_posterior_predictive = generative_process(x, posterior_predictive)
    return y_posterior_predictive

# Define unlearning process function
def unlearning_process(mean_prior, cov_prior, x, y, mean_likelihood, cov_likelihood, x_unlearn, y_unlearn):
    cov_posterior = np.linalg.inv(np.linalg.inv(cov_prior) + 1/cov_likelihood[0,0]*np.outer(x,x))
    mean_posterior = cov_posterior @ (np.linalg.inv(cov_prior) @ mean_prior + y/cov_likelihood[0,0]*x)
    posterior = np.random.multivariate_normal(mean_posterior, cov_posterior)
    y_posterior = generative_process(x, posterior)
    cov_posterior_unlearn = np.linalg.inv(np.linalg.inv(cov_posterior) + 1/cov_likelihood[0,0]*np.outer(x_unlearn,x_unlearn))
    mean_posterior_unlearn = cov_posterior_unlearn @ (np.linalg.inv(cov_posterior) @ mean_posterior - y_unlearn/cov_likelihood[0,0]*x_unlearn)
    posterior_unlearn = np.random.multivariate_normal(mean_posterior_unlearn, cov_posterior_unlearn)
    y_posterior_unlearn = generative_process(x, posterior_unlearn)
    return y_posterior_unlearn



def model_layout():

    # Create Streamlit app
    st.title('Bayesian Linear Regression Unlearning')
    st.sidebar.title('The Control Panel')
    st.sidebar.subheader("Control the model data and parameters from here")
    st.sidebar.text("The model is of the form, ")
    st.sidebar.text("y = w1*x + w2*x^2 + w3*sinx + w4*logx"+ " + w5.")
    st.sidebar.text("You can choose which terms to include in the generative process.")

    mean_prior_lin, mean_prior_quad, mean_prior_sin, mean_prior_log, mean_prior_const = 0.0, 0.0, 0.0, 0.0, 0.0
    var_prior_lin, var_prior_quad, var_prior_sin, var_prior_log, var_prior_const = 1.0, 1.0, 1.0, 1.0, 1.0

    flag_include = [0,0,0,0,1]
    st.sidebar.divider()
    st.sidebar.text("Choose the mean and variance for the prior of the constant term:")

    mean_prior_const=st.sidebar.slider("Mean of prior for constant term", -10.0, 10.0, 0.0)
    var_prior_const =st.sidebar.slider("Variance of prior for constant term", 0.0, 1.0, 1.0)
    st.sidebar.divider()

    st.sidebar.text("Choose whether to include linear term, quadratic term, sin term and log term in the generative process")
    #Checkbox for linear term

    st.sidebar.divider()
    linear_term = st.sidebar.checkbox('Include linear term',value=True)
    
    #If linear term is included only then show the slider for mean of prior for linear term
    if linear_term:
        flag_include[0] = 1
        # Slider for mean of prior for linear term
        mean_prior_lin = st.sidebar.slider('Mean of prior for linear term', -10.0, 10.0, 0.0)
        #Variances of prior for linear term
        var_prior_lin = st.sidebar.slider('Variance of prior for linear term', 0.0, 1.0, 1.0)

    st.sidebar.divider()
    #Checkbox for quadratic term
    quad_term = st.sidebar.checkbox('Include quadratic term')

    #If quadratic term is included only then show the slider for mean of prior for quadratic term
    if quad_term: 
        flag_include[1] = 1
        # Slider for mean of prior for quadratic term
        mean_prior_quad = st.sidebar.slider('Mean of prior for quadratic term', -10.0, 10.0, 0.0)
        #Variances of prior for quadratic term
        var_prior_quad = st.sidebar.slider('Variance of prior for quadratic term', 0.0, 1.0, 1.0)

    st.sidebar.divider()

    #Checkbox for sin term
    sin_term = st.sidebar.checkbox('Include sin term')

    #If sin term is included only then show the slider for mean of prior for sin term
    if sin_term:
        flag_include[2] = 1
        # Slider for mean of prior for sin term
        mean_prior_sin = st.sidebar.slider('Mean of prior for sin term', -10.0, 10.0, 0.0)
        #Variances of prior for sin term
        var_prior_sin = st.sidebar.slider('Variance of prior for sin term', 0.0, 1.0, 1.0)

    st.sidebar.divider()

    #Checkbox for log term
    log_term = st.sidebar.checkbox('Include log term')

    #If log term is included only then show the slider for mean of prior for log term
    if log_term:
        flag_include[3] = 1
        # Slider for mean of prior for log term
        mean_prior_log = st.sidebar.slider('Mean of prior for log term', -10.0, 10.0, 0.0)
        #Variances of prior for log term
        var_prior_log = st.sidebar.slider('Variance of prior for log term', 0.0, 1.0, 1.0)

    st.sidebar.divider()
    st.sidebar.text("Choose the number of datapoints to generate")
    # Slider for number of data points
    num_data_points = st.sidebar.slider('Number of data points', 10, 70, 1)

    st.sidebar.text("Choose the variance of noise for the datapoints")

    #Variance of noise for the datapoints
    noise_var = st.sidebar.slider('Variance of noise for the datapoints', 0.0, 10.0, 0.1)

    st.sidebar.divider()

    st.sidebar.text("Choose the number of points to choose for the unlearning process")
    # Slider for number of points to unlearn
    num_unlearn_points = st.sidebar.slider('Number of points to unlearn', 0, num_data_points, 0)


    #Inputs for covariance matrix of size nxn where n-1 is the number of checkboxes
    cov_prior = np.zeros((5,5))
    var = np.zeros(5)
    if linear_term:
        cov_prior[0,0] = var_prior_lin
        var[0] = var_prior_lin

    if quad_term:
        cov_prior[1,1] = var_prior_quad
        var[1] = var_prior_quad

    if sin_term:
        cov_prior[2,2] = var_prior_sin
        var[2] = var_prior_sin

    if log_term:
        cov_prior[3,3] = var_prior_log
        var[3] = var_prior_log

    cov_prior[4,4] = var_prior_const
    var[4] = var_prior_const

    #Inputs for mean vector of size n where n-1 is the number of checkboxes
    mean_prior = np.zeros(5)

    if linear_term:
        mean_prior[0] = mean_prior_lin

    if quad_term:
        mean_prior[1] = mean_prior_quad

    if sin_term:
        mean_prior[2] = mean_prior_sin

    if log_term:
        mean_prior[3] = mean_prior_log   

    mean_prior[4] = mean_prior_const


    return flag_include,linear_term, quad_term, sin_term, log_term, num_data_points, num_unlearn_points, cov_prior, mean_prior, var_prior_lin, var_prior_quad, var_prior_sin, var_prior_log,var,noise_var

flag_include,linear_term, quad_term, sin_term, log_term, num_data_points, num_unlearn_points, cov_prior, mean_prior, var_prior_lin, var_prior_quad, var_prior_sin, var_prior_log,var,noise_var = model_layout()

flag_include_generative = flag_include

st.subheader("Let's visualise the prior distribution of any two features together.") 
st.text("Make sure to select at most two features to see the plot and that the corresponding checkboxes are selected in the sidebar.")


def choose_plot(options,flag_include):
    # Define the options
    # options = ["Linear", "Quadratic", "Sine", "Log"]
    flag_chosen = [0] * len(options)
    # Initialize variables to keep track of selected options
    selected_options = []
    linear_term = flag_include[0]
    quad_term = flag_include[1]
    sin_term = flag_include[2]
    log_term = flag_include[3]
    const_term = flag_include[4]

    # Create checkboxes for each option
    for option in options:
        selected = st.checkbox(option)

        # If the option is selected, add it to the list
        if selected:
            selected_options.append(option)
            if option == options[0]:
                if linear_term!=True:
                    st.warning("Please select the checkbox for linear term in the sidebar to see the plot")
                flag_chosen[0] = 1
            if option == options[1]:
                if quad_term!=True:
                    st.warning("Please select the checkbox for quadratic term in the sidebar to see the plot")
                flag_chosen[1] = 1
            if option == options[2]:
                if sin_term!=True:
                    st.warning("Please select the checkbox for sin term in the sidebar to see the plot")
                flag_chosen[2] = 1
            if option == options[3]:
                flag_chosen[3] = 1
                if log_term!=True:
                    st.warning("Please select the checkbox for log term in the sidebar to see the plot")
            if option == options[4]:
                flag_chosen[4] = 1
                if const_term!=True:
                    st.warning("Please select the checkbox for constant term in the sidebar to see the plot")

        # Check if more than two options are selected
        if len(selected_options) > 2:
            st.warning("Please select at most two options.")
            # If more than two options are selected, unselect the current option
            selected_options.remove(option)
            if option == options[0]:
                flag_chosen[0] = 0
            if option == options[1]:
                flag_chosen[1] = 0
            if option == options[2]:
                flag_chosen[2] = 0
            if option == options[3] :
                flag_chosen[3] = 0
            if option == options[4] :
                flag_chosen[4] = 0

            st.checkbox(option, value=False)

    return flag_chosen

flag_chosen_prior = choose_plot(["Linear_prior", "Quadratic_prior", "Sine_prior", "Log_prior","Const_prior"],flag_include.copy())

#Make linear_prior and const_prior checkbox selected by default


def plot_density_graph_bivariate(var, mean_prior, flag_chosen):
    #get indices of the chosen options
    indices = [i for i, x in enumerate(flag_chosen) if x == 1]

    #mean vector 
    mean_biv = torch.tensor(mean_prior[indices])

    #covariance matrix as 2x2 matrix

    # var = [var_prior, var_prior_quad, var_prior_sin, var_prior_log]
    var_biv = [var[i] for i in indices]
    cov_biv = torch.diag(torch.tensor(var_biv))
    # Define the grid
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)

    dist = torch.distributions.MultivariateNormal(mean_biv, cov_biv)

    # Create a figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    try:


        # Create a contour plot
        ax.contourf(X, Y, dist.log_prob(torch.stack([torch.Tensor(X), torch.Tensor(Y)], dim=2)).exp())

        # Add a colorbar
        fig.colorbar(ax.contourf(X, Y, dist.log_prob(torch.stack([torch.Tensor(X), torch.Tensor(Y)], dim=2)).exp()), ax=ax)

        # Set title and labels for axes
        ax.set(xlabel='x', ylabel='y', title='Bivariate Normal Distribution')

        #plot the mean_biv as a point and label it as the coordinate values
        ax.plot(mean_biv[0], mean_biv[1], 'o', color='black')
        ax.text(mean_biv[0], mean_biv[1], '({}, {})'.format(mean_biv[0], mean_biv[1]))
        

        # Show the plot
        st.pyplot(fig)
    except:
        st.write("Please select two features to see the plot")

plot_density_graph_bivariate(var, mean_prior, flag_chosen_prior)



st.subheader("Generative Data Plot")
st.text("Below is the plot of the data points generated from the generative process.")
st.text("You can vary the number of data points and the variance of the noise from the sidebar.")
def plot_generative_data(flag_include, num_data_points,noise_var):

    fig,ax = plt.subplots()
    # Generate num_data_points x values randomly between 0.1 and 10
    x = np.random.uniform(0.1, 10, num_data_points)

    # Generate y values from generative process
    w = [3, 1, 9, 2, 2]
    
    #Make those weights zero which are not chosen
    for i in range(4):
        if flag_include[i] == 0:
            w[i] = 0

    y = generative_process(x, w, noise_var)

    #Plot the generative data
    plt.plot(x, y, 'o')
    plt.title('Generative Data')
    plt.xlabel('x')
    plt.ylabel('y')
    st.pyplot(fig)

    return x,y

X_data, y_data = plot_generative_data(flag_include_generative, num_data_points,noise_var)

var_likelihood = 1
def plot_posterior(flag_include, num_data_points, mean_prior, cov_prior, var, flags_chosen_posterior,X_data,y_data):


    indices = [i for i, x in enumerate(flag_include) if x == 1]
    # "Indices:"
    # indices
    
    #mean vector
    mean_biv = torch.tensor(mean_prior[indices])
    # "Mean_biv"
    # mean_biv

    # "Var_biv"

    var_biv = [var[i] for i in indices]
    # var_biv

    cov_biv = torch.diag(torch.tensor(var_biv))
    # "Meandbiv and Covbiv"
    # mean_biv
    # cov_biv
    #covariance matrix as 2x2 matrix
    cov_likelihood = torch.diag(torch.tensor([var_likelihood]*len(indices)))

    x_data = torch.empty(num_data_points,0)
    #Make x_data from 1x100 to number of flags included -1 x 100 by addding adding transformed x values
    for i in range(len(indices)):
        if indices[i] == 0:
            x_data = torch.cat((x_data,torch.tensor(X_data).reshape(-1,1)),1)
        if indices[i] == 1:
            x_data = torch.cat((x_data,torch.tensor(X_data**2).reshape(-1,1)),1)
        if indices[i] == 2:
            x_data = torch.cat((x_data,torch.tensor(np.sin(X_data)).reshape(-1,1)),1)
        if indices[i] == 3:
            x_data = torch.cat((x_data,torch.tensor(np.log(X_data)).reshape(-1,1)),1)
        if indices[i] == 4:
            x_data = torch.cat((x_data,torch.tensor([1]*num_data_points).reshape(-1,1)),1)

    
    
    #conver x_data type to float64
    x_data = x_data.type(torch.float64)
    # x_data
    y_data = torch.tensor(y_data)
    # y_data
    cov_posterior = torch.linalg.inv(torch.linalg.inv(cov_biv) + (1/var_likelihood)*torch.matmul(x_data.T , x_data))
    mean_posterior = torch.matmul(cov_posterior , (torch.linalg.inv(cov_biv)@mean_biv) + (1/var_likelihood)*torch.matmul(x_data.T ,y_data))

    map_original_to_new = {}
    a=0
    for i in range(5):
        if flag_include[i] == 1:
            map_original_to_new[i] = a
            a+=1

    mean_posterior_total = torch.zeros(5)
    
    for i in range(5):
        if flag_include[i] == 1:
            mean_posterior_total[i] = mean_posterior[map_original_to_new[i]]
        else:
            mean_posterior_total[i] = 0

    # mean_posterior = mean_posterior_total
    # "Mean Posterior Total"
    # mean_posterior
    cov_posterior_total = torch.zeros((5,5))

    for i in range(5):
        for j in range(5):
            if flag_include[i] == 1 and flag_include[j] == 1:
                cov_posterior_total[i,j] = cov_posterior[map_original_to_new[i],map_original_to_new[j]]
            else:
                cov_posterior_total[i,j] = 0

    # cov_posterior = cov_posterior_total
    # "Cov_posterior_total"
    # cov_posterior

    indices_plot = [i for i, x in enumerate(flags_chosen_posterior) if x == 1]

    mean_posterior_plot= mean_posterior_total[indices_plot]

    cov_posterior_plot = torch.zeros((len(indices_plot),len(indices_plot)))

    for i in range(len(indices_plot)):
        for j in range(len(indices_plot)):
            cov_posterior_plot[i,j] = cov_posterior_total[indices_plot[i],indices_plot[j]]


    # Define the grid
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)

    dist = torch.distributions.MultivariateNormal(mean_posterior_plot, cov_posterior_plot)
    
    # Create a figure and axes
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))

    try:
        n1,n2 = np.round(mean_posterior_plot[0],2), np.round(mean_posterior_plot[1],2)
        # st.subheader(n1,n2)

        # Create a contour plot
        ax2.contourf(X, Y, dist.log_prob(torch.stack([torch.Tensor(X), torch.Tensor(Y)], dim=2)).exp())

        # Add a colorbar
        fig2.colorbar(ax2.contourf(X, Y, dist.log_prob(torch.stack([torch.Tensor(X), torch.Tensor(Y)], dim=2)).exp()), ax=ax2)

        # Set title and labels for axes
        ax2.set(xlabel='x', ylabel='y', title='Bivariate Normal Distribution')

        ax2.plot(mean_posterior_plot[0], mean_posterior_plot[1], 'o', color='black')
        # ax2.text(n1, n2, '({}, {})'.format(np.round(n1,2), np.round(n2,2)))

        #coordinate values upto two decimal places

        # Show the plot
        st.pyplot(fig2)
    except:
        st.write("Please select two features to see the plot")

    return mean_posterior,cov_posterior, mean_posterior_total, cov_posterior_total, map_original_to_new


# try:
st.subheader("Posterior Distribution Plot")
flags_chosen_posterior = choose_plot(["Linear_posterior", "Quadratic_posterior", "Sine_posterior", "Log_posterior","Const_posterior"],flag_include.copy())
# flags_chosen_posterior
mean_posterior, cov_posterior, mean_posterior_total, cov_posterior_total,map_original_to_new=plot_posterior(flag_include, num_data_points, torch.tensor(mean_prior), torch.tensor(cov_prior), torch.tensor(var),flags_chosen_posterior,X_data,y_data)
# except:
#     st.write("Please select two features to see the plot")
"Cov_posterior"
cov_posterior

"Mean_posterior"
mean_posterior


def plot_posterior_predictive(flag_include, num_data_points, mean_prior, cov_prior, flags_chosen_posterior,X_data,y_data,mean_posterior,cov_posterior,mean_posterior_total,cov_posterior_total,map_original_to_new,var_likelihood):

    '''
    Here we will scatter plot the data points, 
    Take 1000 samples from the posterior distribution,
    Find mean and variance of y for each x from the 1000 samples,
    Plot mean and fill between mean+ 1.96 variance and mean-1.96 variance
    '''

    #Obtain 1000 samples from mean_posterior and cov_posterior
    samples = np.random.multivariate_normal(mean_posterior, cov_posterior, 1000)

    #Resize samples to 1000x5 with the columns corresponding to the chosen features, and the rest as zero
    samples_total = np.zeros((1000,5))

    for i in range(5):
        if flag_include[i] == 1:
            samples_total[:,i] = samples[:,map_original_to_new[i]]
        else:
            samples_total[:,i] = 0

    #Define x and y
    x = torch.tensor(X_data)
    y = torch.tensor(y_data)

    #Scatter plot the data points
    fig4, ax4 = plt.subplots(1, 1, figsize=(6, 6))
    ax4.scatter(x, y, color='black', alpha=0.5)

    x_predict = np.linspace(0.1, 10, 100)

    #Predict y using the 1000 samples parameters
    y_predict = np.zeros((1000,100))

    for i in range(1000):
        y_predict[i] = generative_process(x_predict, samples_total[i])

    #Find mean and variance of y for each x from the 1000 samples
    mean_y_predict = np.mean(y_predict, axis=0)
    var_y_predict = np.var(y_predict, axis=0)

    #Plot mean and fill between mean+ 1.96 variance and mean-1.96 variance
    ax4.plot(x_predict, mean_y_predict, color='black', linewidth=2)
    ax4.fill_between(x_predict, mean_y_predict + 1.96*np.sqrt(var_y_predict), mean_y_predict - 1.96*np.sqrt(var_y_predict), alpha=0.2, color='black')
    ax4.set_title('Posterior Predictive Distribution')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    st.pyplot(fig4)

    return samples_total, mean_y_predict, var_y_predict

st.subheader("Posterior Predictive Distribution Plot")
plot_posterior_predictive(flag_include, num_data_points, torch.tensor(mean_prior), torch.tensor(cov_prior), flags_chosen_posterior,X_data,y_data,mean_posterior,cov_posterior,mean_posterior_total,cov_posterior_total,map_original_to_new,var_likelihood)
    




def unlearning(num_data_points,num_unlearn_points,X_data,y_data,mean_posterior,cov_posterior,flag_include,map_original_to_new,var_likelihood):
    #Randomly choose indices to unlearn
    indices_unlearn = np.random.choice(num_data_points, num_unlearn_points, replace=False)

    #Define x_unlearn and y_unlearn
    x_unlearn = X_data[indices_unlearn]
    y_unlearn = y_data[indices_unlearn]

    X_del = torch.tensor(x_unlearn)
    Y_del = torch.tensor(y_unlearn)

    x_del = torch.empty(num_unlearn_points,0)
    #Make x_data from 1x100 to number of flags included -1 x 100 by addding adding transformed x values
    for i in range(5):
        if flag_include[i] == 1:
            if i == 0:
                x_del = torch.cat((x_del,torch.tensor(x_unlearn).reshape(-1,1)),1)
            if i == 1:
                x_del = torch.cat((x_del,torch.tensor(x_unlearn**2).reshape(-1,1)),1)
            if i == 2:
                x_del = torch.cat((x_del,torch.tensor(np.sin(x_unlearn)).reshape(-1,1)),1)
            if i == 3:
                x_del = torch.cat((x_del,torch.tensor(np.log(x_unlearn)).reshape(-1,1)),1)
            if i == 4:
                x_del = torch.cat((x_del,torch.tensor([1]*num_unlearn_points).reshape(-1,1)),1)

    x_del = x_del.type(torch.float64)

    unlearned_posterior_cov = torch.linalg.inv(torch.linalg.inv(cov_posterior) - (1/var_likelihood)*torch.matmul(x_del.T , x_del))
    unlearned_posterior_mean = torch.matmul(unlearned_posterior_cov , (torch.linalg.inv(cov_posterior)@mean_posterior) - (1/var_likelihood)*torch.matmul(x_del.T , Y_del))

    unlearned_posterior_mean_total = torch.zeros(5)

    for i in range(5):
        if flag_include[i] == 1:
            unlearned_posterior_mean_total[i] = unlearned_posterior_mean[map_original_to_new[i]]
        else:
            unlearned_posterior_mean_total[i] = 0

    unlearned_posterior_cov_total = torch.zeros((5,5))

    for i in range(5):
        for j in range(5):
            if flag_include[i] == 1 and flag_include[j] == 1:
                unlearned_posterior_cov_total[i,j] = unlearned_posterior_cov[map_original_to_new[i],map_original_to_new[j]]
            else:
                unlearned_posterior_cov_total[i,j] = 0

    return unlearned_posterior_cov_total, unlearned_posterior_mean_total, unlearned_posterior_mean, unlearned_posterior_cov, indices_unlearn

unlearned_posterior_cov_total, unlearned_posterior_mean_total, unlearned_posterior_mean, unlearned_posterior_cov,indices_unlearn = unlearning(num_data_points,num_unlearn_points,X_data,y_data,mean_posterior,cov_posterior,flag_include,map_original_to_new,var_likelihood)


def plot_unlearned_posterior(unlearned_posterior_cov_total, unlearned_posterior_mean_total,flag_chosen_unlearned_posterior):

    indices = [i for i, x in enumerate(flag_chosen_unlearned_posterior) if x == 1]
    #mean vector
    unlearned_posterior_mean_plot = unlearned_posterior_mean_total[indices]
    # var_biv = [var[i] for i in indices]
    #covariance matrix as 2x2 matrix
    unlearned_posterior_cov_plot = torch.zeros((len(indices),len(indices)))

    for i in range(len(indices)):
        for j in range(len(indices)):
            unlearned_posterior_cov_plot[i,j] = unlearned_posterior_cov_total[indices[i],indices[j]]


    # "Unlearned posterior_mean"
    # unlearned_posterior_mean
    # unlearned_posterior_cov
    # Define the grid
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)

    dist = torch.distributions.MultivariateNormal(unlearned_posterior_mean_plot, unlearned_posterior_cov_plot)

    # Create a figure and axes
    fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6))

    try:


        # Create a contour plot
        ax3.contourf(X, Y, dist.log_prob(torch.stack([torch.Tensor(X), torch.Tensor(Y)], dim=2)).exp())

        # Add a colorbar
        fig3.colorbar(ax3.contourf(X, Y, dist.log_prob(torch.stack([torch.Tensor(X), torch.Tensor(Y)], dim=2)).exp()), ax=ax3)

        # Set title and labels for axes
        ax3.set(xlabel='x', ylabel='y', title='Bivariate Normal Distribution')

        ax3.plot(unlearned_posterior_mean_plot[0], unlearned_posterior_mean_plot[1], 'o', color='black')
        ax3.text(unlearned_posterior_mean_plot[0], unlearned_posterior_mean_plot[1], '({}, {})'.format((np.round(unlearned_posterior_mean_plot[0],1)), (np.round(unlearned_posterior_mean_plot[1],1))))

        #coordinate values upto two decimal places

        # Show the plot
        st.pyplot(fig3)
    except:
        st.write("Please select two features to see the plot")

    # return unlearned_posterior_mean, unlearned_posterior_cov


st.subheader("Unlearned Posterior Distribution Plot")
st.text("You can vary the number of points to unlearn from the sidebar and observe the change of the posterior accordingly.")
flag_chosen_unlearned_posterior = choose_plot(["Linear_unlearned_posterior", "Quadratic_unlearned_posterior", "Sine_unlearned_posterior", "Log_unlearned_posterior","Const_unlearned_posterior"],flag_include.copy())
plot_unlearned_posterior(unlearned_posterior_cov_total, unlearned_posterior_mean_total,flag_chosen_unlearned_posterior)
"Cov Posterior vs Cov Unlearned Posterior"
cov_posterior
unlearned_posterior_cov

"Mean Posterior vs Mean Unlearned Posterior"
mean_posterior
unlearned_posterior_mean

def plot_unlearned_posterior_predictive(flag_include, X_data,y_data,unlearned_posterior_mean,unlearned_posterior_cov,map_original_to_new, indices_unlearn):

    '''
    Here we will scatter plot the data points, 
    Take 1000 samples from the posterior distribution,
    Find mean and variance of y for each x from the 1000 samples,
    Plot mean and fill between mean+ 1.96 variance and mean-1.96 variance
    '''

    #Obtain 1000 samples from mean_posterior and cov_posterior
    samples = np.random.multivariate_normal(unlearned_posterior_mean, unlearned_posterior_cov, 1000)

    #Resize samples to 1000x5 with the columns corresponding to the chosen features, and the rest as zero
    samples_total = np.zeros((1000,5))

    for i in range(5):
        if flag_include[i] == 1:
            samples_total[:,i] = samples[:,map_original_to_new[i]]
        else:
            samples_total[:,i] = 0

    #Define x and y
    x = torch.tensor(X_data)
    y = torch.tensor(y_data)

    #Scatter plot the data points
    fig5, ax5 = plt.subplots(1, 1, figsize=(6, 6))
    ax5.scatter(x, y, color='black', alpha=0.5)

    #Highligh the unlearned points
    ax5.scatter(x[indices_unlearn], y[indices_unlearn], color='red', alpha=0.5)

    x_predict = np.linspace(0.1, 10, 100)

    #Predict y using the 1000 samples parameters
    y_predict = np.zeros((1000,100))

    for i in range(1000):
        y_predict[i] = generative_process(x_predict, samples_total[i])

    #Find mean and variance of y for each x from the 1000 samples
    mean_y_predict = np.mean(y_predict, axis=0)
    var_y_predict = np.var(y_predict, axis=0)

    #Plot mean and fill between mean+ 1.96 variance and mean-1.96 variance
    ax5.plot(x_predict, mean_y_predict, color='black', linewidth=2)
    ax5.fill_between(x_predict, mean_y_predict + 1.96*np.sqrt(var_y_predict), mean_y_predict - 1.96*np.sqrt(var_y_predict), alpha=0.2, color='black')
    ax5.set_title('Posterior Predictive Distribution')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    st.pyplot(fig5)

    return samples_total, mean_y_predict, var_y_predict

st.subheader("Unlearned Posterior Predictive Distribution Plot")
plot_unlearned_posterior_predictive(flag_include, X_data,y_data,unlearned_posterior_mean,unlearned_posterior_cov,map_original_to_new, indices_unlearn)



