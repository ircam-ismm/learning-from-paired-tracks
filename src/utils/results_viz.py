#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
#%%

#open files containing the results from w2v evaluation
root="../results/domain_shift"
fnames = os.listdir(root)
file_paths=[os.path.join(root,fname) for fname in fnames if fname.endswith(".npy")]

#print(file_paths)
# %%
data={}
for file in file_paths:
    bincount=np.load(file, allow_pickle=True)
    name = os.path.basename(file).strip("cosine_sim_music_")[:-4]
    data[name]=bincount
data =dict(sorted(data.items()))
# %%
def plot_normalised_bins_and_gaussian(data, gauss=True,save_fig=False):
    """
    Plots the normalized bincounts and the corresponding Gaussian curve.

    Args:
        bincounts: A numpy array of size 20 containing the bincounts.
        metric_name: The name of the metric for the plot title.
    """
    colors = ["b","tab:orange","r","c","tab:brown","k"]
    plt.figure(figsize=(10,6), dpi=300)
    for i,(metric_name, bincounts) in enumerate(data.items()): 
        # Normalize the bincounts
        total_count = sum(bincounts)
        normalized_bincounts=bincount/total_count

        # Define bin centers
        bin_centers = np.linspace(-1, 1, len(bincounts))  # Adjust 1.01 for slight overlap
        
        # Fit a Gaussian curve to the data
        # Compute the weighted mean
        mu = np.sum(bincounts * bin_centers) / np.sum(bincounts)
        sigma = np.sum((bincounts * (bin_centers - mu) ** 2)) / np.sum(bincounts)
        sigma=sigma**0.5
        #print(metric_name,mu,sigma)
        x=np.linspace(-1,2,200)
        fitted_curve = 1/(sigma*(2*np.pi)**0.5)*np.exp(-1*(x-mu)**2/(2*sigma**2))   #norm.pdf(bin_centers, mu, sigma)

        # Create the plot
        #plt.figure(figsize=(10, 6))  # Adjust figure size as needed
        #plt.bar(bin_centers, normalized_bincounts, width=0.1, label='Normalized Bincounts')
        if gauss:
            plt.plot(x[np.where(x<=1)], fitted_curve[np.where(x<=1)], label=metric_name,color=colors[i])
            plt.plot(x[np.where(x>1)], fitted_curve[np.where(x>1)],'--',color=colors[i])
        else :
            plt.bar(bin_centers, normalized_bincounts, width=0.1, label=metric_name, alpha=0.5)
        #plt.title('Normalized Bincounts and Gaussian Fit for {}'.format(metric_name))
        plt.legend()
        plt.grid(True)
    
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Probability Density')    
    plt.title("Domain shift for Wav2Vec 2.0 (music)")
    #plt.xlim(0,1)
    if save_fig:
        plt.savefig("w2v_domain_shift.png")
    plt.show()

#%%
plot_normalised_bins_and_gaussian(data, gauss=True, save_fig=True)



# %%
