import numpy as np
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Script to generate data for later latent space analysis via CataRT or dimensionality reduction algorithms")
    parser.add_argument("--is_from", type=str, default="baseline", choices=["baseline","with_adaptation","speech"])
    parser.add_argument("--is_reduced",action="store_true")
    parser.add_argument("--reduction",type=str,default="",choices=["pca","umap","t-sne"])
    #combine is_reduced and --reduction in one arg
    
    args=parser.parse_args()
    
    if args.is_reduced:
        assert args.reduction!="", "reduction algorithm has to be specified if is_reduced is true."
    
    is_from=args.is_from
    is_reduced=args.is_reduced
    reduction=args.reduction
    suffix="_reduced_"+reduction if is_reduced else ""
    fname=f"CataRT data/{is_from}/data_{is_from}{suffix}.txt"

    root=f"CataRT data/{is_from}"
    data_name=f"data_{is_from}.npy"
    if is_reduced:
        data_name=f"reduced_data_{is_from}_{reduction}.npy" 
        
    data_path=os.path.join(root,data_name)
    data = np.load(data_path,allow_pickle=True).item() #get dict from npy
    
    if type(data['latents'])==list:
        X=np.concatenate(data['latents'],axis=0)
        Y=np.concatenate(data['labels'],axis=0)
        data = {key:item for key,item in zip(data.keys(),[X,Y])}

    #take all samples from directory and be sure they are ordered by idx
    #latents and labels are stored in that order so we need to be sure the correct data is extracted from the correspondig file
    chunks=sorted(os.listdir("CataRT data/samples"),key=lambda x: int(x[6:-4])) 

    #write a file with space separated values
    head = ["FileName"]+ [f"d{i}" for i in range(len(data['latents'][0]))] + ["instrument"]
    
    lines = []
    for chunk,latent,instrument in zip(chunks,data['latents'],data['labels']):
        chunk=chunk.split("/")[-1] #not necessary but good precaution
        line=[chunk]+ [str(d) for d in latent] + [str(instrument)]
        lines.append(line)


    #text = head+values
    with open(fname,'w') as f:
        for col in head:
            f.write(col+" ") #space separated value
        f.write("\n") #newline
        for line in lines:
            for col in line:
                f.write(col+" ") #space separated value
            f.write("\n")

if __name__=="__main__":
    main()
        