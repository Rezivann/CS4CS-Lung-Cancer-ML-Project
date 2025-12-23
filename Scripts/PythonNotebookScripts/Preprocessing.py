import pandas as pd
df = pd.read_excel('Datasets\\NewDataset\\BlackLineHospitalLungCancerDataset.xlsx', header=0) 
df.to_csv('Datasets\\NewDataset\\ProcessedDataset.csv', index=False, quotechar="'")
#Most of the preprocessing for this new dataset was done in Endalie and Abebe's paper.