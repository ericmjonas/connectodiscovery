Connectivity and attribute motif experiments

Goal: 
Try building a mixture model with additional metadata including: 

location of soma
dendritic arbor shape

Implement NIG in code

Test mixture model 

Extract features from data

run mixture model 

Estimate runtime performance

Add in connectivity



Pipeline:

1. select the data you want
2. Construct the model, pick HP gridding
3. Random init the data
4. 



To set up cluster

fab s.launch:10,1.68

fab -R slaves s.push_aws_creds s.worker_disks add_packages deploy_files

fab -R master s.push_aws_creds s.connect_master_data add_packages
