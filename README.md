# bozen-tputorial
Toying around with TPU on GKE

Following the Tutorial https://cloud.google.com/tpu/docs/kubernetes-engine-setup

# Setting up the TPU enabled K8S Cluster

```
gcloud beta container --project "bozen2021cloudhack" clusters create "tpu-cluster-1-clone-1" --zone "europe-west4-a" --no-enable-basic-auth --cluster-version "1.19.9-gke.1900" --release-channel "regular" --machine-type "e2-medium" --image-type "COS_CONTAINERD" --disk-type "pd-standard" --disk-size "100" --metadata disable-legacy-endpoints=true --scopes "https://www.googleapis.com/auth/cloud-platform" --max-pods-per-node "110" --num-nodes "3" --enable-ip-alias --network "projects/bozen2021cloudhack/global/networks/default" --subnetwork "projects/bozen2021cloudhack/regions/europe-west4/subnetworks/default" --no-enable-intra-node-visibility --default-max-pods-per-node "110" --no-enable-master-authorized-networks --addons HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver --enable-autoupgrade --enable-autorepair --max-surge-upgrade 1 --max-unavailable-upgrade 0 --enable-tpu --enable-shielded-nodes --node-locations "europe-west4-a"
```

# Running a TPU job:

`kubectl apply -f tpu-mnist.yaml`
