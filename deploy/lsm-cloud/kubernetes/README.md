# Kubernetes LSM configuration example

`10-configmap.yaml` is a provider-neutral configuration example for the Helion
LSM backend. Replace the bucket, region, namespace, service names, image, and
resource sizing for your environment.

The example contains no credentials. Supply object-store credentials through a
Kubernetes Secret, workload identity, or the equivalent mechanism supported by
your cloud provider.

For a runnable local Kubernetes example, use the Minikube manifests and
`deploy/lsm-cloud/minikube/verify.sh`.
