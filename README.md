# Elastic Training
Elastic Training on Kubernetes

*This repository holds end-to-end documents on how to start elastic training on Kubernetes clusters.*


It is a concept of elastic training is familiar to you, please jump to the section of [how to use Elastic Training](#how-to-use-elastic-training?).

## What is Elastic Training?

In brief, elastic training belongs to enhanced distributed training where the job persists regardless changes from computation resources.

### Distributed training

When a training task requires computation resources beyond a single node, whether because of dataset or the model size, distributed training divide the computation into several parts. According to scheme to decompose computation, we have *ParameterServer-Worker* mode where there will be participants of parameter servers as well as workers and *AllReduce* mode uniformly consists of workers.

### Elastic Training

When a distributed training task continues its computation regardless changes of participants unless it's catastrophically broken, we call this training task **Elastic Training**.

Without elastic training, a training job able to continue ***after a reboot*** from checkpoint as well. However, elastic training saves the work of reboot and considerable resource idleness.  

Given different training mode (PS-Worker/AllReduce), the elasticity could be difined differently. At this moment,  an elastic training of allreduce can lost workers unless there is at lease one left (otherwise, it's not recoverable).

## Who would need Elastic Training?

Elastic Training benefits groups with limit resources on deep learning tasks.

### With preemptible resources

Not every group can afford distributed training in large scale, even with public cloud service. However, spot instance (preempitble resource) generally costs only 20% ~ 30% of regular resource. Elastic training enables distributed training compatible with spot instances as researchers need not worry failures after instances recycled.

### With collocation system

For users work on IDC (Internet Data Center), the fixed resource pool contradicts with periodically fluctuating online request. With elastic training collocated with workloads processing online request, the training task works as a sponge, squeezing out resources when requests surge and soaking resources with plunge of requests. 

## How to use Elastic Training?

### Setup a Kubernetes cluster

We recommend a Kubernetes version of 1.18+.

### Deploy Kubeflow Training Operator

[training-operator](https://github.com/kubeflow/training-operator) from [Kubeflow](https://www.kubeflow.org/) community offer controllers for the following APIs:

| API       | API Version | Support Elastic Training |
| --------- | ----------- | ------------------------ |
| TFJob     | `v1`        | Yes (worker only*)        |
| MPIJob    | `v1`        | Yes                      |
| PyTorchJob| `v1`        | In Progress              |
| MXNetJob  | `v1`        | No                       |
| XGBoostJob| `v1`        | No                       |

**For TFJob (with ps-worker training mode), training-operator only support dynamic worker, not parameter server.*

Please use the following script to deploy kubeflow/training-operator (**master branch**):

```shell
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone"
```

### Elastic training with PriorityClass

We introduce how to train with Kubeflow MPIJob with PriorityClass in this section.

#### Prepare training script

To prepare an elastic training task with `Horovod`, please refer this [tutorial](https://horovod.readthedocs.io/en/stable/elastic_include.html#elastic-tensorflow).

For the sake of converge, user might need to wrapper the `lr_scheduler` to scaling the learning rate proportionally to the contemporary worker counts.

#### (Optional) Define PriorityClass

User may assign PriorityClass to Pods. Generally, we assign Pods from workloads for online service with **High** Priority, Pods from training jobs that should be kept under high resource stress with **Medium** Priority, Pods from training jobs that could be evicted with **Low** Priority.

`priority_classes.yaml` only serves as an example. Users can redefine values in these PriorityClass.


```yaml
apiVersion: scheduling.k8s.io/v1
description: Used for online inference related pods that is capable to make low priority pods preempted
kind: PriorityClass
metadata:
  name: inference-pc
preemptionPolicy: PreemptLowerPriority
value: 100000000
---
apiVersion: scheduling.k8s.io/v1
description: Used for training pods that should be non-preempting
kind: PriorityClass
metadata:
  name: preserved-pc
preemptionPolicy: Never
value: 90000000
---
apiVersion: scheduling.k8s.io/v1
description: Used for training pods that are preemptible
kind: PriorityClass
metadata:
  name: preemptible-pc
preemptionPolicy: Never
value: 80000001
```

#### Launch & Go!

Here is an example of elastic training job with Horovod Elastic:

```yaml
apiVersion: kubeflow.org/v1
kind: MPIJob
metadata:
  name: tensorflow-mnist-elastic
spec:
  slotsPerWorker: 1
  cleanPodPolicy: Running
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        spec:
          priorityClassName: preserved-pc
          containers:
          - image: horovod/horovod:0.20.0-tf2.3.0-torch1.6.0-mxnet1.5.0-py3.7-cpu
            name: mpi-launcher
            command:
            - horovodrun
            args:
            - -np
            - "2"
            - --min-np
            - "1"
            - --max-np
            - "3"
            - --host-discovery-script
            - /etc/mpi/discover_hosts.sh
            - python
            - /examples/elastic/tensorflow2_mnist_elastic.py
            resources:
              requests:
                cpu: 1
                memory: 2Gi
              limits:
                cpu: 1
                memory: 2Gi
    Worker:
      replicas: 2
      template:
        spec:
          priorityClassName: preemptible-pc
          containers:
          - image: horovod/horovod:0.20.0-tf2.3.0-torch1.6.0-mxnet1.5.0-py3.7-cpu
            name: mpi-worker
            resources:
              requests:
                cpu: 2
                memory: 4Gi
              limits:
                cpu: 2
                memory: 4Gi
```

As long as Pods for workload with high Priority are assigned with proper PriorityClass, the default scheduler is able to evict worker Pods and make room for other pending Pods. Here is a [demo](./demo.cast).

### Elastic training with HorizontalPodAutoscaler

Training with HorizontalPodAutoscaler is also supported. In this section, we present an example with Kubeflow PyTorchJob.

#### Prepare training script

PyTorch 1.10 supports elastic training. Users should manage the checkpoints as below. Please refer to [PyTorch Elastic documentation](https://pytorch.org/docs/stable/distributed.elastic.html) for details.

```python
def main():
     args = parse_args(sys.argv[1:])
     state = load_checkpoint(args.checkpoint_path)
     initialize(state)

     # torch.distributed.run ensures that this will work
     # by exporting all the env vars needed to initialize the process group
     torch.distributed.init_process_group(backend=args.backend)

     for i in range(state.epoch, state.total_num_epochs)
          for batch in iter(state.dataset)
              train(batch, state.model)

          state.epoch += 1
          save_checkpoint(state)
```

#### Run PyTorchJob with ElasticPolicy

Elastic training with PyTorchJob is now supported in master branch of [Kubeflow training operator](https://github.com/kubeflow/training-operator). The example above can be used to deploy an elastic PyTorchJob.

```yaml
apiVersion: "kubeflow.org/v1"
kind: PyTorchJob
metadata:
  name: elastic-example-imagenet
spec:
  elasticPolicy:
    rdzvBackend: c10d
    minReplicas: 1
    maxReplicas: 2
    maxRestarts: 100
  pytorchReplicaSpecs:
    Worker:
      replicas: 2
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: kubeflow/pytorch-elastic-example-imagenet:1.0.0-sigterm
              imagePullPolicy: IfNotPresent
              env:
              - name: LOGLEVEL
                value: DEBUG
              command:
                - python
                - -m
                - torch.distributed.run
                - /workspace/examples/imagenet.py
                - "--arch=resnet18"
                - "--epochs=20"
                - "--batch-size=32"
                - "--workers=0"
                - "/workspace/data/tiny-imagenet-200"
```

Please have a look at the demo video [here](https://asciinema.org/a/446932).
