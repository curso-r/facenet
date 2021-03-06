% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/inception_resnet_v1.R
\name{nn_inception_resnet_v1}
\alias{nn_inception_resnet_v1}
\title{Inception Resnet V1}
\usage{
nn_inception_resnet_v1(
  pretrained = NULL,
  classify = FALSE,
  num_classes = NULL,
  dropout_prob = 0.6,
  device = NULL
)
}
\arguments{
\item{pretrained}{{str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'. (default: {NULL})}

\item{classify}{{bool} -- Whether the model should output classification probabilities or feature
embeddings.  (default: {FALSE})}

\item{num_classes}{{int} -- Number of output classes. If 'pretrained' is set and num_classes not
equal to that used for the pretrained model, the final linear layer will be randomly
initialized.  (default: {NULL})}

\item{dropout_prob}{{float} -- Dropout probability.  (default: {0.6})}

\item{device}{(torch_device) The device. See \link[torch:torch_device]{torch::torch_device}}
}
\value{
torch.tensor -- Batch of embedding vectors or multinomial logits.
}
\description{
Inception Resnet V1 model with optional loading of pretrained weights.
}
\details{
Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
requested and cached in the torch cache. Subsequent instantiations use the cache rather than
redownloading.
}
\section{forward method}{


Calculate embeddings or logits given a batch of input image tensors.
x {torch.tensor} -- Batch of image tensors representing faces.
}

