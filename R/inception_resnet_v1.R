
#' @keywords internal
nn_basic_conv2d <- torch::nn_module(
  "BasicConv2d",
  initialize = function( in_planes, out_planes, kernel_size, stride, padding=0) {

    self$conv = torch::nn_conv2d(
      in_planes, out_planes,
      kernel_size=kernel_size, stride=stride,
      padding=padding, bias=FALSE
    ) # verify bias FALSE
    self$bn = torch::nn_batch_norm2d(
      out_planes,
      eps=0.001, # value found in tensorflow
      momentum=0.1, # default pytorch value
      affine=TRUE
    )
    self$relu = torch::nn_relu(inplace=FALSE)
  },
  forward = function(x) {
    x = self$conv(x)
    x = self$bn(x)
    x = self$relu(x)
    return(x)
  }
)

nn_block35 <- torch::nn_module(
  "Block35",
  initialize = function(scale=1.0) {

    self$scale = scale

    self$branch0 = nn_basic_conv2d(256, 32, kernel_size=1, stride=1)

    self$branch1 = torch::nn_sequential(
      nn_basic_conv2d(256, 32, kernel_size=1, stride=1),
      nn_basic_conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    )

    self$branch2 = torch::nn_sequential(
      nn_basic_conv2d(256, 32, kernel_size=1, stride=1),
      nn_basic_conv2d(32, 32, kernel_size=3, stride=1, padding=1),
      nn_basic_conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    )

    self$conv2d = torch::nn_conv2d(96, 256, kernel_size=1, stride=1)
    self$relu = torch::nn_relu(inplace=FALSE)
  },

  forward = function(x) {
    x0 = self$branch0(x)
    x1 = self$branch1(x)
    x2 = self$branch2(x)
    out = torch::torch_cat(list(x0, x1, x2), 2)
    out = self$conv2d(out)
    out = out * self$scale + x
    out = self$relu(out)
    return(out)
  }
)

nn_block17 <- torch::nn_module(
  "Block17",
  initialize = function( scale=1.0) {
    self$scale = scale
    self$branch0 = nn_basic_conv2d(896, 128, kernel_size=1, stride=1)
    self$branch1 = torch::nn_sequential(
      nn_basic_conv2d(896, 128, kernel_size=1, stride=1),
      nn_basic_conv2d(128, 128, kernel_size=c(1,7), stride=1, padding=c(0,3)),
      nn_basic_conv2d(128, 128, kernel_size=c(7,1), stride=1, padding=c(3,0))
    )
    self$conv2d = torch::nn_conv2d(256, 896, kernel_size=1, stride=1)
    self$relu = torch::nn_relu(inplace=FALSE)
  },
  forward = function(x) {
    x0 = self$branch0(x)
    x1 = self$branch1(x)
    out = torch::torch_cat(list(x0, x1), 2)
    out = self$conv2d(out)
    out = out * self$scale + x
    out = self$relu(out)
    return(out)
  }
)

nn_block8 <- torch::nn_module(
  "Block8",
  initialize = function(scale=1.0, noReLU=FALSE) {
    self$scale = scale
    self$noReLU = noReLU
    self$branch0 = nn_basic_conv2d(1792, 192, kernel_size=1, stride=1)
    self$branch1 = torch::nn_sequential(
      nn_basic_conv2d(1792, 192, kernel_size=1, stride=1),
      nn_basic_conv2d(192, 192, kernel_size=c(1,3), stride=1, padding=c(0,1)),
      nn_basic_conv2d(192, 192, kernel_size=c(3,1), stride=1, padding=c(1,0))
    )
    self$conv2d = torch::nn_conv2d(384, 1792, kernel_size=1, stride=1)
    if(!self$noReLU)
      self$relu = torch::nn_relu(inplace=FALSE)
  },
  forward = function(x) {
    x0 = self$branch0(x)
    x1 = self$branch1(x)
    out = torch::torch_cat(list(x0, x1), 2)
    out = self$conv2d(out)
    out = out * self$scale + x
    if(!self$noReLU)
      out = self$relu(out)
    return(out)
  }
)

nn_ixed_6a <- torch::nn_module(
  "Mixed_6a",
  initialize = function(self) {
    self$branch0 = nn_basic_conv2d(256, 384, kernel_size=3, stride=2)

    self$branch1 = torch::nn_sequential(
      nn_basic_conv2d(256, 192, kernel_size=1, stride=1),
      nn_basic_conv2d(192, 192, kernel_size=3, stride=1, padding=1),
      nn_basic_conv2d(192, 256, kernel_size=3, stride=2)
    )

    self$branch2 = torch::nn_max_pool2d(3, stride=2)
  },
  forward = function(x) {
    x0 = self$branch0(x)
    x1 = self$branch1(x)
    x2 = self$branch2(x)
    out = torch::torch_cat(list(x0, x1, x2), 2)
    return(out)
  }
)

nn_mixed_7a <- torch::nn_module(
  "Mixed_7a",
  initialize = function(self) {
    self$branch0 = torch::nn_sequential(
      nn_basic_conv2d(896, 256, kernel_size=1, stride=1),
      nn_basic_conv2d(256, 384, kernel_size=3, stride=2)
    )

    self$branch1 = torch::nn_sequential(
      nn_basic_conv2d(896, 256, kernel_size=1, stride=1),
      nn_basic_conv2d(256, 256, kernel_size=3, stride=2)
    )

    self$branch2 = torch::nn_sequential(
      nn_basic_conv2d(896, 256, kernel_size=1, stride=1),
      nn_basic_conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn_basic_conv2d(256, 256, kernel_size=3, stride=2)
    )

    self$branch3 = torch::nn_max_pool2d(3, stride=2)
  },
  forward = function(x) {
    x0 = self$branch0(x)
    x1 = self$branch1(x)
    x2 = self$branch2(x)
    x3 = self$branch3(x)
    out = torch::torch_cat(list(x0, x1, x2, x3), 2)
    return(out)
  }
)

#' Inception Resnet V1
#'
#' Inception Resnet V1 model with optional loading of pretrained weights.
#'
#' @param pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'. (default: {NULL})
#' @param classify {bool} -- Whether the model should output classification probabilities or feature
#' embeddings.  (default: {FALSE})
#' @param num_classes {int} -- Number of output classes. If 'pretrained' is set and num_classes not
#'  equal to that used for the pretrained model, the final linear layer will be randomly
#'  initialized.  (default: {NULL})
#' @param dropout_prob {float} -- Dropout probability.  (default: {0.6})
#' @param device (torch_device) The device. See [torch::torch_device]
#'
#' @section forward method:
#'
#' Calculate embeddings or logits given a batch of input image tensors.
#'  x {torch.tensor} -- Batch of image tensors representing faces.
#'
#' @return torch.tensor -- Batch of embedding vectors or multinomial logits.
#'
#' @details
#' Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
#'    datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
#'    requested and cached in the torch cache. Subsequent instantiations use the cache rather than
#'    redownloading.
#'
#' @export
nn_inception_resnet_v1 <- torch::nn_module(
  "InceptionResnetV1",
  initialize = function(pretrained=NULL, classify=FALSE, num_classes=NULL, dropout_prob=0.6, device=NULL) {
    # Set simple attributes
    self$pretrained = pretrained
    self$classify = classify
    self$num_classes = num_classes

    if(pretrained == 'vggface2') {
      tmp_classes = 8631
    } else if(pretrained == 'casia-webface') {
      tmp_classes = 10575
    } else if(is.null(pretrained) & self$classify & is.null(self$num_classes)) {
      value_error('If "pretrained" is not specified and "classify" is TRUE, "num_classes" must be specified')
    }


    # Define layers
    self$conv2d_1a = nn_basic_conv2d(3, 32, kernel_size=3, stride=2)
    self$conv2d_2a = nn_basic_conv2d(32, 32, kernel_size=3, stride=1)
    self$conv2d_2b = nn_basic_conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self$maxpool_3a = torch::nn_max_pool2d(3, stride=2)
    self$conv2d_3b = nn_basic_conv2d(64, 80, kernel_size=1, stride=1)
    self$conv2d_4a = nn_basic_conv2d(80, 192, kernel_size=3, stride=1)
    self$conv2d_4b = nn_basic_conv2d(192, 256, kernel_size=3, stride=2)
    self$repeat_1 = torch::nn_sequential(
      nn_block35(scale=0.17),
      nn_block35(scale=0.17),
      nn_block35(scale=0.17),
      nn_block35(scale=0.17),
      nn_block35(scale=0.17),
    )
    self$mixed_6a = Mixed_6a()
    self$repeat_2 = torch::nn_sequential(
      nn_block17(scale=0.10),
      nn_block17(scale=0.10),
      nn_block17(scale=0.10),
      nn_block17(scale=0.10),
      nn_block17(scale=0.10),
      nn_block17(scale=0.10),
      nn_block17(scale=0.10),
      nn_block17(scale=0.10),
      nn_block17(scale=0.10),
      nn_block17(scale=0.10),
    )
    self$mixed_7a = Mixed_7a()
    self$repeat_3 = torch::nn_sequential(
      nn_block8(scale=0.20),
      nn_block8(scale=0.20),
      nn_block8(scale=0.20),
      nn_block8(scale=0.20),
      nn_block8(scale=0.20),
    )
    self$block8 = nn_block8(noReLU=TRUE)
    self$avgpool_1a = nn.AdaptiveAvgPool2d(1)
    self$dropout = torch::nn_dropout(dropout_prob)
    self$last_linear = torch::nn_linear(1792, 512, bias=FALSE)
    self$last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=TRUE)

    if(!is.null(pretrained)) {
      self$logits = torch::nn_linear(512, tmp_classes)
      load_weights(self, pretrained)
    }

    if(self$classify & !is.null(self$num_classes))
      self$logits = torch::nn_linear(512, self$num_classes)

    self$device = torch::torch_device('cpu')
    if(!is.null(device)) {
      self$device = device
      self$to(device = device)
    }
  },

  forward = function(x) {
    x = self$conv2d_1a(x)
    x = self$conv2d_2a(x)
    x = self$conv2d_2b(x)
    x = self$maxpool_3a(x)
    x = self$conv2d_3b(x)
    x = self$conv2d_4a(x)
    x = self$conv2d_4b(x)
    x = self$repeat_1(x)
    x = self$mixed_6a(x)
    x = self$repeat_2(x)
    x = self$mixed_7a(x)
    x = self$repeat_3(x)
    x = self$block8(x)
    x = self$avgpool_1a(x)
    x = self$dropout(x)
    x = self$last_linear(x$view(c(x$shape[1], -1)))
    x = self$last_bn(x)
    if(self$classify) {
      x = self$logits(x)
    } else {
      x = torch::nnf_normalize(x, p=2, dim=2)
    }
    return(x)
  }
)


#' Load Pretrained Weights
#'
#' Download pretrained state_dict and load into model.
#'
#'
#' @param mdl {torch nn_module} -- torch model.
#' @param name {str} -- Must be either 'vggface2' or 'casia-webface'. Name of dataset that was used to generate
#' pretrained state_dict.
#' @param cache_dir (NULL or str) dir to cache weights. If NULL, disable cache. Defaul: '.' (working directory)
#'
#' @export
load_weights <- function(mdl, name, cache_dir = ".") {
  if(name == 'vggface2') {
    path = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt'
  } else if(name == 'casia-webface') {
    path = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt'
  } else {
    value_error('Pretrained models only exist for "vggface2" and "casia-webface"')
  }

  model_dir <- file.path(cache_dir, paste0('cached_',name,'_weigths'))
  if(!dir.exists(model_dir))
    dir.create(model_dir)

  cached_file <- file.path(model_dir, basename(path))
  if(!file.exists(cached_file))
    download_url_to_file(path, cached_file)

  state_dict <- torch::load_state_dict(cached_file)
  mdl$load_state_dict(state_dict)
}



