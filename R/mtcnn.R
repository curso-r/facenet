#' MTCNN PNet.
#'
#' @param pretrained {bool} Whether or not to load saved pretrained weights  (default: {TRUE})
#'
#' @export
nn_pnet <- torch::nn_module(
  "MTCNN PNet",
  initialize = function(pretrained = TRUE) {
    self$conv1 = torch::nn_conv2d(3, 10, kernel_size=3)
    self$prelu1 = torch::nn_prelu(10)
    self$pool1 = torch::nn_max_pool2d(2, 2, ceil_mode=TRUE)
    self$conv2 = torch::nn_conv2d(10, 16, kernel_size=3)
    self$prelu2 = torch::nn_prelu(16)
    self$conv3 = torch::nn_conv2d(16, 32, kernel_size=3)
    self$prelu3 = torch::nn_prelu(32)
    self$conv4_1 = torch::nn_conv2d(32, 2, kernel_size=1)
    self$softmax4_1 = torch::nn_softmax(dim=2)
    self$conv4_2 = torch::nn_conv2d(32, 4, kernel_size=1)

    self$training = FALSE

    if(pretrained) {
      state_dict_path = system.file("pnet_reload.pt", package = "facenet")
      state_dict = torch::load_state_dict(state_dict_path)
      self$load_state_dict(state_dict)
    }
  },

  forward = function(x) {
    x = x %>%
      self$conv1() %>%
      self$prelu1() %>%
      self$pool1() %>%
      self$conv2() %>%
      self$prelu2() %>%
      self$conv3() %>%
      self$prelu3()

    a = x %>%
      self$conv4_1() %>%
      self$softmax4_1()

    b = self$conv4_2(x)

    return(list(b, a))
  }
)


#' MTCNN RNet.
#'
#' @param pretrained {bool} Whether or not to load saved pretrained weights  (default: {TRUE})
#'
#' @export
nn_rnet <- torch::nn_module(
  "MTCNN RNet",
  initialize = function(pretrained = TRUE) {

    self$conv1 = torch::nn_conv2d(3, 28, kernel_size=3)
    self$prelu1 = torch::nn_prelu(28)
    self$pool1 = torch::nn_max_pool2d(3, 2, ceil_mode=TRUE)
    self$conv2 = torch::nn_conv2d(28, 48, kernel_size=3)
    self$prelu2 = torch::nn_prelu(48)
    self$pool2 = torch::nn_max_pool2d(3, 2, ceil_mode=TRUE)
    self$conv3 = torch::nn_conv2d(48, 64, kernel_size=2)
    self$prelu3 = torch::nn_prelu(64)
    self$dense4 = torch::nn_linear(576, 128)
    self$prelu4 = torch::nn_prelu(128)
    self$dense5_1 = torch::nn_linear(128, 2)
    self$softmax5_1 = torch::nn_softmax(dim=2)
    self$dense5_2 = torch::nn_linear(128, 4)

    self$training = FALSE

    if(pretrained) {
      state_dict_path = system.file("rnet_reload.pt", package = "facenet")
      state_dict = torch::load_state_dict(state_dict_path)
      self$load_state_dict(state_dict)
    }
  },

  forward = function( x) {
    x = x %>%
      self$conv1() %>%
      self$prelu1() %>%
      self$pool1() %>%
      self$conv2() %>%
      self$prelu2() %>%
      self$pool2() %>%
      self$conv3() %>%
      self$prelu3()

    x = x$permute(c(1, 4, 3, 2))$contiguous()
    x = self$dense4(x$view(c(x$shape[1], -1)))
    x = self$prelu4(x)

    a = x %>%
      self$dense5_1() %>%
      self$softmax5_1()

    b = self$dense5_2(x)

    return(list(b, a))
  }
)

#' MTCNN ONet.
#'
#' @param pretrained {bool} Whether or not to load saved pretrained weights  (default: {TRUE})
#'
#' @export
nn_onet <- torch::nn_module(
  "MTCNN ONet",
  initialize = function( pretrained=TRUE) {

    self$conv1 = torch::nn_conv2d(3, 32, kernel_size=3)
    self$prelu1 = torch::nn_prelu(32)
    self$pool1 = torch::nn_max_pool2d(3, 2, ceil_mode=TRUE)
    self$conv2 = torch::nn_conv2d(32, 64, kernel_size=3)
    self$prelu2 = torch::nn_prelu(64)
    self$pool2 = torch::nn_max_pool2d(3, 2, ceil_mode=TRUE)
    self$conv3 = torch::nn_conv2d(64, 64, kernel_size=3)
    self$prelu3 = torch::nn_prelu(64)
    self$pool3 = torch::nn_max_pool2d(2, 2, ceil_mode=TRUE)
    self$conv4 = torch::nn_conv2d(64, 128, kernel_size=2)
    self$prelu4 = torch::nn_prelu(128)
    self$dense5 = torch::nn_linear(1152, 256)
    self$prelu5 = torch::nn_prelu(256)
    self$dense6_1 = torch::nn_linear(256, 2)
    self$softmax6_1 = torch::nn_softmax(dim=2)
    self$dense6_2 = torch::nn_linear(256, 4)
    self$dense6_3 = torch::nn_linear(256, 10)

    self$training = FALSE

    if(pretrained) {
      state_dict_path = system.file("onet_reload.pt", package = "facenet")
      state_dict = torch::load_state_dict(state_dict_path)
      self$load_state_dict(state_dict)
    }
  },

  forward = function( x) {
    x <- x %>%
      self$conv1() %>%
      self$prelu1() %>%
      self$pool1() %>%
      self$conv2() %>%
      self$prelu2() %>%
      self$pool2() %>%
      self$conv3() %>%
      self$prelu3() %>%
      self$pool3() %>%
      self$conv4() %>%
      self$prelu4()

    x = x$permute(c(1, 4, 3, 2))$contiguous()
    x = self$dense5(x$view(c(x$shape[1], -1)))
    x = self$prelu5(x)

    a = x %>%
      self$dense6_1() %>%
      self$softmax6_1()

    b = self$dense6_2(x)

    c = self$dense6_3(x)

    return(list(b, c, a))
  }

)

fixed_image_standardization <- function(image_tensor) {
  (image_tensor - 127.5) / 128.0
}

prewhiten <- function(x) {
  mean = x$mean()
  std = x$std()
  std_adj = std$clamp(min = 1.0/(as.numeric(x$numel())^0.5))
  y = (x - mean) / std_adj
  return(y)
}




#' MTCNN face detection module.
#'
#' @param - numpy.ndarray  (uint8) representing either a single image (3D) or a batch of images (4D).
#'    Cropped faces can optionally be saved to file
#'    also.
#'
#'    Keyword Arguments:
#' @param image_size {int} -- Output image size in pixels. The image will be square.  (default: {160})
#'        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image.
#'            Note that the application of the margin differs slightly from the davidsandberg/facenet
#'            repo, which applies the margin to the original image before resizing, making the margin
#' @param dependent on the original image size  (this is a bug in davidsandberg/facenet).
#' @param   (default: {0})
#' @param min_face_size {int} -- Minimum face size to search for.  (default: {20})
#' @param thresholds {list} -- MTCNN face detection thresholds  (default: {[0.6, 0.7, 0.7]})
#' @param factor {float} -- Factor used to create a scaling pyramid of face sizes.  (default: {0.709})
#'        post_process {bool} -- Whether or not to post process images tensors before returning.
#' @param   (default: {TRUE})
#'        select_largest {bool} -- If TRUE, if multiple faces are detected, the largest is returned.
#'            If FALSE, the face with the highest detection probability is returned.
#' @param   (default: {TRUE})
#'        selection_method {string} -- Which heuristic to use for selection. Default NULL. If
#'            specified, will override select_largest:
#'                    "probability": highest probability selected
#'                    "largest": largest box selected
#'                    "largest_over_theshold": largest box over a certain probability selected
#'                    "center_weighted_size": box size minus weighted squared offset from image center
#' @param   (default: {NULL})
#'        keep_all {bool} -- If TRUE, all detected faces are returned, in the order dictated by the
#'            select_largest parameter. If a save_path is specified, the first face is saved to that
#'            path and the remaining faces are saved to <save_path>1, <save_path>2 etc.
#' @param   (default: {FALSE})
#'        device {torch.device} -- The device on which to run neural net passes. Image tensors and
#' @param models are copied to this device before running forward passes.  (default: {NULL})
#'
#'
#'
#'
#' @section Detect method
#'
#'#' Detect all faces in PIL image and return bounding boxes and optional facial landmarks.
#'        This method is used by the forward method and is also useful for face detection tasks
#' @param that require lower-level handling of bounding boxes and facial landmarks  (e.g., face
#'        tracking). The functionality of the forward function can be emulated by using this method
#' @param followed by the extract_face () function.
#'
#'        Arguments:
#'            img {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, torch.Tensor, or list.
#'        Keyword Arguments:
#'            landmarks {bool} -- Whether to return facial landmarks in addition to bounding boxes.
#' @param   (default: {FALSE})
#'
#' @return tuple(numpy.ndarray, list) -- For N detected faces, a tuple containing an
#'                Nx4 array of bounding boxes and a length N list of detection probabilities.
#'                Returned boxes will be sorted in descending order by detection probability if
#'                self$select_largest=FALSE, otherwise the largest face will be returned first.
#'                If `img` is a list of images, the items returned have an extra dimension
#' @param   (batch) as the first dimension. Optionally, a third item, the facial landmarks,
#'                are returned if `landmarks=TRUE`.
#'        Example:
#'        >>> from PIL import Image, ImageDraw
#'        >>> from facenet_pytorch import MTCNN, extract_face
#' @param >>> mtcnn = MTCNN (keep_all=TRUE)
#' @param >>> boxes, probs, points = mtcnn.detect (img, landmarks=TRUE)
#'        >>> # Draw boxes and save faces
#' @param >>> img_draw = img.copy ()
#' @param >>> draw = ImageDraw.Draw (img_draw)
#' @param >>> for i,  (box, point) in enumerate(zip(boxes, points)):
#' @param ...     draw.rectangle (box.tolist(), width=5)
#'        ...     for p in point:
#' @param ...         draw.rectangle ((p - 10).tolist() + (p + 10).tolist(), width=10)
#' @param ...     extract_face (img, box, save_path='detected_face_{}.png'.format(i))
#' @param >>> img_draw.save ('annotated_faces.png')
#'
#'
#' @section Forward method
#'
#' Run MTCNN face detection on a PIL image or numpy array. This method performs both
#' detection and extraction of faces, returning tensors representing detected faces rather
#' than the bounding boxes. To access bounding boxes, see the MTCNN.detect () method below.
#'
#'        Arguments:
#'            img {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, torch.Tensor, or list.
#'
#'        Keyword Arguments:
#'            save_path {str} -- An optional save path for the cropped image. Note that when
#'                self$post_process=TRUE, although the returned tensor is post processed, the saved
#'                face image is not, so it is a TRUE representation of the face in the input image.
#'                If `img` is a list of images, `save_path` should be a list of equal length.
#'
#' @param   (default: {NULL})
#'            return_prob {bool} -- Whether or not to return the detection probability.
#' @param   (default: {FALSE})
#'
#' @return Union[torch.Tensor, tuple(torch.tensor, float)] -- If detected, cropped image of a face
#'                with dimensions 3 x image_size x image_size. Optionally, the probability that a
#'                face was detected. If self$keep_all is TRUE, n detected faces are returned in an
#'                n x 3 x image_size x image_size tensor with an optional list of detection
#' @param probabilities. If `img` is a list of images, the item (s) returned have an extra
#' @param dimension  (batch) as the first dimension.
#'        Example:
#'        >>> from facenet_pytorch import MTCNN
#' @param >>> mtcnn = MTCNN ()
#' @param >>> face_tensor, prob = mtcnn (img, save_path='face.png', return_prob=TRUE)
#'
#'
#' @return
#' This class loads pretrained P-, R-, and O-nets and returns images cropped to include the face
#'    only, given raw input images of one of the following types:
#'    - magick image or list of magick images
#'
#' @export
# nn_mtcnn <- torch::nn_module(
#   "MTCNN",
#   initialize = function(
#     image_size = 160, margin = 0, min_face_size = 20,
#     thresholds = c(0.6, 0.7, 0.7), factor = 0.709, post_process = TRUE,
#     select_largest = TRUE, selection_method = NULL, keep_all = FALSE, device = NULL
#   ) {
#     self$image_size = image_size
#     self$margin = margin
#     self$min_face_size = min_face_size
#     self$thresholds = thresholds
#     self$factor = factor
#     self$post_process = post_process
#     self$select_largest = select_largest
#     self$keep_all = keep_all
#     self$selection_method = selection_method
#
#     self$pnet = nn_pnet()
#     self$rnet = nn_rnet()
#     self$onet = nn_onet()
#
#     self$device = torch::torch_device('cpu')
#     if(!is.null(device)) {
#       self$device = device
#       self$to(device = device)
#     }
#
#     if(!self$selection_method) {
#       self$selection_method <- if(self$select_largest) 'largest' else 'probability'
#     }
#   },
#
#   forward = function(img, save_path = NULL, return_prob = FALSE) {
#
#     # Detect faces
#     c(batch_boxes, batch_probs, batch_points) %<-% self$detect(img, landmarks = TRUE)
#     # Select faces
#     if(!self$keep_all) {
#       c(batch_boxes, batch_probs, batch_points) %<-% self$select_boxes(
#         batch_boxes, batch_probs, batch_points, img, method=self$selection_method
#       )
#     }
#     # Extract faces
#     faces = self$extract(img, batch_boxes, save_path)
#
#     if(return_prob) {
#       return(list(faces = faces, batch_probs = batch_probs))
#     } else {
#       return(list(faces = faces))
#     }
#   },
#
#   detect = function(img, landmarks=FALSE) {
#     torch::with_no_grad({
#       batch_boxes_batch_points <- detect_face(
#         img, self$min_face_size,
#         self$pnet, self$rnet, self$onet,
#         self$thresholds, self$factor,
#         self$device
#       )
#     })
#
#
#       c(boxes, probs, points) %<-% list(list(), list(), list())
#
#       for(box, point in batch_boxes_batch_points) {
#         box = np.array(box)
#         point = np.array(point)
#         if(len(box) == 0) {
#           boxes.append(NULL)
#           probs.append([NULL])
#           points.append(NULL)
#         } else if(self$select_largest) {
#           box_order = np.argsort((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]))[::-1]
#           box = box[box_order]
#           point = point[box_order]
#           boxes.append(box[:, :4])
#           probs.append(box[:, 4])
#           points.append(point)
#         } else {
#           boxes.append(box[:, :4])
#           probs.append(box[:, 4])
#           points.append(point)
#           boxes = np.array(boxes)
#           probs = np.array(probs)
#           points = np.array(points)
#         }
#       }
#
#
#         if((
#           not isinstance(img, (list, tuple)) and
#           not (isinstance(img, np.ndarray) and len(img.shape) == 4) and
#           not (isinstance(img, torch) {:torch_Tensor) and len(img.shape) == 4)
#         ):
#           boxes = boxes[0]
#         probs = probs[0]
#         points = points[0]
#
#         if(landmarks) {
#           return(boxes, probs, points)
#
#           return boxes, probs
#         }
#           }
# )
