#' @keywords internal
fixed_batch_process <- function(im_data, model) {
  batch_size = 512
  out = list()
  for(i in seq(0, im_data$shape[1] - 1, batch_size) + 1) {
    batch = im_data[i:(i+batch_size)]
    out <- c(out, model(batch))
  }

  return(lapply(out, function(v) torch::torch_cat(v, dim=1)))
}

#' @keywords internal
imresample <- function(img, sz) {
  torch::nnf_interpolate(img, size = sz, mode = "area")
}

#' @keywords internal
generate_bounding_box <- function(reg, probs, scale, thresh) {
  stride = 2
  cellsize = 12

  reg = reg$permute(c(2, 1, 3, 4))

  mask = probs >= thresh
  mask_inds = mask$nonzero()
  image_inds = mask_inds[, 1]
  score = probs[mask]
  reg = reg[, mask]$permute(c(2, 1))
  bb = mask_inds[, 1:N]$to(dtype = reg$dtype)$flip(2)
  q1 = ((stride * bb + 1) / scale)$floor()
  q2 = ((stride * bb + cellsize - 1 + 1) / scale)$floor()
  boundingbox = torch::torch_cat(list(q1, q2, score$unsqueeze(2), reg), dim=2)
  return(list(boundingbox = boundingbox, image_inds = image_inds))
}

#' @keywords internal
nms_r <- function(boxes, scores, threshold, method) {

  if(length(boxes) == 0)
    return(array(dim = c(0,3)))

  x1 = boxes[, 1]
  y1 = boxes[, 2]
  x2 = boxes[, 3]
  y2 = boxes[, 4]
  s = scores
  area = (x2 - x1 + 1) * (y2 - y1 + 1)

  I = order(s)
  pick = integer(length(s))
  counter = 0
  while(length(I) > 0) {
    i <- I[length(I)]
    pick[counter+1] <- i
    counter <- counter + 1
    idx = I[-length(I)]

    xx1 = pmax(x1[i], x1[idx])
    yy1 = pmax(y1[i], y1[idx])
    xx2 = pmin(x2[i], x2[idx])
    yy2 = pmin(y2[i], y2[idx])

    w = pmax(0.0, xx2 - xx1 + 1)
    h = pmax(0.0, yy2 - yy1 + 1)

    inter = w * h
    if(method == "Min") {
      o = inter / pmin(area[i], area[idx])
    } else {
      o = inter / (area[i] + area[idx] - inter)
    }
    I = I[which(o <= threshold)]
  }

  pick = pick[1:counter]
  return(pick)
}

batched_nms_r <- function(boxes, scores, idxs, threshold, method) {
  device = boxes$device
  if(boxes$numel() == 0)
    return(torch::torch_empty(0, dtype=torch::torch_int64(), device=device))
  # strategy: in order to perform NMS independently per class.
  # we add an offset to all the boxes. The offset is dependent
  # only on the class idx, and is large enough so that boxes
  # from different classes do not overlap
  max_coordinate = boxes$max()
  offsets = idxs$to(device = device, dtype = boxes$dtype) * (max_coordinate + 1)
  boxes_for_nms = boxes + offsets[, NULL]
  boxes_for_nms = torch::as_array(boxes_for_nms$to(device = "cpu"))
  scores = torch::as_array(scores$to(device = "cpu"))
  keep = nms_r(boxes_for_nms, scores, threshold, method)
  return(torch::torch_tensor(keep, dtype = torch::torch_long(), device=device))
}

#' @keywords internal
detect_face <- function(imgs, minsize, pnet, rnet, onet, threshold, factor, device) {

  imgs <- torch_ones(2,7,6,3)
  minsize <- 2
  pnet <- facenet::pnet()
  rnet <- facenet::rnet()
  onet <- facenet::onet()
  threshold <- 0.7
  factor <- 0.5
  device <- "cpu"

  if(inherits(imgs, c("array", "matrix", "torch_tensor"))) {
    if(inherits(imgs, c("array", "matrix")))
      imgs <- torch::torch_tensor(imgs, device = device)

    if(inherits(imgs, "torch_tensor"))
      imgs <- imgs$to(device = device)

    if(length(imgs$shape) == 3)
      imgs <- imgs.unsqueeze(1)

  } else {
    if(!inherits(imgs, c("list")))
      imgs = list(imgs)

    if(any(sapply(imgs, function(x) !identical(x$size(), imgs[[1]]$size()))))
      value_error("MTCNN batch processing only compatible with equal-dimension images.")

    imgs <- torch::torch_stack(imgs, 1)$to(device = device)
  }

  model_dtype <- pnet$parameters$conv1.weight$dtype
  imgs <- imgs$permute(c(1, 4, 2, 3))$to(dtype = model_dtype)

  batch_size <- imgs$shape[1]
  c(h, w) %<-% imgs$shape[3:4]
  m <- 12.0 / minsize
  minl = min(h, w)
  minl = minl * m

  # Create scale pyramid
  scale_i = m
  scales = list()
  while(minl >= 12) {
    scales <- c(scales, scale_i)
    scale_i = scale_i * factor
    minl = minl * factor
  }

  # First stage
  boxes = list()
  image_inds = list()
  scale_picks = list()

  all_i = 0
  offset = 0
  for(scale in scales) {
    im_data = imresample(imgs, c(as.integer(h * scale + 1), as.integer(w * scale + 1)))
    im_data = (im_data - 127.5) * 0.0078125
    c(reg, probs) %<-% pnet(im_data)

    c(boxes_scale, image_inds_scale) %<-% generate_bounding_box(reg, probs[, 1], scale, threshold[1])
    boxes <- c(boxes, boxes_scale)
    image_inds <- c(image_inds, image_inds_scale)

    pick = batched_nms(boxes_scale[, 1:4], boxes_scale[, 5], image_inds_scale, 0.5)
    pick <- c(pick, pick + offset)
    offset = offset + boxes_scale$shape[1]
  }

  boxes = torch::torch_cat(boxes, dim=1)
  image_inds = torch::torch_cat(image_inds, dim=1)

  scale_picks = torch::torch_cat(scale_picks, dim=1)

  # NMS within each scale + image
  c(boxes, image_inds) %<-% list(boxes[scale_picks], image_inds[scale_picks])

  # NMS within each image
  pick = batched_nms(boxes[, 1:4], boxes[, 5], image_inds, 0.7)
  c(boxes, image_inds) %<-% list(boxes[pick], image_inds[pick])

  regw = boxes[, 3] - boxes[, 1]
  regh = boxes[, 4] - boxes[, 2]
  qq1 = boxes[, 1] + boxes[, 6] * regw
  qq2 = boxes[, 2] + boxes[, 7] * regh
  qq3 = boxes[, 3] + boxes[, 8] * regw
  qq4 = boxes[, 4] + boxes[, 9] * regh
  boxes = torch::torch_stack(list(qq1, qq2, qq3, qq4, boxes[, 5]))$permute(2, 1)
  boxes = rerec(boxes)
  c(y, ey, x, ex) %<-% pad(boxes, w, h)

  # Second stage
  if(length(boxes) > 0) {
    im_data = list()
    for(k in seq_along(y)) {
      if(ey[k] > (y[k] - 1) & ex[k] > (x[k] - 1)) {
        img_k = imgs[image_inds[k], , (y[k] - 1):(ey[k]), (x[k] - 1):(ex[k])]$unsqueeze(1)
        im_data$append(imresample(img_k, c(24, 24)))
      }
    }
    im_data = torch::torch_cat(im_data, dim=1)
    im_data = (im_data - 127.5) * 0.0078125

    # This is equivalent to out = rnet(im_data) to avoid GPU out of memory.
    out = fixed_batch_process(im_data, rnet)

    out0 = out[1]$permute(2, 1)
    out1 = out[2]$permute(2, 1)
    score = out1[2, ]
    ipass = (score > threshold[2])  + 1
    boxes = torch::torch_cat(list(boxes[ipass, 1:4], score[ipass]$unsqueeze(2)), dim=2)
    image_inds = image_inds[ipass]
    mv = out0[, ipass]$permute(2, 1)

    # NMS within each image
    pick = batched_nms(boxes[, 1:4], boxes[, 5], image_inds, 0.7)
    c(boxes, image_inds, mv) %<-% list(boxes[pick], image_inds[pick], mv[pick])
    boxes = bbreg(boxes, mv)
    boxes = rerec(boxes)
  }

  # Third stage
  points = torch::torch_zeros(0, 5, 2, device=device)
  if(length(boxes) > 0) {
    c(y, ey, x, ex) %<-% pad(boxes, w, h)
    im_data = list()
    for(k in seq_along(y)) {
      if(ey[k] > (y[k] - 1) & ex[k] > (x[k] - 1)) {
        img_k = imgs[image_inds[k],  , (y[k] - 1):(ey[k]), (x[k] - 1):(ex[k])]$unsqueeze(1)
        im_data <- c(im_data, imresample(img_k, c(48, 48)))
      }
    }
    im_data = torch::torch_cat(im_data, dim=1)
    im_data = (im_data - 127.5) * 0.0078125

    # This is equivalent to out = onet(im_data) to avoid GPU out of memory.
    out = fixed_batch_process(im_data, onet)

    out0 = out[1]$permute(2, 1)
    out1 = out[2]$permute(2, 1)
    out2 = out[3]$permute(2, 1)
    score = out2[2, ]
    points = out1
    ipass = (score > threshold[2]) + 1
    points = points[, ipass]
    boxes = torch::torch_cat(list(boxes[ipass, 1:4], score[ipass]$unsqueeze(2)), dim=2)
    image_inds = image_inds[ipass]
    mv = out0[, ipass]$permute(2, 1)

    w_i = boxes[, 3] - boxes[, 1] + 1
    h_i = boxes[, 4] - boxes[, 2] + 1
    points_x = w_i$`repeat`(c(5, 1)) * points[1:5, ] + boxes[, 1]$`repeat`(c(5, 1)) - 1
    points_y = h_i$`repeat`(c(5, 1)) * points[6:10, ] + boxes[, 2]$`repeat`(c(5, 1)) - 1
    points = torch::torch_stack(c(points_x, points_y))$permute(3, 2, 1)
    boxes = bbreg(boxes, mv)

    # NMS within each image using "Min" strategy
    # pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
    pick = batched_nms_r(boxes[, 1:4], boxes[, 4], image_inds, 0.7, 'Min')
    c(boxes, image_inds, points) %<-% list(boxes[pick], image_inds[pick], points[pick])
  }
  boxes = torch::as_array(boxes$to(device = "cpu"))
  points = torch::as_array(points$to(device = "cpu"))

  image_inds = image_inds$to(device = "cpu")

  batch_boxes = list()
  batch_points = list()
  for(b_i in seq_len(batch_size)){
    b_i_inds = np.where(image_inds == b_i)
    batch_boxes <- c(batch_boxes, boxes[b_i_inds]$copy())
    batch_points <- c(batch_points, points[b_i_inds]$copy())
  }

  c(batch_boxes, batch_points) %<-% list(np.array(batch_boxes), np.array(batch_points))

  return(batch_boxes, batch_points)
}

#' @keywords internal
bbreg <- function(boundingbox, reg) {
  if(reg$shape[2] == 1)
    reg = torch::torch_reshape(reg, c(reg$shape[3], reg$shape[4]))

  w = boundingbox[, 3] - boundingbox[, 1] + 1
  h = boundingbox[, 4] - boundingbox[, 2] + 1
  b1 = boundingbox[, 1] + reg[, 1] * w
  b2 = boundingbox[, 2] + reg[, 2] * h
  b3 = boundingbox[, 3] + reg[, 3] * w
  b4 = boundingbox[, 4] + reg[, 4] * h
  boundingbox[, 1:4] = torch.stack(list(b1, b2, b3, b4))$permute(2, 1)

  return(boundingbox)
}

#' @keywords internal
pad <- function(boxes, w, h) {
  boxes = torch::as_array(boxes$trunc()$to(dtype = torch::torch_int(), device = "cpu"))
  x <- boxes[, 1]
  y <- boxes[, 2]
  ex <- boxes[, 3]
  ey <- boxes[, 4]

  x[x < 1] <- 1
  y[y < 1] <- 1
  ex[ex > w] <- w
  ey[ey > h] <- h

  return(list(y, ey, x, ex))
}

#' @keywords internal
rerec <- function(bboxA) {

  h = bboxA[, 4] - bboxA[, 2]
  w = bboxA[, 3] - bboxA[, 1]

  l = torch$max(w, h)
  bboxA[, 1] = bboxA[, 1] + w * 0.5 - l * 0.5
  bboxA[, 2] = bboxA[, 2] + h * 0.5 - l * 0.5
  bboxA[, 3:4] = bboxA[, 1:2] + l$`repeat`(c(2, 1))$permute(2, 1)

  return(bboxA)
}

#' @keywords internal
# crop_resize <- function(img, box, image_size) {
#
#   img = matrix(runif(8*6), 8,6)
#   box = c(1,1,3,4)
#   image_size = 5
#   if(is.array(img)) {
#     img = img[box[2]:box[4], box[1]:box[3]]
#     out = magick::image_resize(
#       img,
#       geometry = paste0(image_size, "x", image_size)
#     )
#   } else if(isinstance(img, torch) {
#     img = img[box[1]:box[3], box[0]:box[2]]
#     out = imresample(
#       img.permute(2, 0, 1).unsqueeze(0).float(),
#       (image_size, image_size)
#     ).byte().squeeze(0).permute(1, 2, 0)
#   } else {
#     out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
#   }
#   return(out)
# }
