boxes = matrix(c(0,0,1,1,
                 0,0,2,2,
                 3,3,4,4), 3,4, byrow = TRUE)

test_that("nms_r", {
  x <-nms_r(
    boxes,
    scores = c(0.9, 0.9, 0.9),
    threshold = 0.7,
    method = "Min"
  )
  expect_equal(x, c(3,2))
})

test_that("batched_nms_r", {
  x <- batched_nms_r(
    boxes =  torch_tensor(boxes),
    scores = torch_tensor(c(0.9, 0.9, 0.9)),
    idxs = torch_tensor(c(1,2,1)),
    threshold = 0.7,
    method = "Min"
  )
  expect_equal_to_r(x, c(3,2,1))
})
