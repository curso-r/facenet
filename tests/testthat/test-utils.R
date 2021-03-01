test_that("imresample", {
  imgs <- torch::torch_rand(4,3,2,3)
  expect_tensor(imresample(imgs, 3))
})
