test_that("inception_resnet_v1", {

  expect_no_error(facenet:::nn_basic_conv2d(3, 3, kernel_size=3, stride=1, padding=1)(torch_rand(2, 3, 4, 5)))
  expect_no_error(facenet:::nn_block35()(torch_rand(2, 256, 4, 5)))
  expect_no_error(facenet:::nn_block17()(torch_rand(2, 896, 4, 5)))
  expect_no_error(facenet:::nn_block8()(torch_rand(2, 1792, 4, 5)))
  expect_no_error(facenet:::nn_mixed_6a()(torch_rand(2, 256, 4, 5)))
  expect_no_error(facenet:::nn_mixed_7a()(torch_rand(2, 896, 4, 5)))
  expect_no_error(facenet:::nn_inception_resnet_v1()(torch_rand(2, 3, 75, 75)))
  # expect_no_error(facenet:::nn_inception_resnet_v1(pretrained = "casia-webface")(torch_rand(2, 3, 75, 75)))
  # expect_no_error(facenet:::nn_inception_resnet_v1(pretrained = "vggface2")(torch_rand(2, 3, 75, 75)))

})
