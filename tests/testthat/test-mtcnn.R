test_that("p o r net", {
  pnet_1 <- pnet()
  pnet_2 <- pnet()
  pnet_3 <- pnet(FALSE)
  expect_true(sum(torch::as_array(pnet_1$parameters$conv1.weight) - torch::as_array(pnet_2$parameters$conv1.weight)) == 0)
  expect_true(sum(torch::as_array(pnet_1$parameters$conv1.weight) != torch::as_array(pnet_3$parameters$conv1.weight)) != 0)

  rnet_1 <- rnet()
  rnet_2 <- rnet()
  rnet_3 <- rnet(FALSE)
  expect_true(sum(torch::as_array(rnet_1$parameters$conv1.weight) - torch::as_array(rnet_2$parameters$conv1.weight)) == 0)
  expect_true(sum(torch::as_array(rnet_1$parameters$conv1.weight) != torch::as_array(rnet_3$parameters$conv1.weight)) != 0)

  onet_1 <- onet()
  onet_2 <- onet()
  onet_3 <- onet(FALSE)
  expect_true(sum(torch::as_array(onet_1$parameters$conv1.weight) - torch::as_array(onet_2$parameters$conv1.weight)) == 0)
  expect_true(sum(torch::as_array(onet_1$parameters$conv1.weight) != torch::as_array(onet_3$parameters$conv1.weight)) != 0)

})
